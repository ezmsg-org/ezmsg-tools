import pygame
import pygame.locals
import typer

from ezmsg.vis.pygame.dag import VisDAG
from ezmsg.vis.pygame.timeseries import Sweep
from ezmsg.vis.proc import EZProcManager
from ezmsg.vis.mirror import EZShmMirror


SHMEM_NAME = "ezmsg-vis-monitor"
GRAPH_IP = "127.0.0.1"
GRAPH_PORT = 25978


def monitor(
    graph_ip: str = GRAPH_IP, graph_port: int = GRAPH_PORT, shmem_name: str = SHMEM_NAME
):
    pygame.init()

    # Screen
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    screen_width, screen_height = screen.get_size()
    screen = pygame.display.set_mode(
        (screen_width, screen_height), pygame.locals.RESIZABLE
    )
    screen.fill((0, 0, 0))  # Fill the screen with black

    # Interactive ezmsg graph. Its purpose is to show the graph (w/ scrolling)
    #  and get the name of the node that was clicked on, and we want to visualize.
    dag = VisDAG(screen_height=screen_height, graph_ip=graph_ip, graph_port=graph_port)

    # Data Plotter. Puts a surface on the screen, accepts data chunks, plots 2D lines
    #  with some basic auto-scaling.
    sweep = Sweep(
        (screen_width - dag.size[0], screen_height), tl_offset=(dag.size[0], 0)
    )

    # ezmsg process manager -- process runs ezmsg context to attach a node to running pipeline.
    #  We don't have the name of the target node yet so the manager does not start the proc yet.
    #  If you have a pre-existing pipeline with ShMemCircBuff already in it then this is not needed.
    ez_proc_man = EZProcManager(
        graph_ip=graph_ip, graph_port=graph_port, shmem_name=shmem_name
    )

    # Local object that mirrors the shared memory in a ShMemCircBuff node in a running ezmsg pipeline.
    #  Calls to ez_mirror's methods won't do anything until the ezmsg pipeline is running.
    ez_mirror = EZShmMirror(shmem_name=shmem_name)

    running = True
    while running:
        new_node_path = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                # Keyboard presses
                if event.key == pygame.K_ESCAPE:
                    # Close the application when Esc key is pressed
                    running = False
                    break
            new_node_path = dag.handle_event(event)
            _ = sweep.handle_event(event)  # Currently does nothing

        if new_node_path is not None and new_node_path != ez_proc_man.node_path:
            # Clicked on a new node to monitor
            ez_mirror.reset()  # Clean up mirror's shared memory
            ez_proc_man.reset(new_node_path)  # Close subprocess and start a new one
            sweep.reset(new_node_path)  # Reset state and blank surface
            # Remaining initialization must wait until subprocess has seen data.

        # Refresh / scroll dag image if required
        rects = dag.update(screen)

        # If ez_mirror has yet to materialize shmem, try again.
        if not ez_mirror.connected:
            ez_mirror.connect()  # Has a built-in rate-limiter

        # If we have yet to pass the metadata to sweep, try that now
        if ez_mirror.connected and not sweep.has_meta:
            sweep.set_meta(ez_mirror.meta)

        # Get a view of the shared buffer. Will be None if requested samples are not available.
        #  Will always be None if ez_mirror has yet to connect.
        view = ez_mirror.view_samples(n=None)
        if view is not None:
            rects += sweep.update(screen, view)
        pygame.display.update(rects)

    # sweep.cleanup()
    ez_mirror.reset()
    ez_proc_man.cleanup()

    pygame.quit()


def main():
    typer.run(monitor)


if __name__ == "__main__":
    main()
