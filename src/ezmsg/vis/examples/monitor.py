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
PLOT_DUR = 2.0


def monitor(
    graph_addr: str = ":".join((GRAPH_IP, str(GRAPH_PORT))),
    shmem_name: str = SHMEM_NAME
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
    graph_ip, graph_port = graph_addr.split(":")
    graph_port = int(graph_port)
    dag = VisDAG(screen_height=screen_height, graph_ip=graph_ip, graph_port=graph_port)

    # ezmsg process manager -- process runs ezmsg context to attach a node to running pipeline.
    #  We don't have the name of the target node yet so the manager does not start the proc yet.
    #  If you have a pre-existing pipeline with ShMemCircBuff already in it then this is not needed.
    #  All that's needed is to know the name of the shared memory.
    ez_proc_man = EZProcManager(
        graph_ip=graph_ip,
        graph_port=graph_port,
        shmem_name=shmem_name,
        buf_dur=PLOT_DUR,
    )

    # Data Plotter. Puts a surface on the screen, plots 2D lines
    #  with some basic auto-scaling. We pass it shmem_name so it
    #  may create its own reference to the ez msg shared memory.
    sweep = Sweep(
        shmem_name,
        (screen_width - dag.size[0], screen_height),
        tl_offset=(dag.size[0], 0),
        dur=PLOT_DUR,
    )

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
            sweep.reset(new_node_path)  # Reset state (incl shmem) and blank surface
            ez_proc_man.reset(new_node_path)  # Close subprocess and start a new one
            # Remaining initialization must wait until subprocess has seen data.

        # Refresh / scroll dag image if required
        rects = dag.update(screen)

        # Update the sweep plot (internally it uses shmem)
        rects += sweep.update(screen)

        pygame.display.update(rects)

    sweep.reset(None)
    ez_proc_man.cleanup()

    pygame.quit()


def main():
    typer.run(monitor)


if __name__ == "__main__":
    main()
