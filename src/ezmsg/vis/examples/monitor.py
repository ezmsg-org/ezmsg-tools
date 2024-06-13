import pygame
import pygame.locals
import typer

from ezmsg.vis.pygame.dag import VisDAG
from ezmsg.vis.pygame.timeseries import Sweep
from ezmsg.vis.proc import EZProcManager
from bolt.utils.shmem_mirror import EZShmMirror


GRAPH_IP = "127.0.0.1"
GRAPH_PORT = 25978
PLOT_DUR = 2.0


def monitor(
    graph_addr: str = ":".join((GRAPH_IP, str(GRAPH_PORT))),
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

    # ezmsg process manager -- the process runs a mini ezmsg pipeline
    #  that attaches a single node to an existing pipeline. We don't
    #  know the attachment point yet, so we do not start the pipeline.
    # Note: not required if pipeline with ShMemCircBuff already exists.
    ez_proc_man = EZProcManager(
        graph_ip=graph_ip,
        graph_port=graph_port,
        buf_dur=PLOT_DUR,
    )

    # We need a local mirror to the remote ShMemCircBuff.
    #  It initializes in a waiting state because the remote
    #  unit does not exist until EZProcManager starts up.
    #  Other than the shmem, the mirror communicates with
    #  the remote unit via a Pipe. (one end is a Connection object).
    mirror = EZShmMirror()

    # Data Plotter. Puts a surface on the screen, plots 2D lines
    #  with some basic auto-scaling. ezmsg-vis renderers are
    #  highly customized to use the mirror object as it uses
    #  the mirror's shmem buffer as its own rendering buffer.
    sweep = Sweep(
        mirror,
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
            mirror.cleanup()
            ez_proc_man.reset(new_node_path)  # Close subprocess and start a new one.
            mirror.connect("buff_" + new_node_path)  # "buff_" convention used by EZProcManager
            sweep.reset(new_node_path)  # Reset renderer. We provide the path just so it can print it.

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
