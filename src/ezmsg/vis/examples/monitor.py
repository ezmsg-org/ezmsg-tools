from multiprocessing import Manager
import typing

import pygame
import pygame.locals
import typer

from ezmsg.vis.pygame.dag import VisDAG
from ezmsg.vis.renderer.timeseries import Sweep


def monitor(graph_ip: str = "127.0.0.1", graph_port: int = 25978):
    pygame.init()

    # Screen
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    screen_width, screen_height = screen.get_size()
    screen = pygame.display.set_mode(
        (screen_width, screen_height), pygame.locals.RESIZABLE
    )
    screen.fill((0, 0, 0))  # Fill the screen with black

    # Interactive ezmsg graph
    dag = VisDAG(screen_height=screen_height, graph_ip=graph_ip, graph_port=graph_port)

    # Data Plotter
    sweep: typing.Optional[Sweep] = None

    # Plots

    sweep = Sweep(
        (screen_width - dag.size[0], screen_height),
        tl_offset=(dag.size[0], 0),
        graph_ip=graph_ip,
        graph_port=graph_port,
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
            sweep.handle_event(event)

        sweep.reset(new_node_path)  # Will ignore None or repeated path

        rects = dag.update(screen)
        rects += sweep.update(screen)
        pygame.display.update(rects)

    sweep.cleanup()

    pygame.quit()


def main():
    typer.run(monitor)


if __name__ == "__main__":
    main()
