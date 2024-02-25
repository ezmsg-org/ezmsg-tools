import asyncio
from multiprocessing import Manager
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import tempfile
import time
import typing

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
import pygame
import pygame.locals
import typer

from ezmsg.vis.dag import get_graph, pgv2pd
from ezmsg.vis.proc import (
    AttachShmProcess,
    ShMemCircBuffSettings,
)


SCROLL_STEP = 50
PLOT_BG_COLOR = (255, 255, 255)
PLOT_LINE_COLOR = (0, 0, 0)
PLOT_DUR = 2.0
PLOT_Y_RANGE = 1e4  # Raw units per channel


def start_shmem_buf_proc(
    topic,
    shared_state: dict,
    shmem_name: str = "ezmsg-vis-temp",
    buf_dur: float = PLOT_DUR,
    axis: str = "time",
):
    unit_settings = ShMemCircBuffSettings(
        shmem_name=shmem_name,
        shared_state=shared_state,
        topic=topic,
        buf_dur=buf_dur,
        axis=axis,
    )
    process = AttachShmProcess(unit_settings)
    process.start()
    return process


def main(graph_ip: str = "127.0.0.1", graph_port: int = 25978):
    pygame.init()

    # Screen
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    screen_width, screen_height = screen.get_size()
    screen = pygame.display.set_mode(
        (screen_width, screen_height), pygame.locals.RESIZABLE
    )
    screen.fill((0, 0, 0))  # Fill the screen with black

    # GraphViz
    G = get_graph((graph_ip, graph_port))
    G.layout(prog="dot")
    # svg gives the easiest to use coordinates.
    svg_path = Path(tempfile.gettempdir()) / "ezmsg-vis.svg"
    G.draw(svg_path, format="svg:cairo")
    node_df = pgv2pd(G)
    # Unfortunately, pygame cannot render svg very well. Get the png for rendering.
    img_path = Path(tempfile.gettempdir()) / "ezmsg-vis.png"
    G.draw(img_path)
    image = pygame.image.load(img_path)
    image_height = image.get_rect().height
    # Scale the svg coordinates by png size / svg size
    _svg = pygame.image.load(svg_path)
    node_df["y"] *= image_height / _svg.get_rect().height
    node_df["x"] *= image.get_rect().width / _svg.get_rect().width
    image_rect = image.get_rect(topleft=(0, 0))
    image_y = 0  # Initial position of the image
    # Render the image
    screen.blit(image, (0, image_y))
    pygame.display.update(image_rect)

    # IPC
    node_path = ""
    proc = None
    shm_arr: typing.Optional[SharedMemory] = None
    arr: typing.Optional[npt.NDArray] = None
    write_index: typing.Optional[npt.NDArray] = None

    # Plots
    xvec: typing.Optional[npt.NDArray] = None
    x2px: float = 1.0  # Convert time (seconds) to pixels
    yoffsets: typing.Optional[npt.NDArray] = None
    y2px: float = 1.0
    read_index: int = 0
    plot_size = (screen_width - image.get_rect().width, screen_height)
    plot_surface = pygame.Surface(plot_size)
    plot_surface.fill(PLOT_BG_COLOR)
    plot_rect = plot_surface.get_rect(topleft=(image.get_rect().width, 0))

    # TEMP: Setup font for displaying full path to clicked node
    font = pygame.font.Font(None, 36)  # Default font and size 36

    with Manager() as manager:
        running = True
        shared_state = manager.dict()

        def _cleanup_proc(_proc, _shm_arr, _write_index, _arr):
            if _proc is not None:
                shared_state["kill"] = True
                _proc.join()  # Wait for process to close
                _proc = None
                # TODO: Somehow closing the proc isn't enough to clear the VISBUFF connections.
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    ez.GraphService(address=(graph_ip, graph_port)).disconnect(
                        node_path, "VISBUFF/INPUT_SIGNAL"
                    )
                )
            if _shm_arr is not None:
                _shm_arr.close()
                _shm_arr = None
                _write_index = None
                _arr = None
            shared_state.clear()
            shared_state["kill"] = False
            return _proc, _shm_arr, _write_index, _arr

        while running:
            new_node_path = node_path
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    # Keyboard presses
                    if event.key == pygame.K_ESCAPE:
                        # Close the application when Esc key is pressed
                        running = False
                elif event.type == pygame.MOUSEWHEEL:
                    mouse_pos = pygame.mouse.get_pos()
                    if image_rect.left <= mouse_pos[0] <= image_rect.right:
                        if event.y > 0:
                            # scroll graph up
                            image_y = min(0, image_y + SCROLL_STEP)
                        elif event.y < 0:
                            # scroll graph down
                            image_y = max(
                                -(image_height - screen_height), image_y - SCROLL_STEP
                            )
                        screen.blit(image, (0, image_y))
                        pygame.display.update(image_rect)
                    else:
                        pass
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Mouse events
                    if event.button == 1:
                        mouse_pos = pygame.mouse.get_pos()
                        if image_rect.left <= mouse_pos[0] <= image_rect.right:
                            # Over the graph png
                            if event.button == 1:
                                # Clicked on a node
                                graph_pos = (
                                    mouse_pos[0],
                                    image_height - (mouse_pos[1] - image_y),
                                )
                                min_row = (
                                    (node_df.x - graph_pos[0]) ** 2
                                    + (node_df.y - graph_pos[1]) ** 2
                                ).argmin()
                                new_node_path = f"{node_df.iloc[min_row]['upstream']}"
                        else:
                            # TODO: else over the plot
                            pass

            if new_node_path != node_path:
                proc, shm_arr, write_index, arr = _cleanup_proc(
                    proc, shm_arr, write_index, arr
                )
                proc = start_shmem_buf_proc(new_node_path, shared_state)
                node_path = new_node_path

                #  TEMP: Render the node_path
                text_surface = font.render(node_path, True, PLOT_LINE_COLOR)
                text_rect = text_surface.get_rect(midtop=plot_rect.midtop)
                # Optional: draw a background rectangle for the text
                pygame.draw.rect(screen, (200, 200, 200), plot_rect)
                screen.blit(text_surface, text_rect)
                pygame.display.update(text_rect)

            if arr is None and shared_state.get("setup", False):
                shm_arr = SharedMemory("ezmsg-vis-temp", create=False)
                write_index = np.ndarray((1,), dtype=np.uint64, buffer=shm_arr.buf[:8])
                arr = np.ndarray(
                    shared_state["shape"],
                    dtype=shared_state["dtype"],
                    buffer=shm_arr.buf[8:],
                )

                # Reset plot parameters
                plot_samples = int(PLOT_DUR * shared_state["srate"])
                xvec = np.arange(plot_samples)
                x2px = plot_size[0] / plot_samples
                y_span = (shared_state["shape"][1] + 1) * PLOT_Y_RANGE
                yoffsets = (np.arange(shared_state["shape"][1]) + 0.5) * PLOT_Y_RANGE
                y2px = plot_size[1] / y_span
                read_index: int = 0
                # Blank the surface
                plot_surface.fill(PLOT_BG_COLOR)
                pygame.display.update(plot_rect)

                #  TEMP: Render the node_path
                text_surface = font.render(
                    node_path + f" {arr.shape}, {arr.dtype}", True, PLOT_LINE_COLOR
                )
                text_rect = text_surface.get_rect(midtop=plot_rect.midtop)
                # Optional: draw a background rectangle for the text
                pygame.draw.rect(screen, (200, 200, 200), plot_rect)
                screen.blit(text_surface, text_rect)
                pygame.display.update(text_rect)

            # Render the plot
            if arr is not None and read_index != write_index[0]:
                _write_idx = int(write_index[0])
                if _write_idx < read_index:
                    n_samples = len(xvec) - read_index
                else:
                    n_samples = _write_idx - read_index

                if n_samples > 1 or (n_samples == 1 and read_index != 0):
                    # Establish the minimum rectangle for the update
                    _x = xvec[max(0, read_index - 1) : read_index + n_samples]
                    _rect_x = int(_x[0] * x2px), int(np.ceil(_x[-1] * x2px))
                    update_rect = pygame.Rect(
                        (_rect_x[0], 0), (_rect_x[1] - _rect_x[0] + 5, plot_size[1])
                    )
                    # Blank the rectangle with bgcolor
                    pygame.draw.rect(plot_surface, PLOT_BG_COLOR, update_rect)

                    # Plot the lines
                    for ch_ix, ch_offset in enumerate(yoffsets):
                        plot_dat = (
                            arr[max(0, read_index - 1) : read_index + n_samples, ch_ix]
                            + ch_offset
                        ) * y2px
                        pygame.draw.lines(
                            plot_surface,
                            PLOT_LINE_COLOR,
                            0,
                            np.column_stack((_x * x2px, plot_dat)),
                        )

                    read_index = (read_index + n_samples) % len(xvec)

                    # Draw cursor
                    curs_x = int(((read_index + 1) % len(xvec)) * x2px)
                    pygame.draw.line(
                        plot_surface,
                        PLOT_LINE_COLOR,
                        (curs_x, 0),
                        (curs_x, plot_size[1]),
                    )
                    # Update
                    # _rect = screen.blit(plot_surface, (image_rect.width, 0))
                    _rect = screen.blit(
                        plot_surface,
                        (image_rect.width + update_rect.x, 0),
                        update_rect,
                    )
                    pygame.display.update(_rect)
                else:
                    time.sleep(0.005)

        if proc is not None:
            proc, shm_arr, write_index, arr = _cleanup_proc(
                proc, shm_arr, write_index, arr
            )

    pygame.quit()


if __name__ == "__main__":
    typer.run(main)
