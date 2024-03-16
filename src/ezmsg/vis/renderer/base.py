import asyncio
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import typing

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
import pygame

from ..proc import AttachShmProcess, ShMemCircBuffSettings


PLOT_BG_COLOR = (255, 255, 255)
PLOT_FONT_COLOR = (0, 0, 0)
PLOT_DUR = 2.0


class ShmemRenderer(pygame.Surface):
    """
    This is an abstract class representing a pygame.Surface that also manages
    a subprocess running ezmsg as well as shared memory to communicate with that
    subprocess.
    """

    def __init__(
        self,
        *args,
        shmem_name: str = "ezmsg-vis-temp",
        tl_offset: typing.Tuple[int, int] = (0, 0),
        graph_ip: str = "127.0.0.1",
        graph_port: int = 25978,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fill(PLOT_BG_COLOR)
        self._shmem_name = shmem_name
        self._graph_addr: typing.Tuple[str, int] = (graph_ip, graph_port)
        self._rend_conn, self._ez_conn = multiprocessing.Pipe()
        self._proc = None
        self._node_path: typing.Optional[str] = None
        self._shm_arr: typing.Optional[SharedMemory] = None
        self._arr: typing.Optional[npt.NDArray] = None
        self._write_index: typing.Optional[npt.NDArray] = None
        self._read_index: typing.Optional[int] = None
        self._tl_offset: typing.Tuple[int, int] = tl_offset
        self._plot_rect = self.get_rect(topleft=self._tl_offset)
        self._font = pygame.font.Font(None, 36)  # Default font and size 36

    def _reset_plot(self):
        raise NotImplementedError

    def reset(self, node_path: typing.Optional[str]) -> None:
        if node_path is not None and node_path != self._node_path:
            self.cleanup()
            self._node_path = node_path
            self._init_subprocess()
            self.fill(PLOT_BG_COLOR)
            # The subprocess cannot prepare the shared memory
            #  until it has received a message with data.
            #  So we will attach the shmem in our update loop
            #  after receiving a signal.

    def cleanup(self):
        self._cleanup_subprocess()
        self._cleanup_shmem()

    def _cleanup_subprocess(self):
        if self._proc is not None:
            self._rend_conn.send({"kill": True})
            self._proc.join()  # Wait for process to close
            self._proc = None
            # TODO: Somehow closing the proc isn't enough to clear the VISBUFF connections.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                ez.GraphService(address=self._graph_addr).disconnect(
                    self._node_path, "VISBUFF/INPUT_SIGNAL"
                )
            )

    def _cleanup_shmem(self):
        self._cleanup_subprocess()
        if self._shm_arr is not None:
            self._shm_arr.close()
            self._shm_arr = None
            self._write_index = None
            self._arr = None

    def _init_subprocess(self, buf_dur: float = PLOT_DUR, axis: str = "time"):
        unit_settings = ShMemCircBuffSettings(
            shmem_name=self._shmem_name,
            ipconn=self._ez_conn,
            topic=self._node_path,
            buf_dur=buf_dur,
            axis=axis,
        )
        self._proc = AttachShmProcess(unit_settings)
        self._proc.start()

    def _attach_shmem(self, shmem_meta: dict):
        self._shm_arr = SharedMemory(self._shmem_name, create=False)
        self._write_index = np.ndarray(
            (1,), dtype=np.uint64, buffer=self._shm_arr.buf[:8]
        )
        self._arr = np.ndarray(
            shmem_meta["shape"],
            dtype=shmem_meta["dtype"],
            buffer=self._shm_arr.buf[8:],
        )
        self._read_index = 0
        self._shmem_meta = shmem_meta

    def _print_node_path(self, surface: pygame.Surface) -> pygame.Rect:
        #  TEMP: Render the node_path
        text_surface = self._font.render(
            f"{self._node_path} {self._arr.shape}, {self._arr.dtype}",
            True,
            PLOT_FONT_COLOR,
        )
        text_rect = text_surface.get_rect(midtop=self._plot_rect.midtop)
        # Draw a background rectangle for the text
        pygame.draw.rect(surface, (200, 200, 200), self._plot_rect)
        # Draw the actual text
        surface.blit(text_surface, text_rect)
        pygame.display.update(text_rect)
        return text_rect

    def handle_event(self, event: pygame.event.Event):
        if event.type in [pygame.MOUSEWHEEL, pygame.MOUSEBUTTONDOWN]:
            mouse_pos = pygame.mouse.get_pos()
            # TODO: Check if mouse_pos is over self
            # TODO: Respond to mouse.

    def update(self, surface: pygame.Surface) -> typing.List[pygame.Rect]:
        rects = []
        if self._arr is None and self._rend_conn.poll():
            shmem_meta = self._rend_conn.recv()
            self._attach_shmem(shmem_meta)
            _ = self._print_node_path(surface)
            self._reset_plot()
            rects.append(self._plot_rect)  # Render the whole plot after a reset.

        return rects
