import typing

import pygame

from ...shmem.shmem_mirror import EZShmMirror

PLOT_BG_COLOR = (255, 255, 255)
PLOT_FONT_COLOR = (0, 0, 0)
PLOT_DUR = 2.0


class BaseRenderer(pygame.Surface):
    """
    This is an abstract class representing a pygame.Surface that also manages
    a subprocess running ezmsg as well as shared memory to communicate with that
    subprocess.
    """

    def __init__(
        self,
        mirror: EZShmMirror,
        *args,
        tl_offset: typing.Tuple[int, int] = (0, 0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._mirror = mirror
        self._tl_offset: typing.Tuple[int, int] = tl_offset
        self._plot_rect = self.get_rect(topleft=self._tl_offset)
        self._node_path: typing.Optional[str] = None
        self._font = pygame.font.Font(None, 36)  # Default font and size 36
        self._refresh_text = True
        self._plot_needs_reset = True
        self.fill(PLOT_BG_COLOR)

    def handle_event(self, event: pygame.event.Event):
        if event.type in [pygame.MOUSEWHEEL, pygame.MOUSEBUTTONDOWN]:
            pass
            # TODO: Check if mouse_pos is over self
            # mouse_pos = pygame.mouse.get_pos()
            # TODO: Respond to mouse.

    def _reset_plot(self):
        raise NotImplementedError

    def reset(self, node_path: typing.Optional[str]) -> None:
        self._mirror.disconnect()
        self.fill(PLOT_BG_COLOR)
        if node_path is not None and node_path != self._node_path:
            self._node_path = node_path
        self._refresh_text = True
        self._plot_needs_reset = True
        # This is all we can do until the metadata becomes available.

    def _print_node_path(self, surface: pygame.Surface) -> pygame.Rect:
        #  TEMP: Render the node_path
        meta = self._mirror.meta
        if meta is not None:
            self._mirror.connect("buff_" + self._node_path)

            import numpy as np

            buf_shape = meta.shape[: meta.ndim]
            buf_dtype = np.dtype(meta.dtype).name
            src_str = f"{self._node_path} {buf_shape}, {buf_dtype}"
        else:
            src_str = self._node_path
        text_surface = self._font.render(
            src_str,
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

    def update(self, surface: pygame.Surface) -> typing.List[pygame.Rect]:
        rects = []

        if not self._mirror.connected and self._node_path is not None:
            self._mirror.connect("buff_" + self._node_path)

        if self._mirror.connected and self._plot_needs_reset:
            self._reset_plot()
            self._plot_needs_reset = False
            self._refresh_text = True

        if self._refresh_text:
            rects.append(self._print_node_path(surface))
            self._refresh_text = False

        return rects
