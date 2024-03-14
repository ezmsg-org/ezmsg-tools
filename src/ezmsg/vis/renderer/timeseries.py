import typing

import numpy as np
import pygame

from .base import ShmemRenderer, PLOT_DUR


PLOT_BG_COLOR = (255, 255, 255)
PLOT_LINE_COLOR = (0, 0, 0)
PLOT_Y_RANGE = 1e4  # Raw units per channel


class Sweep(ShmemRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._xvec: typing.Optional[npt.NDArray] = None
        x2px: float = 1.0  # Convert time (seconds) to pixels
        yoffsets: typing.Optional[npt.NDArray] = None
        y2px: float = 1.0

    def _reset_plot(self):
        # Reset plot parameters
        plot_samples = int(PLOT_DUR * self._shared_state["srate"])
        xvec = np.arange(plot_samples)
        x2px = self._plot_rect.width / plot_samples
        y_span = (self._shared_state["shape"][1] + 1) * PLOT_Y_RANGE
        yoffsets = (np.arange(self._shared_state["shape"][1]) + 0.5) * PLOT_Y_RANGE
        y2px = self._plot_rect.height / y_span
        self._plot_cfg = {
            "xvec": xvec,
            "x2px": x2px,
            "y2px": y2px,
            "yoffsets": yoffsets,
        }
        self._read_index = 0
        # Blank the surface
        self.fill(PLOT_BG_COLOR)
        pygame.display.update(self._plot_rect)

    def update(self, surface: pygame.Surface) -> typing.List[pygame.Rect]:
        rects = super().update(surface)
        if self._arr is not None and self._read_index != self._write_index[0]:
            _write_idx = int(self._write_index[0])
            if _write_idx < self._read_index:
                n_samples = len(self._plot_cfg["xvec"]) - self._read_index
            else:
                n_samples = _write_idx - self._read_index

            if n_samples > 1 or (n_samples == 1 and self._read_index != 0):
                # Establish the minimum rectangle for the update
                _x = self._plot_cfg["xvec"][
                    max(0, self._read_index - 1) : self._read_index + n_samples
                ]
                _rect_x = (
                    int(_x[0] * self._plot_cfg["x2px"]),
                    int(np.ceil(_x[-1] * self._plot_cfg["x2px"])),
                )
                update_rect = pygame.Rect(
                    (_rect_x[0], 0),
                    (_rect_x[1] - _rect_x[0] + 5, self._plot_rect.height),
                )
                # Blank the rectangle with bgcolor
                pygame.draw.rect(self, PLOT_BG_COLOR, update_rect)

                # Plot the lines
                for ch_ix, ch_offset in enumerate(self._plot_cfg["yoffsets"]):
                    plot_dat = (
                        self._arr[
                            max(0, self._read_index - 1) : self._read_index + n_samples,
                            ch_ix,
                        ]
                        + ch_offset
                    )
                    pygame.draw.lines(
                        self,
                        PLOT_LINE_COLOR,
                        0,
                        np.column_stack(
                            (
                                _x * self._plot_cfg["x2px"],
                                plot_dat * self._plot_cfg["y2px"],
                            )
                        ),
                    )

                self._read_index = (self._read_index + n_samples) % len(
                    self._plot_cfg["xvec"]
                )

                # Draw cursor
                curs_x = int(
                    ((self._read_index + 1) % len(self._plot_cfg["xvec"]))
                    * self._plot_cfg["x2px"]
                )
                pygame.draw.line(
                    self,
                    PLOT_LINE_COLOR,
                    (curs_x, 0),
                    (curs_x, self._plot_rect.height),
                )
                # Update
                _rect = surface.blit(
                    self,
                    (
                        self._tl_offset[0] + update_rect.x,
                        self._tl_offset[1],
                    ),
                    update_rect,
                )
                rects.append(_rect)

        return rects
