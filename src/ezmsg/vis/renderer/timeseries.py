import typing

import numpy as np
import numpy.typing as npt
import pygame

from .base import ShmemRenderer, PLOT_DUR


PLOT_BG_COLOR = (255, 255, 255)
PLOT_LINE_COLOR = (0, 0, 0)
INIT_Y_RANGE = 1e4  # Raw units per channel


def running_stats(
    fs: float,
    time_constant: float = PLOT_DUR,
) -> typing.Generator[typing.Tuple[npt.NDArray, npt.NDArray], npt.NDArray, None]:
    arr_in = np.array([])
    tuple_out = (np.array([]), np.array([]))
    means = vars_means = vars_sq_means = None
    alpha = 1 - np.exp(-1 / (fs * time_constant))

    def _ew_update(arr, prev, _alpha):
        if np.all(prev == 0):
            return arr
        # return _alpha * arr + (1 - _alpha) * prev
        # Micro-optimization: sub, mult, add (below) is faster than sub, mult, mult, add (above)
        return prev + _alpha * (arr - prev)

    while True:
        arr_in = yield tuple_out

        if means is None:
            vars_sq_means = np.zeros_like(arr_in[0], dtype=float)
            vars_means = np.zeros_like(arr_in[0], dtype=float)
            means = np.zeros_like(arr_in[0], dtype=float)

        for sample in arr_in:
            # Update step
            vars_means = _ew_update(sample, vars_means, alpha)
            vars_sq_means = _ew_update(sample**2, vars_sq_means, alpha)
            means = _ew_update(sample, means, alpha)
        tuple_out = means, np.sqrt(vars_sq_means - vars_means**2)


class Sweep(ShmemRenderer):
    def __init__(
        self, *args, yrange: float = INIT_Y_RANGE, autoscale: bool = True, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._xvec: typing.Optional[npt.NDArray] = None
        self._plot_cfg = {
            "xvec": np.array([]),
            "yrange": yrange,
            "stats_gen": None,
            "autoscale": autoscale,
        }

    def _reset_plot(self):
        # Reset plot parameters
        plot_samples = int(PLOT_DUR * self._shared_state["srate"])
        self._plot_cfg["xvec"] = np.arange(plot_samples)
        self._plot_cfg["x2px"] = self._plot_rect.width / plot_samples
        # self._plot_cfg["yrange"] = INIT_Y_RANGE
        self._plot_cfg["stats_gen"] = running_stats(
            self._shared_state["srate"], PLOT_DUR
        )
        self._plot_cfg["stats_gen"].send(None)  # Prime the generator
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
                t_slice = np.s_[
                    max(0, self._read_index - 1) : self._read_index + n_samples
                ]

                if self._plot_cfg["autoscale"]:
                    # Check if the scale has changed.
                    means, stds = self._plot_cfg["stats_gen"].send(
                        self._arr[t_slice, :]
                    )
                    new_y_range = 3 * np.mean(stds)
                    b_reset_scale = (
                        new_y_range < 0.8 * self._plot_cfg["yrange"]
                        or new_y_range > 1.2 * self._plot_cfg["yrange"]
                    )
                    if b_reset_scale:
                        t_slice = np.s_[:]
                        self._plot_cfg["yrange"] = new_y_range

                n_chs = self._shared_state["shape"][1]
                yoffsets = (np.arange(n_chs) + 0.5) * self._plot_cfg["yrange"]
                y_span = (n_chs + 1) * self._plot_cfg["yrange"]
                y2px = self._plot_rect.height / y_span

                # Establish the minimum rectangle for the update
                _x = self._plot_cfg["xvec"][t_slice]
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
                for ch_ix, ch_offset in enumerate(yoffsets):
                    plot_dat = self._arr[t_slice, ch_ix] + ch_offset
                    pygame.draw.lines(
                        self,
                        PLOT_LINE_COLOR,
                        0,
                        np.column_stack((_x * self._plot_cfg["x2px"], plot_dat * y2px)),
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
