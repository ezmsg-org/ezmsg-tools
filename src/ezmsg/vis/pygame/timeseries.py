import typing

import numpy as np
import numpy.typing as npt
import pygame

from .base import BaseRenderer, PLOT_DUR


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


class Sweep(BaseRenderer):
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
            "x_index": 0,  # index into xvec where the next plot starts.
        }
        self._last_y_vec: typing.Optional[npt.NDArray] = None

    def _reset_plot(self):
        # Reset plot parameters
        meta = self._mirror.meta
        plot_samples = int(PLOT_DUR * meta.srate)
        self._plot_cfg["xvec"] = np.arange(plot_samples)
        self._plot_cfg["x2px"] = self._plot_rect.width / plot_samples
        # self._plot_cfg["yrange"] = INIT_Y_RANGE
        self._plot_cfg["stats_gen"] = running_stats(meta.srate, PLOT_DUR)
        self._plot_cfg["stats_gen"].send(None)  # Prime the generator
        self._plot_cfg["x_index"] = 0
        self._last_y_vec = None
        # Blank the surface
        self.fill(PLOT_BG_COLOR)
        pygame.display.update(self._plot_rect)

    def update(self, surface: pygame.Surface) -> typing.List[pygame.Rect]:
        rects = super().update(surface)
        data = self._mirror.view_samples(n=None)
        if data is not None:
            if self._plot_cfg["autoscale"]:
                # Check if the scale has changed.
                means, stds = self._plot_cfg["stats_gen"].send(data)
                new_y_range = 3 * np.mean(stds)
                b_reset_scale = (
                    new_y_range < 0.8 * self._plot_cfg["yrange"]
                    or new_y_range > 1.2 * self._plot_cfg["yrange"]
                )
                if b_reset_scale:
                    self._plot_cfg["yrange"] = new_y_range
                    # TODO: We should also redraw the entire plot at the new scale.
                    #  However, we do not have a copy of all visible data.

            n_chs = data.shape[1]
            yoffsets = (np.arange(n_chs) + 0.5) * self._plot_cfg["yrange"]
            y_span = (n_chs + 1) * self._plot_cfg["yrange"]
            y2px = self._plot_rect.height / y_span

            # Establish the minimum rectangle for the update
            n_samps = data.shape[0]
            dat_offset = 0
            while n_samps > 0:
                x0 = self._plot_cfg["x_index"]
                b_prepend = x0 != 0 and self._last_y_vec is not None
                if b_prepend:
                    xvec = self._plot_cfg["xvec"][x0 - 1 : x0 + n_samps]
                    if dat_offset == 0:
                        _data = np.concatenate(
                            [self._last_y_vec, data[: xvec.shape[0] - 1]], axis=0
                        )
                    else:
                        _data = data[dat_offset - 1 : dat_offset + xvec.shape[0] - 1]
                else:
                    xvec = self._plot_cfg["xvec"][x0 : x0 + n_samps]
                    _data = data[dat_offset : dat_offset + xvec.shape[0]]

                # Identify the rectangle that we will be plotting over.
                _rect_x = (
                    int(xvec[0] * self._plot_cfg["x2px"]),
                    int(np.ceil(xvec[-1] * self._plot_cfg["x2px"])),
                )
                update_rect = pygame.Rect(
                    (_rect_x[0], 0),
                    (_rect_x[1] - _rect_x[0] + 5, self._plot_rect.height),
                )

                # Blank the rectangle with bgcolor
                pygame.draw.rect(self, PLOT_BG_COLOR, update_rect)

                # Plot the lines
                if _data.shape[0] > 1:
                    for ch_ix, ch_offset in enumerate(yoffsets):
                        plot_dat = _data[:, ch_ix] + ch_offset
                        try:
                            xy = np.column_stack(
                                (xvec * self._plot_cfg["x2px"], plot_dat * y2px)
                            )
                        except ValueError:
                            print("DEBUG")
                        pygame.draw.lines(self, PLOT_LINE_COLOR, 0, xy)

                # Blit the surface
                _rect = surface.blit(
                    self,
                    (
                        self._tl_offset[0] + update_rect.x,
                        self._tl_offset[1],
                    ),
                    update_rect,
                )
                rects.append(_rect)

                n_new = (xvec.shape[0] - 1) if b_prepend else xvec.shape[0]
                self._plot_cfg["x_index"] += n_new
                self._plot_cfg["x_index"] %= self._plot_cfg["xvec"].shape[0]
                n_samps -= n_new
                dat_offset += n_new
                self._last_y_vec = _data[-1:].copy()

            # Draw cursor
            curs_x = int(
                ((self._plot_cfg["x_index"] + 1) % self._plot_cfg["xvec"].shape[0])
                * self._plot_cfg["x2px"]
            )
            curs_rect = pygame.draw.line(
                self,
                PLOT_LINE_COLOR,
                (curs_x, 0),
                (curs_x, self._plot_rect.height),
            )
            _rect = surface.blit(
                self,
                (
                    self._tl_offset[0] + curs_rect.x,
                    self._tl_offset[1],
                ),
                curs_rect,
            )
            rects.append(_rect)

        return rects
