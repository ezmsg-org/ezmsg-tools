import typing

import numpy as np
import numpy.typing as npt
import pygame

from .base import PLOT_DUR, BaseRenderer

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
        self,
        *args,
        yrange: float = INIT_Y_RANGE,
        autoscale: bool = True,
        dur: float = PLOT_DUR,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._y_range = yrange
        self._autoscale = autoscale
        self._dur = dur
        self._xvec = np.array([])  # Vector of indices
        self._plot_x_idx = 0  # index into xvec where the next plot starts.
        self._read_index = 0  # Index into shmem buffer
        self._stats_gen: typing.Optional[typing.Generator] = None
        self._last_y_vec: typing.Optional[npt.NDArray] = None
        self._x2px: float = 1.0

    def _reset_plot(self):
        # Reset plot parameters
        meta = self._mirror.meta
        plot_samples = int(self._dur * meta.srate)
        self._xvec = np.arange(plot_samples)
        self._x2px = self._plot_rect.width / plot_samples
        self._stats_gen = running_stats(meta.srate, time_constant=self._dur)
        self._stats_gen.send(None)  # Prime the generator
        self._plot_x_idx = 0
        self._read_index = 0
        self._last_y_vec = None
        # Blank the surface
        self.fill(PLOT_BG_COLOR)
        pygame.display.update(self._plot_rect)
        if meta.ndim > 2:
            # Monkey-patch udpate func to do nothing
            print("timeseries does not support > 2 dimensions")

    def update_with_copy(self, surface: pygame.Surface) -> typing.List[pygame.Rect]:
        rects = super().update(surface)
        data = self._mirror.auto_view(n=None)
        if data is not None:
            if self._autoscale:
                # Check if the scale has changed.
                means, stds = self._stats_gen.send(data)
                new_y_range = 3 * np.mean(stds)
                b_reset_scale = new_y_range < 0.8 * self._y_range or new_y_range > 1.2 * self._y_range
                if b_reset_scale:
                    self._y_range = new_y_range
                    # TODO: We should also redraw the entire plot at the new scale.
                    #  However, we do not have a copy of all visible data.

            n_chs = data.shape[1]
            yoffsets = (np.arange(n_chs) + 0.5) * self._y_range
            y_span = (n_chs + 1) * self._y_range
            y2px = self._plot_rect.height / y_span

            # Establish the minimum rectangle for the update
            n_samps = data.shape[0]
            dat_offset = 0
            while n_samps > 0:
                x0 = self._plot_x_idx
                b_prepend = x0 != 0 and self._last_y_vec is not None
                if b_prepend:
                    xvec = self._xvec[x0 - 1 : x0 + n_samps]
                    if dat_offset == 0:
                        _data = np.concatenate([self._last_y_vec, data[: xvec.shape[0] - 1]], axis=0)
                    else:
                        _data = data[dat_offset - 1 : dat_offset + xvec.shape[0] - 1]
                else:
                    xvec = self._xvec[x0 : x0 + n_samps]
                    _data = data[dat_offset : dat_offset + xvec.shape[0]]

                # Identify the rectangle that we will be plotting over.
                _rect_x = (
                    int(xvec[0] * self._x2px),
                    int(np.ceil(xvec[-1] * self._x2px)),
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
                            xy = np.column_stack((xvec * self._x2px, plot_dat * y2px))
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
                self._plot_x_idx += n_new
                self._plot_x_idx %= self._xvec.shape[0]
                n_samps -= n_new
                dat_offset += n_new
                self._last_y_vec = _data[-1:].copy()

            # Draw cursor
            curs_x = int(((self._plot_x_idx + 1) % self._xvec.shape[0]) * self._x2px)
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

    def update(self, surface: pygame.Surface) -> typing.List[pygame.Rect]:
        rects = super().update(surface)

        res, b_overflow = self._mirror.auto_view()
        if res.size == 0:
            return rects

        if self._plot_needs_reset:
            return rects

        meta = self._mirror.meta
        if meta.ndim > 2:
            return rects
        n_samples = res.shape[0]

        t_slice = np.s_[max(0, self._read_index - 1) : self._read_index + n_samples]
        if self._autoscale:
            means, stds = self._stats_gen.send(self._mirror.buffer[t_slice])
            new_y_range = max(3 * np.mean(stds), 1e-12)
            b_reset_scale = new_y_range < 0.8 * self._y_range or new_y_range > 1.2 * self._y_range
            if b_reset_scale:
                self._y_range = new_y_range
                t_slice = np.s_[:]

        n_chs = res.shape[1]
        yoffsets = (np.arange(n_chs) + 0.5) * self._y_range
        y_span = (n_chs + 1) * self._y_range
        y2px = self._plot_rect.height / y_span

        _x = self._xvec[t_slice]
        _rect_x = (int(_x[0] * self._x2px), int(np.ceil(_x[-1] * self._x2px)))
        update_rect = pygame.Rect(
            (_rect_x[0], 0),
            (_rect_x[1] - _rect_x[0] + 5, self._plot_rect.height),
        )
        # Blank the rectangle with bgcolor
        pygame.draw.rect(self, PLOT_BG_COLOR, update_rect)

        # Plot the lines
        for ch_ix, ch_offset in enumerate(yoffsets):
            plot_dat = self._mirror.buffer[t_slice, ch_ix] + ch_offset
            try:
                xy = np.column_stack((_x * self._x2px, plot_dat * y2px))
            except ValueError:
                print(_x.shape, plot_dat.shape)
                raise
            pygame.draw.lines(self, PLOT_LINE_COLOR, 0, xy)

        self._read_index = (self._read_index + n_samples) % self._xvec.shape[0]

        # Draw cursor
        curs_x = int(((self._read_index + 1) % self._xvec.shape[0]) * self._x2px)
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

    def handle_event(self, event: pygame.event.Event):
        if event.type in [pygame.KEYDOWN]:
            if event.key == pygame.K_a:
                # Toggle autoscale with 'a' key
                self._autoscale = not self._autoscale
            elif not self._autoscale:
                # When autoscale is disabled, allow manual y-range adjustment
                if event.key == pygame.K_MINUS:
                    # Zoom-Out: Increase y-range by 20% with '-' key
                    self._y_range *= 1.2
                elif event.key == pygame.K_EQUALS:
                    # Zoom-in: Decrease y-range by 20% with '=' key
                    self._y_range *= 0.8
