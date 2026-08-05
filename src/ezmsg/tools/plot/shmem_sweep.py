"""A sweep plot fed from a shared-memory ring.

:class:`~ezmsg.tools.shmem.shmem.ShMemCircBuff` writes samples into shared
memory and :class:`~ezmsg.tools.shmem.shmem_mirror.EZShmMirror` reads them back
in another process; this is the Qt widget that sits on the far end and draws
them. It exists so that consumers stop writing their own: the lazy build, the
poll timer, the metadata handling and the envelope unpacking are the same
problem every time, and each reimplementation has got a different subset of it
right.

Why shared memory at all: the plot needs every sample the pipeline produces, and
routing a 30 kHz multichannel stream through ezmsg's message transport to reach
a GUI in another process costs a serialization round-trip per message. The ring
is written once and read in place.
"""

from __future__ import annotations

import logging
import typing

import numpy as np
from phosphor import ChannelPlotControlsWidget
from phosphor.sweep_widget import SweepConfig, SweepWidget
from PySide6 import QtCore, QtWidgets

from ..shmem.shmem_mirror import EZShmMirror
from .describe import (
    StreamShape,
    UnsupportedMetricError,
    describe_mirror,
    flatten_for_plot,
    require_sweep_renderable,
)

logger = logging.getLogger(__name__)

__all__ = ["ShmemSweepWidget"]

# Poll rate used when the render rate is uncapped, so there is no draw cadence
# to match.
DEFAULT_POLL_HZ: float = 60.0


class ShmemSweepWidget(QtWidgets.QWidget):
    """Mirrors a shmem ring and draws it, building the plot on first data.

    The plot cannot be built up front: channel count, sample rate and channel
    names are properties of the stream, and the stream may not exist yet when
    the window opens. So this shows a placeholder, polls, and builds once the
    writer has published something real.
    """

    def __init__(
        self,
        shmem_name: str,
        *,
        display_dur: float = 5.0,
        n_visible: int | None = None,
        poll_hz: float | None = None,
        max_fps: float | None = None,
        n_columns: int | None = None,
        label_fields: typing.Sequence[str] = ("label",),
        show_controls: bool = True,
        placeholder_text: str = "Waiting for data…",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._display_dur = display_dur
        self._n_visible = n_visible
        self._max_fps = max_fps
        self._n_columns = n_columns
        self._label_fields = tuple(label_fields)
        self._show_controls = show_controls

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._layout = layout

        self._placeholder = QtWidgets.QLabel(placeholder_text)
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._placeholder)

        self._sweep: SweepWidget | None = None
        self._controls: ChannelPlotControlsWidget | None = None
        self._shape: StreamShape | None = None
        self._error: str | None = None

        self._shmem_name = shmem_name
        self._mirror = EZShmMirror(shmem_name)

        poll_hz = self._effective_poll_hz(poll_hz, max_fps)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(max(1, int(1000.0 / poll_hz)))
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

        self._idle_ticks = 0
        self._idle_log_every = max(1, int(3.0 * poll_hz))

    # ---- Public API ----------------------------------------------------

    @property
    def sweep(self) -> SweepWidget | None:
        """The inner plot, or None before the first data arrives."""
        return self._sweep

    @property
    def stream_shape(self) -> StreamShape | None:
        """What the widget most recently understood the stream to be."""
        return self._shape

    def shutdown(self) -> None:
        """Stop polling, release the mirror, and close the figure.

        The figure has to go before the Qt widget is destroyed, or rendercanvas
        keeps painting into a deleted canvas.
        """
        self._timer.stop()
        self._mirror.disconnect()
        self._close_figure()

    @property
    def error(self) -> str | None:
        """Why the widget gave up, or None if it has not."""
        return self._error

    # ---- Internals -----------------------------------------------------

    def _fail(self, message: str) -> None:
        """Stop polling and say why, in the widget and in the log."""
        if self._error is not None:
            return
        self._error = message
        logger.error("%s", message)
        self._timer.stop()
        if self._placeholder is not None:
            self._placeholder.setText(message)
            self._placeholder.setWordWrap(True)

    @staticmethod
    def _effective_poll_hz(poll_hz: float | None, max_fps: float | None) -> float:
        """Resolve how often to read the ring.

        An explicit rate wins. Otherwise match the render cap: reading faster
        than the plot draws buys nothing but copies. With no cap there is no
        cadence to match, so fall back to a default.
        """
        if poll_hz is not None and poll_hz > 0:
            return float(poll_hz)
        if max_fps is not None and max_fps > 0:
            return float(max_fps)
        return DEFAULT_POLL_HZ

    def _close_figure(self) -> None:
        if self._sweep is None:
            return
        figure = getattr(self._sweep, "_figure", None)
        if figure is not None:
            try:
                figure.close()
            except Exception:
                logger.exception("closing the sweep figure raised; continuing teardown")

    def _on_tick(self) -> None:
        samples, _overflow = self._mirror.auto_view()

        shape = describe_mirror(self._mirror, label_fields=self._label_fields)
        if shape is None or shape.srate <= 0:
            self._idle_ticks += 1
            if self._idle_ticks % self._idle_log_every == 0:
                logger.info("Waiting for data on shmem %r — nothing published yet.", self._shmem_name)
            return
        if self._idle_ticks:
            logger.info("Connected to shmem %r; data is flowing.", self._shmem_name)
            self._idle_ticks = 0

        try:
            require_sweep_renderable(shape)
        except UnsupportedMetricError as exc:
            # Stop rather than draw it as something it is not. Reported once and
            # the timer stopped, because raising out of a Qt slot would repeat
            # this every tick for as long as the window is open.
            self._fail(str(exc))
            return

        self._apply_shape(shape)

        if samples is not None and samples.size:
            self._sweep.push_data(np.ascontiguousarray(flatten_for_plot(samples, shape), dtype=np.float32))

    def _apply_shape(self, shape: StreamShape) -> None:
        """Build the plot, or reconfigure it if the stream changed underneath."""
        previous, self._shape = self._shape, shape
        if self._sweep is None:
            self._build(shape)
            return
        if previous == shape:
            return
        # A changed rate or envelope mode means the buffer's whole layout is
        # wrong; anything else resizes in place, which avoids the plot flashing
        # every time a user narrows the channel selection.
        if previous is None or previous.srate != shape.srate or previous.envelope != shape.envelope:
            self._build(shape)
        else:
            self._sweep.update_config(self._config_for(shape))
            self._sweep.set_channel_labels(self._labels_for(shape))

    def _labels_for(self, shape: StreamShape) -> list[str]:
        labels = shape.channel_labels
        if labels is not None and len(labels) >= shape.n_channels:
            return list(labels[: shape.n_channels])
        return [f"ch{i}" for i in range(shape.n_channels)]

    def _config_for(self, shape: StreamShape) -> SweepConfig:
        kwargs: dict[str, typing.Any] = {}
        if self._n_columns is not None:
            kwargs["n_columns"] = self._n_columns
        if self._max_fps is not None:
            kwargs["max_fps"] = self._max_fps
        return SweepConfig(
            n_channels=shape.n_channels,
            # For an envelope this is the bucket rate, which is what the buffer
            # must be sized with -- describe_mirror reads it off the ring header,
            # so it is already post-decimation.
            srate=shape.srate,
            display_dur=self._display_dur,
            n_visible=self._n_visible if self._n_visible is not None else shape.n_channels,
            channel_labels=self._labels_for(shape),
            envelope=shape.envelope,
            **kwargs,
        )

    def _build(self, shape: StreamShape) -> None:
        logger.info(
            "Building sweep: %d channels @ %.1f Hz%s",
            shape.n_channels,
            shape.srate,
            " (min/max envelope)" if shape.envelope else "",
        )
        # Keep whatever time span the user had scrolled to across a rebuild.
        if self._sweep is not None:
            buf = getattr(self._sweep, "sweep_buffer", None)
            dur = getattr(buf, "display_dur", None)
            if dur:
                self._display_dur = dur
            self._close_figure()
            self._layout.removeWidget(self._sweep)
            self._sweep.deleteLater()
            self._sweep = None
        if self._controls is not None:
            self._layout.removeWidget(self._controls)
            self._controls.deleteLater()
            self._controls = None
        if self._placeholder is not None:
            self._layout.removeWidget(self._placeholder)
            self._placeholder.deleteLater()
            self._placeholder = None

        self._sweep = SweepWidget(self._config_for(shape), parent=self)
        self._layout.addWidget(self._sweep)
        self._sweep.set_channel_labels_visible(True)
        if self._show_controls:
            self._controls = ChannelPlotControlsWidget(self._sweep, parent=self)
            self._layout.addWidget(self._controls)
