"""ezmsg-viewer — plot a specific ezmsg topic without the graph inspector."""

import logging
import sys
from enum import Enum

import numpy as np
import typer
from ezmsg.qt import EzSession, EzSubscriber
from ezmsg.util.messages.axisarray import AxisArray
from phosphor import (
    ScatterConfig,
    ScatterWidget,
    SpectrumConfig,
    SpectrumWidget,
    SweepConfig,
    SweepEvent,
    SweepWidget,
)
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

from ezmsg.tools.plot.describe import (
    describe_axisarray,
    flatten_for_plot,
    require_sweep_renderable,
)

logger = logging.getLogger(__name__)

GRAPH_IP = "127.0.0.1"
GRAPH_PORT = 25978

# Event color palette (deterministic by label hash)
EVENT_COLORS = [
    (1.0, 1.0, 0.4),  # yellow
    (0.4, 1.0, 1.0),  # cyan
    (1.0, 0.4, 1.0),  # magenta
    (1.0, 0.7, 0.3),  # orange
    (0.4, 1.0, 0.4),  # green
    (1.0, 0.4, 0.4),  # red
]


class PlotMode(str, Enum):
    timeseries = "timeseries"
    spectral = "spectral"
    scatter = "scatter"


def _extract_channel_meta(msg) -> tuple[list[str] | None, np.ndarray | None]:
    """Extract channel labels and 2D positions from AxisArray channel metadata."""
    if "ch" not in msg.dims:
        return None, None

    ch_axis = msg.get_axis("ch")
    ch_data = getattr(ch_axis, "data", None)
    if ch_data is None or ch_data.dtype.names is None:
        return None, None

    labels = None
    if "label" in ch_data.dtype.names:
        labels = [str(v) for v in ch_data["label"]]

    positions = None
    if "x" in ch_data.dtype.names and "y" in ch_data.dtype.names:
        x = ch_data["x"].astype(np.float32)
        y = ch_data["y"].astype(np.float32)
        if np.any(x != 0) or np.any(y != 0):
            positions = np.column_stack([x, y])

    return labels, positions


def _event_label(msg) -> str:
    """Try to extract a human-readable label from an event message."""
    if hasattr(msg, "dims") and hasattr(msg, "get_axis"):
        if "ch" in msg.dims:
            ch_axis = msg.get_axis("ch")
            ch_data = getattr(ch_axis, "data", None)
            if ch_data is not None and ch_data.dtype.names and "label" in ch_data.dtype.names:
                labels = ch_data["label"]
                if len(labels) > 0:
                    return str(labels[0])
    return ""


class ViewerWindow(QMainWindow):
    def __init__(
        self,
        session: EzSession,
        mode: PlotMode,
        data_topic: str,
        event_topic: str | None = None,
        event_filter: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"ezmsg Viewer — {data_topic}")
        self._mode = mode
        self._event_filter = event_filter

        self._data_sub = EzSubscriber(topic=data_topic, parent=self, session=session)
        self._data_sub.connect(self._on_data)

        self._event_sub: EzSubscriber | None = None
        if event_topic:
            self._event_sub = EzSubscriber(topic=event_topic, parent=self, session=session)
            self._event_sub.connect(self._on_event)

        self._plot_widget: QWidget | None = None
        self._first_message = True
        self._channel_labels: list[str] | None = None
        self._channel_positions: np.ndarray | None = None
        # What describe_axisarray made of the stream; rebuilt on topic change.
        self._shape = None

    # ------------------------------------------------------------------
    # Data handling
    # ------------------------------------------------------------------

    def _on_data(self, msg) -> None:
        if self._first_message:
            self._channel_labels, self._channel_positions = _extract_channel_meta(msg)
            self._create_plot_widget(msg)
            self._first_message = False
        self._push_message(msg)

    def _create_plot_widget(self, msg) -> None:
        labels = self._channel_labels

        if self._mode == PlotMode.timeseries:
            shape = describe_axisarray(msg)
            if not shape.srate:
                logger.warning("No usable 'time' axis — assuming 1 kHz")
                shape = shape._replace(srate=1000.0)
            require_sweep_renderable(shape)
            self._shape = shape
            config = SweepConfig(
                n_channels=shape.n_channels,
                srate=shape.srate,
                channel_labels=labels or shape.channel_labels,
                envelope=shape.envelope,
            )
            widget = SweepWidget(config)

        elif self._mode == PlotMode.spectral:
            if "freq" not in msg.dims:
                logger.error("Spectral mode requires 'freq' dimension in data")
                sys.exit(1)

            freq_axis = msg.get_axis("freq")
            freq_idx = msg.get_axis_idx("freq")
            n_bins = msg.shape[freq_idx]
            srate = 2.0 * freq_axis.gain * n_bins
            n_channels = msg.data.size // n_bins

            config = SpectrumConfig(n_channels=n_channels, srate=srate, n_bins=n_bins, channel_labels=labels)
            widget = SpectrumWidget(config)

        elif self._mode == PlotMode.scatter:
            if self._channel_positions is None:
                logger.error("Scatter mode requires channel position metadata (x, y fields in ch axis)")
                sys.exit(1)

            config = ScatterConfig(positions=self._channel_positions, channel_labels=labels)
            widget = ScatterWidget(config)

        self._plot_widget = widget
        self.setCentralWidget(widget)

    def _push_message(self, msg) -> None:
        widget = self._plot_widget

        if isinstance(widget, SweepWidget):
            time_idx = msg.get_axis_idx("time") if "time" in msg.dims else 0
            shape = self._shape or describe_axisarray(msg)
            data = flatten_for_plot(np.moveaxis(msg.data, time_idx, 0), shape)
            # Pass the AxisArray time-axis offset so the sweep buffer
            # tracks the same clock as the event timestamps.
            ts = msg.get_axis("time").offset if "time" in msg.dims else None
            widget.push_data(data.astype(np.float32), timestamps=ts)

        elif isinstance(widget, SpectrumWidget):
            freq_idx = msg.get_axis_idx("freq") if "freq" in msg.dims else 0
            n_bins = msg.shape[freq_idx]
            n_channels = msg.data.size // n_bins if n_bins > 0 else 1
            data_2d = np.moveaxis(msg.data, freq_idx, 0).reshape(n_bins, n_channels)
            widget.push_data(data_2d.astype(np.float32))

        elif isinstance(widget, ScatterWidget):
            if len(msg.shape) > 1:
                targ_idx = 0
                if "time" in msg.dims or "freq" in msg.dims:
                    targ_idx = msg.get_axis_idx("time") if "time" in msg.dims else msg.get_axis_idx("freq")
                n_items = msg.shape[targ_idx]
                n_channels = msg.data.size // n_items if n_items > 0 else 1
                data_2d = np.moveaxis(msg.data, targ_idx, 0).reshape(n_items, n_channels)
            else:
                data_2d = msg.data.reshape(1, msg.data.size)
            widget.push_data(data_2d.astype(np.float32))

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _on_event(self, msg: AxisArray) -> None:
        widget = self._plot_widget
        if not isinstance(widget, SweepWidget):
            return

        if "time" not in msg.dims:
            logger.warning("Event message must have 'time' dimension")
            return

        time_axis = msg.get_axis("time")
        time_idx = msg.get_axis_idx("time")
        timestamps = time_axis.value(list(range(msg.shape[time_idx])))
        events: list[SweepEvent] = []
        for ev_ix, ts in enumerate(timestamps):
            label = msg.data[ev_ix, 0]
            if self._event_filter and self._event_filter not in label:
                continue
            color = EVENT_COLORS[hash(label) % len(EVENT_COLORS)]
            events.append(SweepEvent(t_elapsed=ts, label=label, color=color))
        widget.push_events(events)


def _run(
    data_topic: str = typer.Argument(..., help="ezmsg topic for continuous data"),
    mode: PlotMode = typer.Option(PlotMode.timeseries, help="Plot mode"),
    event_topic: str | None = typer.Option(None, "--events", help="ezmsg topic for event markers"),
    event_filter: str | None = typer.Option(
        None, "--event-filter", help="Only show events whose label contains this string"
    ),
    graph_addr: str = typer.Option(
        ":".join((GRAPH_IP, str(GRAPH_PORT))),
        help="ezmsg graph address (ip:port)",
    ),
) -> None:
    graph_ip, graph_port_str = graph_addr.split(":")
    graph_address = (graph_ip, int(graph_port_str))

    app = QApplication.instance() or QApplication(sys.argv)
    session = EzSession(graph_address=graph_address)
    window = ViewerWindow(
        session=session,
        mode=mode,
        data_topic=data_topic,
        event_topic=event_topic,
        event_filter=event_filter,
    )
    window.showMaximized()
    with session:
        app.exec()


def main() -> None:
    typer.run(_run)


if __name__ == "__main__":
    main()
