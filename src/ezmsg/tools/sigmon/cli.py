"""Sigmon — real-time ezmsg graph inspector using Qt + phosphor."""

import logging
import sys

import numpy as np
import typer
from ezmsg.qt import EzSession, EzSubscriber
from phosphor import (
    ScatterConfig,
    ScatterWidget,
    SpectrumConfig,
    SpectrumWidget,
    SweepConfig,
    SweepWidget,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QMainWindow, QSplitter, QWidget

from ezmsg.tools.plot.describe import (
    describe_axisarray,
    flatten_for_plot,
    require_sweep_renderable,
)
from ezmsg.tools.sigmon.dag_widget import DAGWidget

logger = logging.getLogger(__name__)

GRAPH_IP = "127.0.0.1"
GRAPH_PORT = 25978


def _extract_channel_meta(msg) -> tuple[list[str] | None, np.ndarray | None]:
    """Extract channel labels and 2D positions from AxisArray channel metadata.

    Returns ``(labels, positions)`` where *positions* is ``(n_channels, 2)``
    float32 or ``None`` if no location fields are present.
    """
    if "ch" not in msg.dims:
        return None, None

    ch_axis = msg.get_axis("ch")
    ch_data = getattr(ch_axis, "data", None)
    if ch_data is None or ch_data.dtype.names is None:
        return None, None

    # Labels
    labels = None
    if "label" in ch_data.dtype.names:
        labels = [str(v) for v in ch_data["label"]]

    # Positions — need at least x and y
    positions = None
    if "x" in ch_data.dtype.names and "y" in ch_data.dtype.names:
        x = ch_data["x"].astype(np.float32)
        y = ch_data["y"].astype(np.float32)
        if np.any(x != 0) or np.any(y != 0):
            positions = np.column_stack([x, y])

    return labels, positions


class SigmonWindow(QMainWindow):
    def __init__(
        self,
        graph_address: tuple[str, int],
        session: EzSession,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("ezmsg Signal Monitor")
        self._graph_address = graph_address

        # Dynamic subscriber — switches topics when the user clicks a graph node.
        self._data_sub = EzSubscriber(topic=None, parent=self, session=session)
        self._data_sub.connect(self._on_data)

        # Layout: splitter with DAG on left, plot on right.
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self._splitter)

        self._dag_widget = DAGWidget(graph_address)
        self._dag_widget.node_selected.connect(self._on_node_selected)
        self._splitter.addWidget(self._dag_widget)

        self._plot_widget: QWidget = QWidget()  # placeholder
        self._splitter.addWidget(self._plot_widget)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 3)

        self._first_message = True

        # Channel metadata cached from the first message of each topic.
        self._channel_labels: list[str] | None = None
        self._channel_positions: np.ndarray | None = None
        # What describe_axisarray made of the stream; rebuilt on topic change.
        self._shape = None
        # Cached parameters for rebuilding the primary (sweep/spectrum) widget.
        self._primary_config: SweepConfig | SpectrumConfig | None = None
        self._showing_scatter = False

        # Hotkey: "M" toggles scatter/map view (only effective when positions exist).
        shortcut = QShortcut(QKeySequence("M"), self)
        shortcut.activated.connect(self._toggle_scatter)

    def _on_node_selected(self, topic: str) -> None:
        # logger.debug("Switching to topic: %s", topic)
        print(f"Switching to topic: {topic}")
        self._data_sub.set_topic(topic)
        self._first_message = True
        self._channel_labels = None
        self._channel_positions = None
        self._shape = None
        self._primary_config = None
        self._showing_scatter = False

    def _on_data(self, msg) -> None:
        """Handle a message delivered by the dynamic subscriber."""
        if self._first_message:
            self._channel_labels, self._channel_positions = _extract_channel_meta(msg)
            self._create_plot_widget(msg)
            self._first_message = False

        self._push_message(msg)

    def _create_plot_widget(self, msg) -> None:
        """Detect data type from AxisArray dims and create the appropriate widget."""
        labels = self._channel_labels

        if "time" in msg.dims:
            shape = describe_axisarray(msg)
            require_sweep_renderable(shape)
            self._shape = shape
            config = SweepConfig(
                n_channels=shape.n_channels,
                srate=shape.srate,
                channel_labels=labels or shape.channel_labels,
                envelope=shape.envelope,
            )
            widget = SweepWidget(config)

        elif "freq" in msg.dims:
            freq_axis = msg.get_axis("freq")
            freq_idx = msg.get_axis_idx("freq")
            n_bins = msg.shape[freq_idx]
            srate = 2.0 * freq_axis.gain * n_bins
            n_channels = msg.data.size // n_bins

            config = SpectrumConfig(
                n_channels=n_channels,
                srate=srate,
                n_bins=n_bins,
                channel_labels=labels,
            )
            widget = SpectrumWidget(config)

        elif "ch" in msg.dims and self._channel_positions is not None:
            # ch but no time or freq; assume scatter
            config = ScatterConfig(
                positions=self._channel_positions,
                channel_labels=labels,
            )
            widget = ScatterWidget(config)
        else:
            logger.warning("Unknown AxisArray dims: %s — defaulting to sweep", msg.dims)
            n_samples = msg.shape[0]
            n_channels = msg.data.size // n_samples if n_samples > 0 else 1
            config = SweepConfig(
                n_channels=n_channels,
                srate=1000.0,
                channel_labels=labels,
            )
            widget = SweepWidget(config)

        self._primary_config = config
        self._showing_scatter = isinstance(widget, ScatterWidget)
        self._replace_plot_widget(widget)

    def _toggle_scatter(self) -> None:
        """Toggle between primary (sweep/spectrum) view and scatter/map view."""
        if self._channel_positions is None:
            return  # no locations — nothing to toggle

        if self._showing_scatter:
            # Switch back to primary widget.
            if isinstance(self._primary_config, SweepConfig):
                widget = SweepWidget(self._primary_config)
            elif isinstance(self._primary_config, SpectrumConfig):
                widget = SpectrumWidget(self._primary_config)
            else:
                return
            self._showing_scatter = False
        else:
            config = ScatterConfig(
                positions=self._channel_positions,
                channel_labels=self._channel_labels,
            )
            widget = ScatterWidget(config)
            self._showing_scatter = True

        self._replace_plot_widget(widget)

    def _replace_plot_widget(self, widget: QWidget) -> None:
        """Swap the right pane of the splitter."""
        sizes = self._splitter.sizes()
        old = self._splitter.widget(1)
        if old is not None:
            # Stop the render loop before destroying the Qt widget,
            # otherwise fastplotlib keeps painting a deleted canvas.
            if hasattr(old, "_figure"):
                old._figure.close()
            old.setParent(None)
            old.deleteLater()
        self._splitter.insertWidget(1, widget)
        self._splitter.setStretchFactor(1, 3)
        self._splitter.setSizes(sizes)
        self._plot_widget = widget

    def _push_message(self, msg) -> None:
        """Extract 2D data from AxisArray and push to the plot widget."""
        widget = self._plot_widget

        if isinstance(widget, SweepWidget):
            time_idx = msg.get_axis_idx("time") if "time" in msg.dims else 0
            shape = self._shape or describe_axisarray(msg)
            data = flatten_for_plot(np.moveaxis(msg.data, time_idx, 0), shape)
            widget.push_data(data.astype(np.float32))

        elif isinstance(widget, SpectrumWidget):
            freq_idx = msg.get_axis_idx("freq") if "freq" in msg.dims else 0
            n_bins = msg.shape[freq_idx]
            n_channels = msg.data.size // n_bins if n_bins > 0 else 1
            data_2d = np.moveaxis(msg.data, freq_idx, 0).reshape(n_bins, n_channels)
            widget.push_data(data_2d.astype(np.float32))

        elif isinstance(widget, ScatterWidget):
            # Scatter expects (n_channels,) or (n_samples, n_channels).
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


def _run(
    graph_addr: str = ":".join((GRAPH_IP, str(GRAPH_PORT))),
) -> None:
    graph_ip, graph_port_str = graph_addr.split(":")
    graph_address = (graph_ip, int(graph_port_str))

    app = QApplication.instance() or QApplication(sys.argv)
    session = EzSession(graph_address=graph_address)
    window = SigmonWindow(graph_address, session)
    window.showMaximized()
    with session:
        app.exec()


def main() -> None:
    typer.run(_run)


if __name__ == "__main__":
    main()
