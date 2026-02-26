"""Sigmon — real-time ezmsg graph inspector using Qt + phosphor."""

import logging
import sys

import numpy as np
import typer
from ezmsg.qt import EzDynamicSubscriber, EzGuiBridge
from phosphor import SpectrumConfig, SpectrumWidget, SweepConfig, SweepWidget
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QSplitter, QWidget

from ezmsg.tools.sigmon.dag_widget import DAGWidget

logger = logging.getLogger(__name__)

GRAPH_IP = "127.0.0.1"
GRAPH_PORT = 25978


class SigmonWindow(QMainWindow):
    def __init__(
        self,
        graph_address: tuple[str, int],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("ezmsg Signal Monitor")
        self._graph_address = graph_address

        # Dynamic subscriber — switches topics when the user clicks a graph node.
        self._data_sub = EzDynamicSubscriber(parent=self)
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

    def _on_node_selected(self, topic: str) -> None:
        self._data_sub.subscribe(topic)
        self._first_message = True

    def _on_data(self, msg) -> None:
        """Handle a message delivered by the dynamic subscriber."""
        if self._first_message:
            self._create_plot_widget(msg)
            self._first_message = False

        self._push_message(msg)

    def _create_plot_widget(self, msg) -> None:
        """Detect data type from AxisArray dims and create the appropriate widget."""
        if "time" in msg.dims:
            time_axis = msg.get_axis("time")
            srate = 1.0 / time_axis.gain
            time_idx = msg.get_axis_idx("time")
            n_samples = msg.shape[time_idx]
            n_channels = msg.data.size // n_samples

            config = SweepConfig(n_channels=n_channels, srate=srate)
            widget = SweepWidget(config)

        elif "freq" in msg.dims:
            freq_axis = msg.get_axis("freq")
            freq_idx = msg.get_axis_idx("freq")
            n_bins = msg.shape[freq_idx]
            srate = 2.0 * freq_axis.gain * n_bins
            n_channels = msg.data.size // n_bins

            config = SpectrumConfig(n_channels=n_channels, srate=srate, n_bins=n_bins)
            widget = SpectrumWidget(config)

        else:
            logger.warning("Unknown AxisArray dims: %s — defaulting to sweep", msg.dims)
            # Fallback: treat first axis as time-like.
            n_samples = msg.shape[0]
            n_channels = msg.data.size // n_samples if n_samples > 0 else 1
            config = SweepConfig(n_channels=n_channels, srate=1000.0)
            widget = SweepWidget(config)

        self._replace_plot_widget(widget)

    def _replace_plot_widget(self, widget: QWidget) -> None:
        """Swap the right pane of the splitter."""
        sizes = self._splitter.sizes()
        old = self._splitter.widget(1)
        if old is not None:
            # Stop the rendercanvas scheduler before destroying the widget,
            # otherwise it keeps calling update() on a deleted C++ object.
            if hasattr(old, "canvas"):
                old.canvas.close()
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
            n_samples = msg.shape[time_idx]
            n_channels = msg.data.size // n_samples if n_samples > 0 else 1
            data_2d = np.moveaxis(msg.data, time_idx, 0).reshape(n_samples, n_channels)
            widget.push_data(data_2d.astype(np.float32))

        elif isinstance(widget, SpectrumWidget):
            freq_idx = msg.get_axis_idx("freq") if "freq" in msg.dims else 0
            n_bins = msg.shape[freq_idx]
            n_channels = msg.data.size // n_bins if n_bins > 0 else 1
            data_2d = np.moveaxis(msg.data, freq_idx, 0).reshape(n_bins, n_channels)
            widget.push_data(data_2d.astype(np.float32))


def _run(
    graph_addr: str = ":".join((GRAPH_IP, str(GRAPH_PORT))),
) -> None:
    graph_ip, graph_port_str = graph_addr.split(":")
    graph_address = (graph_ip, int(graph_port_str))

    app = QApplication.instance() or QApplication(sys.argv)
    window = SigmonWindow(graph_address)
    window.showMaximized()
    with EzGuiBridge(app, graph_address=graph_address):
        app.exec()


def main() -> None:
    typer.run(_run)


if __name__ == "__main__":
    main()
