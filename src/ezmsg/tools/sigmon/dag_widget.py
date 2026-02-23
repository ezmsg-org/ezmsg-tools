"""Qt widget for interactive DAG visualization."""

import logging
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QEvent, QObject, QSize, Signal
from PySide6.QtGui import QMouseEvent, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..dag import get_graph, pgv2pd

logger = logging.getLogger(__name__)


class DAGWidget(QWidget):
    """Interactive DAG graph view with click-to-select nodes."""

    node_selected = Signal(str)

    def __init__(
        self,
        graph_address: tuple[str, int],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._graph_address = graph_address
        self._node_df = pd.DataFrame(columns=["name", "x", "y", "upstream"])
        self._full_pixmap: QPixmap | None = None
        self._display_scale: float = 1.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._refresh_graph)
        layout.addWidget(self._refresh_btn)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(False)
        layout.addWidget(self._scroll_area)

        self._label = QLabel()
        self._label.setScaledContents(False)
        self._scroll_area.setWidget(self._label)
        self._scroll_area.viewport().installEventFilter(self)

        self._refresh_graph()

    def _refresh_graph(self) -> None:
        G = get_graph(self._graph_address)

        if len(G.nodes()) == 0:
            self._node_df = pd.DataFrame(columns=["name", "x", "y", "upstream"])
            self._full_pixmap = None
            self._label.setText("No graph connected")
            return

        G.layout(prog="dot")

        # Render SVG for coordinate extraction.
        svg_path = Path(tempfile.gettempdir()) / "ezmsg-graphviz.svg"
        G.draw(svg_path, format="svg:cairo")

        # Extract node positions from the layout.
        self._node_df = pgv2pd(G)

        # Render PNG for display.
        img_path = Path(tempfile.gettempdir()) / "ezmsg-graphviz.png"
        G.draw(img_path)
        self._full_pixmap = QPixmap(str(img_path))

        # Compute coordinate scale from SVG (points) to PNG (pixels).
        # Parse SVG viewBox for native graphviz coordinate dimensions —
        # avoids Qt's DPI-dependent SVG rasterization (wrong on Retina).
        tree = ET.parse(svg_path)
        root = tree.getroot()
        viewbox = root.get("viewBox")
        if viewbox:
            parts = viewbox.split()
            svg_width = float(parts[2])
            svg_height = float(parts[3])
        else:
            # Fallback: parse width/height attributes (e.g. "400pt").
            svg_width = float("".join(c for c in root.get("width", "1") if c.isdigit() or c == "."))
            svg_height = float("".join(c for c in root.get("height", "1") if c.isdigit() or c == "."))
        x_scale = self._full_pixmap.width() / svg_width if svg_width else 1.0
        y_scale = self._full_pixmap.height() / svg_height if svg_height else 1.0

        self._node_df["x"] *= x_scale
        self._node_df["y"] *= y_scale
        # Invert Y so origin is top-left (matching PNG pixel coords).
        self._node_df["y"] = self._full_pixmap.height() - self._node_df["y"]

        self._scale_and_display()

    def _scale_and_display(self) -> None:
        if self._full_pixmap is None:
            return
        available_width = self._scroll_area.viewport().width()
        if available_width <= 0:
            available_width = self.width()
        scaled = self._full_pixmap.scaledToWidth(max(available_width, 1))
        self._display_scale = scaled.width() / self._full_pixmap.width() if self._full_pixmap.width() else 1.0
        self._label.setPixmap(scaled)
        self._label.adjustSize()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._scale_and_display()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is self._scroll_area.viewport() and event.type() == QEvent.Type.MouseButtonPress:
            self._handle_click(event)
            return True
        return super().eventFilter(obj, event)

    def _handle_click(self, event: QMouseEvent) -> None:
        if self._full_pixmap is None or len(self._node_df) == 0:
            return

        # Viewport click + scroll offset = position in scaled image.
        vx = event.position().x()
        vy = event.position().y()
        sx = self._scroll_area.horizontalScrollBar().value()
        sy = self._scroll_area.verticalScrollBar().value()
        img_x = vx + sx
        img_y = vy + sy

        # Convert to full-pixmap coords.
        px_x = img_x / self._display_scale if self._display_scale > 0 else img_x
        px_y = img_y / self._display_scale if self._display_scale > 0 else img_y

        # Find nearest node via Euclidean distance.
        dist_sq = (self._node_df["x"] - px_x) ** 2 + (self._node_df["y"] - px_y) ** 2
        min_idx = dist_sq.argmin()
        topic = str(self._node_df.iloc[min_idx]["upstream"])
        self.node_selected.emit(topic)

    def sizeHint(self) -> QSize:
        return QSize(300, 600)
