"""In-process ezmsg bridge for sigmon.

Runs a minimal ezmsg Unit in a daemon thread that subscribes to a topic
and appends incoming AxisArray messages to a collections.deque, which
the Qt main thread drains via QTimer.
"""

import asyncio
import collections
import logging
import threading

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray

logger = logging.getLogger(__name__)


class BridgeUnit(ez.Unit):
    # Leaky so that sigmon never applies backpressure to the real-time system,
    # and so that handle_subscriber uses recv() (deepcopy) — the deque stores
    # messages beyond the recv_zero_copy context, so they must be independent copies.
    INPUT = ez.InputStream(AxisArray, leaky=True, max_queue=2)

    # _data_deque and _shutdown_event are set by EZBridge before ez.run().
    # They must NOT be set in initialize() — ezmsg calls initialize() after
    # the bridge wires them in, which would overwrite with fresh instances.

    @ez.subscriber(INPUT)
    async def on_data(self, msg: AxisArray) -> None:
        self._data_deque.append(msg)

    @ez.task
    async def check_shutdown(self) -> None:
        while not self._shutdown_event.is_set():
            await asyncio.sleep(0.05)
        raise ez.NormalTermination


class EZBridge:
    """Manages BridgeUnit lifecycle and provides a deque-based data interface."""

    def __init__(self, graph_address: tuple[str, int]) -> None:
        self._graph_address = graph_address
        self._unit = BridgeUnit()
        self._deque: collections.deque[AxisArray] = collections.deque(maxlen=4096)
        self._shutdown_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_topic: str | None = None

    def start(self) -> None:
        # Wire deque and shutdown event into the unit before starting the thread.
        self._unit._data_deque = self._deque
        self._unit._shutdown_event = self._shutdown_event

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            ez.run(
                components={"SIGMON": self._unit},
                connections=[],
                graph_address=self._graph_address,
            )
        except Exception:
            logger.exception("EZBridge thread exited with error")

    def subscribe(self, topic: str) -> None:
        """Subscribe to a new topic. Called from the Qt main thread."""
        gs = ez.graphserver.GraphService(address=self._graph_address)

        # Disconnect old topic if any.
        if self._current_topic is not None:
            try:
                asyncio.run(gs.disconnect(self._current_topic, "SIGMON/INPUT"))
            except Exception:
                logger.warning("Failed to disconnect %s", self._current_topic)

        self._deque.clear()

        # Connect new topic.
        try:
            asyncio.run(gs.connect(topic, "SIGMON/INPUT"))
            logger.debug("Connected %s -> SIGMON/INPUT", topic)
        except Exception:
            logger.error("Failed to connect %s", topic, exc_info=True)
            return

        self._current_topic = topic

    def try_get(self) -> AxisArray | None:
        """Non-blocking pop from the deque. Returns None if empty."""
        try:
            return self._deque.popleft()
        except IndexError:
            return None

    def stop(self) -> None:
        """Shutdown the bridge thread cleanly."""
        # Disconnect current topic.
        if self._current_topic is not None:
            gs = ez.graphserver.GraphService(address=self._graph_address)
            try:
                asyncio.run(gs.disconnect(self._current_topic, "SIGMON/INPUT"))
            except Exception:
                logger.debug("Failed to disconnect on stop")
            self._current_topic = None

        # Signal the unit to terminate.
        self._shutdown_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
