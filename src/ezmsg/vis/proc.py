import asyncio
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import typing

import ezmsg.core as ez
import numpy.typing as npt

from .unit import ShMemCircBuffSettings, ShMemCircBuff, ShmemArrMeta


BUF_DUR = 3.0


class EzMonitorProcess(multiprocessing.Process):
    settings: ShMemCircBuffSettings

    def __init__(
        self,
        settings: ShMemCircBuffSettings,
        topic: str,
        address: typing.Optional[typing.Tuple[str, int]] = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self._topic = topic
        self._graph_address = address

    def run(self) -> None:
        components = {"VISBUFF": ShMemCircBuff(self.settings)}
        ez.run(
            components=components,
            connections=(
                (
                    self._topic,
                    components["VISBUFF"].INPUT_SIGNAL,
                ),
            ),
            graph_address=self._graph_address,
        )


class EZProcManager:
    """
    Manages the subprocess that runs an ezmsg pipeline comprising a single ShMemCircBuff unit connected to a pipeline.
    The unit must be parameterized with the correct shared memory name.
    We do not actually interact with the shared memory in this class. See .mirror.EzmsgShmMirror.
    """

    def __init__(
        self, graph_ip: str, graph_port: int, shmem_name: str, buf_dur: float = BUF_DUR
    ) -> None:
        self._graph_addr: typing.Tuple[str, int] = (graph_ip, graph_port)
        self._shmem_name = shmem_name
        self._buf_dur = buf_dur
        self._proc = None
        self._node_path: typing.Optional[str] = None

    @property
    def node_path(self) -> str:
        return self._node_path

    def reset(self, node_path: typing.Optional[str]) -> None:
        self._cleanup_subprocess()
        self._node_path = node_path
        self._init_subprocess()

    def cleanup(self):
        self._cleanup_subprocess()

    def _cleanup_subprocess(self) -> None:
        if self._proc is not None:
            # Send message to kill process. self._shmem_meta.running = False
            old_shm = None
            try:
                old_shm = SharedMemory(name=self._shmem_name, create=False)
                meta = ShmemArrMeta.from_buffer(old_shm.buf)
                meta.running = False
            except FileNotFoundError:
                # Not sure how we can get to this state... proc is running but shmem doesn't exist.
                pass
            # Close process
            self._proc.join()
            self._proc = None
            if old_shm is not None:
                meta = None
                try:
                    old_shm.close()
                except BufferError as e:
                    print("EZProcManager._cleanup_subprocess():", e)
            # TODO: Somehow closing the proc isn't enough to clear the VISBUFF connections.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                ez.GraphService(address=self._graph_addr).disconnect(
                    self._node_path, "VISBUFF/INPUT_SIGNAL"
                )
            )

    def _init_subprocess(self, axis: str = "time"):
        unit_settings = ShMemCircBuffSettings(
            shmem_name=self._shmem_name,
            buf_dur=self._buf_dur,
            axis=axis,
        )
        self._proc = EzMonitorProcess(
            unit_settings, self._node_path, address=self._graph_addr
        )
        self._proc.start()

    # if self._rend_conn.poll(): msg = self._rend_conn.recv()
