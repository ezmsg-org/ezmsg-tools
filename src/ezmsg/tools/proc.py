import asyncio
import multiprocessing
import multiprocessing.connection
import typing

import ezmsg.core as ez

from .shmem.shmem import ShMemCircBuff, ShMemCircBuffSettings

BUF_DUR = 3.0


class EzMonitorProcess(multiprocessing.Process):
    def __init__(
        self,
        settings: ShMemCircBuffSettings,
        topic: str,
        address: typing.Optional[typing.Tuple[str, int]] = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._topic = topic
        self._graph_address = address

    def run(self) -> None:
        comps = {"SHMEM": ShMemCircBuff(self._settings)}
        conns = ((self._topic, comps["SHMEM"].INPUT_SIGNAL),)
        ez.run(components=comps, connections=conns, graph_address=self._graph_address)


class EZProcManager:
    """
    Manages the subprocess that runs an ezmsg pipeline comprising a single ShMemCircBuff unit connected to a pipeline.
    The unit must be parameterized with the correct shared memory name.
    We do not actually interact with the shared memory in this class. See .mirror.EzmsgShmMirror.
    """

    def __init__(self, graph_ip: str, graph_port: int, buf_dur: float = BUF_DUR) -> None:
        self._graph_addr: typing.Tuple[str, int] = (graph_ip, graph_port)
        self._buf_dur = buf_dur
        self._proc = None
        self._node_path: typing.Optional[str] = None
        self._remote_conn, self._conn = multiprocessing.Pipe()

    @property
    def node_path(self) -> str:
        return self._node_path

    @property
    def conn(self) -> typing.Optional[multiprocessing.connection.Connection]:
        return self._conn

    def reset(self, node_path: typing.Optional[str]) -> None:
        self._cleanup_subprocess()
        self._node_path = node_path
        self._init_subprocess()

    def cleanup(self):
        self._cleanup_subprocess()

    def _cleanup_subprocess(self) -> None:
        if self._proc is not None:
            self._conn.send("quit")
            # Close process
            self._proc.join()
            self._proc = None

            # TODO: Somehow closing the proc doesn't always clear the VISBUFF connections.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                ez.graphserver.GraphService(address=self._graph_addr).disconnect(
                    self._node_path, "VISBUFF/INPUT_SIGNAL"
                )
            )

    def _init_subprocess(self, axis: str = "time"):
        unit_settings = ShMemCircBuffSettings(
            shmem_name="buff_" + self._node_path,
            buf_dur=self._buf_dur,
            conn=self._remote_conn,
            axis=axis,
        )
        self._proc = EzMonitorProcess(unit_settings, self._node_path, address=self._graph_addr)
        self._proc.start()

    # if self._rend_conn.poll(): msg = self._rend_conn.recv()
