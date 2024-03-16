import asyncio
from dataclasses import dataclass
import multiprocessing.connection
from multiprocessing.shared_memory import SharedMemory
import typing

import numpy as np
import numpy.typing as npt
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray


UINT64_SIZE = 8
BYTEORDER = "little"


def to_bytes(data: typing.Any) -> bytes:
    if isinstance(data, bool):
        return data.to_bytes(2, byteorder=BYTEORDER, signed=False)
    elif isinstance(data, int):
        return np.int64(data).to_bytes(UINT64_SIZE, BYTEORDER, signed=False)


class ShMemCircBuffSettings(ez.Settings):
    shmem_name: typing.Optional[str]
    ipconn: multiprocessing.connection.Connection
    topic: str
    buf_dur: float
    axis: str = "time"


class ShMemCircBuffState(ez.State):
    cur_settings: ShMemCircBuffSettings
    write_index: typing.Optional[npt.NDArray] = None
    shm_arr: typing.Optional[SharedMemory] = None
    buffer: typing.Optional[npt.NDArray] = None


class ShMemCircBuff(ez.Unit):
    SETTINGS: ShMemCircBuffSettings
    STATE: ShMemCircBuffState

    INPUT_SIGNAL = ez.InputStream(AxisArray)

    def initialize(self) -> None:
        self.STATE.cur_settings = self.SETTINGS

    def reset_shmem(self, msg: typing.Optional[AxisArray] = None) -> None:
        # Now that we have a message, calculate the metadata and init the shmem
        ax_idx = msg.get_axis_idx(self.SETTINGS.axis)
        axis = msg.axes[self.STATE.cur_settings.axis]
        n_frames = int(np.ceil(self.STATE.cur_settings.buf_dur / axis.gain))
        frame_shape = msg.data.shape[:ax_idx] + msg.data.shape[ax_idx + 1 :]
        shmem_meta = {
            "dtype": msg.data.dtype,
            "srate": 1 / axis.gain,
            "shape": (n_frames,) + frame_shape,
        }

        # And create and fill-in shm_arr.
        shm_arr_size = 8 + n_frames * np.prod(frame_shape) * msg.data.itemsize
        try:
            self.STATE.shm_arr = SharedMemory(
                name=self.STATE.cur_settings.shmem_name,
                create=True,
                size=shm_arr_size,
            )
        except FileExistsError:
            old_shm = SharedMemory(
                name=self.STATE.cur_settings.shmem_name, create=False
            )
            old_shm.close()
            old_shm.unlink()
            # Failed cleanup on previous run. Reuse the location.
            self.STATE.shm_arr = SharedMemory(
                name=self.STATE.cur_settings.shmem_name,
                create=True,
                size=shm_arr_size,
            )
        if self.STATE.cur_settings.shmem_name is None:
            self.STATE.cur_settings.shmem_name = self.STATE.shm_arr.name
        self.STATE.write_index = np.ndarray(
            (1,), dtype=np.uint64, buffer=self.STATE.shm_arr.buf[:8]
        )
        self.STATE.buffer = np.ndarray(
            shmem_meta["shape"],
            dtype=shmem_meta["dtype"],
            buffer=self.STATE.shm_arr.buf[8:],
        )
        self.STATE.cur_settings.ipconn.send(shmem_meta)

    @ez.task
    async def check_continue(self):
        while True:
            if self.STATE.cur_settings.ipconn.poll():
                in_msg = self.STATE.cur_settings.ipconn.recv()
                if "kill" in in_msg and in_msg["kill"]:
                    if self.STATE.shm_arr is not None:
                        self.STATE.shm_arr.close()
                        self.STATE.shm_arr.unlink()
                    break
            else:
                await asyncio.sleep(0.2)
        raise ez.NormalTermination

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    async def handle_message(self, msg: AxisArray):
        if not type(msg) is AxisArray:
            return
        if self.STATE.cur_settings.axis not in msg.dims:
            return
        ax_idx = msg.get_axis_idx(self.STATE.cur_settings.axis)
        data = np.moveaxis(msg.data, ax_idx, 0)
        if self.STATE.buffer is None or self.STATE.buffer.shape[1:] != data.shape[1:]:
            self.reset_shmem(msg)
        n_samples = data.shape[0]
        write_stop = int(self.STATE.write_index[0] + n_samples)

        if write_stop > self.STATE.buffer.shape[0]:
            overflow = write_stop - self.STATE.buffer.shape[0]
            self.STATE.buffer[self.STATE.write_index[0] :] = data[
                : n_samples - overflow
            ]
            self.STATE.buffer[:overflow] = data[n_samples - overflow :]
            self.STATE.write_index[0] = overflow
        else:
            self.STATE.buffer[self.STATE.write_index[0] : write_stop] = data[:]
            self.STATE.write_index[0] = write_stop
