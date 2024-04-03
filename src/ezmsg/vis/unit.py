import asyncio
import ctypes
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


class ShmemArrMeta(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("running", ctypes.c_bool),
        ("dtype", ctypes.c_char),
        ("srate", ctypes.c_double),
        ("ndim", ctypes.c_uint32),
        ("shape", ctypes.c_uint32 * 64),
        ("write_index", ctypes.c_uint64),
    ]


class ShMemCircBuffSettings(ez.Settings):
    shmem_name: typing.Optional[str]
    buf_dur: float
    axis: str = "time"


class ShMemCircBuffState(ez.State):
    cur_settings: ShMemCircBuffSettings
    shmem: typing.Optional[SharedMemory] = None
    meta: typing.Optional[ShmemArrMeta] = None
    buffer: typing.Optional[npt.NDArray] = None


class ShMemCircBuff(ez.Unit):
    SETTINGS: ShMemCircBuffSettings
    STATE: ShMemCircBuffState

    INPUT_SIGNAL = ez.InputStream(AxisArray)

    def initialize(self) -> None:
        self.STATE.cur_settings = self.SETTINGS

    def reset_shmem(self, msg: AxisArray) -> None:
        # Now that we have a message, calculate the metadata and init the shmem
        ax_idx = msg.get_axis_idx(self.SETTINGS.axis)
        axis = msg.axes[self.STATE.cur_settings.axis]
        n_frames = int(np.ceil(self.STATE.cur_settings.buf_dur / axis.gain))
        frame_shape = msg.data.shape[:ax_idx] + msg.data.shape[ax_idx + 1 :]

        # And create and fill-in shmem.
        meta_size = ctypes.sizeof(ShmemArrMeta)
        shmem_size = int(meta_size + n_frames * np.prod(frame_shape) * msg.data.itemsize)
        try:
            self.STATE.shmem = SharedMemory(
                name=self.STATE.cur_settings.shmem_name,
                create=True,
                size=shmem_size,
            )
        except FileExistsError:
            if self.STATE.meta is not None:
                self.STATE.meta.running = False
            old_shm = SharedMemory(
                name=self.STATE.cur_settings.shmem_name, create=False
            )
            old_shm.close()
            old_shm.unlink()
            # Failed cleanup on previous run. Reuse the location.
            self.STATE.shmem = SharedMemory(
                name=self.STATE.cur_settings.shmem_name,
                create=True,
                size=shmem_size,
            )
        if self.STATE.cur_settings.shmem_name is None:
            self.STATE.cur_settings.shmem_name = self.STATE.shmem.name
        self.STATE.meta = ShmemArrMeta.from_buffer(self.STATE.shmem.buf)
        self.STATE.meta.dtype = msg.data.dtype.char.encode("utf8")
        self.STATE.meta.srate = 1 / axis.gain
        self.STATE.meta.ndim = 1 + len(frame_shape)
        self.STATE.meta.shape[: self.STATE.meta.ndim] = (n_frames,) + frame_shape
        self.STATE.meta.write_index = 0
        self.STATE.buffer = np.ndarray(
            self.STATE.meta.shape[: self.STATE.meta.ndim],
            dtype=np.dtype(self.STATE.meta.dtype.decode("utf8")),
            buffer=self.STATE.shmem.buf[meta_size:],
        )
        self.STATE.meta.running = True

    @ez.task
    async def check_continue(self):
        while True:
            if self.STATE.meta is not None and not self.STATE.meta.running:
                if self.STATE.shmem is not None:
                    self.STATE.buffer = None
                    self.STATE.meta = None
                    self.STATE.shmem.close()
                    self.STATE.shmem.unlink()
                    self.STATE.shmem = None
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
        write_stop = self.STATE.meta.write_index + n_samples

        if write_stop > self.STATE.buffer.shape[0]:
            overflow = write_stop - self.STATE.buffer.shape[0]
            self.STATE.buffer[self.STATE.meta.write_index :] = data[
                : n_samples - overflow
            ]
            self.STATE.buffer[:overflow] = data[n_samples - overflow :]
            self.STATE.meta.write_index = overflow
        else:
            self.STATE.buffer[self.STATE.meta.write_index : write_stop] = data[:]
            self.STATE.meta.write_index = write_stop
