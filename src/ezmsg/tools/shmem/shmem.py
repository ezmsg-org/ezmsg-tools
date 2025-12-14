"""
It is possible to move data from ezmsg to non-ezmsg processes using shared memory. This module contains the ezmsg half
of that communication. The non-ezmsg half is in the .shmem_mirror module.
The same `shmem_name` must be passed to both the ShMemCircBuff and the EZShmMirror objects!

The ShMemCircBuff class is a sink node that receives AxisArray messages and writes them to a shared memory buffer.

Upon initialization, or upon receiving updated settings with a different shmem_name value, the node creates a shared
memory object located at {shorten_shmem_name(shmem_name)} to hold the metadata initialized with placeholder values
(e.g., srate = -1).
Additionally, the node has a convenience handle to the metadata via
`self.STATE.meta_struct = ShmemArrMeta.from_buffer(shmem.buf)`.

Upon receiving a data message, its metadata is checked, and if it does not match the shmem metadata
 (which will always be true for the first message) then the node first updates the metadata, then it (re-)creates
 a shared memory buffer to hold the data, located at shorten_shmem_name(f"{shmem_name}/buffer{buffer_generation}"),
 where `buffer_generation` is an integer that tracks how many times the buffer has been reset. This corresponds to the
 same integer stored in the metadata.

The other half must monitor the metadata shared memory to see if it changes, and if it does then it must recreate
the data shared memory buffer reader at the new location.
"""

import asyncio
import base64
import ctypes
import hashlib
import multiprocessing.connection
import time
import typing
from multiprocessing.shared_memory import SharedMemory

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt
from ezmsg.util.messages.axisarray import AxisArray, AxisBase

UINT64_SIZE = 8
BYTEORDER = "little"


def to_bytes(data: typing.Any) -> bytes:
    if isinstance(data, bool):
        return data.to_bytes(2, byteorder=BYTEORDER, signed=False)
    elif isinstance(data, int):
        return np.int64(data).to_bytes(UINT64_SIZE, BYTEORDER, signed=False)


def shorten_shmem_name(long_name: str) -> str:
    """
    Convert a potentially long shared memory name to a shorter, fixed-length name.

    Args:
        long_name: The original, potentially long shared memory name

    Returns:
        A shortened, deterministic name suitable for shared memory
    """
    if long_name is None:
        return None

    # Create a hash of the original name
    hash_obj = hashlib.sha256(long_name.encode("utf-8"))
    # Convert to URL-safe base64 and limit to 20 characters (plus 'sm_' prefix)
    # The 'sm_' prefix helps identify this as a shared memory name
    short_name = "sm_" + base64.urlsafe_b64encode(hash_obj.digest()).decode("ascii")[:20]

    return short_name


MAXKEYLEN = 1024


class ShmemArrMeta(ctypes.Structure):
    """
    Structure containing the metadata describing the separate shmem buffer.

    The SharedMemory object is expected to have allocated enough
    memory for this header + the memory required for the buffer
    described by this header. i.e.,
    meta_size = ctypes.sizeof(ShmemArrMeta)
    item_size = np.dtype(dtype).itemsize
    shmem_size = int(meta_size + np.prod(shape) * item_size)
    shmem = SharedMemory(name="...", create=True, size=shmem_size)
    meta = ShmemArrMeta.from_buffer(shmem)
    meta.dtype = dtype
    meta.ndim = len(shape)
    meta.shape[:meta.ndim] = shape
    circ_buff = np.ndarray(shape, dtype=dtype, buffer=shmem.buf[meta_size:])
    """

    _pack_ = 1
    _fields_ = [
        ("bvalid", ctypes.c_bool),
        ("dtype", ctypes.c_char),
        ("srate", ctypes.c_double),
        ("ndim", ctypes.c_uint32),
        ("shape", ctypes.c_uint32 * 64),
        ("buffer_generation", ctypes.c_uint32),
        ("wrap_counter", ctypes.c_uint64),
        ("_key_bytes", ctypes.c_byte * MAXKEYLEN),
        ("_key_len", ctypes.c_uint32),
        ("write_index", ctypes.c_uint64),
    ]

    @property
    def key(self) -> str:
        return ctypes.string_at(self._key_bytes, self._key_len).decode("utf8")

    @key.setter
    def key(self, value: str) -> None:
        key_bytes = value.encode("utf8")
        self._key_len = min(len(key_bytes), MAXKEYLEN)
        ctypes.memmove(self._key_bytes, key_bytes[: self._key_len], self._key_len)


class ShMemCircBuffSettings(ez.Settings):
    shmem_name: typing.Optional[str]
    buf_dur: float
    conn: typing.Optional[multiprocessing.connection.Connection] = None
    axis: str = "time"


class ShMemCircBuffState(ez.State):
    meta_shmem: typing.Optional[SharedMemory] = None
    meta_struct: typing.Optional[ShmemArrMeta] = None
    buffer_shmem: typing.Optional[SharedMemory] = None
    buffer_arr: typing.Optional[npt.NDArray] = None
    meta_hash: int = -1


def _persist_create_shmem(name: str, size: int) -> SharedMemory:
    """
    Create a shared memory object, retrying if necessary.
    Args:
        name: The name of the shared memory object.
        size: The size of the shared memory object.

    Returns: The SharedMemory object.
    """
    t0 = time.time()
    n_attempts = 0
    while True:
        try:
            result = SharedMemory(
                name=name,
                create=True,
                size=size,
            )
            break
        except FileExistsError:
            tmp_shmem = SharedMemory(
                name=name,
                create=False,
            )
            tmp_shmem.close()
            tmp_shmem.unlink()
    ez.logger.info(f"Created shmem at {name} in {n_attempts} attempts after {time.time() - t0:.2f} s.")
    return result


class ShMemCircBuff(ez.Unit):
    SETTINGS = ShMemCircBuffSettings
    STATE = ShMemCircBuffState

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    INPUT_SETTINGS = ez.InputStream(ShMemCircBuffSettings)

    async def initialize(self) -> None:
        # Prophylactic cleanup. These should mostly be a no-ops
        #  because the shared memory objects should not exist yet.
        self._cleanup_buffer()
        self._cleanup_meta()

        # Create the metadata shared memory object.
        #  Our SETTINGS need valid values for shmem_name and buf_dur.
        #  Even then, the meta_struct will be invalid until we receive
        #  a data packet.
        self._reset_meta()

    @ez.subscriber(INPUT_SETTINGS)
    def on_settings(self, msg: ShMemCircBuffSettings) -> None:
        b_reset_meta = msg.shmem_name != self.SETTINGS.shmem_name
        b_reset_buff = msg.buf_dur != self.SETTINGS.buf_dur
        b_reset_buff = b_reset_buff or msg.axis != self.SETTINGS.axis
        self.apply_settings(msg)

        if b_reset_buff or b_reset_meta:
            # First we destroy the data buffer, because it is no
            #  longer valid with the new settings.
            self._cleanup_buffer()
            # It will be recreated with the next data packet.

        if b_reset_meta:
            # Destroy the metadata and its shared memory object because the name has changed.
            self._cleanup_meta()
            # Then we reset the metadata to the new name.
            self._reset_meta(reset_generation=False)

        # Do not reset the buffer. We will wait for a new data packet.

    async def shutdown(self) -> None:
        self._cleanup_buffer()
        self._cleanup_meta()
        if self.SETTINGS.conn is not None:
            self.SETTINGS.conn.send("close")
            self.SETTINGS.conn.close()

    def _cleanup_meta(self):
        """
        Destroy the metadata and its shared memory object.
        This is called during initialization and shutdown,
        or if the SETTINGS name has changed.
        """
        if self.SETTINGS.conn is not None:
            self.SETTINGS.conn.send("meta cleanup")

        self.STATE.meta_struct = None

        if self.STATE.meta_shmem is not None:
            self.STATE.meta_shmem.close()
            try:
                self.STATE.meta_shmem.unlink()
            except FileNotFoundError:
                pass
            del self.STATE.meta_shmem
        self.STATE.meta_shmem = None

    def _cleanup_buffer(self):
        """
        Destroy the data buffer and the shared memory object.
        This is called during initialization and shutdown,
        as well as whenever the incoming data changes its statistics
        requiring a new buffer.
        """
        if self.SETTINGS.conn is not None:
            self.SETTINGS.conn.send("buffer cleanup")

        if self.STATE.meta_struct is not None:
            # Mark the metadata as invalid
            self.STATE.meta_struct.bvalid = False

        # Destroy the buffer
        self.STATE.buffer_arr = None

        # Destroy the shared memory object
        if self.STATE.buffer_shmem is not None:
            self.STATE.buffer_shmem.close()
            try:
                self.STATE.buffer_shmem.unlink()
            except FileNotFoundError:
                pass
        if self.STATE.buffer_shmem is not None:
            del self.STATE.buffer_shmem
        self.STATE.buffer_shmem = None

    def _reset_meta(self, reset_generation: bool = True) -> None:
        """
        Crete the metadata shared memory object.
        This is called during initialization and whenever the SETTINGS.shmem_name changes.
        """
        if self.SETTINGS.conn is not None:
            self.SETTINGS.conn.send("meta reset")

        # Create the metadata shared memory object.
        meta_size = int(ctypes.sizeof(ShmemArrMeta))
        short_name = shorten_shmem_name(self.SETTINGS.shmem_name)
        self.STATE.meta_shmem = _persist_create_shmem(short_name, meta_size)

        if self.SETTINGS.shmem_name is None:
            # If the name is None, then we need to get the name from the shared memory object.
            self.SETTINGS.shmem_name = self.STATE.meta_shmem.name

        # Build the metadata structure.
        self.STATE.meta_struct = ShmemArrMeta.from_buffer(self.STATE.meta_shmem.buf)
        self.STATE.meta_struct.bvalid = False
        if reset_generation:
            self.STATE.meta_struct.buffer_generation = -1
        # We will wait for a data packet before we modify the remaining fields.

    def _n_frames_for_axis(self, axis: AxisBase) -> int:
        """
        Utility function to calculate the number of frames to allocate for the buffer.
        Args:
            axis: The axis object containing the metadata for the axis along
              which we are buffering.

        Returns: number of frames we should buffer based on the axis and settings.
        """
        if hasattr(axis, "data"):
            fs = 1 / np.median(np.diff(axis.data)) if len(axis.data) > 1 else 100.0
        else:
            fs = 1 / axis.gain
        return int(np.ceil(self.SETTINGS.buf_dur * fs))

    def _get_msg_meta(self, msg: AxisArray) -> typing.Tuple[bytes, float, int, typing.Tuple[int, ...]]:
        """
        Utility function to extract relevant metadata from the incoming message.

        Args:
            msg: The incoming AxisArray message.

        Returns:
            A tuple of metadata extracted from the message.
            msg_dtype, msg_srate, n_frames, frame_shape
        """
        ax_idx = msg.get_axis_idx(self.SETTINGS.axis)
        axis = msg.axes[self.SETTINGS.axis]
        n_frames = self._n_frames_for_axis(axis)
        frame_shape = msg.data.shape[:ax_idx] + msg.data.shape[ax_idx + 1 :]
        data = np.moveaxis(msg.data, ax_idx, 0)
        msg_dtype = data.dtype.char.encode("utf8")
        msg_srate = 1 / axis.gain if hasattr(axis, "gain") else 0.0
        return msg_dtype, msg_srate, n_frames, frame_shape

    def _update_meta_if_needed(self, msg: AxisArray) -> bool:
        """
        Update the metadata structure if the incoming message has different metadata.

        Args:
            msg: The incoming AxisArray message.

        Returns: True if the metadata was updated, False otherwise.
        """
        # Extract the metadata from the incoming message
        msg_dtype, msg_srate, n_frames, frame_shape = self._get_msg_meta(msg)
        # Get its hash for quick comparison, and we will reuse the hash.
        new_hash = hash(
            (
                msg_dtype,
                msg_srate,
                n_frames,
            )
            + frame_shape
            + (msg.key,)
        )
        b_update = self.STATE.meta_hash != new_hash
        if b_update:
            if self.SETTINGS.conn is not None:
                self.SETTINGS.conn.send("begin update")

            self.STATE.meta_struct.bvalid = False
            self.STATE.meta_struct.dtype = msg_dtype
            self.STATE.meta_struct.srate = msg_srate
            self.STATE.meta_struct.ndim = 1 + len(frame_shape)
            self.STATE.meta_struct.shape[: self.STATE.meta_struct.ndim] = (n_frames,) + frame_shape
            self.STATE.meta_struct.key = msg.key
            self.STATE.meta_struct.write_index = 0
            self.STATE.meta_struct.wrap_counter = 0
            self.STATE.meta_hash = new_hash

            if self.SETTINGS.conn is not None:
                self.SETTINGS.conn.send("meta updated")

        return b_update

    def _reset_buffer(self, msg: AxisArray) -> None:
        """
        Reset the buffer to accommodate the new metadata and new message.
        Args:
            msg: The incoming AxisArray message.
        """
        self.STATE.meta_struct.buffer_generation += 1
        msg_dtype, msg_srate, n_frames, frame_shape = self._get_msg_meta(msg)
        buff_size = int(n_frames * np.prod(frame_shape) * msg.data.itemsize)
        buff_shm_name = self.SETTINGS.shmem_name + "/buffer" + str(self.STATE.meta_struct.buffer_generation)
        short_name = shorten_shmem_name(buff_shm_name)
        self.STATE.buffer_shmem = _persist_create_shmem(short_name, buff_size)
        self.STATE.buffer_arr = np.ndarray(
            self.STATE.meta_struct.shape[: self.STATE.meta_struct.ndim],
            dtype=np.dtype(self.STATE.meta_struct.dtype.decode("utf8")),
            buffer=self.STATE.buffer_shmem.buf[:],
        )
        self.STATE.meta_struct.write_index = 0
        self.STATE.meta_struct.wrap_counter = 0
        self.STATE.meta_struct.bvalid = True

        if self.SETTINGS.conn is not None:
            self.SETTINGS.conn.send("buffer reset")

    @ez.task
    async def check_continue(self):
        while True:
            if self.SETTINGS.conn is not None and self.SETTINGS.conn.poll():
                obj = self.SETTINGS.conn.recv()
                if obj == "quit":
                    self.shutdown()
                    break
                else:
                    print(f"Unhandled object received on connection: {obj}")
            else:
                await asyncio.sleep(0.05)
        raise ez.NormalTermination

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    async def on_message(self, msg: AxisArray):
        # Sanity check the input
        if not isinstance(msg, AxisArray):
            return

        if self.SETTINGS.axis not in msg.dims:
            return

        ax_idx = msg.get_axis_idx(self.SETTINGS.axis)
        data = np.moveaxis(msg.data, ax_idx, 0)

        # Check if we need to update the metadata, and if so, reset the buffer.
        if self._update_meta_if_needed(msg):
            self._reset_buffer(msg)

        n_samples = data.shape[0]
        write_stop = self.STATE.meta_struct.write_index + n_samples

        if write_stop > self.STATE.buffer_arr.shape[0]:
            overflow = write_stop - self.STATE.buffer_arr.shape[0]
            self.STATE.buffer_arr[self.STATE.meta_struct.write_index :] = data[: n_samples - overflow]
            self.STATE.buffer_arr[:overflow] = data[n_samples - overflow :]
            self.STATE.meta_struct.write_index = overflow
            self.STATE.meta_struct.wrap_counter += 1
        else:
            self.STATE.buffer_arr[self.STATE.meta_struct.write_index : write_stop] = data[:]
            self.STATE.meta_struct.write_index = write_stop
