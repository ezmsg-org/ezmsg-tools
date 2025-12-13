"""
It is possible to move data from ezmsg to non-ezmsg processes using shared memory. This module contains the non-ezmsg
half of that communication. The ezmsg half is found in .shmem.
The same `shmem_name` must be passed to both the ShMemCircBuff and the EZShmMirror objects!
"""

import copy
import time
import typing
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import numpy.typing as npt

from .shmem import ShmemArrMeta, ShMemCircBuffState, shorten_shmem_name

CONNECT_RETRY_INTERVAL = 0.5


class EZShmMirror:
    """
    An object that has a local (in-client-process) representation of the shared memory from
    another process' .shmem.ShMemCircBuff Unit.

    There are 2 pieces of shared memory: the metadata and the data buffer.
    The ezmsg node is responsible for creating both pieces. Here we only connect to them.
    We cannot know if the shared memory exists before we try to connect to it, so we
    must try the connection -- sometimes repeatedly while handling connection errors.
    """

    def __init__(self, shmem_name: typing.Optional[str] = None):
        self._mirror_state: ShMemCircBuffState = ShMemCircBuffState()
        self._shmem_name: typing.Optional[str] = None
        self._change_callback: typing.Optional[typing.Callable] = None
        self._last_meta: typing.Optional[ShmemArrMeta] = None
        self._read_index = 0  # Used by auto_view
        self._last_connect_try = -np.inf
        # If shmem_name is None then this will simply not connect to anything.
        self.connect(shmem_name)

    def __del__(self):
        self.disconnect()

    def disconnect(self):
        self._cleanup_buffer()
        self._cleanup_meta()
        self._shmem_name = None

    @property
    def meta(self) -> typing.Optional[ShmemArrMeta]:
        if self._mirror_state.meta_struct is None:
            return None
        return copy.deepcopy(self._mirror_state.meta_struct)

    @property
    def buffer(self) -> typing.Optional[npt.NDArray]:
        return self._mirror_state.buffer_arr

    @property
    def write_index(self) -> typing.Optional[int]:
        return self._mirror_state.meta_struct.write_index

    @property
    def connected(self) -> bool:
        return self.buffer is not None

    def _cleanup_meta(self):
        if self._mirror_state.meta_shmem is not None:
            del self._mirror_state.meta_struct
        self._mirror_state.meta_struct = None

        if self._mirror_state.meta_shmem is not None:
            # Note: Uncommenting the following does not eliminate the resource_tracker warnings.
            try:
                self._mirror_state.meta_shmem.close()
            except Exception as e:
                print(f"Error closing meta: {e}")
            del self._mirror_state.meta_shmem
        self._mirror_state.meta_shmem = None
        self._read_index = 0

    def _cleanup_buffer(self):
        if self._mirror_state.buffer_arr is not None:
            del self._mirror_state.buffer_arr
        self._mirror_state.buffer_arr = None

        if self._mirror_state.buffer_shmem is not None:
            # Note: Uncommenting the following does not eliminate the resource_tracker warnings.
            try:
                self._mirror_state.buffer_shmem.close()
            except Exception as e:
                print(f"Error closing buffer: {e}")
            del self._mirror_state.buffer_shmem
        self._mirror_state.buffer_shmem = None
        self._read_index = 0

    def register_change_callback(self, callback: typing.Callable) -> None:
        self._change_callback = callback

    def unregister_change_callback(self) -> None:
        self._change_callback = None

    def _connect_meta(self):
        # Attempt to connect to the meta shmem
        try:
            short_name = shorten_shmem_name(self._shmem_name)
            self._mirror_state.meta_shmem = SharedMemory(short_name, create=False)
            self._mirror_state.meta_struct = ShmemArrMeta.from_buffer(self._mirror_state.meta_shmem.buf)
        except FileNotFoundError:
            self._mirror_state.meta_struct = None
            self._mirror_state.meta_shmem = None

    def _reset_buffer(self) -> bool:
        if self._mirror_state.buffer_shmem is not None:
            # We might enter here if input data changed shape or dtype,
            #  meaning we are reconnecting to the same _name_ but different layout.
            self._cleanup_buffer()

        if self._mirror_state.meta_struct is None or not self._mirror_state.meta_struct.bvalid:
            # Cannot connect to buffer without valid meta.
            return False

        try:
            buff_name = self._shmem_name + "/buffer" + str(self._mirror_state.meta_struct.buffer_generation)
            short_name = shorten_shmem_name(buff_name)
            self._mirror_state.buffer_shmem = SharedMemory(short_name, create=False)
            self._mirror_state.buffer_arr = np.ndarray(
                self._mirror_state.meta_struct.shape[: self._mirror_state.meta_struct.ndim],
                dtype=np.dtype(self._mirror_state.meta_struct.dtype),
                buffer=self._mirror_state.buffer_shmem.buf[:],
            )
            self._last_meta = self.meta  # Copy
            if self._change_callback is not None:
                self._change_callback()
            return True
        except FileNotFoundError:
            self._mirror_state.buffer_arr = None
            self._mirror_state.buffer_shmem = None
        except TypeError:
            # buffer is too small for requested array
            self._mirror_state.buffer_arr = None
            self._mirror_state.buffer_shmem = None
            print("DEBUG!")
        return False

    def connect(self, name: str) -> None:
        if self._shmem_name is None or self._shmem_name != name:
            # Clear connection
            self._cleanup_buffer()
            self._cleanup_meta()

        self._shmem_name = name

        if self._shmem_name is None:
            # Provided name was None. Do not connect.
            return

        if (time.time() - self._last_connect_try) <= CONNECT_RETRY_INTERVAL:
            # Delay retrying the connection to avoid spamming the system.
            return

        if self._mirror_state.meta_struct is None:
            # The only way we can enter this `connect` method and not enter this logical block
            #  is if the provided `name` was the same as the last name.
            self._connect_meta()

        self._last_connect_try = time.time()

    def auto_view(self, n: typing.Optional[int] = None) -> typing.Tuple[npt.NDArray, bool]:
        if self._mirror_state.meta_struct is None:
            self.connect(self._shmem_name)

        if self._mirror_state.meta_struct is None or not self._mirror_state.meta_struct.bvalid:
            # Still not connected
            #  or we are connected but the buffer data is invalid.
            return np.array([[]]), False

        b_connected = True
        # Determine if we need to reset the buffer
        if (
            self._last_meta is None
            or self._mirror_state.meta_struct.buffer_generation != self._last_meta.buffer_generation
            or self._mirror_state.buffer_arr is None
        ):
            b_connected = self._reset_buffer()

        if not b_connected:
            # We STILL aren't connected.
            return np.array([[]]), False

        # -- From here, we should know we have a good connection to a valid buffer -- #

        wrapped_since_last_read = self._mirror_state.meta_struct.wrap_counter - self._last_meta.wrap_counter
        b_overflow = wrapped_since_last_read > 1 or (
            wrapped_since_last_read == 1 and self._mirror_state.meta_struct.write_index >= self._read_index
        )

        if b_overflow:
            # In case of overflow, start reading from the oldest available data
            self._read_index = (self._mirror_state.meta_struct.write_index + 1) % self._mirror_state.meta_struct.shape[
                0
            ]
            self._last_meta.wrap_counter = self._mirror_state.meta_struct.wrap_counter

        # Calculate how many samples are available
        n_available = 0
        if self._mirror_state.buffer_arr is not None:
            if self._mirror_state.meta_struct.write_index >= self._read_index:
                n_available = self._mirror_state.meta_struct.write_index - self._read_index
            else:
                n_available = (
                    self._mirror_state.meta_struct.shape[0]
                    - self._read_index
                    + self._mirror_state.meta_struct.write_index
                )

        if n_available <= 1 or (n is not None and n_available < n):
            # Not enough samples available.
            # Return a null-slice of the buffer. This provides correct dimensions.
            return self._mirror_state.buffer_arr[:0], b_overflow

        # We have enough samples.
        if n is None:
            n = n_available

        if (self._read_index + n) <= self._mirror_state.meta_struct.shape[0]:
            # Return a contiguous chunk
            t_slice = np.s_[max(0, self._read_index) : self._read_index + n]
            result = self._mirror_state.buffer_arr[t_slice, :]
        else:
            # Split read into two chunks
            n_after_wrap = n - (self._mirror_state.meta_struct.shape[0] - self._read_index)
            result = np.concatenate(
                (
                    self._mirror_state.buffer_arr[self._read_index :],
                    self._mirror_state.buffer_arr[:n_after_wrap],
                ),
                axis=0,
            )

        self._read_index = (self._read_index + n) % self._mirror_state.meta_struct.shape[0]
        self._last_meta.wrap_counter = self._mirror_state.meta_struct.wrap_counter

        return result, b_overflow
