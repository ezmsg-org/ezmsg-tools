import copy
import ctypes
from multiprocessing.shared_memory import SharedMemory
import time
import typing

import numpy as np
import numpy.typing as npt

from .unit import ShmemArrMeta


CONNECT_RETRY_INTERVAL = 0.5


class EZShmMirror:
    """
    An object that has a local (in-client-process) representation of the shared memory from
    an ezmsg.vis.unit.ShMemCircBuff Unit.
    """

    def __init__(self, shmem_name: str):
        self._shmem_name = shmem_name
        self._shmem: typing.Optional[SharedMemory] = None
        self._meta: typing.Optional[ShmemArrMeta] = None
        self._buffer: typing.Optional[npt.NDArray] = None
        self._read_index = 0
        self._last_connect_try = -np.inf

    @property
    def meta(self) -> typing.Optional[ShmemArrMeta]:
        if self._meta is None:
            return None
        return copy.deepcopy(self._meta)

    @property
    def connected(self) -> bool:
        return self._buffer is not None

    def reset(self):
        self._cleanup_shmem()

    def _cleanup_shmem(self):
        if self._shmem is not None:
            self._meta = None
            self._buffer = None
            self._shmem.close()
            self._shmem = None
        self._read_index = 0

    def _try_connect(self) -> None:
        if self._shmem is not None:
            self._cleanup_shmem()
        try:
            self._shmem = SharedMemory(self._shmem_name, create=False)
            self._meta = ShmemArrMeta.from_buffer(self._shmem.buf)
            _meta_size = ctypes.sizeof(ShmemArrMeta)
            self._buffer = np.ndarray(
                self._meta.shape[: self._meta.ndim],
                dtype=np.dtype(self._meta.dtype),
                buffer=self._shmem.buf[_meta_size:],
            )
        except FileNotFoundError:
            self._shmem = None

    def connect(self) -> None:
        if (
            self._shmem is None
            and (time.time() - self._last_connect_try) > CONNECT_RETRY_INTERVAL
        ):
            self._try_connect()
            self._last_connect_try = time.time()

    def view_samples(
        self, n: typing.Optional[int] = None
    ) -> typing.Optional[npt.NDArray]:
        if self._shmem is None:
            return None

        n_available = 0
        # Calculate how many samples are available
        if self._buffer is not None and self._read_index != self._meta.write_index:
            if self._meta.write_index < self._read_index:
                n_available = self._meta.shape[0] - self._read_index
            else:
                n_available = self._meta.write_index - self._read_index
            # if n_samples > 1 or (n_samples == 1 and self._read_index != 0):

        if n_available == 0 or (n is not None and n_available < n):
            return None

        # We have enough samples.
        if n is None:
            n = n_available
        t_slice = np.s_[max(0, self._read_index - 1) : self._read_index + n]
        self._read_index = (self._read_index + n) % self._meta.shape[0]

        return self._buffer[t_slice, :]
