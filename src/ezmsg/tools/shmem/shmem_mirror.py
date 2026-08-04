"""
It is possible to move data from ezmsg to non-ezmsg processes using shared memory. This module contains the non-ezmsg
half of that communication. The ezmsg half is found in .shmem.
The same `shmem_name` must be passed to both the ShMemCircBuff and the EZShmMirror objects!

Besides the sample data, the mirror exposes the source AxisArray's static metadata -- its coordinate axes (e.g. a `ch`
axis naming each channel), axis units, and `attrs` -- via the `axes`, `attrs`, and `dims` properties. These are plain
dicts rather than ezmsg objects; see .aux_meta for why. They read None until the writer publishes, and update in place
if it ever republishes, so poll them (or register_metadata_callback) rather than reading once.
"""

import copy
import time
import typing
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import numpy.typing as npt

from .aux_meta import decode_aux
from .shmem import SHMEM_META_STRUCT_VERSION, ShmemArrMeta, ShMemCircBuffState, shorten_shmem_name

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
        self._metadata_callback: typing.Optional[typing.Callable] = None
        self._last_meta: typing.Optional[ShmemArrMeta] = None
        self._read_index = 0  # Used by auto_view
        self._last_connect_try = -np.inf
        # Decoded static metadata (see .aux_meta) and the generation it came
        # from. 0 means we have not read one; the writer never publishes gen 0.
        self._aux: typing.Optional[dict] = None
        self._aux_generation: int = 0
        # Remembered so an undecodable blob is reported once, not every poll.
        self._aux_error_generation: int = 0
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

    # ---- Static metadata (the non-buffered axes, units, and attrs) ----------

    @property
    def axes(self) -> typing.Optional[typing.Dict[str, dict]]:
        """The source AxisArray's axes as plain dicts, or None if unavailable.

        Keyed by axis name. Each value is
        ``{"kind": "linear", "unit", "gain", "offset"}`` or
        ``{"kind": "coord", "unit", "dims", "data"}`` -- see :mod:`.aux_meta`
        for why these are dicts rather than ezmsg axis objects.

        The buffered axis (whatever the sink was configured to buffer along,
        normally ``"time"``) appears here with only its static descriptors: its
        position along the stream lives in the ring's write index, not here.

        None means either that the metadata has not arrived yet or that the
        writer predates this feature; poll again or check
        :attr:`metadata_available`.
        """
        self._refresh_aux()
        return None if self._aux is None else self._aux["axes"]

    @property
    def attrs(self) -> typing.Optional[dict]:
        """The source AxisArray's ``attrs``, minus any non-transportable values."""
        self._refresh_aux()
        return None if self._aux is None else self._aux["attrs"]

    @property
    def dims(self) -> typing.Optional[typing.List[str]]:
        """The source AxisArray's dimension names, in the *sender's* order.

        Note the buffer itself is rolled so the buffered axis comes first; this
        is the message's original ordering.
        """
        self._refresh_aux()
        return None if self._aux is None else self._aux["dims"]

    @property
    def metadata_available(self) -> bool:
        """Whether a decoded metadata blob is currently held."""
        self._refresh_aux()
        return self._aux is not None

    def register_metadata_callback(self, callback: typing.Callable) -> None:
        """Call ``callback`` whenever a new metadata generation is decoded.

        Separate from :meth:`register_change_callback`, which fires when the
        *data buffer* is rebuilt. The two are independent: channel labels can
        arrive without the buffer changing, and vice versa.
        """
        self._metadata_callback = callback

    def unregister_metadata_callback(self) -> None:
        self._metadata_callback = None

    def _cleanup_aux(self):
        if self._mirror_state.aux_shmem is not None:
            try:
                self._mirror_state.aux_shmem.close()
            except Exception as e:
                print(f"Error closing metadata segment: {e}")
            del self._mirror_state.aux_shmem
        self._mirror_state.aux_shmem = None
        self._aux = None
        self._aux_generation = 0
        self._aux_error_generation = 0

    def _refresh_aux(self) -> None:
        """Attach to and decode the metadata segment if the writer bumped it.

        Cheap and idempotent: in the steady state this is one integer compare,
        so the properties above can call it unconditionally.
        """
        meta = self._mirror_state.meta_struct
        if meta is None:
            return
        if meta.struct_version < SHMEM_META_STRUCT_VERSION:
            # Writer predates the metadata segment. Nothing to read, ever.
            return
        generation = int(meta.meta_generation)
        if generation == 0 or generation == self._aux_generation or generation == self._aux_error_generation:
            return

        nbytes = int(meta.aux_nbytes)
        aux_name = shorten_shmem_name(self._shmem_name + "/meta" + str(generation))
        try:
            shm = SharedMemory(aux_name, create=False)
        except FileNotFoundError:
            # The writer has moved on to a newer generation and unlinked this
            # one. Leave the old decode in place; the next poll picks up the new
            # generation.
            return

        try:
            blob = bytes(shm.buf[:nbytes])
            payload = decode_aux(blob)
        except ValueError as e:
            # A blob we cannot read is not a transient condition -- the writer
            # is a version we do not speak. Remember the generation so we
            # complain once rather than on every poll.
            self._aux_error_generation = generation
            shm.close()
            print(f"Ignoring shmem metadata generation {generation}: {e}")
            return

        # Only now release the previous segment, so a decode failure above
        # leaves the last good metadata intact.
        if self._mirror_state.aux_shmem is not None:
            try:
                self._mirror_state.aux_shmem.close()
            except Exception as e:
                print(f"Error closing metadata segment: {e}")
        self._mirror_state.aux_shmem = shm
        self._aux = payload
        self._aux_generation = generation

        if self._metadata_callback is not None:
            self._metadata_callback()

    def _cleanup_meta(self):
        self._cleanup_aux()
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

        # Poll the metadata here too, so a consumer that only ever calls
        # auto_view still gets its metadata callback fired.
        self._refresh_aux()

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
