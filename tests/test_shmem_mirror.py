import os
import tempfile
import threading
import time
import typing
from dataclasses import replace
from pathlib import Path

import ezmsg.core as ez
import numpy as np
import pytest
from ezmsg.sigproc.synth import Clock, Oscillator
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTotal

from ezmsg.tools.shmem.shmem import ShMemCircBuff
from ezmsg.tools.shmem.shmem_mirror import EZShmMirror


class CrazyUnitSettings(ez.Settings):
    change_after: int = 1e9
    change_type: str = "shape"


class CrazyUnitState(ez.State):
    msg_count: int = 0
    b_mod: bool = False


class CrazyUnit(ez.Unit):
    SETTINGS = CrazyUnitSettings
    STATE = CrazyUnitState

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_signal(self, message: AxisArray) -> typing.AsyncGenerator:
        if self.STATE.b_mod:
            if self.SETTINGS.change_type == "shape":
                # Drop the last channel. So crazy!
                message = replace(
                    message,
                    data=message.data[:, :-1],
                    axes={
                        **message.axes,
                        "ch": replace(message.axes["ch"], data=message.axes["ch"].data[:-1]),
                    },
                )
            elif self.SETTINGS.change_type == "irregular":
                # Convert the time axis to a coordinate axis, implying the signal is irregular (no fs).
                tvec = message.axes["time"].value(np.arange(message.data.shape[0]))
                message = replace(
                    message,
                    axes={
                        **message.axes,
                        "time": AxisArray.CoordinateAxis(data=tvec, dims=["time"], unit="s"),
                    },
                )
            elif self.SETTINGS.change_type == "dtype":
                # Change the data type to float16.
                message = replace(message, data=message.data.astype(np.float16))
        yield self.OUTPUT_SIGNAL, message

        self.STATE.msg_count += 1
        if self.STATE.msg_count >= self.SETTINGS.change_after:
            self.STATE.b_mod = not self.STATE.b_mod
            self.STATE.msg_count = 0


SHMEM_NAME = "graphviz" + str(os.getpid())
SR = 2000.0
CHANNEL_COUNT = 128
CHUNK_SIZE = 64
TOTAL_DURATION = 5.0


def app(file_path) -> None:
    change_type = "dtype"
    chunk_rate = 10.0
    chunk_size = SR // chunk_rate
    n_messages = int(TOTAL_DURATION * chunk_rate)

    comps = {
        "CLOCK": Clock(dispatch_rate=chunk_rate),
        "SYNTH": Oscillator(n_time=chunk_size, fs=SR, n_ch=CHANNEL_COUNT, dispatch_rate="ext_clock"),
        "CRAZY": CrazyUnit(change_after=n_messages // 2, change_type=change_type),
        "SINK": ShMemCircBuff(SHMEM_NAME, 2.0, conn=None, axis="time"),
        "LOGGER": MessageLogger(output=file_path),
        "TERM": TerminateOnTotal(total=n_messages),
    }
    conns = (
        (comps["CLOCK"].OUTPUT_SIGNAL, comps["SYNTH"].INPUT_SIGNAL),
        (comps["SYNTH"].OUTPUT_SIGNAL, comps["CRAZY"].INPUT_SIGNAL),
        (comps["CRAZY"].OUTPUT_SIGNAL, comps["SINK"].INPUT_SIGNAL),
        (comps["CRAZY"].OUTPUT_SIGNAL, comps["LOGGER"].INPUT_MESSAGE),
        (comps["LOGGER"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
    )
    ez.run(components=comps, connections=conns)

    print("Pipeline terminated.")


# Receive chunk of data using shm mirror
def get_chunk(mirror, chunk_size=CHUNK_SIZE):
    res, b_overflow = mirror.auto_view(chunk_size)
    if b_overflow:
        print("Overflow!")
    return res if res.size else None


# Get data from the shmem mirror for `duration` seconds
def collect_data(mirror, duration):
    start_time = time.time()
    buffer = []
    while (time.time() - start_time) < duration:
        chunk = get_chunk(mirror, chunk_size=CHUNK_SIZE)
        if chunk is not None:
            buffer.append(chunk.copy())
        time.sleep(CHUNK_SIZE / SR)
    return buffer


@pytest.mark.skipif("CI" in os.environ, reason="Try skipping this test for CI runner.")
def test_shmem_mirror_switch_buffer():
    file_path = Path(tempfile.gettempdir())
    file_path = file_path / Path("test_shmem_min.txt")
    file_path.unlink(missing_ok=True)

    # Shared memory mirror
    mirror = EZShmMirror()
    mirror.connect(SHMEM_NAME)

    # Start a pipeline with a data stream sinking to a ShMemCircBuff.
    #  Note that ShMemCircBuff automatically hashes the name to guarantee < 20 character shmem filenames.
    app_thread = threading.Thread(target=app, args=(file_path,))
    app_thread.start()

    START_TIME = time.time()
    while get_chunk(mirror) is None:
        time.sleep(0.1)
    print(f"*** Pipeline started in {time.time() - START_TIME:.2f} seconds")

    # Receive data from shared memory for a few seconds.
    data_received = collect_data(mirror, TOTAL_DURATION)

    # Stop bolt and LSL stream
    app_thread.join()

    messages: typing.List[AxisArray] = [_ for _ in message_log(file_path)]
    file_path.unlink(missing_ok=True)

    # Now let's look at what we got...
    print("Logger")
    chunk_lens = [_.data.shape[0] for _ in messages]
    print(f"\t{np.sum(chunk_lens)} samples.")
    dt0 = messages[0].data.dtype
    switch_idx = np.where(~np.array([_.data.dtype == dt0 for _ in messages]))[0][0]
    switch_dt = messages[switch_idx].data.dtype
    print(f"\tFrom {dt0} to {switch_dt} after {np.sum(chunk_lens[:switch_idx])} samples")

    print("Shmem")
    chunk_lens = [_.shape[0] for _ in data_received]
    print(f"\t{np.sum(chunk_lens)} samples.")
    dt0 = data_received[0].dtype
    switch_idx = np.where(~np.array([_.dtype == dt0 for _ in data_received]))[0]
    if len(switch_idx) == 0:
        print("No switch found")
    else:
        switch_idx = switch_idx[0]
        switch_dt = data_received[switch_idx].dtype
        print(f"\tFrom {dt0} to {switch_dt} after {np.sum(chunk_lens[:switch_idx])} samples")
    assert dt0 == np.float64
    assert switch_dt == np.float16
