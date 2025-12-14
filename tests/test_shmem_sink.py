import tempfile
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


@pytest.mark.parametrize("change_type", ["irregular", "shape", "dtype"])
def test_shmem_change(change_type: str):
    """
    In this test we are simply verifying that the ShMemCircBuff node does not crash.
    In the second iteration (change_shape = True), we are verifying that the ShMemCircBuff node
    does not crash when the incoming message changes shape.
    """
    n_messages = 10
    n_ch = 32
    SHMEM_NAME = "TESTSHMEM"
    file_path = Path(tempfile.gettempdir())
    file_path = file_path / Path("test_outlet_system.txt")
    file_path.unlink(missing_ok=True)

    comps = {
        "CLOCK": Clock(dispatch_rate=100.0),
        "SYNTH": Oscillator(n_time=10, fs=1000, n_ch=n_ch, dispatch_rate="ext_clock"),
        "CRAZY": CrazyUnit(change_after=n_messages // 2, change_type=change_type),
        "SINK": ShMemCircBuff(SHMEM_NAME, 2.0, conn=None, axis="time"),
        "LOGGER": MessageLogger(output=file_path),
        "TERM": TerminateOnTotal(total=n_messages),
    }
    conns = (
        (comps["CLOCK"].OUTPUT_SIGNAL, comps["SYNTH"].INPUT_SIGNAL),
        (comps["SYNTH"].OUTPUT_SIGNAL, comps["CRAZY"].INPUT_SIGNAL),
        (comps["CRAZY"].OUTPUT_SIGNAL, comps["LOGGER"].INPUT_MESSAGE),
        (comps["LOGGER"].OUTPUT_MESSAGE, comps["TERM"].INPUT_MESSAGE),
        (comps["CRAZY"].OUTPUT_SIGNAL, comps["SINK"].INPUT_SIGNAL),
    )
    ez.run(components=comps, connections=conns)

    messages: typing.List[AxisArray] = [_ for _ in message_log(file_path)]
    if change_type == "shape":
        assert all(msg.shape[1] == n_ch - 1 for msg in messages[5:10])
    else:
        assert all(msg.shape[1] == n_ch for msg in messages)

    if change_type == "irregular":
        assert all(hasattr(msg.axes["time"], "data") for msg in messages[5:10])
    else:
        assert all(not hasattr(msg.axes["time"], "data") for msg in messages)

    if change_type == "dtype":
        assert all(msg.data.dtype == np.float16 for msg in messages[5:10])
    else:
        assert all(msg.data.dtype == float for msg in messages)

    file_path.unlink(missing_ok=True)
