import tempfile
import typing
from dataclasses import replace
from pathlib import Path

import ezmsg.core as ez
import numpy as np
import pytest
from ezmsg.simbiophys.eeg import EEGSynth
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis
from ezmsg.util.terminate import TerminateOnTotal

from ezmsg.tools.shmem.shmem import ShMemCircBuff, ShMemCircBuffSettings


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
        "SYNTH": EEGSynth(fs=1000, n_time=10, n_ch=n_ch),
        "CRAZY": CrazyUnit(change_after=n_messages // 2, change_type=change_type),
        "SINK": ShMemCircBuff(SHMEM_NAME, 2.0, conn=None, axis="time"),
        "LOGGER": MessageLogger(output=file_path),
        "TERM": TerminateOnTotal(total=n_messages),
    }
    conns = (
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


# ---------------------------------------------------------------------------
# Which dimension the ring is a history along
# ---------------------------------------------------------------------------


def _windowed_msg(n_win: int = 4, n_lag: int = 10, n_ch: int = 3, stream_dim: str | None = "win") -> AxisArray:
    """`(win, time, ch)` -- what a windowing stage emits.

    `time` is the *within-window* lag dimension. Both it and `win` are
    LinearAxes, so nothing distinguishes them but `stream_dim`.
    """
    kwargs = {"stream_dim": stream_dim} if stream_dim else {}
    return AxisArray(
        np.zeros((n_win, n_lag, n_ch), np.float32),
        dims=["win", "time", "ch"],
        axes={
            "win": AxisArray.TimeAxis(fs=10.0),
            "time": AxisArray.TimeAxis(fs=100.0),
            "ch": CoordinateAxis(data=np.array(["a", "b", "c"]), dims=["ch"]),
        },
        key="dev",
        **kwargs,
    )


def _plain_msg(n_time: int = 20, n_ch: int = 3, stream_dim: str | None = "time") -> AxisArray:
    kwargs = {"stream_dim": stream_dim} if stream_dim else {}
    return AxisArray(
        np.zeros((n_time, n_ch), np.float32),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=100.0),
            "ch": CoordinateAxis(data=np.array(["a", "b", "c"]), dims=["ch"]),
        },
        key="dev",
        **kwargs,
    )


def _sink(axis=None):
    unit = ShMemCircBuff(ShMemCircBuffSettings(shmem_name=None, buf_dur=1.0, axis=axis))
    unit.STATE = ShMemCircBuff.STATE()
    return unit


class TestTheBufferedAxisFollowsTheMessage:
    """The ring is a history of the stream, so it has to be the dimension
    messages accumulate along. `"time"` is present in a windowed message but is
    the wrong one, so nothing rejected the old default -- it just buffered the
    within-window samples and reported a 10x wrong sample rate.
    """

    def test_a_windowed_stream_resolves_to_win(self):
        assert _sink()._resolve_axis(_windowed_msg()) == "win"

    def test_a_plain_stream_resolves_to_time(self):
        assert _sink()._resolve_axis(_plain_msg()) == "time"

    def test_an_explicit_setting_still_wins(self):
        """For a producer that declares nothing, an operator must still be able
        to say which dimension to buffer."""
        assert _sink(axis="time")._resolve_axis(_windowed_msg()) == "time"

    def test_an_undeclared_producer_falls_back_to_time(self):
        """Nothing better is available. A windowed producer that declares no
        `stream_dim` still gets the old, wrong answer -- the fix is for it to
        declare one, which every ezmsg source now does."""
        assert _sink()._resolve_axis(_plain_msg(stream_dim=None)) == "time"
        assert _sink()._resolve_axis(_windowed_msg(stream_dim=None)) == "time"

    def test_a_message_with_neither_is_skipped(self):
        msg = AxisArray(
            np.zeros((4, 3), np.float32),
            dims=["freq", "ch"],
            axes={"freq": AxisArray.LinearAxis(gain=1.0)},
            key="dev",
        )
        assert _sink()._resolve_axis(msg) is None

    def test_what_the_old_default_did_to_a_windowed_stream(self):
        """Buffering `time` puts the window *count* inside the frame, so the
        buffer is reallocated whenever the window count jitters, and the rate
        reported to the viewer is the within-window rate."""
        for axis, want_frame, want_rate in (("time", (4, 3), 100.0), ("win", (10, 3), 10.0)):
            msg = _windowed_msg(n_win=4)
            ax_idx = msg.get_axis_idx(axis)
            frame_shape = msg.data.shape[:ax_idx] + msg.data.shape[ax_idx + 1 :]
            assert frame_shape == want_frame
            assert 1 / msg.axes[axis].gain == want_rate

        # ...and the window count is not stable, so `time` reshapes the frame.
        shapes = set()
        for n_win in (4, 7, 5):
            msg = _windowed_msg(n_win=n_win)
            ax_idx = msg.get_axis_idx("time")
            shapes.add(msg.data.shape[:ax_idx] + msg.data.shape[ax_idx + 1 :])
        assert len(shapes) == 3, "frame shape must vary with window count when buffering `time`"
