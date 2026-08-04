"""The static-metadata side channel: codec, change detection, and end-to-end.

The point of the feature is that a consumer on the far side of the shared-memory
boundary can learn what the channels *are*, so these tests care about two things
in roughly equal measure: that the metadata arrives intact, and that carrying it
costs nothing when it is not changing -- a republish per message would defeat the
purpose.
"""

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
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.tools.chmeta import available_fields, channel_names
from ezmsg.tools.shmem.aux_meta import (
    AUX_FORMAT_VERSION,
    attrs_equal,
    axes_equal,
    decode_aux,
    encode_aux,
)
from ezmsg.tools.shmem.shmem import ShMemCircBuff
from ezmsg.tools.shmem.shmem_mirror import EZShmMirror

CHANNEL_DTYPE = np.dtype([("bank", "U2"), ("elec", "<i4"), ("label", "U16")])


def make_ch_axis(n_ch: int) -> AxisArray.CoordinateAxis:
    data = np.zeros(n_ch, dtype=CHANNEL_DTYPE)
    for i in range(n_ch):
        data["bank"][i] = "AB"[i // 32]
        data["elec"][i] = (i % 32) + 1
        data["label"][i] = f"elec{i:03d}"
    return AxisArray.CoordinateAxis(data=data, dims=["ch"], unit="")


def make_msg(n_time: int = 8, n_ch: int = 4, offset: float = 0.0, **attrs) -> AxisArray:
    return AxisArray(
        data=np.zeros((n_time, n_ch), dtype=np.float32),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=1000.0, offset=offset),
            "ch": make_ch_axis(n_ch),
        },
        attrs=dict(attrs),
        key="test",
    )


# ---------------------------------------------------------------- codec -----


def test_encode_decode_round_trip():
    msg = make_msg(n_ch=64, unit="uV")
    blob, dropped = encode_aux(msg.dims, msg.axes, msg.attrs, msg.key, "time")
    assert dropped == []

    payload = decode_aux(blob)
    assert payload["version"] == AUX_FORMAT_VERSION
    assert payload["dims"] == ["time", "ch"]
    assert payload["key"] == "test"
    assert payload["attrs"] == {"unit": "uV"}

    ch = payload["axes"]["ch"]
    assert ch["kind"] == "coord"
    assert ch["dims"] == ["ch"]
    np.testing.assert_array_equal(ch["data"], msg.axes["ch"].data)


def test_buffered_axis_carries_no_position():
    """The time axis's offset advances every message; it must not ride along.

    If it did, the blob would differ on every message and the side channel would
    republish at the sample rate.
    """
    msg = make_msg(offset=1.0)
    blob_a, _ = encode_aux(msg.dims, msg.axes, msg.attrs, msg.key, "time")
    later = replace(msg, axes={**msg.axes, "time": AxisArray.TimeAxis(fs=1000.0, offset=99.0)})
    blob_b, _ = encode_aux(later.dims, later.axes, later.attrs, later.key, "time")

    assert blob_a == blob_b
    time_axis = decode_aux(blob_a)["axes"]["time"]
    assert "offset" not in time_axis
    assert time_axis["gain"] == pytest.approx(0.001)


def test_coordinate_time_axis_carries_no_data():
    """Same argument for an irregular stream, where time is a CoordinateAxis
    whose data is wholly new each message."""
    msg = make_msg()
    a = replace(msg, axes={**msg.axes, "time": AxisArray.CoordinateAxis(data=np.arange(8.0), dims=["time"], unit="s")})
    b = replace(
        msg, axes={**msg.axes, "time": AxisArray.CoordinateAxis(data=np.arange(100.0, 108.0), dims=["time"], unit="s")}
    )
    blob_a, _ = encode_aux(a.dims, a.axes, a.attrs, a.key, "time")
    blob_b, _ = encode_aux(b.dims, b.axes, b.attrs, b.key, "time")

    assert blob_a == blob_b
    assert "data" not in decode_aux(blob_a)["axes"]["time"]


def test_non_plain_attrs_are_dropped_not_pickled():
    class Opaque:
        pass

    msg = make_msg(unit="uV", handle=Opaque(), count=3)
    blob, dropped = encode_aux(msg.dims, msg.axes, msg.attrs, msg.key, "time")

    assert dropped == ["handle"]
    assert decode_aux(blob)["attrs"] == {"unit": "uV", "count": 3}


def test_decode_rejects_foreign_payloads():
    import pickle

    with pytest.raises(ValueError, match="could not decode"):
        decode_aux(b"not a pickle at all")
    with pytest.raises(ValueError, match="expected dict"):
        decode_aux(pickle.dumps([1, 2, 3]))
    with pytest.raises(ValueError, match="format version"):
        decode_aux(pickle.dumps({"version": AUX_FORMAT_VERSION + 1}))


# ------------------------------------------------------ change detection -----


def test_axes_equal_is_true_for_passed_through_axes():
    msg = make_msg()
    forwarded = replace(msg, data=msg.data * 2)  # axes dict passed through unchanged
    assert axes_equal(msg.axes, forwarded.axes)


def test_axes_equal_is_true_for_rebuilt_but_identical_axes():
    """A producer that rebuilds an equal ch axis must not look like a change."""
    a, b = make_msg(), make_msg()
    assert a.axes["ch"] is not b.axes["ch"]
    assert axes_equal(a.axes, b.axes)


def test_axes_equal_detects_real_changes():
    a = make_msg(n_ch=4)
    relabelled = make_ch_axis(4)
    relabelled.data["label"][2] = "CHANGED"
    b = replace(a, axes={**a.axes, "ch": relabelled})
    assert not axes_equal(a.axes, b.axes)

    assert not axes_equal(a.axes, {k: v for k, v in a.axes.items() if k != "ch"})


def test_attrs_equal_tolerates_array_values():
    """dict == would raise on an ndarray value; identity comparison must not."""
    arr = np.arange(4)
    a = {"m": arr}
    assert attrs_equal(a, a)
    assert attrs_equal(a, {"m": arr})
    assert not attrs_equal(a, {"m": np.arange(4)})  # conservative: re-encode, then blob-compare


# ------------------------------------------------------------ chmeta --------


def test_channel_names_defaults_to_label():
    ch = make_ch_axis(4)
    assert channel_names(ch.data) == ["elec000", "elec001", "elec002", "elec003"]
    assert available_fields(ch.data) == ["bank", "elec", "label"]


def test_channel_names_with_alternate_fields():
    ch = make_ch_axis(34)
    names = channel_names(ch.data, fields=("bank", "elec"))
    assert names[0] == "A-1"
    assert names[32] == "B-1"


def test_channel_names_fallbacks():
    assert channel_names(None, 3) == ["ch0", "ch1", "ch2"]
    # Unstructured axis data.
    assert channel_names(np.arange(2.0), 2) == ["ch0", "ch1"]
    # Requested field is absent from the dtype.
    ch = make_ch_axis(2)
    assert channel_names(ch.data, fields=("nonexistent",)) == ["ch0", "ch1"]
    # Field present but empty for one channel only.
    partial = make_ch_axis(2)
    partial.data["label"][1] = ""
    assert channel_names(partial.data) == ["elec000", "ch1"]


# ------------------------------------------------------------- e2e ----------


class MetaCountSettings(ez.Settings):
    relabel_at: int = -1
    n_messages: int = 20


class MetaCountState(ez.State):
    count: int = 0


class Source(ez.Unit):
    """Emits a fixed stream, optionally relabelling the ch axis partway."""

    SETTINGS = MetaCountSettings
    STATE = MetaCountState

    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    @ez.publisher(OUTPUT_SIGNAL)
    async def generate(self) -> typing.AsyncGenerator:
        n_ch = 8
        base_ch = make_ch_axis(n_ch)
        alt_ch = make_ch_axis(n_ch)
        alt_ch.data["label"] = [f"NEW{i:03d}" for i in range(n_ch)]
        while self.STATE.count < self.SETTINGS.n_messages:
            i = self.STATE.count
            relabelled = 0 <= self.SETTINGS.relabel_at <= i
            yield (
                self.OUTPUT_SIGNAL,
                AxisArray(
                    data=np.full((10, n_ch), float(i), dtype=np.float32),
                    dims=["time", "ch"],
                    # A fresh axes dict every message, as a real producer builds.
                    axes={
                        "time": AxisArray.TimeAxis(fs=1000.0, offset=i * 0.01),
                        "ch": alt_ch if relabelled else base_ch,
                    },
                    attrs={"unit": "uV"},
                    key="e2e",
                ),
            )
            self.STATE.count += 1
            await __import__("asyncio").sleep(0.01)
        raise ez.NormalTermination


def _run_graph(shmem_name: str, relabel_at: int, n_messages: int) -> None:
    comps = {
        "SRC": Source(relabel_at=relabel_at, n_messages=n_messages),
        "SINK": ShMemCircBuff(shmem_name, 2.0, conn=None, axis="time"),
    }
    conns = ((comps["SRC"].OUTPUT_SIGNAL, comps["SINK"].INPUT_SIGNAL),)
    ez.run(components=comps, connections=conns)


@pytest.mark.skipif("CI" in os.environ, reason="Timing-sensitive; matches the existing mirror test.")
@pytest.mark.parametrize("relabel_at", [-1, 100])
def test_metadata_reaches_the_mirror(relabel_at: int):
    shmem_name = f"auxtest{os.getpid()}{relabel_at}"
    n_messages = 200  # ~2 s at the source's 10 ms cadence

    mirror = EZShmMirror()
    mirror.connect(shmem_name)

    generations = []
    mirror.register_metadata_callback(lambda: generations.append(mirror._aux_generation))

    thread = threading.Thread(target=_run_graph, args=(shmem_name, relabel_at, n_messages))
    thread.start()

    # Snapshots taken while the graph runs: the writer unlinks its segments at
    # shutdown, so anything not read now is gone.
    seen_labels = []
    last_attrs = last_dims = last_time_axis = None
    deadline = time.time() + 30.0
    while thread.is_alive() and time.time() < deadline:
        mirror.auto_view()
        axes = mirror.axes
        if axes is not None:
            labels = list(axes["ch"]["data"]["label"])
            if not seen_labels or labels != seen_labels[-1]:
                seen_labels.append(labels)
            last_attrs, last_dims, last_time_axis = mirror.attrs, mirror.dims, axes["time"]
        time.sleep(0.005)
    thread.join(timeout=10.0)

    assert seen_labels, "no metadata ever arrived at the mirror"
    assert seen_labels[0][0] == "elec000"
    assert last_attrs == {"unit": "uV"}
    assert last_dims == ["time", "ch"]

    # The buffered axis is present but positionless.
    assert "offset" not in last_time_axis
    assert last_time_axis["gain"] == pytest.approx(0.001)

    if relabel_at < 0:
        # Steady metadata over 200 messages must publish exactly one generation.
        assert seen_labels == [seen_labels[0]]
        assert generations == [1]
    else:
        assert len(seen_labels) == 2, f"expected one relabel, saw {len(seen_labels)} distinct label sets"
        assert seen_labels[-1][0] == "NEW000"
        assert generations == [1, 2]

    mirror.disconnect()


@pytest.mark.skipif("CI" in os.environ, reason="Timing-sensitive; matches the existing mirror test.")
def test_mirror_without_metadata_is_not_broken():
    """A stream whose sink never publishes metadata still mirrors data fine."""
    shmem_name = f"auxnone{os.getpid()}"
    file_path = Path(tempfile.gettempdir()) / "test_aux_none.txt"
    file_path.unlink(missing_ok=True)

    mirror = EZShmMirror()
    mirror.connect(shmem_name)
    # Nothing published yet: the properties must answer, not raise.
    assert mirror.axes is None
    assert mirror.attrs is None
    assert not mirror.metadata_available
    mirror.disconnect()
