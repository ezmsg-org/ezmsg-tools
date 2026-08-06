"""Working out what a stream is, so a plot can draw it.

The case that matters is the envelope. A ``(time, ch, 2)`` block folded naively
into two dimensions renders as twice as many traces, alternating lower and upper
bounds, against channel labels that are now off by a factor of two -- and it
looks like data, so nothing complains. That is what both CLIs did before this
module existed, and it is what these tests exist to prevent coming back.
"""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.tools.plot.describe import (
    UnsupportedMetricError,
    describe_axisarray,
    flatten_for_plot,
    metric_axis,
    require_sweep_renderable,
)

CHANNEL_DTYPE = np.dtype([("bank", "U2"), ("elec", "<i4"), ("label", "U16")])


def ch_axis(n: int) -> AxisArray.CoordinateAxis:
    data = np.zeros(n, dtype=CHANNEL_DTYPE)
    for i in range(n):
        data["bank"][i] = "A"
        data["elec"][i] = i + 1
        data["label"][i] = f"e{i}"
    return AxisArray.CoordinateAxis(data=data, dims=["ch"], unit="")


def metric_ax(labels=("min", "max")) -> AxisArray.CoordinateAxis:
    return AxisArray.CoordinateAxis(data=np.array(list(labels)), dims=["metric"], unit="")


def signal(n_time=10, n_ch=4, fs=30000.0, unit="uV") -> AxisArray:
    return AxisArray(
        data=np.zeros((n_time, n_ch), dtype=np.float32),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=fs), "ch": ch_axis(n_ch)},
        attrs={"unit": unit},
        key="sig",
    )


def envelope(n_time=10, n_ch=4, fs=1000.0, labels=("min", "max")) -> AxisArray:
    return AxisArray(
        data=np.zeros((n_time, n_ch, 2), dtype=np.float32),
        dims=["time", "ch", "metric"],
        axes={"time": AxisArray.TimeAxis(fs=fs), "ch": ch_axis(n_ch), "metric": metric_ax(labels)},
        attrs={"unit": "uV"},
        key="env",
    )


# ---- plain signals ---------------------------------------------------------


def test_describes_a_plain_signal():
    shape = describe_axisarray(signal(n_ch=8))
    assert (shape.n_channels, shape.srate, shape.envelope) == (8, 30000.0, False)
    assert shape.channel_labels == [f"e{i}" for i in range(8)]
    assert shape.unit == "uV"


def test_extra_dimensions_fold_into_channels():
    """A (time, ch, band) block has no envelope axis, so band multiplies out."""
    msg = AxisArray(
        data=np.zeros((10, 4, 3), dtype=np.float32),
        dims=["time", "ch", "band"],
        axes={"time": AxisArray.TimeAxis(fs=100.0), "ch": ch_axis(4)},
        key="multi",
    )
    shape = describe_axisarray(msg)
    assert shape.n_channels == 12
    assert not shape.envelope


def test_missing_ch_metadata_yields_no_labels():
    msg = AxisArray(
        data=np.zeros((10, 4), dtype=np.float32),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=100.0)},
        key="bare",
    )
    assert describe_axisarray(msg).channel_labels is None


# ---- envelopes -------------------------------------------------------------


def test_describes_an_envelope():
    shape = describe_axisarray(envelope(n_ch=4))
    assert shape.envelope
    # The pair axis must not be counted as channels.
    assert shape.n_channels == 4
    assert shape.channel_labels == ["e0", "e1", "e2", "e3"]


def test_metric_kind_comes_from_labels_not_width():
    """A 2-wide trailing axis says nothing on its own: (min, max) and
    (mean, std) are the same shape and mean entirely different things."""
    dims = ["time", "ch", "metric"]

    assert metric_axis(dims, {"metric": metric_ax(("min", "max"))}).kind == "minmax"
    assert metric_axis(dims, {"metric": metric_ax(("mean", "std"))}).kind == "mean_std"
    assert metric_axis(dims, {"metric": metric_ax(("MIN", "MAX"))}).kind == "minmax"
    # Not a vocabulary we know: treat as ordinary extra dimensions.
    assert metric_axis(dims, {"metric": metric_ax(("p5", "p95"))}) is None


def test_only_minmax_is_renderable_today():
    """Others are recognised so they fail with an explanation rather than
    being drawn as if they were an envelope."""
    minmax = describe_axisarray(envelope())
    require_sweep_renderable(minmax)  # does not raise

    dispersion = describe_axisarray(envelope(labels=("mean", "std")))
    assert dispersion.metric.kind == "mean_std"
    assert not dispersion.envelope
    with pytest.raises(UnsupportedMetricError, match="mean_std"):
        require_sweep_renderable(dispersion)


def test_unrenderable_metric_still_describes_cleanly():
    """Describing is not drawing: a caller that only wants to know what
    arrived should not have to catch anything."""
    shape = describe_axisarray(envelope(n_ch=4, labels=("mean", "std")))
    assert shape.n_channels == 4
    assert shape.metric.labels == ("mean", "std")
    assert shape.channel_labels == ["e0", "e1", "e2", "e3"]


def test_metric_axis_must_be_trailing_and_named():
    assert metric_axis(["time", "metric", "ch"], {"metric": metric_ax()}) is None
    assert metric_axis(["time", "ch", "other"], {"other": metric_ax()}) is None
    assert metric_axis([], {}) is None


def test_metric_axis_of_unknown_width_is_rejected():
    wide = AxisArray.CoordinateAxis(data=np.array(["min", "max", "mean"]), dims=["metric"], unit="")
    assert metric_axis(["time", "ch", "metric"], {"metric": wide}) is None


def test_axes_may_be_plain_dicts():
    """EZShmMirror hands back dicts, not ezmsg axis objects."""
    as_dict = {"kind": "coord", "unit": "", "dims": ["metric"], "data": np.array(["min", "max"])}
    assert metric_axis(["time", "ch", "metric"], {"metric": as_dict}).kind == "minmax"


# ---- reshaping -------------------------------------------------------------


def test_envelope_keeps_its_pair_axis():
    """The regression this module exists for."""
    shape = describe_axisarray(envelope(n_ch=4))
    raw = np.arange(10 * 4 * 2, dtype=np.float32).reshape(10, 4, 2)

    out = flatten_for_plot(raw, shape)

    assert out.shape == (10, 4, 2)
    # Naive flattening would have produced (10, 8) with min/max interleaved.
    np.testing.assert_array_equal(out, raw)


def test_plain_signal_flattens_to_two_dimensions():
    shape = describe_axisarray(signal(n_ch=4))
    out = flatten_for_plot(np.zeros((10, 4), dtype=np.float32), shape)
    assert out.shape == (10, 4)


def test_extra_dimensions_flatten_into_channels():
    msg = AxisArray(
        data=np.zeros((10, 4, 3), dtype=np.float32),
        dims=["time", "ch", "band"],
        axes={"time": AxisArray.TimeAxis(fs=100.0), "ch": ch_axis(4)},
        key="multi",
    )
    shape = describe_axisarray(msg)
    assert flatten_for_plot(np.zeros((10, 4, 3), dtype=np.float32), shape).shape == (10, 12)


def test_empty_blocks_keep_their_rank():
    """A zero-length block still has to match the shape of its neighbours."""
    env = describe_axisarray(envelope(n_ch=4))
    sig = describe_axisarray(signal(n_ch=4))
    assert flatten_for_plot(np.zeros((0, 4, 2), dtype=np.float32), env).shape == (0, 4, 2)
    assert flatten_for_plot(np.zeros((0, 4), dtype=np.float32), sig).shape == (0, 4)


def test_envelope_srate_is_the_post_decimation_rate():
    """What a sweep buffer must be sized with. Using the pre-decimation rate
    makes the ring far longer than the data arriving to fill it."""
    assert describe_axisarray(envelope(fs=1000.0)).srate == pytest.approx(1000.0)


def test_transposed_source_is_described_by_the_buffer_order():
    """The sink rolls the buffered axis to the front, so a source that sent
    (ch, time) is held as (time, ch) -- and dims must say so, since a reader
    has no way to know it should re-roll them."""
    from ezmsg.tools.plot.describe import describe_mirror

    class FakeMeta:
        bvalid, ndim, srate = True, 3, 1000.0
        shape = (2000, 4, 2)  # rolled: time first

    class FakeMirror:
        meta = FakeMeta()
        # What ShMemCircBuff now records: the order the ring actually holds.
        dims = ["time", "ch", "metric"]
        axes = {
            "ch": {"kind": "coord", "data": np.zeros(4, dtype=CHANNEL_DTYPE)},
            "metric": {"kind": "coord", "data": np.array(["min", "max"])},
        }
        attrs = {"unit": "uV"}

    shape = describe_mirror(FakeMirror())
    assert shape.n_channels == 4
    assert shape.envelope
    assert shape.srate == pytest.approx(1000.0)
