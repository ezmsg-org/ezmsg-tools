"""Reading a stream's shape well enough to plot it.

Every consumer that puts ezmsg data on a phosphor widget has to answer the same
questions -- how many channels, at what rate, what are they called, and is this
a signal or an envelope -- and each has been answering them slightly
differently. This module answers them once.

It deliberately does not import phosphor or Qt: the inputs are dims, axes and
attrs, and the outputs are plain numbers and arrays. That keeps it usable from a
topic subscriber, from a shared-memory mirror, and from a test with neither.
"""

from __future__ import annotations

import typing

import numpy as np

from ..chmeta import channel_names

__all__ = [
    "METRIC_AXIS_CANDIDATES",
    "METRIC_KINDS",
    "SWEEP_RENDERABLE_METRICS",
    "MetricSpec",
    "StreamShape",
    "UnsupportedMetricError",
    "describe_axisarray",
    "describe_mirror",
    "flatten_for_plot",
    "metric_axis",
    "require_sweep_renderable",
]

# Axis names an upstream aggregator might use for its per-sample tuple.
# ezmsg-sigproc's BinnedAggregate calls it "metric" by default but the name is a
# setting, so recognising a couple of obvious alternatives costs nothing.
METRIC_AXIS_CANDIDATES = ("metric", "minmax", "bound", "stat")

# Label tuples we recognise, and what to call the thing they describe. Keyed on
# labels rather than width because width says nothing: (min, max) and
# (mean, std) are both 2-wide and mean entirely different things, and drawing
# one as the other is silently wrong rather than visibly broken.
#
# Adding a kind here is the cheap half. The expensive half is teaching a
# renderer to draw it -- see SWEEP_RENDERABLE_METRICS.
METRIC_KINDS: dict[tuple[str, ...], str] = {
    ("min", "max"): "minmax",
    ("mean", "std"): "mean_std",
    ("mean", "sem"): "mean_sem",
}

# What a sweep plot can actually draw today.
#
# "minmax" maps onto phosphor's envelope input directly: the pair *is* the band,
# so the existing column reduction (min of mins, max of maxes) is correct.
#
# A dispersion pair like "mean_std" needs different drawing -- a semi-transparent
# band from mean-std to mean+std with an opaque line at the mean -- and
# different column reduction, since averaging a mean is not the same as taking
# extremes. Recognised here so it fails with an explanation instead of being
# drawn as if it were an envelope.
SWEEP_RENDERABLE_METRICS = frozenset({"minmax"})


class MetricSpec(typing.NamedTuple):
    """A trailing per-sample tuple: what it is called and what it holds."""

    axis: str
    """Name of the trailing axis."""

    labels: tuple[str, ...]
    """Its coordinate values, lowercased, in order."""

    kind: str
    """The entry in :data:`METRIC_KINDS` these labels matched."""


class UnsupportedMetricError(NotImplementedError):
    """A recognised metric axis that no renderer here can draw yet."""


class StreamShape(typing.NamedTuple):
    """What a plot needs to know about an incoming stream."""

    n_channels: int
    """Channels, excluding any envelope axis."""

    srate: float
    """Samples per second of the *pushed* stream. For an envelope this is the
    bucket rate, not the rate before decimation -- which is what a sweep buffer
    must be sized with, or its ring is longer than the data arriving to fill
    it."""

    channel_labels: list[str] | None
    """One name per channel, or None if the stream does not say."""

    metric: MetricSpec | None
    """The trailing per-sample tuple, if the stream carries one."""

    unit: str | None
    """The signal's amplitude unit, if it declares one."""

    @property
    def envelope(self) -> bool:
        """Whether each sample carries a (min, max) pair -- phosphor's envelope."""
        return self.metric is not None and self.metric.kind == "minmax"


def metric_axis(dims: typing.Sequence[str], axes: typing.Mapping[str, typing.Any]) -> MetricSpec | None:
    """Describe the trailing per-sample tuple, or None if there is not one.

    Recognised by *labels*, not by name or width. The name only narrows the
    search; the labels are what distinguish a (min, max) envelope from a
    (mean, std) dispersion pair, which is the same shape and must not be drawn
    the same way.

    Returns a spec for any tuple in :data:`METRIC_KINDS`, including ones no
    renderer here supports yet -- describing a stream is not the same as being
    able to draw it, and a caller that only wants to know what arrived should
    not have to catch an exception. See :func:`require_sweep_renderable` for
    the capability check.
    """
    if not dims:
        return None
    name = dims[-1]
    if name not in METRIC_AXIS_CANDIDATES:
        return None
    data = _axis_data(axes.get(name))
    if data is None:
        return None
    labels = tuple(str(v).lower() for v in data)
    kind = METRIC_KINDS.get(labels)
    return None if kind is None else MetricSpec(axis=name, labels=labels, kind=kind)


def require_sweep_renderable(shape: StreamShape) -> None:
    """Raise if a sweep plot cannot draw this stream's metric axis.

    :raises UnsupportedMetricError: for a recognised metric a sweep cannot draw.
    """
    metric = shape.metric
    if metric is None or metric.kind in SWEEP_RENDERABLE_METRICS:
        return
    raise UnsupportedMetricError(
        f"stream carries a {metric.kind!r} metric axis {metric.labels} on {metric.axis!r}, "
        f"which a sweep plot cannot draw yet (supported: {sorted(SWEEP_RENDERABLE_METRICS)}). "
        "Aggregate the stream differently upstream, or add rendering for it."
    )


def _axis_data(axis: typing.Any) -> np.ndarray | None:
    """Coordinate values of an axis given either as a dict or an ezmsg object."""
    if axis is None:
        return None
    if isinstance(axis, dict):
        data = axis.get("data")
    else:
        data = getattr(axis, "data", None)
    return None if data is None else np.asarray(data)


def _axis_gain(axis: typing.Any) -> float | None:
    if axis is None:
        return None
    gain = axis.get("gain") if isinstance(axis, dict) else getattr(axis, "gain", None)
    return None if gain in (None, 0) else float(gain)


def _describe(
    dims: typing.Sequence[str],
    axes: typing.Mapping[str, typing.Any],
    attrs: typing.Mapping[str, typing.Any],
    shape: typing.Sequence[int],
    srate: float | None,
    *,
    time_axis: str = "time",
    label_fields: typing.Sequence[str] = ("label",),
) -> StreamShape:
    dims = list(dims)
    metric = metric_axis(dims, axes)
    metric_name = metric.axis if metric is not None else None

    # Channel count is everything that is neither time nor the metric tuple.
    n_channels = 1
    for name, size in zip(dims, shape):
        if name in (time_axis, metric_name):
            continue
        n_channels *= int(size)

    if srate is None:
        gain = _axis_gain(axes.get(time_axis))
        srate = 1.0 / gain if gain else 0.0

    ch_data = _axis_data(axes.get("ch"))
    labels = None
    if ch_data is not None and ch_data.dtype.fields is not None:
        labels = channel_names(ch_data, n_channels, fields=label_fields)

    unit = attrs.get("unit") if attrs else None
    return StreamShape(
        n_channels=max(1, n_channels),
        srate=float(srate or 0.0),
        channel_labels=labels,
        metric=metric,
        unit=None if unit is None else str(unit),
    )


def describe_axisarray(
    msg: typing.Any,
    *,
    time_axis: str = "time",
    label_fields: typing.Sequence[str] = ("label",),
) -> StreamShape:
    """Describe a stream from one of its ``AxisArray`` messages."""
    return _describe(
        msg.dims,
        msg.axes,
        getattr(msg, "attrs", None) or {},
        msg.data.shape,
        None,
        time_axis=time_axis,
        label_fields=label_fields,
    )


def describe_mirror(
    mirror: typing.Any,
    *,
    time_axis: str = "time",
    label_fields: typing.Sequence[str] = ("label",),
) -> StreamShape | None:
    """Describe a stream from a connected :class:`EZShmMirror`.

    Returns None until the writer has published both a valid buffer header and
    its metadata -- the two arrive independently, and a description built from
    only one of them would be missing either the shape or the names.
    """
    meta = mirror.meta
    if meta is None or not meta.bvalid or meta.ndim < 2:
        return None
    axes = mirror.axes
    if axes is None:
        return None
    shape = tuple(int(v) for v in meta.shape[: meta.ndim])
    # dims and meta.shape describe the same ordering -- the sink records the
    # order the ring actually holds, not the order the message arrived in.
    return _describe(
        list(mirror.dims or []),
        axes,
        mirror.attrs or {},
        shape,
        float(meta.srate),
        time_axis=time_axis,
        label_fields=label_fields,
    )


def flatten_for_plot(data: np.ndarray, shape: StreamShape) -> np.ndarray:
    """Reshape a block to what a plot's ``push_data`` expects.

    ``(n_samples, n_channels, k)`` when the stream carries a k-wide metric
    tuple, ``(n_samples, n_channels)`` otherwise, with any other dimensions
    folded into channels.

    The metric case is the reason this exists. Folding a ``(time, ch, 2)``
    block into ``(time, ch * 2)`` -- which is what a naive ``reshape`` does --
    renders as twice as many traces, alternating the two metrics, with every
    channel label off by a factor of two. It looks like data, so nothing
    complains.
    """
    width = len(shape.metric.labels) if shape.metric is not None else None
    tail = (shape.n_channels,) if width is None else (shape.n_channels, width)
    if data.size == 0:
        return data.reshape((0,) + tail)
    return data.reshape((data.shape[0],) + tail)
