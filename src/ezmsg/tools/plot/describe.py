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
    "ENVELOPE_AXIS_CANDIDATES",
    "StreamShape",
    "describe_axisarray",
    "describe_mirror",
    "envelope_axis",
    "flatten_for_plot",
]

# Axis names an upstream min/max decimator might use for its (min, max) pair.
# ezmsg-sigproc's BinnedAggregate calls it "metric" by default but the name is a
# setting, so recognising a couple of obvious alternatives costs nothing.
ENVELOPE_AXIS_CANDIDATES = ("metric", "minmax", "bound")

# Aggregation-function labels that make an axis an envelope rather than, say,
# a (mean, std) pair -- which is 2-wide and named the same way but must not be
# drawn as an upper and lower bound.
_ENVELOPE_LABELS = ("min", "max")


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

    envelope: bool
    """Whether each sample carries a (min, max) pair on a trailing axis."""

    unit: str | None
    """The signal's amplitude unit, if it declares one."""


def envelope_axis(dims: typing.Sequence[str], axes: typing.Mapping[str, typing.Any]) -> str | None:
    """Name of the trailing (min, max) axis, or None if this is a plain signal.

    Identified by its *labels* rather than its name or width. A length-2
    trailing axis could as easily be (mean, std), which would be nonsense to
    draw as an envelope, so the coordinate values have to say ``min`` and
    ``max``. The name is only used to narrow the search.
    """
    if not dims:
        return None
    name = dims[-1]
    if name not in ENVELOPE_AXIS_CANDIDATES:
        return None
    data = _axis_data(axes.get(name))
    if data is None or len(data) != 2:
        return None
    return name if tuple(str(v).lower() for v in data) == _ENVELOPE_LABELS else None


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
    env_axis = envelope_axis(dims, axes)

    # Channel count is everything that is not time and not the envelope pair.
    n_channels = 1
    for name, size in zip(dims, shape):
        if name in (time_axis, env_axis):
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
        envelope=env_axis is not None,
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
    # The ring rolls the buffered axis to the front; dims record the sender's
    # order, so rebuild the order the buffer is actually in.
    dims = list(mirror.dims or [])
    if time_axis in dims:
        dims.insert(0, dims.pop(dims.index(time_axis)))
    return _describe(
        dims,
        axes,
        mirror.attrs or {},
        shape,
        float(meta.srate),
        time_axis=time_axis,
        label_fields=label_fields,
    )


def flatten_for_plot(data: np.ndarray, shape: StreamShape) -> np.ndarray:
    """Reshape a block to what phosphor's ``push_data`` expects.

    ``(n_samples, ..., 2)`` for an envelope, ``(n_samples, n_channels)``
    otherwise, with any extra dimensions folded into channels.

    The envelope case is the reason this exists. Folding a ``(time, ch, 2)``
    block into ``(time, ch * 2)`` -- which is what a naive ``reshape`` does --
    renders as twice as many traces, alternating lower and upper bounds, with
    every channel label off by a factor of two. It looks like data, so nothing
    complains.
    """
    if data.size == 0:
        return data.reshape((0, shape.n_channels, 2) if shape.envelope else (0, shape.n_channels))
    n_samples = data.shape[0]
    if shape.envelope:
        return data.reshape(n_samples, shape.n_channels, 2)
    return data.reshape(n_samples, shape.n_channels)
