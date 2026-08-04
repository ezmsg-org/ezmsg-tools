"""Codec for the *static* half of an AxisArray, carried alongside the shmem ring.

The circular buffer in :mod:`.shmem` transports raw samples plus the little that
fits in a fixed ctypes header: dtype, shape, sample rate, and ``key``. Everything
else an :class:`~ezmsg.util.messages.axisarray.AxisArray` knows -- the ``ch``
coordinate axis holding per-channel ``bank``/``elec``/``label``, the units, the
message ``attrs`` -- is dropped at the boundary. A consumer on the far side can
therefore plot the signal but cannot say what any of it *is*.

This module encodes that dropped metadata into a self-describing byte blob which
:class:`~.shmem.ShMemCircBuff` publishes into its own shared-memory segment, and
which :class:`~.shmem_mirror.EZShmMirror` decodes on the far side.

Wire format
-----------
A pickled ``dict`` of **plain Python and numpy types only** -- never ezmsg
classes. The two halves of a shmem link are separate processes and may be
separate environments with different ezmsg versions installed; pinning the wire
format to ezmsg's dataclass layout would make an upgrade on one side a silent
decode failure on the other. Plain dicts cost one down-conversion and buy
version independence. ``AUX_FORMAT_VERSION`` guards the shape of the dict
itself.

Axes decode to::

    {"kind": "linear", "unit": str, "gain": float, "offset": float}
    {"kind": "coord",  "unit": str, "dims": list[str], "data": np.ndarray}

The buffered axis (normally ``time``) is a deliberate special case: its
``offset`` advances with every message and a coordinate time axis's ``data`` is
wholly new each message, so including either would make the metadata change
continuously and defeat the point of a low-rate side channel. Only its static
descriptors are kept -- see :func:`encode_aux`.
"""

import pickle
import typing

import numpy as np

AUX_FORMAT_VERSION = 1

# Values we are willing to put on the wire. Anything else in ``attrs`` is
# dropped rather than pickled: an arbitrary object would force the decoding
# process to import the class that defines it, which is exactly the coupling
# this format exists to avoid.
_PLAIN_SCALARS = (str, bytes, int, float, bool, complex, type(None))


def _is_plain(value: typing.Any) -> bool:
    """Whether ``value`` is safe to pickle into the blob."""
    if isinstance(value, _PLAIN_SCALARS):
        return True
    if isinstance(value, np.ndarray):
        # Object arrays hold arbitrary picklable classes; same objection.
        return value.dtype != np.dtype("O")
    if isinstance(value, np.generic):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_plain(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_plain(v) for k, v in value.items())
    return False


def axis_to_plain(axis: typing.Any, *, static_only: bool = False) -> dict:
    """Down-convert one ezmsg axis to plain types.

    ``static_only`` keeps just the descriptors that do not change from message to
    message -- used for the buffered axis, whose position along the stream is
    carried by the ring's write index rather than by this blob.
    """
    unit = getattr(axis, "unit", "")
    if hasattr(axis, "data"):  # CoordinateAxis
        if static_only:
            return {"kind": "coord", "unit": unit, "dims": list(axis.dims)}
        return {
            "kind": "coord",
            "unit": unit,
            "dims": list(axis.dims),
            "data": np.array(axis.data, copy=True),
        }
    out = {"kind": "linear", "unit": unit, "gain": float(axis.gain)}
    if not static_only:
        out["offset"] = float(axis.offset)
    return out


def encode_aux(
    dims: typing.Sequence[str],
    axes: typing.Mapping[str, typing.Any],
    attrs: typing.Mapping[str, typing.Any],
    key: str,
    buffered_axis: str,
) -> tuple[bytes, list[str]]:
    """Serialize an AxisArray's static metadata.

    Returns the blob and the list of ``attrs`` keys that were dropped for not
    being plain types, so the caller can log them once rather than per message.
    """
    plain_axes = {name: axis_to_plain(ax, static_only=(name == buffered_axis)) for name, ax in axes.items()}
    plain_attrs = {}
    dropped = []
    for name, value in attrs.items():
        if isinstance(name, str) and _is_plain(value):
            plain_attrs[name] = value
        else:
            dropped.append(str(name))
    payload = {
        "version": AUX_FORMAT_VERSION,
        "dims": list(dims),
        "axes": plain_axes,
        "attrs": plain_attrs,
        "key": key,
        "buffered_axis": buffered_axis,
    }
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), dropped


def decode_aux(blob: bytes) -> dict:
    """Inverse of :func:`encode_aux`.

    :raises ValueError: if the blob is unreadable or was written by a format
        version this build does not understand.
    """
    try:
        payload = pickle.loads(blob)
    except Exception as exc:  # noqa: BLE001 - any unpickling failure is the same failure to us
        raise ValueError(f"could not decode shmem metadata blob: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"shmem metadata blob decoded to {type(payload).__name__}, expected dict")
    version = payload.get("version")
    if version != AUX_FORMAT_VERSION:
        raise ValueError(
            f"shmem metadata blob is format version {version!r}, this build understands {AUX_FORMAT_VERSION}"
        )
    return payload


def _axis_equal(a: typing.Any, b: typing.Any) -> bool:
    """Value equality for one axis, compared field by field.

    Deliberately does not use ``==``. As of ezmsg 3.6, ``CoordinateAxis.__eq__``
    resolves through the MRO to the dataclass-generated ``AxisBase.__eq__``,
    which compares ``unit`` and nothing else -- ``ArrayWithNamedDims.__eq__``,
    written to compare ``dims`` and ``data``, is shadowed and never runs. Two
    coordinate axes with different data, or even different lengths, therefore
    compare equal. Relying on that would mean a channel relabelling silently
    never reaching the far side of the shmem link, which is the one thing this
    module exists to deliver. Comparing explicitly also keeps the check correct
    across ezmsg versions, which matters given the two halves of a link need not
    share one.
    """
    a_data = getattr(a, "data", None)
    b_data = getattr(b, "data", None)
    if (a_data is None) != (b_data is None):
        return False
    if getattr(a, "unit", "") != getattr(b, "unit", ""):
        return False
    if a_data is None:
        return a.gain == b.gain and a.offset == b.offset
    if list(a.dims) != list(b.dims):
        return False
    if a_data is b_data:
        return True
    if a_data.shape != b_data.shape or a_data.dtype != b_data.dtype:
        return False
    return bool(np.array_equal(a_data, b_data))


def axes_equal(a: typing.Mapping[str, typing.Any], b: typing.Mapping[str, typing.Any]) -> bool:
    """Cheap "have the axes changed?" test, for the per-message hot path.

    Identity is checked before value at every level, which is what makes this
    affordable at kHz rates: an ezmsg processor that leaves an axis alone passes
    the *same object* through, so the common case costs one pointer comparison
    per axis. An element-wise comparison happens only when a producer rebuilt an
    axis -- rare, and precisely the case we must not get wrong.
    """
    if a is b:
        return True
    if a.keys() != b.keys():
        return False
    for name, av in a.items():
        bv = b[name]
        if av is bv:
            continue
        if type(av) is not type(bv):
            return False
        if not _axis_equal(av, bv):
            return False
    return True


def attrs_equal(a: typing.Mapping[str, typing.Any], b: typing.Mapping[str, typing.Any]) -> bool:
    """Cheap "have the attrs changed?" test.

    Identity only. ``attrs`` values are arbitrary -- a numpy array's ``==``
    returns an array, and a user class's ``__eq__`` could be arbitrarily
    expensive -- so value equality is deliberately not attempted here. A producer
    that rebuilds equal attrs every message trips this check; the caller absorbs
    that by comparing the encoded blob before it republishes anything.
    """
    if a is b:
        return True
    if a.keys() != b.keys():
        return False
    return all(a[name] is b[name] for name in a)
