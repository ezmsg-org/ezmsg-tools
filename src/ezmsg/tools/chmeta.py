"""Turning a structured ``ch`` coordinate axis into per-channel display names.

An AxisArray's ``ch`` axis usually carries a structured array with one row per
channel. What is *in* those rows depends on the acquisition system: an LSL or
NWB source typically offers a ``label``; a Blackrock source additionally offers
``bank`` and ``elec``, which is what its users actually read off the front panel.
A plot needs one string per channel, so something has to choose.

This module makes that choice explicit and configurable rather than hard-coding
one system's convention. ``label`` is the default because it is the field most
sources populate and the one most likely to be meaningful to whoever recorded
the data; a GUI that knows better can ask for other fields.
"""

import typing

import numpy as np

__all__ = ["available_fields", "channel_names"]


def available_fields(ch_axis_data: typing.Optional[np.ndarray]) -> typing.List[str]:
    """Field names present on a structured ``ch`` axis, for building a chooser."""
    if ch_axis_data is None:
        return []
    return list((ch_axis_data.dtype.fields or {}).keys())


def _field_text(row_value: typing.Any) -> str:
    """One field of one channel as display text, or "" if it carries nothing.

    Numeric fields use 0 as their "unset" value -- a Blackrock ``elec`` of 0
    means no electrode, not electrode zero -- so it is treated as absent. Bytes
    arrive from numpy ``S`` dtypes and are decoded leniently: a mangled label is
    still more use than an exception.
    """
    if isinstance(row_value, bytes):
        row_value = row_value.decode("utf8", errors="replace")
    if isinstance(row_value, np.generic):
        row_value = row_value.item()
        if isinstance(row_value, bytes):
            row_value = row_value.decode("utf8", errors="replace")
    if isinstance(row_value, (int, float)) and not isinstance(row_value, bool):
        return "" if row_value == 0 else f"{row_value:g}"
    return str(row_value).strip()


def channel_names(
    ch_axis_data: typing.Optional[np.ndarray],
    n_ch: typing.Optional[int] = None,
    *,
    fields: typing.Sequence[str] = ("label",),
    sep: str = "-",
    fallback: str = "ch{index}",
) -> typing.List[str]:
    """Per-channel display names from a structured ``ch`` axis.

    :param ch_axis_data: The axis's structured array, or None if the source
        provided no ``ch`` axis.
    :param n_ch: Channel count, used only when ``ch_axis_data`` is None or
        unstructured. Otherwise the axis's own length governs.
    :param fields: Field names to join, in order. Fields absent from the dtype
        are skipped; fields present but empty for a given channel are skipped for
        that channel only, so a partially-populated column degrades per-row
        rather than for the whole array.
    :param sep: Joins the field values, e.g. ``("bank", "elec")`` -> ``"A-1"``.
    :param fallback: Format string used when no requested field yields anything
        for a channel. ``{index}`` is the channel's index.

    :return: One name per channel. Never empty strings, so a caller can render
        the result without further guarding.
    """
    if ch_axis_data is None or ch_axis_data.dtype.fields is None:
        n = int(n_ch or 0) if ch_axis_data is None else int(ch_axis_data.shape[0])
        return [fallback.format(index=i) for i in range(n)]

    present = [f for f in fields if f in ch_axis_data.dtype.fields]
    n = int(ch_axis_data.shape[0])
    if not present:
        return [fallback.format(index=i) for i in range(n)]

    columns = [ch_axis_data[f] for f in present]
    names = []
    for i in range(n):
        parts = [t for t in (_field_text(col[i]) for col in columns) if t]
        names.append(sep.join(parts) if parts else fallback.format(index=i))
    return names
