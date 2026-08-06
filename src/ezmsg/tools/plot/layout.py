"""Turning a structured ``ch`` coordinate axis into a per-channel grid layout.

The geometry counterpart to :mod:`ezmsg.tools.chmeta`, which does the same job
for names. An AxisArray's ``ch`` axis may carry electrode coordinates; a grid
plot wants positions, sizes and labels. Which fields those live in is a
property of the acquisition system, so the decoding belongs here rather than in
phosphor, whose grids take plain arrays and have no opinion about where they
came from.

Everything degrades: a source with no ``ch`` axis at all, or one carrying names
but no coordinates, still gets a layout -- a square-ish tiling -- because a plot
that draws nothing is less useful than one that draws the right number of cells
in the wrong places.
"""

import typing

import numpy as np

from ..chmeta import channel_names

__all__ = ["ChannelLayoutCache", "channel_layout"]

DEFAULT_POSITION_FIELDS = ("x", "y")
DEFAULT_SIZE_FIELD = "size"
DEFAULT_GROUP_FIELD = "headstage"


def channel_layout(
    ch_axis_data: typing.Optional[np.ndarray],
    n_ch: int,
    *,
    position_fields: typing.Tuple[str, str] = DEFAULT_POSITION_FIELDS,
    size_field: str = DEFAULT_SIZE_FIELD,
    group_field: typing.Optional[str] = DEFAULT_GROUP_FIELD,
    label_fields: typing.Sequence[str] = ("label",),
) -> typing.Tuple[np.ndarray, typing.Optional[np.ndarray], typing.List[str]]:
    """Per-channel ``(positions, sizes, labels)`` for a grid plot.

    :param ch_axis_data: The ``ch`` axis' structured data, or ``None`` when the
        stream carries no channel metadata.
    :param n_ch: Channels in the data, used when the axis cannot say.
    :param position_fields: Fields holding each channel's coordinates. Both must
        be present, or the layout falls back to a tiling.
    :param size_field: Field holding each channel's extent. Absent gives
        ``None``, which lets the renderer size cells by the inferred pitch.
    :param group_field: Field identifying which device a channel belongs to.
        Devices commonly number their electrodes from a shared origin, so
        without this two of them draw on top of each other. ``None`` skips the
        check.
    :param label_fields: Passed to :func:`~ezmsg.tools.chmeta.channel_names`.

    :returns: ``positions`` as ``(n, 2)`` float32, ``sizes`` as ``(n,)`` float32
        or ``None``, and one label per channel.
    """
    from phosphor.grid_layout import tile_by_group, tiled_grid_positions

    if ch_axis_data is None:
        return tiled_grid_positions(n_ch), None, channel_names(None, n_ch, fields=label_fields)

    fields = ch_axis_data.dtype.fields or {}
    actual_n = ch_axis_data.shape[0]
    x_field, y_field = position_fields

    if {x_field, y_field} <= set(fields):
        positions = np.column_stack([ch_axis_data[x_field], ch_axis_data[y_field]]).astype(np.float32)
        if group_field and group_field in fields:
            positions = tile_by_group(positions, ch_axis_data[group_field])
    else:
        positions = tiled_grid_positions(actual_n)

    sizes = ch_axis_data[size_field].astype(np.float32) if size_field in fields else None
    return positions, sizes, channel_names(ch_axis_data, actual_n, fields=label_fields)


class ChannelLayoutCache:
    """:func:`channel_layout`, recomputed only when the axis actually changes.

    A grid that derives its layout per message pays for it per message, while
    the answer changes about once a session. Fingerprinting the axis costs
    around 1 us against the 75 us the derivation takes, so this is worth having
    wherever messages arrive faster than the geometry does.

    Deliberately a cache rather than a shared instance: two widgets watching one
    stream each keep their own, so neither has to know the other exists, and the
    derivation stays where any application can call it.
    """

    def __init__(self) -> None:
        self._key: typing.Optional[tuple] = None
        self._value: typing.Optional[tuple] = None

    @staticmethod
    def _fingerprint(ch_axis_data: typing.Optional[np.ndarray], n_ch: int, kwargs: dict) -> tuple:
        """Enough to tell one layout apart from another, cheaply.

        The axis' bytes rather than its identity: it arrives deserialized from
        another process, so a new object every message describes the same
        electrodes.
        """
        options = tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in kwargs.items()))
        if ch_axis_data is None:
            return (n_ch, options, None)
        return (n_ch, options, str(ch_axis_data.dtype), ch_axis_data.shape, ch_axis_data.tobytes())

    def __call__(
        self,
        ch_axis_data: typing.Optional[np.ndarray],
        n_ch: int,
        **kwargs: typing.Any,
    ) -> typing.Tuple[np.ndarray, typing.Optional[np.ndarray], typing.List[str]]:
        key = self._fingerprint(ch_axis_data, n_ch, kwargs)
        if key != self._key:
            self._value = channel_layout(ch_axis_data, n_ch, **kwargs)
            self._key = key
        return self._value
