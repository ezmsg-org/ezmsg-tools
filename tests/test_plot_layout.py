"""Deriving a grid layout from a ``ch`` coordinate axis.

The decoding half of the grid: which fields hold coordinates, and what to do
when they are missing. Every fallback here exists because a plot that draws
nothing is less useful than one that draws the right number of cells in the
wrong places -- and because in practice the geometry is often absent, partial,
or shared between two devices that each numbered from their own origin.
"""

import numpy as np
import pytest

# The geometry helpers this builds on live in phosphor, which is an optional
# extra -- so a runner that installs only the test group cannot reach them.
# Same guard as test_shmem_sweep, and skipped for the same reason.
# Keyed on phosphor itself, not on the module under test: layout.py imports it
# inside the function, so the module imports fine without it and only fails when
# called.
pytest.importorskip(
    "phosphor.grid_layout",
    reason="needs phosphor (the 'viewer' or 'sigmon' extra)",
    exc_type=ImportError,
)

from ezmsg.tools.plot import ChannelLayoutCache, channel_layout  # noqa: E402

GEOMETRY = np.dtype([("x", "f4"), ("y", "f4"), ("size", "f4"), ("label", "U8"), ("headstage", "i4")])


def make_axis(n=4, *, fields=("x", "y", "size", "label", "headstage")):
    """A ``ch`` axis carrying only the named fields."""
    dt = np.dtype([(name, GEOMETRY[name]) for name in fields])
    ch = np.zeros(n, dtype=dt)
    if "x" in fields:
        ch["x"] = np.arange(n) % 2
    if "y" in fields:
        ch["y"] = np.arange(n) // 2
    if "size" in fields:
        ch["size"] = 0.5
    if "label" in fields:
        ch["label"] = [f"E{i}" for i in range(n)]
    return ch


def test_coordinates_are_used_verbatim():
    ch = make_axis(4, fields=("x", "y"))
    positions, sizes, labels = channel_layout(ch, 4)

    np.testing.assert_allclose(positions, [[0, 0], [1, 0], [0, 1], [1, 1]])
    assert sizes is None, "no size field means the renderer picks from the pitch"
    assert labels == ["ch0", "ch1", "ch2", "ch3"]


def test_sizes_and_labels_come_along_when_present():
    positions, sizes, labels = channel_layout(make_axis(4), 4)
    assert positions.shape == (4, 2)
    np.testing.assert_allclose(sizes, 0.5)
    assert labels == ["E0", "E1", "E2", "E3"]


def test_devices_sharing_a_coordinate_range_are_separated():
    """Two headstages numbering electrodes from the same origin would otherwise
    render one on top of the other."""
    ch = make_axis(4)
    ch["x"] = [0, 1, 0, 1]
    ch["y"] = [0, 0, 0, 0]
    ch["headstage"] = [0, 0, 1, 1]

    positions, _, _ = channel_layout(ch, 4)
    assert positions[2, 0] > positions[1, 0], "the second device must start clear of the first"


def test_the_grouping_field_can_be_ignored():
    ch = make_axis(4)
    ch["x"] = [0, 1, 0, 1]
    ch["headstage"] = [0, 0, 1, 1]

    positions, _, _ = channel_layout(ch, 4, group_field=None)
    np.testing.assert_allclose(positions[:, 0], [0, 1, 0, 1])


def test_an_axis_without_coordinates_still_gets_a_layout():
    """Channel names but no geometry: common for LSL and plain NWB streams."""
    ch = make_axis(4, fields=("label",))
    positions, sizes, labels = channel_layout(ch, 4)

    assert positions.shape == (4, 2)
    assert len({tuple(p) for p in positions}) == 4, "cells must not stack"
    assert sizes is None
    assert labels == ["E0", "E1", "E2", "E3"], "names survive even with no geometry"


def test_no_channel_axis_at_all_still_gets_a_layout():
    positions, sizes, labels = channel_layout(None, 3)
    assert positions.shape == (3, 2)
    assert sizes is None
    assert labels == ["ch0", "ch1", "ch2"]


def test_only_one_coordinate_is_not_half_a_layout():
    """An x with no y cannot place anything; falling back is the honest move."""
    ch = make_axis(4, fields=("x", "label"))
    positions, _, _ = channel_layout(ch, 4)
    assert len({tuple(p) for p in positions}) == 4


def test_the_coordinate_fields_are_configurable():
    """Nothing says a source must call them x and y."""
    dt = np.dtype([("col", "f4"), ("row", "f4")])
    ch = np.zeros(3, dtype=dt)
    ch["col"] = [5.0, 6.0, 7.0]
    ch["row"] = 1.0

    positions, _, _ = channel_layout(ch, 3, position_fields=("col", "row"))
    np.testing.assert_allclose(positions[:, 0], [5.0, 6.0, 7.0])


def test_labels_follow_the_requested_fields():
    """The same choice the sweep offers: a Blackrock user reads bank and elec
    off the front panel, not a label field."""
    dt = np.dtype([("x", "f4"), ("y", "f4"), ("bank", "U4"), ("elec", "i4")])
    ch = np.zeros(2, dtype=dt)
    ch["bank"] = ["A", "A"]
    ch["elec"] = [1, 2]

    _, _, labels = channel_layout(ch, 2, label_fields=("bank", "elec"))
    assert labels == ["A-1", "A-2"]


def test_the_axis_decides_the_count_when_it_disagrees():
    """The axis is the thing that knows how many rows it has; trusting a stale
    n_ch would index off the end of it."""
    ch = make_axis(4)
    positions, sizes, labels = channel_layout(ch, 99)
    assert positions.shape[0] == 4
    assert len(labels) == 4


@pytest.mark.parametrize("n", [0, 1])
def test_degenerate_channel_counts_do_not_raise(n):
    positions, _, labels = channel_layout(None, n)
    assert positions.shape == (n, 2)
    assert len(labels) == n


# ---- caching ----------------------------------------------------------------
#
# A grid deriving its layout per message pays per message, while the answer
# changes about once a session.


def test_an_unchanged_axis_is_not_derived_twice():
    cache = ChannelLayoutCache()
    ch = make_axis(4)
    assert cache(ch, 4) is cache(ch, 4)


def test_an_equal_but_distinct_axis_still_hits():
    """The axis arrives deserialized from another process, so it is a new object
    every message while describing the same electrodes. Keying on identity would
    make the cache useless exactly where it is needed."""
    cache = ChannelLayoutCache()
    ch = make_axis(4)
    first = cache(ch, 4)
    assert cache(ch.copy(), 4) is first


def test_a_changed_axis_is_derived_again():
    cache = ChannelLayoutCache()
    ch = make_axis(4)
    first = cache(ch, 4)

    moved = ch.copy()
    moved["x"][0] = 99.0
    second = cache(moved, 4)

    assert second is not first
    assert second[0][0, 0] == pytest.approx(99.0)


def test_a_changed_channel_count_is_derived_again():
    cache = ChannelLayoutCache()
    assert cache(None, 4) is not cache(None, 9)


def test_changed_options_are_derived_again():
    """Same axis, different question -- the answer is not the cached one."""
    dt = np.dtype([("x", "f4"), ("y", "f4"), ("bank", "U4"), ("elec", "i4")])
    ch = np.zeros(2, dtype=dt)
    ch["bank"] = ["A", "A"]
    ch["elec"] = [1, 2]

    cache = ChannelLayoutCache()
    assert cache(ch, 2)[2] == ["ch0", "ch1"]
    assert cache(ch, 2, label_fields=("bank", "elec"))[2] == ["A-1", "A-2"]


def test_no_axis_at_all_caches_too():
    cache = ChannelLayoutCache()
    assert cache(None, 3) is cache(None, 3)


def test_two_caches_do_not_share():
    """Each consumer keeps its own, so neither has to know the other exists."""
    a, b = ChannelLayoutCache(), ChannelLayoutCache()
    ch = make_axis(4)
    assert a(ch, 4) is not b(ch, 4)
