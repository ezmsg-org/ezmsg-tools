"""ShmemSweepWidget's decision logic.

The widget itself needs a live ring, a GPU and a display, so what is covered
here is the part that does not: how often to read, and the rule for when a
stream change means rebuilding the plot rather than resizing it.
"""

import pytest

# Reaching the widget needs Qt and a rendering backend; a headless runner has
# neither. Keyed on the import, so it still runs wherever they exist.
_mod = pytest.importorskip(
    "ezmsg.tools.plot.shmem_sweep",
    reason="needs PySide6 + phosphor (the 'viewer' or 'sigmon' extra)",
    # Not the default: since pytest 9.1 importorskip only skips on
    # ModuleNotFoundError, and rendercanvas raises a plain ImportError from a
    # module that is very much installed.
    exc_type=ImportError,
)

from ezmsg.tools.plot.describe import MetricSpec, StreamShape  # noqa: E402

DEFAULT_POLL_HZ = _mod.DEFAULT_POLL_HZ
_poll = _mod.ShmemSweepWidget._effective_poll_hz

MINMAX = MetricSpec("metric", ("min", "max"), "minmax")


def shape(**kw) -> StreamShape:
    base = dict(n_channels=4, srate=1000.0, channel_labels=None, metric=None, unit=None)
    base.update(kw)
    return StreamShape(**base)


@pytest.mark.parametrize(
    ("poll_hz", "max_fps", "expected"),
    [
        (None, 30, 30.0),  # match the draw cadence
        (45, 30, 45.0),  # explicit wins
        (120, None, 120.0),  # explicit honoured with no cap
        (None, None, DEFAULT_POLL_HZ),  # nothing to match
        (None, 0, DEFAULT_POLL_HZ),  # uncapped is not a cadence of zero
        (None, -1, DEFAULT_POLL_HZ),
        (0, 30, 30.0),  # non-positive poll means "not set"
        (-5, None, DEFAULT_POLL_HZ),
    ],
)
def test_poll_rate_precedence(poll_hz, max_fps, expected):
    assert _poll(poll_hz, max_fps) == expected


def test_poll_rate_is_always_float():
    """The interval is computed as 1000/poll_hz, so an int here would still
    work -- but the coercion is what keeps that true for odd inputs."""
    assert isinstance(_poll(45, 30), float)


# ---- rebuild vs resize -----------------------------------------------------
#
# Rebuilding throws away the figure and flashes the plot, so it is reserved for
# changes that invalidate the buffer's layout. Everything else resizes in place.


@pytest.mark.parametrize(
    ("before", "after", "rebuild"),
    [
        (shape(), shape(n_channels=8), False),  # narrower selection: resize
        (shape(), shape(channel_labels=["a"] * 4), False),  # relabel: resize
        (shape(), shape(srate=2000.0), True),  # rate change invalidates the ring
        (shape(), shape(metric=MINMAX), True),  # envelope changes the rank
        (shape(metric=MINMAX), shape(), True),
    ],
)
def test_rebuild_only_when_the_layout_is_invalid(before, after, rebuild):
    assert _mod.ShmemSweepWidget._needs_rebuild(before, after) is rebuild


def test_first_shape_always_builds():
    assert _mod.ShmemSweepWidget._needs_rebuild(None, shape()) is True
