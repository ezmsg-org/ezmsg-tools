"""Putting ezmsg streams onto phosphor plots.

:mod:`.describe` is the pure half -- given dims, axes and attrs, work out what
is being plotted -- and imports neither Qt nor phosphor, so it is usable from a
topic subscriber, a shared-memory mirror, or a test with neither.
:mod:`.shmem_sweep` is the Qt widget built on it.

Requires the ``viewer`` or ``sigmon`` extra (PySide6 and phosphor). Importing
this package pulls in Qt, so import :mod:`.describe` directly if that is all
you need.
"""

from .describe import (
    METRIC_KINDS,
    SWEEP_RENDERABLE_METRICS,
    MetricSpec,
    StreamShape,
    UnsupportedMetricError,
    describe_axisarray,
    describe_mirror,
    flatten_for_plot,
    metric_axis,
    require_sweep_renderable,
)
from .shmem_sweep import ShmemSweepWidget

__all__ = [
    "METRIC_KINDS",
    "SWEEP_RENDERABLE_METRICS",
    "MetricSpec",
    "ShmemSweepWidget",
    "StreamShape",
    "UnsupportedMetricError",
    "describe_axisarray",
    "describe_mirror",
    "flatten_for_plot",
    "metric_axis",
    "require_sweep_renderable",
]
