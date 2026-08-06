"""Putting ezmsg streams onto phosphor plots.

:mod:`.describe` is the pure half -- given dims, axes and attrs, work out what
is being plotted -- and imports neither Qt nor phosphor, so it is usable from a
topic subscriber, a shared-memory mirror, or a test with neither.
:mod:`.shmem_sweep` is the Qt widget built on it, and needs the ``viewer`` or
``sigmon`` extra.

``ShmemSweepWidget`` is resolved lazily so that importing this package, or
anything under it, does not pull in Qt. Eagerly importing it here would make
``from ezmsg.tools.plot.describe import ...`` fail without phosphor installed,
since importing a submodule runs its parent's ``__init__`` first -- which would
put a GPU stack behind a module that deliberately has no rendering dependency
at all.
"""

import typing

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

if typing.TYPE_CHECKING:  # pragma: no cover - import for type checkers only
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


def __getattr__(name: str) -> typing.Any:
    """Resolve the Qt widget on first use (PEP 562)."""
    if name == "ShmemSweepWidget":
        from .shmem_sweep import ShmemSweepWidget

        return ShmemSweepWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
