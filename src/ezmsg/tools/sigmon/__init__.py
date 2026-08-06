"""Signal monitor: a graph inspector with live plots.

The console script points here rather than at :mod:`.cli` so that a missing
``sigmon`` extra produces an explanation instead of a traceback -- see
:mod:`ezmsg.tools._entry`.
"""

from .._entry import run_cli


def main() -> None:
    run_cli("ezmsg.tools.sigmon.cli", "sigmon")
