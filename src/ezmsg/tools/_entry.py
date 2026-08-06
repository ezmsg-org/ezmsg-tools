"""Launching a console script whose dependencies live in an optional extra.

Console scripts are installed unconditionally -- ``[project.scripts]`` has no
notion of extras -- so ``pip install ezmsg-tools`` puts commands on the PATH
whose imports are not satisfied. Run one and you get a bare
``ModuleNotFoundError: dash`` with no hint that an extra exists or what it is
called.

The alternative would be promoting those dependencies to the core install, so
that a Dash web app drags in Qt and a GPU stack for everyone. A clear message
is the cheaper fix.
"""

import importlib
import os
import sys
import typing

__all__ = ["run_cli"]


def run_cli(module: str, extra: str) -> typing.NoReturn:
    """Import ``module`` and call its ``main()``, or explain what is missing.

    :param module: Dotted path of the CLI module to run.
    :param extra: The extra that declares this command's dependencies.
    """
    try:
        cli = importlib.import_module(module)
    except ImportError as exc:
        command = os.path.basename(sys.argv[0]) or module
        missing = getattr(exc, "name", None)
        # Name the module that was actually missing rather than assuming the
        # extra is the whole story: an ImportError from inside the CLI is a
        # different problem, and saying which one it was keeps this honest.
        detail = f" (could not import {missing!r})" if missing else ""
        raise SystemExit(
            f"{command} needs the optional '{extra}' dependencies{detail}.\n"
            f"Install them with:\n\n"
            f"    pip install 'ezmsg-tools[{extra}]'\n"
        ) from exc
    sys.exit(cli.main())
