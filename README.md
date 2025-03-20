# ezmsg-tools

A namespace package for [ezmsg](https://github.com/iscoe/ezmsg) to visualize running graphs and data.

The data visualization is highly fragile. Expect bugs.

## Installation

### Pre-requisites

* [graphviz](https://graphviz.org/download/)

On Mac, you should use brew:

* `brew install graphviz`
* `export CFLAGS="-I $(brew --prefix graphviz)/include"`
* `export LDFLAGS="-L $(brew --prefix graphviz)/lib"`

### Release

Install the latest release from pypi with: `pip install ezmsg-tools` (or `uv add ...` or `poetry add ...`).

More than likely, you will want to include at least one of the extras when installing:

`pip install "ezmsg-tools[all]"`

### Development Version

If you intend to edit `ezmsg-tools` then please refer to the [Developers](#developers) section below.

You can add the development version of ezmsg-tools directly from GitHub:

* Using `pip`: `pip install git+https://github.com/ezmsg-org/ezmsg-tools.git@dev`
* Using `poetry`: `poetry add "git+https://github.com/ezmsg-org/ezmsg-tools.git@dev"`
* Using `uv`: `uv add git+https://github.com/ezmsg-org/ezmsg-tools --branch dev`

You probably want to include the extras when installing the development version:

* `pip install "ezmsg-tools[all] @ git+https://github.com/ezmsg-org/ezmsg-tools.git@dev"`

## Getting Started

This package includes some entrypoints with useful tools.

### ezmsg-performance-monitor

This tool operates on logfiles created by ezmsg. Logfiles will automatically be created when running a pipeline containing nodes decorated with `ezmsg.sigproc.util.profile.profile_subpub`,
and if the `EZMSG_LOGLEVEL` environment variable is set to DEBUG. The logfiles will be created in `~/.ezmsg/profile/ezprofiler.log` by default but this can be changed with the `EZMSG_PROFILE` environment variable.

Most of the nodes provided by `ezmsg.sigproc` are already decorated to enable profiling, as is any custom nodes that inherit from `ezmsg.sigproc.base.GenAxisArray`.
You can decorate other nodes with `ezmsg.sigproc.util.profile.profile_subpub` to enable profiling.

During a run with profiling enabled, the logfiles will be created in the specified location. You may wish to additionally create a graph file: (`uv run`) `EZMSG_LOGLEVEL=WARN ezmsg mermaid > ~/.ezmsg/profile/ezprofiler.mermaid`

During or after a pipeline run with profiling enabled, you can run (`uv run `) `performance-monitor` to visualize the performance of the nodes in the pipeline.

> Unlike `signal-monitor`, this tool does not require the pipeline to attach to an existing graph service because it relies exclusively on the logfile.

### ezmsg-signal-monitor

The pipeline must be running on a graph service exposed on the network. For example, first, run the GraphService on an open port:

`ezmsg --address 127.0.0.1:25978 start`

Then run your usual pipeline but make sure it attaches to the graph address by passing `graph_address=("127.0.0.1", 25978)` as a kwarg to `ez.run`.

While the pipeline is running, you can run the signal-monitor tool with (`uv run`) `signal-monitor --graph-addr 127.0.0.1:25978`.

This launches a window with graph visualized on the left. Click on a node's output box to get a live visualization on the right side of the screen plotting the data as it leaves that node. Use `a` to toggle auto-scaling. With auto-scaling off, use `-`, and `=` to zoom out and in, respectively.

> Currently only 2-D outputs are supported!

Don't forget to shutdown your graph service when you are done, e.g.: `ezmsg --address 127.0.0.1:25978 shutdown` 

## Developers

We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for development. It is not strictly required, but if you intend to contribute to ezmsg-tools then using `uv` will lead to the smoothest collaboration.

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.
2. Fork ezmsg-tools and clone your fork to your local computer.
3. Open a terminal and `cd` to the cloned folder.
4. Make sure `pygraphviz` [pre-requisites](#pre-requisites) are installed.
    * On mac: `export CFLAGS="-I $(brew --prefix graphviz)/include"` and `export LDFLAGS="-L $(brew --prefix graphviz)/lib"`
5. `uv sync --all-extras --python 3.10` to create a .venv and install ezmsg-tools including dev and test dependencies.
6. After editing code and making commits, Run the test suite before making a PR: `uv run pytest`

## Troubleshooting

Graphviz can be difficult to install on some systems. The simplest may be to use conda/mamba: `conda install graphviz`.
If that fails, [see here](https://github.com/pygraphviz/pygraphviz/issues/398#issuecomment-1038476921).
