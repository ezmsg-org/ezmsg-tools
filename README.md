# ezmsg-graphviz

A namespace package for [ezmsg](https://github.com/iscoe/ezmsg) to visualize running graphs and data.

The data visualization is highly fragile. Expect bugs.

## Installation

### Pre-requisites

* [graphviz](https://graphviz.org/download/)

On Mac, you should:

* `brew install graphviz`
* `export CFLAGS="-I $(brew --prefix graphviz)/include"`
* `export LDFLAGS="-L $(brew --prefix graphviz)/lib"`

### Release

Install the latest release from pypi with: `pip install ezmsg-graphviz` (or `uv add ...` or `poetry add ...`).

### Development Version

You can add the development version of `ezmsg-graphviz` to your project's dependencies in one of several ways.

You can clone it and add its path to your project dependencies. You may wish to do this if you intend to edit `ezmsg-graphviz`. If so, please refer to the [Developers](#developers) section below.

You can also add it directly from GitHub:

* Using `pip`: `pip install git+https://github.com/ezmsg-org/ezmsg-graphviz.git@dev`
* Using `poetry`: `poetry add "git+https://github.com/ezmsg-org/ezmsg-graphviz.git@dev"`
* Using `uv`: `uv add git+https://github.com/ezmsg-org/ezmsg-graphviz --branch dev`

## Developers

We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for development. It is not strictly required, but if you intend to contribute to ezmsg-graphviz then using `uv` will lead to the smoothest collaboration.

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.
2. Fork ezmsg-graphviz and clone your fork to your local computer.
3. Open a terminal and `cd` to the cloned folder.
4. `uv sync --all-extras --python 3.10` to create a .venv and install ezmsg-graphviz including dev and test dependencies.
5. After editing code and making commits, Run the test suite before making a PR: `uv run pytest tests`


## Usage

You must have an ezmsg pipeline running on a graph service exposed on the network. For example, first, run the GraphService on an open port:

`ezmsg --address 127.0.0.1:25978 start`

Then run your usual pipeline but pass the graph address `ez.run`:

`ez.run(..., graph_address=("127.0.0.1", 25978))`

> You can use the supplied overly-simple example: `python -m ezmsg.graphviz.examples.ez_count`

And then you can connect to that with `ezmsg-monitor --graph_addr 127.0.0.1:25978`

Don't forget to tear down your GraphService when you are done with it:

`ezmsg --address 127.0.0.1:25978 shutdown`

## Troubleshooting

Graphviz can be difficult to install on some systems. The simplest may be to use conda/mamba: `conda install graphviz`.
If that fails, [see here](https://github.com/pygraphviz/pygraphviz/issues/398#issuecomment-1038476921).
