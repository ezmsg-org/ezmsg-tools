# ezmsg-vis

A namespace package for [ezmsg](https://github.com/iscoe/ezmsg) to visualize running graphs and data.

## Installation and Run

`pip install "git+ssh://git@bitbucket.org/wysscenter/ezmsg-vis.git[app]"`

> TODO: after a public release, use `pip install ezmsg-vis`

After installing, there will be a `ezmsg-monitor` entry point available when the Python environment is active. Call `ezmsg-monitor --help` to see its commandline arguments.

You must have an ezmsg pipeline running on a graph service exposed on the network. For example, first, run the GraphService on an open port:

`ezmsg --address 127.0.0.1:25978 start`

Then run a pipeline passing the ip and port to `ez.run`:

`ez.run(..., graph_address=("127.0.0.1", 25978))`

And then you can connect to that with `ezmsg-monitor --graph-ip 127.0.0.1 --graph-port 25978`

Don't forget to tear down your GraphService when you are done with it:

`ezmsg --address 127.0.0.1:25978 shutdown`

## Setup (Development)

`cd` to this directory (`ezmsg-vis`) and run `pip install -e ".[app]"`

> Note: You may need a more recent version of ezmsg that what is on pypi. `pip install git+https://github.com/iscoe/ezmsg@dev`

* Modules are available under `import ezmsg.vis`
* Entry points are in `ezmsg.vis.examples`; `ezmsg.vis.examples.monitor:main` is the main one.

## Troubleshooting

* Mac users might fail on installing pygraphviz. The simplest solution is to use conda/mamba and install into the environment `conda install graphviz`. Otherwise, [see here](https://github.com/pygraphviz/pygraphviz/issues/398#issuecomment-1038476921).
