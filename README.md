# ezmsg-vis

A namespace package for [ezmsg](https://github.com/iscoe/ezmsg) to visualize running graphs and data.

## Installation

> Maybe one day if this goes public: `pip install ezmsg-vis`

## Setup (Development)

1. Install `ezmsg` either using `pip install ezmsg` or set up the repo for development as described in the `ezmsg` readme.
2. `cd` to this directory (`ezmsg-vis`) and run `pip install -e ".[app]"`
    * Mac users might fail on installing pygraphviz. The simplest solution is to use conda/mamba and install into the environment `conda install graphviz`. Otherwise, [see here](https://github.com/pygraphviz/pygraphviz/issues/398#issuecomment-1038476921).
3. Modules are available under `import ezmsg.vis`
