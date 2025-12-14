import asyncio
import typing
from collections import defaultdict
from typing import TYPE_CHECKING
from uuid import uuid4

import ezmsg.core as ez
import pandas as pd

if TYPE_CHECKING:
    import pygraphviz


def get_graph(graph_address: typing.Tuple[str, int]) -> "pygraphviz.AGraph":
    import pygraphviz as pgv

    # Create a graphviz object with our graph components as nodes and our connections as edges.
    G = pgv.AGraph(name="ezmsg-graphviz", strict=False, directed=True)
    G.graph_attr["label"] = "ezmsg-graphviz"
    G.graph_attr["rankdir"] = "TB"
    # G.graph_attr["outputorder"] = "edgesfirst"
    # G.graph_attr["ratio"] = "1.0"
    # G.node_attr["shape"] = "circle"
    # G.node_attr["fixedsize"] = "true"
    G.node_attr["fontsize"] = "8"
    G.node_attr["fontcolor"] = "#000000"
    G.node_attr["style"] = "filled"
    G.edge_attr["color"] = "#0000FF"
    G.edge_attr["style"] = "setlinewidth(2)"

    # Get the dag from the GraphService
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    dag = loop.run_until_complete(ez.graphserver.GraphService(address=graph_address).dag())

    # Retrieve a description of the graph
    graph_connections = dag.graph.copy()
    # graph_connections is a dict with format
    # {
    #   'apath/unit/port': {'some/other_unit/port', 'yet/another/unit/port'},
    # }
    # where 'port' might be a pub (out) stream or a sub (input) stream.

    b_refresh_dag = False
    for k, v in graph_connections.items():
        if "VISBUFF/INPUT_SIGNAL" in v:
            b_refresh_dag = True
            loop.run_until_complete(
                ez.graphserver.GraphService(address=graph_address).disconnect(k, "VISBUFF/INPUT_SIGNAL")
            )
    if b_refresh_dag:
        dag = loop.run_until_complete(ez.graphserver.GraphService(address=graph_address).dag())
        graph_connections = dag.graph.copy()

    # Generate UUID node names
    node_map = {name: f'"{str(uuid4())}"' for name in set(graph_connections.keys())}

    for node, conns in graph_connections.items():
        for sub in conns:
            G.add_edge(node_map[node], node_map[sub])

    # Make a new dict `graph` with format {component_name: {sub_component: {stream: stream_full_path}}, ...}
    def tree():
        return defaultdict(tree)

    graph: defaultdict = tree()
    for node, conns in graph_connections.items():
        subgraph = graph
        path = node.split("/")
        route = path[:-1]
        stream = path[-1]
        for seg in route:
            subgraph = subgraph[seg]
        subgraph[stream] = node

    # Build out the AGraph recursively
    def build_graph(g: defaultdict, agraph: pgv.AGraph):
        for k, v in g.items():
            if type(v) is defaultdict:
                clust = agraph.add_subgraph(name=f"cluster_{k.lower()}", label=k, cluster=True)
                build_graph(v, clust)
            else:
                agraph.add_node(node_map[v], name=v, label=k)

    build_graph(graph, G)

    return G


def pgv2pd(g: "pygraphviz.AGraph") -> pd.DataFrame:
    df_ps = pd.DataFrame(g.edges(), columns=["pub", "sub"])

    def recurse_upstream(sub):
        pubs = df_ps[df_ps["sub"] == sub]["pub"]
        if len(pubs):
            return recurse_upstream(pubs.iloc[0])
        else:
            return sub

    nodes = []
    for n in g.nodes():
        coords = n.attr["pos"].split(",")
        nodes.append(
            {
                # "id": n.name,
                "name": n.attr["name"],
                "x": float(coords[0]),
                "y": float(coords[1]),
                "upstream": g.get_node(recurse_upstream(n.name)).attr["name"],
            }
        )
    return pd.DataFrame(nodes)


async def crawl_coro(graph_address: tuple):
    graph_service = ez.graphserver.GraphService(address=graph_address)
    dag: ez.dag.DAG = await graph_service.dag()
    graph_connections = dag.graph.copy()

    # Construct the graph
    def tree():
        return defaultdict(tree)

    graph: defaultdict = tree()

    for node, conns in graph_connections.items():
        subgraph = graph
        path = node.split("/")
        route = path[:-1]
        stream = path[-1]
        for seg in route:
            subgraph = subgraph[seg]
        subgraph[stream] = node

    def recurse_get_unit_topics(g: defaultdict) -> list:
        out = []
        sub_graphs = [v for k, v in g.items() if isinstance(v, defaultdict)]
        if len(sub_graphs):
            for sub_graph in sub_graphs:
                out += recurse_get_unit_topics(sub_graph)
        else:
            out.extend(list(g.values()))
        return out

    unit_topics = recurse_get_unit_topics(graph)
    return unit_topics
