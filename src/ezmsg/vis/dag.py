import asyncio
from collections import defaultdict
from pathlib import Path
import tempfile
import typing
from uuid import uuid4

import ezmsg.core as ez
import pandas as pd
import pygame


SCROLL_STEP = 50


def get_graph(graph_address: typing.Tuple[str, int]) -> "pygraphviz.AGraph":
    import pygraphviz as pgv

    # Create a graphviz object with our graph components as nodes and our connections as edges.
    G = pgv.AGraph(name="ezmsg-vis", strict=False, directed=True)
    G.graph_attr["label"] = "ezmsg-vis"
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
    dag = loop.run_until_complete(ez.GraphService(address=graph_address).dag())

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
                ez.GraphService(address=graph_address).disconnect(
                    k, "VISBUFF/INPUT_SIGNAL"
                )
            )
    if b_refresh_dag:
        dag = loop.run_until_complete(ez.GraphService(address=graph_address).dag())
        graph_connections = dag.graph.copy()

    # Let's come up with UUID node names
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
                clust = agraph.add_subgraph(
                    name=f"cluster_{k.lower()}", label=k, cluster=True
                )
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


class VisDAG:
    def __init__(
        self,
        tl_offset: typing.Tuple[int, int] = (0, 0),
        screen_height: int = 1440,
        graph_ip: str = "127.0.0.1",
        graph_port: int = 25978,
    ):
        self._screen_height = screen_height
        G = get_graph((graph_ip, graph_port))
        G.layout(prog="dot")
        # Create SVG to get the correct coordinates
        svg_path = Path(tempfile.gettempdir()) / "ezmsg-vis.svg"
        G.draw(svg_path, format="svg:cairo")
        # Get the graph details as dataframe
        self._node_df = pgv2pd(G)
        # Unfortunately, pygame cannot render svg very well, so we render as png for display
        img_path = Path(tempfile.gettempdir()) / "ezmsg-vis.png"
        G.draw(img_path)
        self._image = pygame.image.load(img_path)
        self._image_rect = self._image.get_rect(topleft=tl_offset)
        self._min_y = -(self._image_rect.height - screen_height)
        # Scale the svg coordinates by png size / svg size
        _svg = pygame.image.load(svg_path)
        self._node_df["y"] *= self._image_rect.height / _svg.get_rect().height
        self._node_df["x"] *= self._image_rect.width / _svg.get_rect().width

        self._image_y = 0  # Initial position of the image
        self._b_update = True

    @property
    def size(self) -> typing.Tuple[int, int]:
        return self._image_rect.size

    def handle_event(self, event: pygame.event.Event) -> typing.Optional[str]:
        new_node_path = None
        if event.type in [pygame.MOUSEWHEEL, pygame.MOUSEBUTTONDOWN]:
            mouse_pos = pygame.mouse.get_pos()
            if self._image_rect.left <= mouse_pos[0] <= self._image_rect.right:
                if event.type == pygame.MOUSEWHEEL:
                    if event.y > 0:
                        # scroll graph up
                        self._image_y = min(0, self._image_y + SCROLL_STEP)
                    elif event.y < 0:
                        # scroll graph down
                        self._image_y = max(self._min_y, self._image_y - SCROLL_STEP)
                    self._b_update = True

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Mouse events
                    if event.button == 1:
                        # Clicked on a node
                        graph_pos = (
                            mouse_pos[0],
                            self._image_rect.height - (mouse_pos[1] - self._image_y),
                        )
                        min_row = (
                            (self._node_df.x - graph_pos[0]) ** 2
                            + (self._node_df.y - graph_pos[1]) ** 2
                        ).argmin()
                        new_node_path = f"{self._node_df.iloc[min_row]['upstream']}"
        return new_node_path

    def update(self, surface: pygame.Surface) -> typing.List[pygame.Rect]:
        res = []
        if self._b_update:
            surface.blit(self._image, (0, self._image_y))
            pygame.display.update(self._image_rect)
            res.append(self._image_rect)
            self._b_update = False
        return res
