import asyncio
from collections import defaultdict
from dataclasses import field
import time
import logging

import typer
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray


logger = logging.getLogger("EZProfiler")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("ezprofiler.log")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(created)f,%(message)s"))
logger.addHandler(fh)


async def coro(graph_address: tuple):
    graph_service = ez.GraphService(address=graph_address)
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
    return [
        topic for topic in unit_topics if topic.split("/")[-1].lower().startswith("out")
    ]


class ProfileLogSettings(ez.Settings):
    source: str
    run_duration: float = 60.0
    track_last_sample: bool = True


class ProfileLogState(ez.State):
    t_start: float = field(default_factory=time.time)


class ProfileLog(ez.Unit):
    SETTINGS = ProfileLogSettings
    STATE = ProfileLogState

    INPUT = ez.InputStream(AxisArray)

    def initialize(self) -> None:
        self.STATE.t_start = time.time()

    @ez.subscriber(INPUT, zero_copy=True)
    async def log(self, msg: AxisArray):
        if hasattr(msg, "get_axis"):
            targ_axis = None
            if "step" in msg.axes:
                targ_axis = "step"
            elif "time" in msg.axes:
                targ_axis = "time"
            if targ_axis is not None:
                ax = msg.get_axis(targ_axis)
                samp_time = ax.offset  # First sample in msg
                if self.SETTINGS.track_last_sample:
                    # Last sample in msg.
                    samp_time += ax.gain * (
                        msg.data.shape[msg.get_axis_idx(targ_axis)] - 1
                    )
                logger.info(f"{self.SETTINGS.source},{samp_time}")

        if (time.time() - self.STATE.t_start) > self.SETTINGS.run_duration:
            raise ez.Complete


def main(
    graph_ip: str = "127.0.0.1",
    graph_port: int = 25978,
    run_duration: float = 30.0,
    track_most_recent: bool = False,
):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    out_topics = loop.run_until_complete(coro((graph_ip, graph_port)))

    loggers = []
    run_kwargs = {}
    connections = []
    for t_ix, topic in enumerate(out_topics):
        loggers.append(
            ProfileLog(
                ProfileLogSettings(
                    source=topic,
                    run_duration=run_duration,
                    track_last_sample=track_most_recent,
                )
            )
        )
        run_kwargs["LOGGER_" + str(t_ix)] = loggers[t_ix]
        connections.append((topic, loggers[t_ix].INPUT))
    run_kwargs["connections"] = tuple(connections)
    ez.run(**run_kwargs)


if __name__ == "__main__":
    typer.run(main)


"""
TODO:
In Spectrogram, the Window unit creates a new `step` axis that has offset=time_of_first_samp_in_window.
Then, the Spectrum Unit destroys the original time axis so we lost knowledge that the Spectrogram output
actually represents data from offset to offset + window length.

The Decoder feeds a lagged history of the feature data to the decoder. The output's time axis' offset will be
the time of the oldest feature, which is the oldest spectrum. So a 500 msec spectrum means the feature-sample
is already 500 msec old, and a 200 msec lagged feature window yields an offset that is an additional 200 msec old.
Is that right?
"""
