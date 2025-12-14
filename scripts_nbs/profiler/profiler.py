import asyncio

import ezmsg.core as ez
import typer

from ezmsg.tools.dag import crawl_coro
from ezmsg.tools.profile import ProfileLog, ProfileLogSettings


def main(
    graph_ip: str = "127.0.0.1",
    graph_port: int = 25978,
    run_duration: float = 10.0,
    track_most_recent: bool = False,
):
    graph_addr = (graph_ip, graph_port)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    unit_topics = loop.run_until_complete(crawl_coro(graph_addr))
    # TODO: Instead of simply choosing connections that end with OUT*, it would be good to identify each node,
    #  then find all of its output streams and attach to each of those.
    out_topics = [_ for _ in unit_topics if _.split("/")[-1].lower().startswith("out")]
    # TODO: Store the graph in a file.

    run_kwargs = {"graph_address": graph_addr, "components": {}}
    connections = []
    for t_ix, topic in enumerate(out_topics):
        profile_logger = ProfileLog(
            ProfileLogSettings(
                source=topic,
                run_duration=run_duration,
                track_last_sample=track_most_recent,
            )
        )
        run_kwargs["components"]["LOGGER_" + str(t_ix)] = profile_logger
        connections.append((topic, profile_logger.INPUT))
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
