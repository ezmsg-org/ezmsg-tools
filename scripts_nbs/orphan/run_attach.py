import asyncio
import logging

import ezmsg.core as ez
import ezmsg.core.graphserver
import ezmsg.core.messagecache
import ezmsg.core.shmserver
import typer
from ezmsg.core.backendprocess import DefaultBackendProcess, new_threaded_event_loop
from ezmsg.util.debuglog import DebugLog

logger = logging.getLogger("attach_ezmsg")


GRAPH_IP = "127.0.0.1"
GRAPH_PORT = 25978
PLOT_DUR = 2.0


def main(
    graph_addr: str = ":".join((GRAPH_IP, str(GRAPH_PORT))),
    node_addr: str = "COUNT/OUTPUT_SIGNAL",
):
    # Below is just a minimal version of ez.run
    # TODO: Create a class that handles all this
    #  * run in a thread
    #  * add method to remove connection without restarting
    #  * add method to add connection without restarting
    # TODO: Different units
    #  *

    components = {"LOG": DebugLog()}
    connections = {(node_addr, components["LOG"].INPUT)}
    graph_addr = graph_addr.split(":")
    graph_addr = (graph_addr[0], int(graph_addr[1]))

    graph_service = ezmsg.core.graphserver.GraphService(graph_addr)
    shm_service = ezmsg.core.shmserver.SHMService()

    with new_threaded_event_loop(ev=None) as loop:
        execution_context = ez.backend.ExecutionContext.setup(
            components,
            graph_service,
            shm_service,
            None,
            connections,
            None,
            DefaultBackendProcess,
            False,
        )

        async def create_graph_context() -> ez.GraphContext:
            return await ez.GraphContext(graph_service, shm_service).__aenter__()

        graph_context = asyncio.run_coroutine_threadsafe(create_graph_context(), loop).result()

        async def cleanup_graph() -> None:
            await graph_context.__aexit__(None, None, None)

        async def setup_graph() -> None:
            for edge in execution_context.connections:
                await graph_context.connect(*edge)

        asyncio.run_coroutine_threadsafe(setup_graph(), loop).result()
        main_process = execution_context.processes[0]
        try:
            main_process.process(loop)
            # This does quite a bit...
            # * await unit.setup()
            # * creates subscribers to input streams and publishers to output streams
            # * (optional) threads for @thread-decorated methods
            # * runs the publisher methods
            # * run coroutines for wrapped tasks (including all unit methods)
        except KeyboardInterrupt:
            execution_context.term_ev.set()
        finally:
            asyncio.run_coroutine_threadsafe(cleanup_graph(), loop).result()


if __name__ == "__main__":
    typer.run(main)
