"""
This is a modified version of ezmsg's examples/ezmsg_intro.py
Refer to that script for an explanation.
"""

import asyncio
from dataclasses import replace
import math
import time
import typing

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
import numpy as np
import typer


class CountSettings(ez.Settings):
    iterations: int
    approx_srate: float = 10.0


class Count(ez.Unit):
    SETTINGS: CountSettings
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    @ez.publisher(OUTPUT_SIGNAL)
    async def count(self) -> typing.AsyncGenerator:
        count = 0
        while count < self.SETTINGS.iterations:
            msg = AxisArray(
                data=np.array([[count]]),
                dims=["time", "ch"],
                axes={
                    "time": AxisArray.Axis.TimeAxis(
                        fs=self.SETTINGS.approx_srate, offset=time.time()
                    )
                },
            )
            yield self.OUTPUT_SIGNAL, msg
            count = count + 1
            await asyncio.sleep(1 / self.SETTINGS.approx_srate)


class Sine(ez.Unit):
    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    @ez.subscriber(INPUT_SIGNAL)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_message(self, message: AxisArray) -> typing.AsyncGenerator:
        yield (
            self.OUTPUT_SIGNAL,
            replace(message, data=np.sin(2 * np.pi * 1.0 * message.data)),
        )


class PrintSettings(ez.Settings):
    iterations: int


class PrintState(ez.State):
    current_iteration: int = 0


class PrintValue(ez.Unit):
    SETTINGS: PrintSettings
    STATE: PrintState
    INPUT_SIGNAL = ez.InputStream(AxisArray)

    @ez.subscriber(INPUT_SIGNAL)
    async def on_message(self, message: AxisArray) -> None:
        print(f"Current Count: {message.data[0, 0]}")

        self.STATE.current_iteration = self.STATE.current_iteration + 1
        if self.STATE.current_iteration == self.SETTINGS.iterations:
            raise ez.NormalTermination


def main(
    runtime: float = 300.0,
    approx_srate: float = 5.0,
    graph_addr: str = "127.0.0.1:25978",
):
    iterations = math.ceil(runtime * approx_srate)
    comps = {
        "COUNT": Count(iterations=iterations, approx_srate=approx_srate),
        "SINE": Sine(),
        "PRINT": PrintValue(iterations=iterations),
    }
    conns = {
        (comps["COUNT"].OUTPUT_SIGNAL, comps["SINE"].INPUT_SIGNAL),
        (comps["SINE"].OUTPUT_SIGNAL, comps["PRINT"].INPUT_SIGNAL),
    }
    graph_addr = graph_addr.split(":") if graph_addr is not None else None
    ez.run(components=comps, connections=conns, graph_address=graph_addr)


if __name__ == "__main__":
    typer.run(main)
