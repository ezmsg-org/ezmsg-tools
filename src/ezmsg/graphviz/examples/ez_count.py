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
from ezmsg.util.debuglog import DebugLog
import numpy as np
import typer


class CountSettings(ez.Settings):
    iterations: int
    approx_srate: float = 10.0


class Count(ez.Unit):
    SETTINGS = CountSettings
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    @ez.publisher(OUTPUT_SIGNAL)
    async def count(self) -> typing.AsyncGenerator:
        count = 0
        template = AxisArray(
            data=np.array([[0.0]]),
            dims=["time", "ch"],
            axes={
                "time": AxisArray.TimeAxis(
                    fs=self.SETTINGS.approx_srate, offset=time.time()
                )
            },
            key="count",
        )
        while count < self.SETTINGS.iterations:
            msg = replace(
                template,
                data=np.array([[count]]),
                axes={
                    **template.axes,
                    "time": replace(template.axes["time"], offset=time.time()),
                },
            )
            yield self.OUTPUT_SIGNAL, msg
            count = count + 1
            await asyncio.sleep(1 / self.SETTINGS.approx_srate)


class SineSettings(ez.Settings):
    freq: float = 1.0


class Sine(ez.Unit):
    SETTINGS = SineSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    @ez.subscriber(INPUT_SIGNAL)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_message(self, message: AxisArray) -> typing.AsyncGenerator:
        tvec = message.axes["time"].value(
            np.arange(message.data.shape[message.get_axis_idx("time")])
        )[:, None]
        yield (
            self.OUTPUT_SIGNAL,
            replace(message, data=np.sin(2 * np.pi * self.SETTINGS.freq * tvec)),
        )


def main(
    runtime: float = 300.0,
    approx_srate: float = 5.0,
    graph_addr: str = "127.0.0.1:25978",
):
    iterations = math.ceil(runtime * approx_srate)
    comps = {
        "COUNT": Count(iterations=iterations, approx_srate=approx_srate),
        "SINE": Sine(),
        "LOG": DebugLog(),
    }
    conns = {
        (comps["COUNT"].OUTPUT_SIGNAL, comps["SINE"].INPUT_SIGNAL),
        (comps["COUNT"].OUTPUT_SIGNAL, comps["LOG"].INPUT),
    }
    graph_addr = graph_addr.split(":") if graph_addr is not None else None
    ez.run(components=comps, connections=conns, graph_address=graph_addr)


if __name__ == "__main__":
    typer.run(main)
