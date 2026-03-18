"""Test graph for sigmon development.

Generates synthetic EEG (time-domain) and routes a copy through
Window + Spectrum to produce frequency-domain data. Both branches
terminate at a no-op Sink so the topics exist in the graph for
sigmon to subscribe to.

Usage:
    uv run python scripts/eeg_generator_graph.py
"""

import ezmsg.core as ez
from ezmsg.sigproc.spectrum import Spectrum, SpectrumSettings
from ezmsg.sigproc.window import Window, WindowSettings
from ezmsg.simbiophys import EEGSynth, EEGSynthSettings
from ezmsg.util.messages.axisarray import AxisArray


class Sink(ez.Unit):
    """Consumes messages and does nothing."""

    INPUT_SIGNAL = ez.InputStream(AxisArray)

    @ez.subscriber(INPUT_SIGNAL)
    async def on_message(self, msg: AxisArray) -> None:
        pass


def main() -> None:
    eeg = EEGSynth(
        EEGSynthSettings(
            fs=2000.0,
            n_time=100,
            n_ch=16,
            alpha_freq=10.5,
        )
    )

    win = Window(
        WindowSettings(
            axis="time",
            window_dur=0.5,
            window_shift=0.2,
        )
    )

    spec = Spectrum(
        SpectrumSettings(
            axis="time",
        )
    )

    time_sink = Sink()
    freq_sink = Sink()

    ez.run(
        components={
            "EEG": eeg,
            "WIN": win,
            "SPEC": spec,
            "TIME_SINK": time_sink,
            "FREQ_SINK": freq_sink,
        },
        connections=(
            (eeg.OUTPUT_SIGNAL, time_sink.INPUT_SIGNAL),
            (eeg.OUTPUT_SIGNAL, win.INPUT_SIGNAL),
            (win.OUTPUT_SIGNAL, spec.INPUT_SIGNAL),
            (spec.OUTPUT_SIGNAL, freq_sink.INPUT_SIGNAL),
        ),
    )


if __name__ == "__main__":
    main()
