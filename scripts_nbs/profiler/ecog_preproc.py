import ezmsg.core as ez
import numpy as np
import typer
from ezmsg.sigproc.affinetransform import CommonRereference
from ezmsg.sigproc.bandpower import BandPower, SpectrogramSettings
from ezmsg.sigproc.butterworthfilter import ButterworthFilter
from ezmsg.sigproc.downsample import Downsample
from ezmsg.sigproc.scaler import AdaptiveStandardScaler
from ezmsg.sigproc.slicer import Slicer
from ezmsg.sigproc.synth import EEGSynth
from ezmsg.sigproc.wavelets import CWT, MinPhaseMode
from ezmsg.sigproc.window import Anchor
from ezmsg.util.terminate import TerminateOnTotal


def main(
    srate: float = 2000.0,
    n_ch: int = 128,
    graph_addr: str = "127.0.0.1:25978",
    run_duration: float = 600.0,
    use_wavelets: bool = True,
):
    chunks_per_sec = 50
    n_time = int(srate / chunks_per_sec)
    chunks_per_sec = int(srate / n_time)
    total_chunks = int(run_duration * chunks_per_sec)
    if graph_addr is not None:
        graph_addr = graph_addr.split(":")
    if use_wavelets:
        freq_node = CWT(
            frequencies=np.geomspace(10, 200, num=20),
            wavelet="morl",
            min_phase=MinPhaseMode.HOMOMORPHIC,
            axis="time",
        )
    else:
        freq_node = BandPower(
            bands=((18, 30), (70, 170)),
            spectrogram_settings=SpectrogramSettings(
                window_dur=0.5,
                window_shift=1 / chunks_per_sec,
                window_anchor=Anchor.END,
            ),
        )
    comps = {
        "ECOG": EEGSynth(fs=srate, n_time=n_time, n_ch=n_ch),
        "SELECT": Slicer(axis="ch", selection="2:"),
        "LP": ButterworthFilter(axis="time", coef_type="sos", order=4, cutoff=srate / 8),
        "DS": Downsample(axis="time", target_rate=srate / 4),
        "CAR": CommonRereference(axis="ch"),
        "FREQ": freq_node,
        "ZSCORE": AdaptiveStandardScaler(axis="time", time_constant=20.0),
        "TERM": TerminateOnTotal(total=total_chunks),
    }
    conns = {
        (comps["ECOG"].OUTPUT_SIGNAL, comps["SELECT"].INPUT_SIGNAL),
        (comps["SELECT"].OUTPUT_SIGNAL, comps["LP"].INPUT_SIGNAL),
        (comps["LP"].OUTPUT_SIGNAL, comps["DS"].INPUT_SIGNAL),
        (comps["DS"].OUTPUT_SIGNAL, comps["CAR"].INPUT_SIGNAL),
        (comps["CAR"].OUTPUT_SIGNAL, comps["FREQ"].INPUT_SIGNAL),
        (comps["FREQ"].OUTPUT_SIGNAL, comps["ZSCORE"].INPUT_SIGNAL),
        (comps["ZSCORE"].OUTPUT_SIGNAL, comps["TERM"].INPUT_MESSAGE),
    }
    ez.run(components=comps, connections=conns, graph_address=graph_addr)


if __name__ == "__main__":
    typer.run(main)
