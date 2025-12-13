import sys

import ezmsg.core as ez
from ezmsg.sigproc.synth import Counter, CounterSettings
from ezmsg.util.terminate import TerminateOnTotal, TerminateOnTotalSettings

sys.path.append("..")
from nodes.dummy import Dummy, DummySettings


def main(
    do_multi: bool = False,
    sleep_time: float = 0.005,
    fs: float = 10.0,
    run_duration: float = 35.0,
):
    n_msgs = int(run_duration * fs)
    comps = {
        "SOURCE": Counter(CounterSettings(n_time=1, fs=fs, dispatch_rate="realtime")),
        "DUMMY1": Dummy(DummySettings(mean=sleep_time, stddev=0.0)),
        "DUMMY2": Dummy(DummySettings(mean=sleep_time, stddev=0.0)),
        "SINK": TerminateOnTotal(TerminateOnTotalSettings(total=n_msgs * 2)),
    }
    conns = (
        (comps["SOURCE"].OUTPUT_SIGNAL, comps["DUMMY1"].INPUT_SIGNAL),
        (comps["SOURCE"].OUTPUT_SIGNAL, comps["DUMMY2"].INPUT_SIGNAL),
        (comps["DUMMY1"].OUTPUT_SIGNAL, comps["SINK"].INPUT_MESSAGE),
        (comps["DUMMY2"].OUTPUT_SIGNAL, comps["SINK"].INPUT_MESSAGE),
    )
    ez.run(
        components=comps,
        connections=conns,
        # graph_address=("127.0.0.1", 25978),
        process_components=(comps["DUMMY2"],) if do_multi else (),
    )


if __name__ == "__main__":
    try:
        import typer

        typer.run(main)
    except ModuleNotFoundError:
        main()
