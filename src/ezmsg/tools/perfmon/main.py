"""
This is a plotly.dash application that monitors and visualizes the performance of an ezmsg system.

Upon page load or changing the logger path, the application reads the CSV file at the given path
and displays the data in a table.
Additionally, every second, the application updates the table with the latest data from the CSV file.

Whenever the table is updated, the application also updates a histogram graph that shows the average
elapsed time for each topic.

Only the last 1 minute of data is used in the table and graphs.
"""

import asyncio
import datetime
import io
import typing
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import pygtail
from dash_extensions import Mermaid, enrich
from ezmsg.baseproc.util.profile import get_logger_path
from ezmsg.core.graphserver import GraphService

PAGE_SIZE = 20

app = dash.Dash("ezmsg Performance Monitor", external_stylesheets=[dbc.themes.CYBORG])

state = dbc.Col(
    [
        dash.dcc.Interval(id="interval", interval=10_000, n_intervals=0),
        dash.dcc.Store(id="df-store"),
        dash.dcc.Store(id="last-dt-store"),
        dash.html.Div(id="onload"),  # this div is used to trigger any functions that need to run on page load
    ]
)

header_ui = dbc.Col(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Input(
                        id="logger-path",
                        type="text",
                        placeholder="logpath",
                        debounce=True,
                        valid=False,
                    )
                ),
                dbc.Col(dbc.Switch(id="live-toggle", label="Live", value=False)),
                # dbc.Col("History (s):", width=1),
                # dbc.Col(dbc.Input(id="history-input", type="number", debounce=True, min=1, value=10)),
            ]
        ),
    ]
)

dag = dbc.Col(
    [
        Mermaid(id="dag", config={"theme": "neutral"}),
    ],
    style={"width": "100%", "backgroundColor": "rgb(200, 200, 200)"},
)

table_summary = dbc.Col(
    [
        dbc.Row(dash.dcc.Graph(id="hist-graph", style={"width": "100%"})),
        dbc.Row(dash.html.Div("Sum:", id="proc-sum", style={"width": "100%"})),
        dbc.Row(
            dash.dash_table.DataTable(
                id="table",
                data=[],
                page_current=-1,
                page_size=PAGE_SIZE,
                page_action="custom",
                style_header={"backgroundColor": "rgb(30, 30, 30)", "color": "white"},
                style_data={"backgroundColor": "rgb(50, 50, 50)", "color": "white"},
            )
        ),
    ]
)

app.layout = dash.html.Div(
    children=[state, header_ui, dag, table_summary],
    id="container",
    className="dash-bootstrap",
)


@dash.callback(
    dash.Output("logger-path", "value"),
    enrich.Trigger("onload", "children"),
    prevent_initial_call=False,
)
def on_load(_):
    return str(get_logger_path())


@dash.callback(
    dash.Output("logger-path", "valid"),
    dash.Input("logger-path", "value"),
    prevent_initial_call=True,
)
def on_logger_path(logger_path: str) -> bool:
    valid = False
    logger_path = Path(logger_path)
    if logger_path.exists():
        offset_path = logger_path.parent / (logger_path.name + ".offset")
        offset_path.unlink(missing_ok=True)
        valid = logger_path.stat().st_size > 0
    return valid


def _trim_df(df: pd.DataFrame, history_sec: float = 10.0) -> pd.DataFrame:
    last_dt = df["Time"].iloc[-1]
    hist_lim = last_dt - datetime.timedelta(seconds=history_sec)
    return df[df["Time"] >= hist_lim]


@dash.callback(
    dash.Output("df-store", "data"),
    dash.Output("last-dt-store", "data"),
    dash.Input("logger-path", "value"),
    # dash.Input("history-input", "value"),
    prevent_initial_call=True,
)
def load_once(
    logger_path: str,
    # history_sec
) -> tuple[list[dict[str, typing.Any]], datetime.datetime]:
    if logger_path is None or not Path(logger_path).exists():
        raise dash.exceptions.PreventUpdate
    try:
        df = pd.read_csv(logger_path, header=0, parse_dates=["Time"])
    except pd.errors.EmptyDataError:
        raise dash.exceptions.PreventUpdate
    # Rewrite logger-path.offset with the current offset.
    tail = pygtail.Pygtail(logger_path)
    tail.read_from_end = True
    tail.update_offset_file()
    # Trim any rows with headers
    b_bad = df["Time"].astype(str) == "Time"
    if b_bad.any():
        df = df[~b_bad]
        # Reinterpret the columns:
        # Time (datetime64), Source (obj), Topic (obj), SampleTime (float64), PerfCounter (float64), Elapsed (float64)
        df["Time"] = pd.to_datetime(df["Time"])
        for col in ["SampleTime", "PerfCounter", "Elapsed"]:
            df[col] = pd.to_numeric(df[col])
    # Trim dataframe to only include the last history_sec of data.
    df = _trim_df(df, history_sec=10.0)  # TODO: Get history_sec from widget
    last_dt = df["Time"].iloc[-1]
    data = df.to_dict("records")
    return data, last_dt


@dash.callback(
    dash.Output("df-store", "data", allow_duplicate=True),
    dash.Output("last-dt-store", "data", allow_duplicate=True),
    [
        dash.Input("interval", "n_intervals"),
        dash.Input("live-toggle", "value"),
        dash.State("logger-path", "value"),
        dash.State("df-store", "data"),
        dash.State("last-dt-store", "data"),
    ],
    prevent_initial_call=True,
)
def interval_callback(_, toggle_state, logger_path, data, last_dt):
    if not toggle_state:
        raise dash.exceptions.PreventUpdate

    tail = pygtail.Pygtail(logger_path)
    new_lines = tail.read()

    if not new_lines:
        raise dash.exceptions.PreventUpdate

    if data is not None:
        df = pd.DataFrame.from_dict(data)
        df["Time"] = pd.to_datetime(df["Time"])
        new_df = pd.read_csv(io.StringIO(new_lines), names=df.columns, parse_dates=["Time"])
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = pd.read_csv(io.StringIO(new_lines), header=0, parse_dates=["Time"])
    df = _trim_df(df, history_sec=10.0)  # TODO: Get history_sec from widget
    last_dt = df["Time"].iloc[-1]
    return df.to_dict("records"), last_dt


@dash.callback(
    dash.Output("dag", "chart"),
    [
        dash.Input("df-store", "data"),
        dash.State("logger-path", "value"),
    ],
    prevent_initial_call=True,
    memoize=True,
)
def update_dag(data, logger_path):
    async def _get_formatted_graph():
        graph_service = GraphService(("127.0.0.1", 25978))
        graph_out = await graph_service.get_formatted_graph(fmt="mermaid", direction="LR")
        return graph_out

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    graph_str = loop.run_until_complete(_get_formatted_graph())
    if not graph_str:
        logger_path = Path(logger_path)
        graph_path = logger_path.parent / (logger_path.stem + ".mermaid")
        if not graph_path.exists():
            return ""
        with graph_path.open() as f:
            graph_str = f.read()

    df = pd.DataFrame.from_dict(data)
    df["Time"] = pd.to_datetime(df["Time"])

    topic_means = df.groupby("Topic")["Elapsed"].mean()
    max_elapsed = topic_means.max()
    for topic, mean in topic_means.items():
        topic_str = topic.split("/")[-1].lower()
        # https://mermaid.js.org/syntax/flowchart.html#styling-a-node
        color = px.colors.find_intermediate_color((0, 0.0, 1.0), (1.0, 0.0, 0.0), mean / max_elapsed)
        fill_str = "".join([f"{int(c * 255):02x}" for c in color])
        # style id2 fill:#bbf,stroke:#f66,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
        graph_str += f"  style {topic_str} fill:#{fill_str}80\n"
    return graph_str


@dash.callback(
    dash.Output("table", "data"),
    dash.Output("table", "page_current"),
    dash.Input("df-store", "data"),
    dash.Input("table", "page_current"),
    dash.Input("table", "page_size"),
    prevent_initial_call=True,
    memoize=True,
)
def update_table(data, page_current, page_size):
    df = pd.DataFrame.from_dict(data)
    df["Time"] = pd.to_datetime(df["Time"])
    if page_current < 0:
        page_current = int(len(df) // PAGE_SIZE) - 1
    out_data = df.iloc[page_current * page_size : (page_current + 1) * page_size].to_dict("records")
    return out_data, page_current


@dash.callback(
    dash.Output("hist-graph", "figure"),
    dash.Output("proc-sum", "children"),
    dash.Input("df-store", "data"),
    prevent_initial_call=True,
    memoize=True,
)
def update_hist(data):
    df = pd.DataFrame.from_dict(data)
    df["Time"] = pd.to_datetime(df["Time"])
    topic_means = df.groupby("Topic")[["PerfCounter", "Elapsed"]].mean()
    fig = px.bar(
        topic_means,
        y="Elapsed",
        hover_data=["Elapsed"],
        color="Elapsed",
        labels={"Elapsed": "Processing time per chunk (ms)"},
        height=400,
        color_continuous_scale="Bluered",
    )
    # px.histogram(df, x="Topic", y="Elapsed", histfunc="avg")
    fig.update_layout(height=400, showlegend=False, template="plotly_dark")
    fig.layout.coloraxis.colorbar.title = None
    proc_sum = topic_means["Elapsed"].sum()
    return fig, f"Sum: {proc_sum:.2f} ms"


if __name__ == "__main__":
    app.run(debug=True)
