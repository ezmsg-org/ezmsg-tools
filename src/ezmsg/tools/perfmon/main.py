"""
This is a plotly.dash application that monitors and visualizes the performance of an ezmsg system.

Upon page load or changing the logger path, the application reads the CSV file at the given path
and displays the data in a table.
Additionally, every second, the application updates the table with the latest data from the CSV file.

Whenever the table is updated, the application also updates a histogram graph that shows the average
elapsed time for each topic.

Only the last 1 minute of data is used in the table and graphs.
"""
from pathlib import Path

import dash
from dash_extensions import Mermaid, enrich
import pandas as pd
import plotly.express as px
# import pygtail
from ezmsg.sigproc.util.profile import get_logger_path


app = dash.Dash("ezmsg Performance Monitor", update_title=None)

app.layout = [
    # dash.dcc.Interval(id="interval", interval=10_000, n_intervals=0),
    dash.dcc.Store(id="df-store"),
    dash.dcc.Store(id="last-dt-store"),
    dash.html.Div(id="onload"),  # this div is used to trigger any functions that need to run on page load
    dash.dcc.Input(id="logger-path", type="text", placeholder="logpath", debounce=True),
    Mermaid(id="dag", config={"theme": "neutral"}),
    dash.dcc.Graph(id="hist-graph"),
    dash.dash_table.DataTable(id="table", data=[], page_size=20, page_current=1),
]


@dash.callback(
    dash.Output("logger-path", "value"),
    enrich.Trigger("onload", "children"),
    prevent_initial_call=False
)
def on_load(_):
    return str(get_logger_path())


@dash.callback(
    dash.Input("logger-path", "value"),
    prevent_initial_call=True,
)
def on_logger_path(logger_path: str):
    logger_path = Path(logger_path)
    if logger_path.exists():
        offset_path = logger_path.parent / (logger_path.name + ".offset")
        offset_path.unlink(missing_ok=True)


@dash.callback(
dash.Output("df-store", "data"),
    dash.Output("last-dt-store", "data"),
    dash.Input("logger-path", "value"),
    prevent_initial_call=True,
)
def load_once(logger_path: str) -> tuple[dict, str]:
    if logger_path is None:
        return {}
    df = pd.read_csv(logger_path, header=0)
    return df.to_dict("records"), df["Time"].iloc[-1]


# @dash.callback(
#     dash.Output("df-store", "data"),
#     dash.Output("last-dt-store", "data"),
#     [
#         dash.Input("interval", "n_intervals"),
#         dash.State("logger-path", "value"),
#         dash.State("df-store", "data"),
#         dash.State("last-dt-store", "data"),
#     ],
#     prevent_initial_call=True,
# )
# def interval_callback(_, logger_path, last_df, last_dt):
#     tail = pygtail.Pygtail(logger_path)
#     try:
#         lines = tail.read()
#     except FileNotFoundError:
#         return last_df, last_dt
#
#     if lines is None:
#         return last_df, last_dt
#
#     df = pd.read_csv(logger_path, header=0)
#
#     # TODO: If the most recent entries in the log file are newer than the last update, replace the data.
#     #  Otherwise, use the previous table.
#     return df.to_dict("records"), df["Time"].iloc[-1]


@dash.callback(
    dash.Output("dag", "chart"),
    [
        dash.Input("last-dt-store", "data"),
        dash.State("df-store", "data"),
        dash.State("logger-path", "value"),
    ],
    prevent_initial_call=True,
    memoize=True,
)
def update_dag(last_dt, data, logger_path):
    logger_path = Path(logger_path)
    graph_path = logger_path.parent / (logger_path.stem + ".mermaid")
    if not graph_path.exists():
        return ""
    with graph_path.open() as f:
        graph_str = f.read()
    df = pd.DataFrame.from_dict(data)
    topic_means = df.groupby("Topic")["Elapsed"].mean()
    max_elapsed = topic_means.max()
    for topic, mean in topic_means.items():
        topic_str = topic.split("/")[-1].lower()
        # https://mermaid.js.org/syntax/flowchart.html#styling-a-node
        color = px.colors.find_intermediate_color((0, 0.0, 1.0), (1.0, 0.0, 0.0), mean / max_elapsed)
        fill_str = "".join([f"{int(c*255):02x}" for c in color])
        # style id2 fill:#bbf,stroke:#f66,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
        graph_str += f"  style {topic_str} fill:#{fill_str}80\n"
    return graph_str


@dash.callback(
    dash.Output("table", "data"),
    dash.Output("table", "page_current"),
    dash.Input("df-store", "data"),
    prevent_initial_call=True,
    memoize=True,
)
def update_table(data):
    df = pd.DataFrame.from_dict(data)
    return df.to_dict("records"), len(df)//20


@dash.callback(
    dash.Output("hist-graph", "figure"),
    dash.Input("df-store", "data"),
    prevent_initial_call=True,
    memoize=True,
)
def update_hist(data):
    df = pd.DataFrame.from_dict(data)
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
    return fig


if __name__ == '__main__':
    app.run(debug=True)
