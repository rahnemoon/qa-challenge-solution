import os
import json
from collections import deque
import requests

import dash
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd

BASE_URL = "http://{host}:{port}".format(
    host=os.getenv("REST_API_HOST", "0.0.0.0"), port=os.getenv("REST_API_PORT", "9000")
)
CLASS_ENDPOINT = "{base_url}/api/v1/classification".format(base_url=BASE_URL)
REGRE_ENDPOINT = "{base_url}/api/v1/regression".format(base_url=BASE_URL)

MODELID_CLASS = deque(maxlen=100)
MODELID_REGRE = deque(maxlen=100)

F1BETA = deque(maxlen=100)
ACCURACY = deque(maxlen=100)
ROCAUC = deque(maxlen=100)
PRECISION = deque(maxlen=100)
MCC = deque(maxlen=100)
DATA_CLASS = deque(maxlen=150)


MSE = deque(maxlen=100)
RMSE = deque(maxlen=100)
MEA = deque(maxlen=100)
DATA_REGRE = deque(maxlen=150)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div(
    children=[
        dbc.Col(
            [
                dbc.Nav(
                    [
                        dbc.NavLink("Home", href="/", active="exact"),
                    ],
                    className="navbar navbar-dark bg-dark",
                ),
                html.H1("Training Dashboard", className="text-center"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H2(
                                    "Classification Report", className="text-center"
                                ),
                                dcc.Graph(id="class-total-diagram"),
                            ],
                            width=8,
                        ),
                        dbc.Col(
                            [
                                html.H2("Confusion Matrix", className="text-center"),
                                dcc.Graph(id="class-confusion-matrix"),
                            ],
                            width=4,
                        ),
                    ]
                ),
                html.H2(
                    "Classification Report Per Each Metric", className="text-center"
                ),
                dbc.Row([dcc.Graph(id="class-each-diagram")]),
                html.H2("Input Data & Prediction Result", className="text-center"),
                dbc.Row([dcc.Graph(id="class-table")]),
                dcc.Interval(
                    id="classification-report-update", interval=1000, n_intervals=0
                ),
                dcc.Store(id="classification-report"),
            ]
        ),
        dbc.Col(
            [
                dbc.Row(
                    [
                        html.H2("Regression Report", className="text-center"),
                        dcc.Graph(id="regre-total-diagram"),
                    ]
                ),
                html.H2("Regression Report Per Each Metric", className="text-center"),
                dbc.Row([dcc.Graph(id="regre-each-diagram")]),
                html.H2("Input Data & Prediction Result", className="text-center"),
                dbc.Row([dcc.Graph(id="regre-table")]),
                dcc.Interval(
                    id="regression-report-update", interval=1000, n_intervals=0
                ),
                dcc.Store(id="regression-report"),
            ]
        ),
    ]
)


@app.callback(
    Output("classification-report", "data"),
    Input("classification-report-update", "n_intervals"),
)
def update_classification_report(n):
    resp = requests.get(CLASS_ENDPOINT)
    data = json.loads(resp.content)

    MODELID_CLASS.append(data.get("model_id"))
    F1BETA.append(data.get("metric_report").get("f1beta"))
    ACCURACY.append(data.get("metric_report").get("accuracy"))
    ROCAUC.append(data.get("metric_report").get("rocauc"))
    PRECISION.append(data.get("metric_report").get("precision"))
    MCC.append(data.get("metric_report").get("mcc"))
    for item in data.get("data_batch"):
        DATA_CLASS.append(item)

    return data.get("metric_report").get("confusion_matrix")


@app.callback(
    Output("class-total-diagram", "figure"),
    [
        Input("classification-report-update", "n_intervals"),
        Input("classification-report", "data"),
    ],
)
def update_class_total_plot(n, classification_report):
    fig_f1beta = go.Scatter(
        x=list(MODELID_CLASS), y=list(F1BETA), mode="lines+markers", name="F1Beta"
    )
    fig_accuracy = go.Scatter(
        x=list(MODELID_CLASS), y=list(ACCURACY), mode="lines+markers", name="Accuracy"
    )
    fig_rocauc = go.Scatter(
        x=list(MODELID_CLASS), y=list(ROCAUC), mode="lines+markers", name="ROCAUC"
    )
    fig_precision = go.Scatter(
        x=list(MODELID_CLASS), y=list(PRECISION), mode="lines+markers", name="Precision"
    )
    fig_mcc = go.Scatter(
        x=list(MODELID_CLASS), y=list(MCC), mode="lines+markers", name="MCC"
    )
    fig_total = go.Figure()
    fig_total.add_trace(fig_f1beta)
    fig_total.add_trace(fig_accuracy)
    fig_total.add_trace(fig_rocauc)
    fig_total.add_trace(fig_precision)
    fig_total.add_trace(fig_mcc)

    return fig_total


@app.callback(
    Output("class-confusion-matrix", "figure"),
    [
        Input("classification-report-update", "n_intervals"),
        Input("classification-report", "data"),
    ],
)
def update_class_conf_plot(n, classification_report):

    x = ["True Positive", "True Negative"]
    y = ["Predicted Positive", "predicted Negative"]

    z = [
        [
            classification_report.get("true_positives"),
            classification_report.get("false_positives"),
        ],
        [
            classification_report.get("false_negatives"),
            classification_report.get("true_negatives"),
        ],
    ]

    z_text = [["True Positive", "False Positive"], ["False Negative", "True Negative"]]

    fig = px.imshow(z, x=x, y=y, color_continuous_scale="Viridis", aspect="auto")
    fig.update_traces(text=z_text, texttemplate="%{text}")
    fig.update_xaxes(side="top")
    return fig


@app.callback(
    Output("class-each-diagram", "figure"),
    [
        Input("classification-report-update", "n_intervals"),
        Input("classification-report", "data"),
    ],
)
def update_class_each_plot(n, classification_report):
    fig = make_subplots(
        rows=5,
        cols=1,
        specs=[
            [{"type": "Scatter"}],
            [{"type": "Scatter"}],
            [{"type": "Scatter"}],
            [{"type": "Scatter"}],
            [{"type": "Scatter"}],
        ],
    )
    fig_f1beta = go.Scatter(
        x=list(MODELID_CLASS), y=list(F1BETA), mode="lines+markers", name="F1Beta"
    )
    fig_accuracy = go.Scatter(
        x=list(MODELID_CLASS), y=list(ACCURACY), mode="lines+markers", name="Accuracy"
    )
    fig_rocauc = go.Scatter(
        x=list(MODELID_CLASS), y=list(ROCAUC), mode="lines+markers", name="ROCAUC"
    )
    fig_precision = go.Scatter(
        x=list(MODELID_CLASS), y=list(PRECISION), mode="lines+markers", name="Precision"
    )
    fig_mcc = go.Scatter(
        x=list(MODELID_CLASS), y=list(MCC), mode="lines+markers", name="MCC"
    )
    fig.add_trace(fig_f1beta, row=1, col=1)
    fig.add_trace(fig_accuracy, row=2, col=1)
    fig.add_trace(fig_rocauc, row=3, col=1)
    fig.add_trace(fig_precision, row=4, col=1)
    fig.add_trace(fig_mcc, row=5, col=1)
    fig.update_layout(autosize=True, height=1000)
    return fig


@app.callback(
    Output("class-table", "figure"),
    [
        Input("classification-report-update", "n_intervals"),
        Input("classification-report", "data"),
    ],
)
def update_class_table(n, classification_report):
    df = pd.DataFrame(list(DATA_CLASS))
    df = df.reindex(
        columns=["id", "competence", "network_ability", "promoted", "pred_promoted"]
    )
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df.columns), fill_color="paleturquoise", align="left"
                ),
                cells=dict(
                    values=df.transpose().values.tolist(),
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )

    return fig


@app.callback(
    Output("regression-report", "data"),
    Input("regression-report-update", "n_intervals"),
)
def update_regression_report(n):
    resp = requests.get(REGRE_ENDPOINT)
    data = json.loads(resp.content)

    MODELID_REGRE.append(data.get("model_id"))
    MSE.append(data.get("metric_report").get("MSE"))
    RMSE.append(data.get("metric_report").get("RMSE"))
    MEA.append(data.get("metric_report").get("MEA"))
    for item in data.get("data_batch"):
        DATA_REGRE.append(item)

    return data


@app.callback(
    Output("regre-total-diagram", "figure"),
    Input("regression-report-update", "n_intervals"),
)
def update_regre_total_plot(n):
    fig_mse = go.Scatter(
        x=list(MODELID_REGRE), y=list(MSE), mode="lines+markers", name="MSE"
    )
    fig_rmse = go.Scatter(
        x=list(MODELID_REGRE), y=list(RMSE), mode="lines+markers", name="RMSE"
    )
    fig_mea = go.Scatter(
        x=list(MODELID_REGRE), y=list(MEA), mode="lines+markers", name="MEA"
    )
    fig_total = go.Figure()
    fig_total.add_trace(fig_mse)
    fig_total.add_trace(fig_rmse)
    fig_total.add_trace(fig_mea)

    return fig_total


@app.callback(
    Output("regre-each-diagram", "figure"),
    Input("regression-report-update", "n_intervals"),
)
def update_regre_each_plot(n):
    fig = make_subplots(
        rows=3,
        cols=1,
        specs=[
            [{"type": "Scatter"}],
            [{"type": "Scatter"}],
            [{"type": "Scatter"}],
        ],
    )
    fig_mse = go.Scatter(
        x=list(MODELID_REGRE), y=list(MSE), mode="lines+markers", name="MSE"
    )
    fig_rmse = go.Scatter(
        x=list(MODELID_REGRE), y=list(RMSE), mode="lines+markers", name="RMSE"
    )
    fig_mea = go.Scatter(
        x=list(MODELID_REGRE), y=list(MEA), mode="lines+markers", name="MEA"
    )
    fig.add_trace(fig_mse, row=1, col=1)
    fig.add_trace(fig_rmse, row=2, col=1)
    fig.add_trace(fig_mea, row=3, col=1)
    fig.update_layout(autosize=True, height=1000)
    return fig


@app.callback(
    Output("regre-table", "figure"), Input("regression-report-update", "n_intervals")
)
def update_regre_table(n):
    df = pd.DataFrame(list(DATA_REGRE))
    df = df.reindex(
        columns=[
            "id",
            "competence",
            "network_ability",
            "pred_network_ability",
            "promoted",
        ]
    )
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df.columns), fill_color="paleturquoise", align="left"
                ),
                cells=dict(
                    values=df.transpose().values.tolist(),
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
