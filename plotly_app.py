from pathlib import Path
import pickle
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

if not (ASSETS_DIR / "figs.pkl").exists():
    raise FileNotFoundError("assets/figs.pkl not found. Run preprocessing notebook.")

if not (ASSETS_DIR / "metrics.csv").exists():
    raise FileNotFoundError("assets/metrics.csv not found. Run preprocessing notebook.")

# ----------------------------
# Load artifacts
# ----------------------------
with open(ASSETS_DIR / "figs.pkl", "rb") as f:
    figs = pickle.load(f)

metrics_df = pd.read_csv(ASSETS_DIR / "metrics.csv")

# ----------------------------
# App
# ----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Nepal Earthquake Damage"

# Tabs for EDA vs Models
tabs = dbc.Tabs(
    [
        dbc.Tab(
            label="EDA",
            children=[
                html.Br(),
                dcc.Graph(figure=figs["fig_class"]),
                dcc.Graph(figure=figs["fig_plinth"]),
                dcc.Graph(figure=figs["fig_roof"]),
            ],
        ),
        dbc.Tab(
            label="Model Metrics",
            children=[
                html.Br(),
                html.H5("Metrics Table"),
                dbc.Table.from_dataframe(metrics_df.round(3), striped=True, bordered=True, hover=True),
                html.Hr(),
                html.H5("Confusion Matrices"),
                dcc.Graph(figure=figs["fig_cm_lr"]),
                dcc.Graph(figure=figs["fig_cm_dt"]),
                html.Hr(),
                html.H5("Decision Tree Feature Importance"),
                dcc.Graph(figure=figs["fig_feat"]),
            ],
        ),
    ],
    id="tabs",
    active_tab="EDA",
)

app.layout = dbc.Container(
    [
        html.H3("🌏 Nepal Earthquake Building Damage Analysis"),
        tabs,
    ],
    fluid=True,
)

if __name__ == "__main__":
    app.run(debug=True)