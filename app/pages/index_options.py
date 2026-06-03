"""Index option implied-volatility surface via Monte Carlo (Cont-Kokholm)."""
import dash
from dash import html, dcc, callback, Input, Output
import numpy as np
import plotly.graph_objects as go

from finpricing.models.vix.model import VixModel
from finpricing.data.fixtures import TENOR_DATES
from finpricing.models.black_scholes.implied_volatility import (
    ImpliedVolatilityCalculator, ImpliedVolatilityParams, Method
)

dash.register_page(__name__, path="/index-options", title="Index Options")

_MODELS: dict[str, VixModel] = {}
_PATHS_STORED: dict[str, int] = {}


def _get_model(model_type: str, num_paths: int) -> VixModel:
    key = f"{model_type}_{num_paths}"
    if key not in _MODELS:
        m = VixModel(model_type=model_type)
        m.store_tenor_data(num_paths)
        _MODELS[key] = m
        _PATHS_STORED[key] = num_paths
    return _MODELS[key]


def _bs_iv(price, S, K, T):
    if price <= 0 or not np.isfinite(price):
        return np.nan
    try:
        p = ImpliedVolatilityParams(call_value=float(price), s=S, k=K, t=T, r=0.0,
                                    method=Method.BRENTQ)
        iv = ImpliedVolatilityCalculator.implied_volatility(p)
        return iv if 0.01 < iv < 2.0 else np.nan
    except Exception:
        return np.nan


# ── Layout ────────────────────────────────────────────────────────────────────

layout = html.Div(className="page", children=[
    html.H2("Index Option Implied Volatility  —  Cont-Kokholm MC"),

    html.Div(className="card controls-inline", children=[
        html.Label("Model"),
        dcc.RadioItems(
            id="idx-model",
            options=[
                {"label": "Merton", "value": "Merton"},
                {"label": "Kou",    "value": "Kou"},
            ],
            value="Merton",
            inline=True,
            className="radio-inline",
        ),
        html.Label("MC paths", style={"marginLeft": "2rem"}),
        dcc.Dropdown(
            id="idx-paths",
            options=[
                {"label": "50 K  (fast)",       "value": 50_000},
                {"label": "200 K (default)",     "value": 200_000},
                {"label": "500 K (slow, smoother)", "value": 500_000},
            ],
            value=200_000,
            clearable=False,
            style={"width": "180px"},
        ),
        html.Label("Tenor", style={"marginLeft": "2rem"}),
        dcc.Dropdown(
            id="idx-tenor",
            options=[
                {"label": f"{int(round(T*12))}M", "value": i+1}
                for i, T in enumerate(TENOR_DATES[1:])
            ],
            value=1,
            clearable=False,
            style={"width": "100px"},
        ),
        html.Label("Spot S₀", style={"marginLeft": "2rem"}),
        dcc.Input(id="idx-S0", type="number", value=1.0, step=0.1,
                  min=0.1, className="input-field input-short"),
    ]),

    html.P("First run will simulate MC paths and may take a few seconds.",
           className="hint-text"),

    dcc.Loading(
        type="circle",
        children=dcc.Graph(id="idx-smile-graph", style={"height": "450px"}),
    ),
])


# ── Callback ──────────────────────────────────────────────────────────────────

@callback(
    Output("idx-smile-graph", "figure"),
    Input("idx-model",  "value"),
    Input("idx-paths",  "value"),
    Input("idx-tenor",  "value"),
    Input("idx-S0",     "value"),
)
def update_index_smile(model_type, num_paths, tenor_idx, S0):
    if S0 is None or S0 <= 0:
        return go.Figure()

    T = TENOR_DATES[tenor_idx]
    model = _get_model(model_type, num_paths)

    # strike grid around S0
    ks = np.linspace(0.55 * S0, 1.45 * S0, 40)
    prices = model.index_option_pricer(S0=S0, strikes=ks, tenor_index=tenor_idx, r=0.0)
    ivs = np.array([_bs_iv(prices[i], S0, ks[i], T) for i in range(len(ks))])

    mask = ~np.isnan(ivs) & (ivs > 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=(ks[mask] / S0), y=ivs[mask],
        mode="lines+markers",
        line={"width": 2},
        marker={"size": 5},
        name=model_type,
    ))
    fig.update_layout(
        title=f"{model_type} index option smile  —  T = {int(round(T*12))}M  ({num_paths:,} paths)",
        xaxis_title="Moneyness  K / S₀",
        yaxis_title="Implied volatility",
        template="plotly_white",
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
    )
    return fig
