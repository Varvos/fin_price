"""VIX option pricer page — Fourier pricing with paper parameters."""
import dash
from dash import html, dcc, callback, Input, Output
import numpy as np
import plotly.graph_objects as go

from finpricing.models.vix.model import VixModel
from finpricing.data.fixtures import TENOR_DATES
from finpricing.models.black_scholes.implied_volatility import (
    ImpliedVolatilityCalculator, ImpliedVolatilityParams, Method
)

dash.register_page(__name__, path="/vix", title="VIX Option Pricer")

_MODELS: dict[str, VixModel] = {}

def _get_model(model_type: str) -> VixModel:
    if model_type not in _MODELS:
        _MODELS[model_type] = VixModel(model_type=model_type)
    return _MODELS[model_type]


def _bs_iv(price: float, S: float, K: float, T: float) -> float | None:
    if price <= 1e-8 or not np.isfinite(price):
        return None
    try:
        p = ImpliedVolatilityParams(call_value=float(price), s=S, k=K, t=T, r=0.0,
                                    method=Method.BRENTQ)
        iv = ImpliedVolatilityCalculator.implied_volatility(p)
        return iv if 0.01 < iv < 5.0 else None
    except Exception:
        return None


# ── Layout ────────────────────────────────────────────────────────────────────

layout = html.Div(className="page", children=[
    html.H2("VIX Option Pricer  —  Cont-Kokholm Model"),

    html.Div(className="two-col", children=[

        html.Div(className="card controls", children=[
            html.H4("Settings"),
            html.Label("Model"),
            dcc.RadioItems(
                id="vix-model",
                options=[
                    {"label": "Merton (Gaussian jumps)",          "value": "Merton"},
                    {"label": "Kou (double-exponential jumps)",    "value": "Kou"},
                    {"label": "Black-Scholes (log-normal)",        "value": "Black-Scholes"},
                ],
                value="Merton",
                className="radio-group",
            ),
            html.Hr(),
            html.Label("Maturity"),
            dcc.Dropdown(
                id="vix-maturity",
                options=[
                    {"label": f"{int(round(T*12))}M", "value": i}
                    for i, T in enumerate(TENOR_DATES[1:])
                ],
                value=0,
                clearable=False,
                className="dropdown",
            ),
            html.Hr(),
            html.Label("Current VIX level (V₀)"),
            dcc.Slider(id="vix-V0", min=0.05, max=0.80, step=0.01, value=0.20,
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("Moneyness range"),
            dcc.RangeSlider(id="vix-krange", min=0.5, max=3.0, step=0.05,
                            value=[0.8, 2.5],
                            tooltip={"placement": "bottom", "always_visible": True}),
            html.Hr(),
            html.Label("Display"),
            dcc.RadioItems(
                id="vix-display",
                options=[
                    {"label": "Option prices",       "value": "price"},
                    {"label": "Implied volatility",  "value": "iv"},
                ],
                value="iv",
                className="radio-group",
            ),
        ]),

        html.Div(className="card", children=[
            dcc.Graph(id="vix-smile-graph", style={"height": "420px"}),
            html.Div(id="vix-params-display", className="params-display"),
        ]),
    ]),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("vix-smile-graph",    "figure"),
    Output("vix-params-display", "children"),
    Input("vix-model",    "value"),
    Input("vix-maturity", "value"),
    Input("vix-V0",       "value"),
    Input("vix-krange",   "value"),
    Input("vix-display",  "value"),
)
def update_vix_smile(model_type, mat_idx, V0, krange, display):
    T = TENOR_DATES[mat_idx + 1]
    ks = np.arange(krange[0], krange[1] + 0.01, 0.02)

    if model_type == "Black-Scholes":
        from finpricing.data.fixtures import MERTON_PARAMS
        from finpricing.utils.bs_utils import BlackScholesCalculator
        prices = np.array([BlackScholesCalculator.call_price(V0, K, T, 0.0, 0.3275) for K in ks])
    else:
        model = _get_model(model_type)
        prices = np.array([model.vix_option_pricer(V0, K * V0, T, r=0.0) for K in ks])

    if display == "iv":
        ys = np.array([_bs_iv(p, V0, K * V0, T) for p, K in zip(prices, ks)])
        ylabel = "Implied volatility"
    else:
        ys = prices
        ylabel = "Call price"

    mask = np.array([y is not None and np.isfinite(y) for y in ys])
    ys_clean = np.array([y if y is not None else np.nan for y in ys])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks[mask], y=ys_clean[mask],
        mode="lines", line={"width": 2},
        name=model_type,
    ))
    fig.update_layout(
        title=f"{model_type} — T = {int(round(T*12))}M",
        xaxis_title="Moneyness  K / V₀",
        yaxis_title=ylabel,
        template="plotly_white",
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
    )

    # parameter display
    if model_type != "Black-Scholes":
        model = _get_model(model_type)
        param_items = [
            html.Span(f"{k} = {v}", className="param-chip")
            for k, v in model.params.model_dump().items()
        ]
        params_div = html.Div(["Parameters (Table 2): "] + param_items)
    else:
        params_div = html.Div("σ = 0.3275 (Black-Scholes)")

    return fig, params_div
