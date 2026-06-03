"""Black-Scholes interactive pricer page."""
import dash
from dash import html, dcc, callback, Input, Output
import numpy as np
from scipy.stats import norm

from finpricing.utils.bs_utils import BlackScholesCalculator
from finpricing.models.black_scholes.implied_volatility import (
    ImpliedVolatilityCalculator, ImpliedVolatilityParams, Method
)

dash.register_page(__name__, path="/", title="Black-Scholes Pricer")

# ── Layout ────────────────────────────────────────────────────────────────────

def _slider(id_, label, min_, max_, step, value, marks=None):
    return html.Div(className="slider-row", children=[
        html.Label(label, className="slider-label"),
        dcc.Slider(id=id_, min=min_, max=max_, step=step, value=value,
                   marks=marks or {min_: str(min_), max_: str(max_)},
                   tooltip={"placement": "bottom", "always_visible": True}),
    ])


layout = html.Div(className="page", children=[
    html.H2("Black-Scholes Option Pricer"),

    html.Div(className="two-col", children=[

        # ── Left: controls ───────────────────────────────────────────────────
        html.Div(className="card controls", children=[
            html.H4("Parameters"),
            _slider("bs-S",     "Spot (S)",              10,  500, 1,    100),
            _slider("bs-K",     "Strike (K)",            10,  500, 1,    100),
            _slider("bs-T",     "Maturity (years)",      0.02, 3,  0.01, 1.0),
            _slider("bs-r",     "Risk-free rate",        0.0,  0.15, 0.001, 0.05),
            _slider("bs-sigma", "Volatility σ",          0.01, 1.5,  0.01, 0.20),
        ]),

        # ── Right: output ────────────────────────────────────────────────────
        html.Div(className="card", children=[
            html.H4("Prices"),
            html.Div(id="bs-prices", className="price-display"),
            html.H4("Greeks", style={"marginTop": "1.5rem"}),
            html.Div(id="bs-greeks"),
        ]),
    ]),

    # ── IV Calculator ─────────────────────────────────────────────────────────
    html.Div(className="card", style={"marginTop": "1.5rem"}, children=[
        html.H4("Implied Volatility Calculator"),
        html.Div(className="iv-row", children=[
            html.Div([
                html.Label("Market call price"),
                dcc.Input(id="bs-market-price", type="number", value=10.0,
                          min=0.001, step=0.01, className="input-field"),
            ]),
            html.Div([
                html.Label("Method"),
                dcc.Dropdown(
                    id="bs-iv-method",
                    options=[
                        {"label": "Iterative bisection", "value": "iterative"},
                        {"label": "Brentq",               "value": "brentq"},
                        {"label": "Newton-Raphson",        "value": "newton"},
                    ],
                    value="brentq",
                    clearable=False,
                    className="dropdown",
                ),
            ]),
            html.Div([
                html.Label("Implied volatility"),
                html.Div(id="bs-iv-result", className="iv-result"),
            ]),
        ]),
    ]),
])

# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("bs-prices", "children"),
    Output("bs-greeks", "children"),
    Input("bs-S", "value"),
    Input("bs-K", "value"),
    Input("bs-T", "value"),
    Input("bs-r", "value"),
    Input("bs-sigma", "value"),
)
def update_prices(S, K, T, r, sigma):
    if any(v is None or v <= 0 for v in [S, K, T, sigma]):
        return "—", "—"

    call = BlackScholesCalculator.call_price(S, K, T, r, sigma)
    put  = BlackScholesCalculator.put_price(S, K, T, r, sigma)
    d1, d2 = BlackScholesCalculator.calculate_d1_d2(S, K, T, r, sigma)

    prices = html.Div([
        html.Span(f"Call: {call:.4f}", className="price-call"),
        html.Span(f"Put:  {put:.4f}",  className="price-put"),
    ])

    delta_call = norm.cdf(d1)
    delta_put  = norm.cdf(d1) - 1
    gamma      = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega       = BlackScholesCalculator.vega(S, K, T, r, sigma)
    theta_call = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(d2))
    rho_call   = K * T * np.exp(-r * T) * norm.cdf(d2)

    greeks = html.Table(className="greeks-table", children=[
        html.Thead(html.Tr([html.Th("Greek"), html.Th("Call"), html.Th("Put")])),
        html.Tbody([
            html.Tr([html.Td("Delta"), html.Td(f"{delta_call:.4f}"), html.Td(f"{delta_put:.4f}")]),
            html.Tr([html.Td("Gamma"), html.Td(f"{gamma:.4f}"),      html.Td(f"{gamma:.4f}")]),
            html.Tr([html.Td("Vega"),  html.Td(f"{vega:.4f}"),       html.Td(f"{vega:.4f}")]),
            html.Tr([html.Td("Theta"), html.Td(f"{theta_call:.4f}"), html.Td("—")]),
            html.Tr([html.Td("Rho"),   html.Td(f"{rho_call:.4f}"),   html.Td("—")]),
        ]),
    ])
    return prices, greeks


@callback(
    Output("bs-iv-result", "children"),
    Input("bs-market-price", "value"),
    Input("bs-S", "value"),
    Input("bs-K", "value"),
    Input("bs-T", "value"),
    Input("bs-r", "value"),
    Input("bs-iv-method", "value"),
)
def update_iv(market_price, S, K, T, r, method_str):
    if any(v is None or v <= 0 for v in [market_price, S, K, T]):
        return "—"
    method_map = {"iterative": Method.ITERATIVE, "brentq": Method.BRENTQ, "newton": Method.NEWTON}
    try:
        params = ImpliedVolatilityParams(
            call_value=float(market_price), s=S, k=K, t=T, r=r,
            method=method_map[method_str],
        )
        iv = ImpliedVolatilityCalculator.implied_volatility(params)
        return f"{iv:.4f}  ({iv*100:.2f}%)"
    except Exception as e:
        return f"Error: {e}"
