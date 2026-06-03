"""Implied volatility surface from live SPX option chain (yfinance)."""
import dash
from dash import html, dcc, callback, Input, Output, State
import numpy as np
import plotly.graph_objects as go

from finpricing.models.black_scholes.implied_volatility import (
    ImpliedVolatilityCalculator, ImpliedVolatilityParams, Method
)

dash.register_page(__name__, path="/vol-surface", title="Vol Surface")


def _bs_iv(price, S, K, T, r=0.02):
    if price <= 0 or not np.isfinite(price):
        return np.nan
    try:
        p = ImpliedVolatilityParams(call_value=float(price), s=S, k=K, t=T, r=r,
                                    method=Method.BRENTQ)
        iv = ImpliedVolatilityCalculator.implied_volatility(p)
        return iv if 0.01 < iv < 3.0 else np.nan
    except Exception:
        return np.nan


layout = html.Div(className="page", children=[
    html.H2("Implied Volatility Surface"),

    html.Div(className="card controls-inline", children=[
        html.Label("Ticker"),
        dcc.Input(id="vs-ticker", type="text", value="SPY",
                  debounce=True, className="input-field input-short"),
        html.Label("Risk-free rate"),
        dcc.Input(id="vs-rate", type="number", value=0.05, step=0.001,
                  min=0.0, max=0.2, className="input-field input-short"),
        html.Button("Fetch", id="vs-fetch-btn", n_clicks=0, className="btn-primary"),
        html.Span(id="vs-status", className="status-text"),
    ]),

    dcc.Loading(
        id="vs-loading",
        type="circle",
        children=html.Div(id="vs-graph-container", children=[
            dcc.Graph(id="vs-surface", style={"height": "550px"}),
        ]),
    ),
])


@callback(
    Output("vs-surface",   "figure"),
    Output("vs-status",    "children"),
    Input("vs-fetch-btn",  "n_clicks"),
    State("vs-ticker",     "value"),
    State("vs-rate",       "value"),
    prevent_initial_call=True,
)
def fetch_surface(n_clicks, ticker, rate):
    if not ticker:
        return go.Figure(), "Enter a ticker."
    try:
        import yfinance as yf
        import pandas as pd
        from datetime import datetime

        tkr = yf.Ticker(ticker.upper())
        spot = tkr.fast_info["lastPrice"]
        expiries = tkr.options[:8]  # up to 8 expiries

        rows = []
        for exp in expiries:
            chain = tkr.option_chain(exp)
            calls = chain.calls
            T = (pd.Timestamp(exp) - pd.Timestamp.now()).days / 365.25
            if T <= 0:
                continue
            for _, row in calls.iterrows():
                K = row["strike"]
                mid = (row["bid"] + row["ask"]) / 2 if row["bid"] > 0 else row["lastPrice"]
                iv = _bs_iv(mid, spot, K, T, rate or 0.02)
                if not np.isnan(iv):
                    rows.append({"K": K / spot, "T": T, "IV": iv})

        if not rows:
            return go.Figure(), "No valid option data returned."

        import pandas as pd
        df = pd.DataFrame(rows)

        fig = go.Figure(data=[go.Scatter3d(
            x=df["T"], y=df["K"], z=df["IV"],
            mode="markers",
            marker=dict(size=3, color=df["IV"], colorscale="Viridis", showscale=True,
                        colorbar=dict(title="IV")),
        )])
        fig.update_layout(
            scene=dict(
                xaxis_title="Maturity (years)",
                yaxis_title="Moneyness K/S",
                zaxis_title="Implied vol",
            ),
            title=f"{ticker.upper()} Implied Volatility Surface  (spot = {spot:.2f})",
            template="plotly_white",
            margin={"l": 0, "r": 0, "t": 50, "b": 0},
        )
        return fig, f"Fetched {len(df)} option quotes across {len(expiries)} expiries."

    except Exception as e:
        return go.Figure(), f"Error: {e}"
