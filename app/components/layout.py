import dash
from dash import html, dcc


def navbar() -> html.Div:
    return html.Div(
        className="navbar",
        children=[
            html.Span("finpricing", className="navbar-brand"),
            html.Nav([
                dcc.Link("Black-Scholes",  href="/",             className="nav-link"),
                dcc.Link("VIX Options",    href="/vix",          className="nav-link"),
                dcc.Link("Vol Surface",    href="/vol-surface",  className="nav-link"),
                dcc.Link("Index Options",  href="/index-options", className="nav-link"),
            ]),
        ],
    )


def page_container() -> html.Div:
    return html.Div([
        navbar(),
        html.Div(dash.page_container, className="page-content"),
    ])
