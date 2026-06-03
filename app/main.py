"""Dash application entry point."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dash
from dash import html, dcc
from components.layout import navbar

app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder=str(Path(__file__).parent / "pages"),
    suppress_callback_exceptions=True,
)
server = app.server  # for deployment (gunicorn/waitress)

app.layout = html.Div([
    navbar(),
    html.Div(dash.page_container, className="page-content"),
    dcc.Location(id="url"),
])

if __name__ == "__main__":
    app.run(debug=True, port=8050)
