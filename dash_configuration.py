import dash
import dash_bootstrap_components as dbc

# Configure Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Retail Inventory Dashboard",
    suppress_callback_exceptions=True,
    assets_folder='assets',  # Look for assets in the assets folder
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

server = app.server  # For Gunicorn to use 