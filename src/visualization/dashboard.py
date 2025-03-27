"""Interactive dashboard for inventory management system."""

from typing import Dict, Optional, Any
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    refresh_interval: int = 3600  # seconds
    max_points: int = 10000
    cache_timeout: int = 300  # seconds
    port: int = 8050
    debug: bool = False

class DashboardError(Exception):
    """Base class for dashboard exceptions."""
    pass

class Dashboard:
    """Interactive inventory management dashboard."""
    
    def __init__(
        self,
        results: Dict[str, Any],
        config: Optional[DashboardConfig] = None
    ):
        """Initialize dashboard.
        
        Args:
            results: Pipeline results
            config: Optional dashboard configuration
        """
        self.results = results
        self.config = config or DashboardConfig()
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.FLATLY],
            suppress_callback_exceptions=True
        )
        
        # Data cache
        self._cache = {}
        self._last_update = None
        
        # Set up layout
        self.app.layout = self._create_layout()
        
        # Register callbacks
        self._register_callbacks()
        
        logger.info("Dashboard initialized")
    
    def _create_layout(self):
        """Create dashboard layout."""
        return dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Inventory Management Dashboard", className="text-primary mb-4")
                ])
            ], className="mt-4"),
            
            # Filters
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Filters"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Date Range:"),
                                    dcc.DatePickerRange(
                                        id='date-filter',
                                        start_date=datetime.now() - timedelta(days=30),
                                        end_date=datetime.now(),
                                        className="mb-3"
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label("Category:"),
                                    dcc.Dropdown(
                                        id='category-filter',
                                        multi=True,
                                        className="mb-3"
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label("Product:"),
                                    dcc.Dropdown(
                                        id='product-filter',
                                        multi=True,
                                        className="mb-3"
                                    )
                                ], md=4)
                            ])
                        ])
                    ], className="mb-4")
                ])
            ]),
            
            # KPIs
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Sales"),
                        dbc.CardBody(id='sales-kpi')
                    ])
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Inventory"),
                        dbc.CardBody(id='inventory-kpi')
                    ])
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Forecast Accuracy"),
                        dbc.CardBody(id='forecast-kpi')
                    ])
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Revenue"),
                        dbc.CardBody(id='revenue-kpi')
                    ])
                ], md=3)
            ], className="mb-4"),
            
            # Main content
            dbc.Tabs([
                # Overview
                dbc.Tab(label="Overview", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='sales-trend')
                        ], md=6),
                        dbc.Col([
                            dcc.Graph(id='inventory-health')
                        ], md=6)
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Key Recommendations"),
                            html.Div(id='recommendations')
                        ])
                    ])
                ]),
                
                # Forecasting
                dbc.Tab(label="Forecasting", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='forecast-trend')
                        ], md=8),
                        dbc.Col([
                            html.H4("Forecast Metrics"),
                            html.Div(id='forecast-metrics')
                        ], md=4)
                    ])
                ]),
                
                # Inventory
                dbc.Tab(label="Inventory", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='inventory-levels')
                        ], md=6),
                        dbc.Col([
                            dcc.Graph(id='reorder-points')
                        ], md=6)
                    ])
                ]),
                
                # Pricing
                dbc.Tab(label="Pricing", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='price-elasticity')
                        ], md=6),
                        dbc.Col([
                            dcc.Graph(id='optimal-prices')
                        ], md=6)
                    ])
                ])
            ], className="mb-4"),
            
            # Footer
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P(
                        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        className="text-muted"
                    )
                ])
            ])
        ], fluid=True)
    
    def _register_callbacks(self):
        """Register all dashboard callbacks."""
        
        @self.app.callback(
            [Output('sales-kpi', 'children'),
             Output('inventory-kpi', 'children'),
             Output('forecast-kpi', 'children'),
             Output('revenue-kpi', 'children')],
            [Input('date-filter', 'start_date'),
             Input('date-filter', 'end_date'),
             Input('category-filter', 'value'),
             Input('product-filter', 'value')]
        )
        def update_kpis(start_date, end_date, categories, products):
            """Update KPI cards."""
            try:
                df = self._get_filtered_data(start_date, end_date, categories, products)
                
                sales = df['units_sold'].sum()
                inventory = df['inventory_level'].mean()
                accuracy = 1 - abs(df['demand_forecast'] - df['units_sold']).mean() / df['units_sold'].mean()
                revenue = df['revenue'].sum()
                
                return [
                    self._format_kpi("Total Sales", sales, "units"),
                    self._format_kpi("Avg Inventory", inventory, "units"),
                    self._format_kpi("Forecast Accuracy", accuracy, "percent"),
                    self._format_kpi("Total Revenue", revenue, "currency")
                ]
            except Exception as e:
                logger.error(f"Error updating KPIs: {str(e)}")
                return ["Error"] * 4
        
        @self.app.callback(
            Output('sales-trend', 'figure'),
            [Input('date-filter', 'start_date'),
             Input('date-filter', 'end_date'),
             Input('category-filter', 'value')]
        )
        def update_sales_trend(start_date, end_date, categories):
            """Update sales trend chart."""
            try:
                df = self._get_filtered_data(start_date, end_date, categories)
                
                # Downsample if needed
                if len(df) > self.config.max_points:
                    df = df.sample(n=self.config.max_points)
                
                fig = px.line(
                    df,
                    x='date',
                    y='units_sold',
                    color='category',
                    title='Sales Trend'
                )
                
                fig.update_layout(
                    template='plotly_white',
                    height=400,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error updating sales trend: {str(e)}")
                return go.Figure()
    
    def _format_kpi(self, title: str, value: float, format_type: str) -> html.Div:
        """Format KPI card content.
        
        Args:
            title: KPI title
            value: KPI value
            format_type: Value format type ('units', 'percent', 'currency')
            
        Returns:
            Formatted KPI card content
        """
        if format_type == 'units':
            formatted_value = f"{value:,.0f}"
        elif format_type == 'percent':
            formatted_value = f"{value:.1%}"
        elif format_type == 'currency':
            formatted_value = f"${value:,.2f}"
        else:
            formatted_value = f"{value}"
        
        return html.Div([
            html.H3(formatted_value, className="text-center"),
            html.P(title, className="text-center text-muted")
        ])
    
    def _get_filtered_data(
        self,
        start_date: str,
        end_date: str,
        categories: Optional[list] = None,
        products: Optional[list] = None
    ) -> pd.DataFrame:
        """Get filtered data with caching.
        
        Args:
            start_date: Start date
            end_date: End date
            categories: Optional list of categories
            products: Optional list of products
            
        Returns:
            Filtered DataFrame
        """
        cache_key = f"{start_date}_{end_date}_{categories}_{products}"
        
        # Check cache
        if (
            cache_key in self._cache and
            self._last_update and
            (datetime.now() - self._last_update).seconds < self.config.cache_timeout
        ):
            return self._cache[cache_key]
        
        try:
            # Get fresh data
            df = pd.DataFrame(self.results['data'])
            
            # Apply filters
            if start_date and end_date:
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            if categories:
                df = df[df['category'].isin(categories)]
            
            if products:
                df = df[df['product_id'].isin(products)]
            
            # Update cache
            self._cache[cache_key] = df
            self._last_update = datetime.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting filtered data: {str(e)}")
            raise DashboardError(f"Error getting filtered data: {str(e)}")
    
    def run(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        debug: Optional[bool] = None
    ) -> None:
        """Run the dashboard server.
        
        Args:
            host: Host address
            port: Port number (defaults to config)
            debug: Debug mode (defaults to config)
        """
        try:
            self.app.run_server(
                host=host,
                port=port or self.config.port,
                debug=debug or self.config.debug
            )
        except Exception as e:
            logger.error(f"Error running dashboard: {str(e)}")
            raise DashboardError(f"Error running dashboard: {str(e)}") 