"""Interactive dashboard for inventory management system."""

from typing import Dict, Optional, Any, List
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

# Color scheme
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#18BC9C',
    'success': '#28B463',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'light': '#ECF0F1',
    'dark': '#2C3E50'
}

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
        """Initialize dashboard."""
        self.results = results
        self.config = config or DashboardConfig()
        
        # Initialize Dash app with professional theme
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.FLATLY,
                'https://use.fontawesome.com/releases/v5.15.4/css/all.css'
            ],
            suppress_callback_exceptions=True
        )
        
        # Data cache
        self._cache = {}
        self._last_update = None
        
        # Initialize data
        self._initialize_data()
        
        # Set up layout
        self.app.layout = self._create_layout()
        
        # Register callbacks
        self._register_callbacks()
        
        logger.info("Dashboard initialized")
    
    def _initialize_data(self):
        """Initialize data and extract metadata."""
        try:
            # Get initial DataFrame
            if isinstance(self.results, dict) and 'data' in self.results:
                self.df = self.results['data']
            else:
                self.df = self.results
            
            if not isinstance(self.df, pd.DataFrame):
                self.df = pd.DataFrame(self.df)
            
            # Ensure date column is datetime
            if 'Date' in self.df.columns:
                self.df['Date'] = pd.to_datetime(self.df['Date'])
            
            # Extract metadata for filters
            self.categories = sorted(self.df['Category'].unique())
            self.products = sorted(self.df['Product_ID'].unique())
            self.date_range = [self.df['Date'].min(), self.df['Date'].max()]
            
            logger.info("Data initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing data: {str(e)}")
            raise DashboardError(f"Error initializing data: {str(e)}")
    
    def _create_layout(self):
        """Create dashboard layout."""
        return dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1([
                        html.I(className="fas fa-chart-line mr-2"),
                        "Inventory Management Dashboard"
                    ], className="text-primary mb-4")
                ])
            ], className="mt-4"),
            
            # Filters Card
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4("Filters", className="mb-0"),
                            dbc.Button(
                                html.I(className="fas fa-sync-alt"),
                                id="refresh-data",
                                color="link",
                                className="float-right"
                            )
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Date Range"),
                                    dcc.DatePickerRange(
                                        id='date-filter',
                                        start_date=self.date_range[0],
                                        end_date=self.date_range[1],
                                        className="mb-3"
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label("Category"),
                                    dcc.Dropdown(
                                        id='category-filter',
                                        options=[{'label': c, 'value': c} for c in self.categories],
                                        multi=True,
                                        placeholder="Select categories...",
                                        className="mb-3"
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label("Product"),
                                    dcc.Dropdown(
                                        id='product-filter',
                                        options=[{'label': p, 'value': p} for p in self.products],
                                        multi=True,
                                        placeholder="Select products...",
                                        className="mb-3"
                                    )
                                ], md=4)
                            ])
                        ])
                    ], className="mb-4 shadow-sm")
                ])
            ]),
            
            # KPIs Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-shopping-cart mr-2"),
                            "Sales"
                        ]),
                        dbc.CardBody(
                            id='sales-kpi',
                            children=dbc.Spinner(color="primary")
                        )
                    ], className="shadow-sm")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-boxes mr-2"),
                            "Inventory"
                        ]),
                        dbc.CardBody(
                            id='inventory-kpi',
                            children=dbc.Spinner(color="primary")
                        )
                    ], className="shadow-sm")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-chart-bar mr-2"),
                            "Forecast Accuracy"
                        ]),
                        dbc.CardBody(
                            id='forecast-kpi',
                            children=dbc.Spinner(color="primary")
                        )
                    ], className="shadow-sm")
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-dollar-sign mr-2"),
                            "Revenue"
                        ]),
                        dbc.CardBody(
                            id='revenue-kpi',
                            children=dbc.Spinner(color="primary")
                        )
                    ], className="shadow-sm")
                ], md=3)
            ], className="mb-4"),
            
            # Main content
            dbc.Tabs([
                # Executive Overview
                dbc.Tab(label="Executive Overview", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Sales Trend"),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id='sales-trend',
                                        config={'displayModeBar': False}
                                    )
                                ])
                            ], className="shadow-sm")
                        ], md=8),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Key Insights"),
                                dbc.CardBody([
                                    html.Div(id='key-insights')
                                ])
                            ], className="shadow-sm")
                        ], md=4)
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Category Performance"),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id='category-performance',
                                        config={'displayModeBar': False}
                                    )
                                ])
                            ], className="shadow-sm")
                        ], md=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Inventory Health"),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id='inventory-health',
                                        config={'displayModeBar': False}
                                    )
                                ])
                            ], className="shadow-sm")
                        ], md=6)
                    ])
                ]),
                
                # Technical Analysis
                dbc.Tab(label="Technical Analysis", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Forecast Analysis"),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id='forecast-analysis',
                                        config={'displayModeBar': True}
                                    )
                                ])
                            ], className="shadow-sm")
                        ], md=8),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Model Metrics"),
                                dbc.CardBody([
                                    html.Div(id='model-metrics')
                                ])
                            ], className="shadow-sm")
                        ], md=4)
                    ], className="mb-4")
                ]),
                
                # Optimization Results
                dbc.Tab(label="Optimization Results", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Inventory Optimization"),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id='inventory-optimization',
                                        config={'displayModeBar': True}
                                    )
                                ])
                            ], className="shadow-sm")
                        ], md=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Pricing Optimization"),
                                dbc.CardBody([
                                    dcc.Graph(
                                        id='pricing-optimization',
                                        config={'displayModeBar': True}
                                    )
                                ])
                            ], className="shadow-sm")
                        ], md=6)
                    ])
                ])
            ], className="mb-4"),
            
            # Footer
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P([
                        html.I(className="fas fa-clock mr-2"),
                        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    ], className="text-muted text-center")
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
             Input('product-filter', 'value'),
             Input('refresh-data', 'n_clicks')]
        )
        def update_kpis(start_date, end_date, categories, products, n_clicks):
            """Update KPI cards."""
            try:
                df = self._get_filtered_data(start_date, end_date, categories, products)
                
                sales = df['Units_Sold'].sum()
                inventory = df['Inventory_Level'].mean()
                accuracy = 1 - abs(df['Demand_Forecast'] - df['Units_Sold']).mean() / df['Units_Sold'].mean()
                revenue = (df['Units_Sold'] * df['Price'] * (1 - df['Discount']/100)).sum()
                
                return [
                    self._format_kpi("Total Sales", sales, "units", "fas fa-shopping-cart"),
                    self._format_kpi("Avg Inventory", inventory, "units", "fas fa-boxes"),
                    self._format_kpi("Forecast Accuracy", accuracy, "percent", "fas fa-chart-line"),
                    self._format_kpi("Total Revenue", revenue, "currency", "fas fa-dollar-sign")
                ]
            except Exception as e:
                logger.error(f"Error updating KPIs: {str(e)}")
                return [self._format_error_card("Error loading KPI")] * 4
        
        @self.app.callback(
            [Output('sales-trend', 'figure'),
             Output('category-performance', 'figure'),
             Output('inventory-health', 'figure')],
            [Input('date-filter', 'start_date'),
             Input('date-filter', 'end_date'),
             Input('category-filter', 'value'),
             Input('refresh-data', 'n_clicks')]
        )
        def update_executive_charts(start_date, end_date, categories, n_clicks):
            """Update executive overview charts."""
            try:
                df = self._get_filtered_data(start_date, end_date, categories)
                
                # Sales Trend
                sales_fig = self._create_sales_trend(df)
                
                # Category Performance
                category_fig = self._create_category_performance(df)
                
                # Inventory Health
                inventory_fig = self._create_inventory_health(df)
                
                return sales_fig, category_fig, inventory_fig
            except Exception as e:
                logger.error(f"Error updating executive charts: {str(e)}")
                return [go.Figure()] * 3
        
        @self.app.callback(
            Output('key-insights', 'children'),
            [Input('date-filter', 'start_date'),
             Input('date-filter', 'end_date'),
             Input('category-filter', 'value'),
             Input('refresh-data', 'n_clicks')]
        )
        def update_key_insights(start_date, end_date, categories, n_clicks):
            """Update key insights."""
            try:
                df = self._get_filtered_data(start_date, end_date, categories)
                insights = self._generate_insights(df)
                
                return [
                    html.Div([
                        html.I(className=insight['icon'], style={'color': insight['color']}),
                        html.Span(f" {insight['text']}")
                    ], className="mb-3")
                    for insight in insights
                ]
            except Exception as e:
                logger.error(f"Error updating insights: {str(e)}")
                return html.Div("Error loading insights", className="text-danger")
    
    def _format_kpi(self, title: str, value: float, format_type: str, icon: str) -> html.Div:
        """Format KPI card content with icon."""
        if format_type == 'units':
            formatted_value = f"{value:,.0f}"
        elif format_type == 'percent':
            formatted_value = f"{value:.1%}"
        elif format_type == 'currency':
            formatted_value = f"${value:,.2f}"
        else:
            formatted_value = f"{value}"
        
        return html.Div([
            html.H3([
                html.I(className=f"{icon} mr-2"),
                formatted_value
            ], className="text-center"),
            html.P(title, className="text-center text-muted")
        ])
    
    def _format_error_card(self, message: str) -> html.Div:
        """Format error message for cards."""
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-danger"),
            html.P(message, className="text-danger mb-0 ml-2")
        ], className="d-flex align-items-center justify-content-center h-100")
    
    def _create_sales_trend(self, df: pd.DataFrame) -> go.Figure:
        """Create sales trend chart."""
        # Group by date and category
        daily_sales = df.groupby(['Date', 'Category'])['Units_Sold'].sum().reset_index()
        
        fig = px.line(
            daily_sales,
            x='Date',
            y='Units_Sold',
            color='Category',
            title='Daily Sales by Category'
        )
        
        fig.update_layout(
            template='plotly_white',
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_title="Date",
            yaxis_title="Units Sold"
        )
        
        return fig
    
    def _create_category_performance(self, df: pd.DataFrame) -> go.Figure:
        """Create category performance chart."""
        category_metrics = df.groupby('Category').agg({
            'Units_Sold': 'sum',
            'Revenue': lambda x: (df['Units_Sold'] * df['Price'] * (1 - df['Discount']/100)).sum(),
            'Inventory_Level': 'mean'
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bars for units sold
        fig.add_trace(
            go.Bar(
                x=category_metrics['Category'],
                y=category_metrics['Units_Sold'],
                name="Units Sold",
                marker_color=COLORS['primary']
            ),
            secondary_y=False
        )
        
        # Add line for revenue
        fig.add_trace(
            go.Scatter(
                x=category_metrics['Category'],
                y=category_metrics['Revenue'],
                name="Revenue",
                marker_color=COLORS['secondary'],
                mode='lines+markers'
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title="Category Performance",
            template='plotly_white',
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_yaxes(title_text="Units Sold", secondary_y=False)
        fig.update_yaxes(title_text="Revenue ($)", secondary_y=True)
        
        return fig
    
    def _create_inventory_health(self, df: pd.DataFrame) -> go.Figure:
        """Create inventory health chart."""
        inventory_status = df.groupby('Category').agg({
            'Inventory_Level': 'mean',
            'Units_Sold': 'mean'
        }).reset_index()
        
        inventory_status['Inventory_Ratio'] = inventory_status['Inventory_Level'] / inventory_status['Units_Sold']
        
        # Create threshold markers
        optimal_ratio = 2  # Example threshold
        inventory_status['Status'] = pd.cut(
            inventory_status['Inventory_Ratio'],
            bins=[-np.inf, 1, 3, np.inf],
            labels=['Low', 'Optimal', 'High']
        )
        
        colors = {
            'Low': COLORS['danger'],
            'Optimal': COLORS['success'],
            'High': COLORS['warning']
        }
        
        fig = go.Figure()
        
        for status in colors:
            mask = inventory_status['Status'] == status
            fig.add_trace(go.Bar(
                x=inventory_status[mask]['Category'],
                y=inventory_status[mask]['Inventory_Ratio'],
                name=status,
                marker_color=colors[status]
            ))
        
        fig.update_layout(
            title="Inventory Health by Category",
            template='plotly_white',
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="Inventory to Sales Ratio"
        )
        
        return fig
    
    def _generate_insights(self, df: pd.DataFrame) -> List[Dict]:
        """Generate key insights from data."""
        insights = []
        
        try:
            # Top performing category
            top_category = df.groupby('Category')['Units_Sold'].sum().idxmax()
            top_category_sales = df[df['Category'] == top_category]['Units_Sold'].sum()
            insights.append({
                'text': f"Top category: {top_category} with {top_category_sales:,.0f} units sold",
                'icon': "fas fa-trophy",
                'color': COLORS['success']
            })
            
            # Inventory efficiency
            avg_inventory_ratio = (df['Inventory_Level'] / df['Units_Sold']).mean()
            if avg_inventory_ratio < 1.5:
                status = "Efficient"
                color = COLORS['success']
            elif avg_inventory_ratio < 3:
                status = "Moderate"
                color = COLORS['warning']
            else:
                status = "High"
                color = COLORS['danger']
            
            insights.append({
                'text': f"Inventory efficiency: {status} ({avg_inventory_ratio:.1f}x coverage)",
                'icon': "fas fa-boxes",
                'color': color
            })
            
            # Forecast accuracy
            accuracy = 1 - abs(df['Demand_Forecast'] - df['Units_Sold']).mean() / df['Units_Sold'].mean()
            insights.append({
                'text': f"Forecast accuracy: {accuracy:.1%}",
                'icon': "fas fa-chart-line",
                'color': COLORS['primary']
            })
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            insights.append({
                'text': "Error generating insights",
                'icon': "fas fa-exclamation-triangle",
                'color': COLORS['danger']
            })
        
        return insights
    
    def _get_filtered_data(
        self,
        start_date: str,
        end_date: str,
        categories: Optional[list] = None,
        products: Optional[list] = None
    ) -> pd.DataFrame:
        """Get filtered data with caching."""
        cache_key = f"{start_date}_{end_date}_{categories}_{products}"
        
        # Check cache
        if (
            cache_key in self._cache and
            self._last_update and
            (datetime.now() - self._last_update).seconds < self.config.cache_timeout
        ):
            return self._cache[cache_key]
        
        try:
            df = self.df.copy()
            
            # Apply filters
            if start_date and end_date:
                df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            
            if categories:
                df = df[df['Category'].isin(categories)]
            
            if products:
                df = df[df['Product_ID'].isin(products)]
            
            # Calculate revenue
            df['Revenue'] = df['Units_Sold'] * df['Price'] * (1 - df['Discount']/100)
            
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
        """Run the dashboard server."""
        try:
            logger.info(f"Starting dashboard server on port {port or self.config.port}")
            self.app.run_server(
                host=host,
                port=port or self.config.port,
                debug=debug or self.config.debug
            )
        except Exception as e:
            logger.error(f"Error running dashboard: {str(e)}")
            raise DashboardError(f"Error running dashboard: {str(e)}") 