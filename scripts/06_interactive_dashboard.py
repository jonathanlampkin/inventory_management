import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import dash_table

# Function to load all necessary data
def load_data():
    """Load all necessary data for the dashboard."""
    try:
        # Load inventory data
        df = pd.read_csv("data/retail_store_inventory.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create necessary derived columns if they don't exist
        if 'Inventory_Sales_Ratio' not in df.columns:
            df['Inventory_Sales_Ratio'] = df['Inventory Level'] / df['Units Sold'].replace(0, np.nan)
        
        if 'Supply_Gap' not in df.columns:
            df['Supply_Gap'] = df['Inventory Level'] - df['Units Sold']
        
        if 'Forecast_Accuracy' not in df.columns:
            df['Forecast_Accuracy'] = 1 - abs(df['Demand Forecast'] - df['Units Sold']) / df['Demand Forecast'].replace(0, np.nan)
        
        # Create Supply_Status if it doesn't exist (using same logic as in the EDA script)
        if 'Supply_Status' not in df.columns:
            df['Optimal_Inventory'] = df['Units Sold'] * 1.5
            conditions = [
                (df['Inventory Level'] < df['Units Sold']),
                (df['Inventory Level'] >= df['Units Sold']) & (df['Inventory Level'] <= df['Optimal_Inventory']),
                (df['Inventory Level'] > df['Optimal_Inventory'])
            ]
            values = ['Undersupplied', 'Optimal', 'Oversupplied']
            df['Supply_Status'] = np.select(conditions, values, default='Unknown')
        
        # Load output files using corrected paths
        forecast_results = {}
        inventory_results = {}
        pricing_results = {}
        supply_results = {}
        
        # Fix path for supply analysis results - it's in eda/supply_analysis not supply_analysis
        try:
            with open("output/eda/supply_analysis/supply_analysis_results.json", 'r') as f:
                supply_results = json.load(f)
                # Add log to help with debugging
                print(f"Successfully loaded supply analysis data with {len(supply_results)} keys")
        except Exception as e:
            print(f"Error loading supply analysis data: {str(e)}")
        
        try:
            with open("output/forecasting/forecast_results.json", 'r') as f:
                forecast_results = json.load(f)
                print(f"Successfully loaded forecast data with {len(forecast_results)} keys")
        except Exception as e:
            print(f"Error loading forecast data: {str(e)}")
        
        try:
            with open("output/inventory_optimization/inventory_optimization_results.json", 'r') as f:
                inventory_results = json.load(f)
                print(f"Successfully loaded inventory data with {len(inventory_results)} keys")
        except Exception as e:
            print(f"Error loading inventory data: {str(e)}")
        
        try:
            with open("output/pricing_strategy/pricing_strategy_results.json", 'r') as f:
                pricing_results = json.load(f)
                print(f"Successfully loaded pricing data with {len(pricing_results)} keys")
        except Exception as e:
            print(f"Error loading pricing data: {str(e)}")
        
        return {
            'df': df,
            'forecast_results': forecast_results,
            'inventory_results': inventory_results,
            'pricing_results': pricing_results,
            'supply_results': supply_results
        }
    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        return {
            'df': pd.DataFrame(),
            'forecast_results': {},
            'inventory_results': {},
            'pricing_results': {},
            'supply_results': {}
        }

# Load data
data = load_data()
df = data['df']

# Create the dashboard app
app = dash.Dash(
    __name__, 
    title="Retail Inventory Management Dashboard",
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)

# Define colors for consistent styling
colors = {
    'background': '#f9f9f9',
    'text': '#333333',
    'primary': '#2C3E50',
    'secondary': '#18BC9C',
    'accent': '#3498DB',
    'warning': '#F39C12',
    'danger': '#E74C3C'
}

# Create filter components
filter_components = dbc.Card([
    dbc.CardHeader("Filters", className="text-white bg-primary"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("Date Range:"),
                dcc.DatePickerRange(
                    id='date-filter',
                    min_date_allowed=df['Date'].min() if not df.empty else None,
                    max_date_allowed=df['Date'].max() if not df.empty else None,
                    start_date=df['Date'].min() if not df.empty else None,
                    end_date=df['Date'].max() if not df.empty else None
                )
            ], width=6),
            dbc.Col([
                html.Label("Category:"),
                dcc.Dropdown(
                    id='category-filter',
                    options=[{'label': cat, 'value': cat} for cat in df['Category'].unique()] if not df.empty else [],
                    multi=True,
                    placeholder="All Categories"
                )
            ], width=3),
            dbc.Col([
                html.Label("Store:"),
                dcc.Dropdown(
                    id='store-filter',
                    options=[{'label': store, 'value': store} for store in df['Store ID'].unique()] if not df.empty else [],
                    multi=True,
                    placeholder="All Stores"
                )
            ], width=3)
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Region:"),
                dcc.Dropdown(
                    id='region-filter',
                    options=[{'label': region, 'value': region} for region in df['Region'].unique()] if not df.empty else [],
                    multi=True,
                    placeholder="All Regions"
                )
            ], width=3),
            dbc.Col([
                html.Label("Product:"),
                dcc.Dropdown(
                    id='product-filter',
                    options=[{'label': prod, 'value': prod} for prod in df['Product ID'].unique()] if not df.empty else [],
                    multi=True,
                    placeholder="All Products"
                )
            ], width=3),
            dbc.Col([
                html.Button('Apply Filters', id='apply-filters', n_clicks=0, 
                           className="btn btn-primary mt-4")
            ], width=3),
            dbc.Col([
                html.Button('Reset Filters', id='reset-filters', n_clicks=0, 
                           className="btn btn-secondary mt-4")
            ], width=3)
        ], className="mt-3")
    ])
], className="mb-4")

# Create the layout for the dashboard
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Retail Inventory Management Dashboard", 
                   className="text-primary text-center my-4")
        ], width=12)
    ]),
    
    # Filters Section
    dbc.Row([
        dbc.Col([
            filter_components
        ], width=12)
    ]),
    
    # KPI Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Total Sales", className="text-white bg-primary"),
                dbc.CardBody([
                    html.H3(id="total-sales-value", className="card-title"),
                    html.P("Units", className="card-text text-muted")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Average Inventory", className="text-white bg-success"),
                dbc.CardBody([
                    html.H3(id="avg-inventory-value", className="card-title"),
                    html.P("Units", className="card-text text-muted")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Forecast Accuracy", className="text-white bg-info"),
                dbc.CardBody([
                    html.H3(id="forecast-accuracy-value", className="card-title"),
                    html.P("MAPE", className="card-text text-muted")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Inventory Turn", className="text-white bg-warning"),
                dbc.CardBody([
                    html.H3(id="inventory-turn-value", className="card-title"),
                    html.P("Turns per year", className="card-text text-muted")
                ])
            ])
        ], width=3)
    ], className="my-4"),
    
    # Tabs for Analysis Sections
    dbc.Tabs([
        # Sales Analysis Tab
        dbc.Tab(label="Sales Analysis", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Sales Trend", className="text-primary mt-3"),
                    dcc.Graph(id="sales-trend-graph")
                ], width=8),
                dbc.Col([
                    html.H4("Sales by Category", className="text-primary mt-3"),
                    dcc.Graph(id="sales-by-category-graph")
                ], width=4)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("Top Selling Products", className="text-primary mt-3"),
                    dash_table.DataTable(
                        id='top-products-table',
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'left',
                            'padding': '8px',
                            'minWidth': '100px'
                        },
                        style_header={
                            'backgroundColor': colors['primary'],
                            'color': 'white',
                            'fontWeight': 'bold'
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': '#f9f9f9'
                            }
                        ],
                        page_size=5
                    )
                ], width=12)
            ])
        ]),
        
        # Inventory Analysis Tab
        dbc.Tab(label="Inventory Analysis", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Inventory Levels", className="text-primary mt-3"),
                    dcc.Graph(id="inventory-levels-graph")
                ], width=6),
                dbc.Col([
                    html.H4("Supply Status", className="text-primary mt-3"),
                    dcc.Graph(id="supply-status-graph")
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("Optimal Inventory Parameters", className="text-primary mt-3"),
                    html.Label("Select Product:"),
                    dcc.Dropdown(
                        id='inventory-product-selector',
                        options=[{'label': prod, 'value': prod} for prod in df['Product ID'].unique()] if not df.empty else [],
                        value=df['Product ID'].iloc[0] if not df.empty else None
                    ),
                    html.Div(id="inventory-parameters-display", className="mt-3")
                ], width=12)
            ])
        ]),
        
        # Forecast Analysis Tab
        dbc.Tab(label="Forecast Analysis", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Sales Forecast", className="text-primary mt-3"),
                    dcc.Graph(id="forecast-graph")
                ], width=6),
                dbc.Col([
                    html.H4("Forecast vs Actual", className="text-primary mt-3"),
                    dcc.Graph(id="forecast-vs-actual-graph")
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("Feature Importance", className="text-primary mt-3"),
                    dcc.Graph(id="feature-importance-graph")
                ], width=12)
            ])
        ]),
        
        # Pricing Analysis Tab
        dbc.Tab(label="Pricing Analysis", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Price Elasticity", className="text-primary mt-3"),
                    dcc.Graph(id="price-elasticity-graph")
                ], width=6),
                dbc.Col([
                    html.H4("Discount Impact", className="text-primary mt-3"),
                    dcc.Graph(id="discount-impact-graph")
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("Competitive Pricing Analysis", className="text-primary mt-3"),
                    dcc.Graph(id="competitive-pricing-graph")
                ], width=12)
            ])
        ])
    ]),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P(f"Dashboard generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                  className="text-muted text-center")
        ], width=12)
    ])
], fluid=True, style={"backgroundColor": colors['background']})

# Callbacks for interactivity

# Callback to filter data
@app.callback(
    [Output('total-sales-value', 'children'),
     Output('avg-inventory-value', 'children'),
     Output('forecast-accuracy-value', 'children'),
     Output('inventory-turn-value', 'children')],
    [Input('apply-filters', 'n_clicks')],
    [State('date-filter', 'start_date'),
     State('date-filter', 'end_date'),
     State('category-filter', 'value'),
     State('store-filter', 'value'),
     State('region-filter', 'value'),
     State('product-filter', 'value')]
)
def update_kpis(n_clicks, start_date, end_date, categories, stores, regions, products):
    """Update KPI values based on filters."""
    filtered_df = df.copy()
    
    # Apply date filter
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]
    
    # Apply category filter
    if categories and len(categories) > 0:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    
    # Apply store filter
    if stores and len(stores) > 0:
        filtered_df = filtered_df[filtered_df['Store ID'].isin(stores)]
    
    # Apply region filter
    if regions and len(regions) > 0:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    # Apply product filter
    if products and len(products) > 0:
        filtered_df = filtered_df[filtered_df['Product ID'].isin(products)]
    
    # Calculate KPIs
    total_sales = filtered_df['Units Sold'].sum()
    avg_inventory = filtered_df['Inventory Level'].mean()
    
    # Calculate forecast accuracy
    if 'Demand Forecast' in filtered_df.columns:
        mape = 100 * abs(filtered_df['Units Sold'] - filtered_df['Demand Forecast']).mean() / filtered_df['Units Sold'].mean()
        forecast_accuracy = f"{100 - mape:.1f}%"
    else:
        forecast_accuracy = "N/A"
    
    # Calculate inventory turn
    if not filtered_df.empty:
        inventory_turn = total_sales / avg_inventory
    else:
        inventory_turn = 0
    
    return f"{total_sales:,.0f}", f"{avg_inventory:.1f}", forecast_accuracy, f"{inventory_turn:.1f}"

# Callback for sales trend graph
@app.callback(
    Output('sales-trend-graph', 'figure'),
    [Input('apply-filters', 'n_clicks')],
    [State('date-filter', 'start_date'),
     State('date-filter', 'end_date'),
     State('category-filter', 'value'),
     State('store-filter', 'value'),
     State('region-filter', 'value'),
     State('product-filter', 'value')]
)
def update_sales_trend(n_clicks, start_date, end_date, categories, stores, regions, products):
    """Update sales trend graph based on filters."""
    filtered_df = df.copy()
    
    # Apply filters
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]
    
    if categories and len(categories) > 0:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    
    if stores and len(stores) > 0:
        filtered_df = filtered_df[filtered_df['Store ID'].isin(stores)]
    
    if regions and len(regions) > 0:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    if products and len(products) > 0:
        filtered_df = filtered_df[filtered_df['Product ID'].isin(products)]
    
    # Aggregate by date
    daily_sales = filtered_df.groupby('Date')['Units Sold'].sum().reset_index()
    
    # Create figure
    fig = px.line(
        daily_sales, 
        x='Date', 
        y='Units Sold',
        title='Daily Sales Trend',
        labels={'Date': 'Date', 'Units Sold': 'Units Sold'},
        template='plotly_white'
    )
    
    fig.update_layout(
        title_font=dict(size=20),
        title_x=0.5,
        xaxis_title="Date",
        yaxis_title="Units Sold",
        legend_title="Legend",
        xaxis_rangeslider_visible=True
    )
    
    return fig

# Callback for sales by category graph
@app.callback(
    Output('sales-by-category-graph', 'figure'),
    [Input('apply-filters', 'n_clicks')],
    [State('date-filter', 'start_date'),
     State('date-filter', 'end_date'),
     State('store-filter', 'value'),
     State('region-filter', 'value'),
     State('product-filter', 'value')]
)
def update_sales_by_category(n_clicks, start_date, end_date, stores, regions, products):
    """Update sales by category graph based on filters."""
    filtered_df = df.copy()
    
    # Apply filters
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]
    
    if stores and len(stores) > 0:
        filtered_df = filtered_df[filtered_df['Store ID'].isin(stores)]
    
    if regions and len(regions) > 0:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    if products and len(products) > 0:
        filtered_df = filtered_df[filtered_df['Product ID'].isin(products)]
    
    # Aggregate by category
    category_sales = filtered_df.groupby('Category')['Units Sold'].sum().reset_index()
    
    # Create figure
    fig = px.pie(
        category_sales, 
        values='Units Sold', 
        names='Category',
        title='Sales by Category',
        hole=0.4,
        template='plotly_white'
    )
    
    fig.update_layout(
        title_font=dict(size=20),
        title_x=0.5,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

# Callback for top products table
@app.callback(
    Output('top-products-table', 'data'),
    [Input('apply-filters', 'n_clicks')],
    [State('date-filter', 'start_date'),
     State('date-filter', 'end_date'),
     State('category-filter', 'value'),
     State('store-filter', 'value'),
     State('region-filter', 'value')]
)
def update_top_products(n_clicks, start_date, end_date, categories, stores, regions):
    """Update top products table based on filters."""
    filtered_df = df.copy()
    
    # Apply filters
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]
    
    if categories and len(categories) > 0:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    
    if stores and len(stores) > 0:
        filtered_df = filtered_df[filtered_df['Store ID'].isin(stores)]
    
    if regions and len(regions) > 0:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    # Aggregate by product
    product_sales = filtered_df.groupby(['Product ID', 'Category']).agg({
        'Units Sold': 'sum',
        'Inventory Level': 'mean',
        'Price': 'mean',
        'Discount': 'mean'
    }).reset_index()
    
    # Get top 10 products by sales
    top_products = product_sales.sort_values('Units Sold', ascending=False).head(10)
    
    # Format the data
    top_products['Units Sold'] = top_products['Units Sold'].round(0).astype(int)
    top_products['Avg Inventory'] = top_products['Inventory Level'].round(1)
    top_products['Avg Price'] = top_products['Price'].round(2)
    top_products['Avg Discount'] = top_products['Discount'].round(2)
    
    # Return data for table
    return top_products[['Product ID', 'Category', 'Units Sold', 'Avg Inventory', 'Avg Price', 'Avg Discount']].to_dict('records')

# Callback for resetting filters
@app.callback(
    [Output('date-filter', 'start_date'),
     Output('date-filter', 'end_date'),
     Output('category-filter', 'value'),
     Output('store-filter', 'value'),
     Output('region-filter', 'value'),
     Output('product-filter', 'value')],
    [Input('reset-filters', 'n_clicks')]
)
def reset_filters(n_clicks):
    """Reset all filters to default values."""
    return (
        df['Date'].min() if not df.empty else None,
        df['Date'].max() if not df.empty else None,
        [], [], [], []
    )

# Callback for inventory parameters display
@app.callback(
    Output('inventory-parameters-display', 'children'),
    [Input('inventory-product-selector', 'value')]
)
def display_inventory_parameters(product_id):
    """Display inventory optimization parameters for selected product."""
    if not product_id or not data['inventory_results']:
        return html.Div("No data available")
    
    try:
        # Get inventory parameters for the selected product
        inventory_params = data['inventory_results']['product_inventory_params']
        
        # Create table with inventory parameters
        inventory_table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Parameter"),
                html.Th("Value")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td("Daily Demand (Mean)"),
                    html.Td(f"{inventory_params['demand_mean'].get(product_id, 'N/A'):.2f}")
                ]),
                html.Tr([
                    html.Td("Daily Demand (Std Dev)"),
                    html.Td(f"{inventory_params['demand_std'].get(product_id, 'N/A'):.2f}")
                ]),
                html.Tr([
                    html.Td("Annual Demand"),
                    html.Td(f"{inventory_params['annual_demand'].get(product_id, 'N/A'):,.0f}")
                ]),
                html.Tr([
                    html.Td("Safety Stock"),
                    html.Td(f"{inventory_params['safety_stock'].get(product_id, 'N/A'):,.0f}")
                ]),
                html.Tr([
                    html.Td("Reorder Point"),
                    html.Td(f"{inventory_params['reorder_point'].get(product_id, 'N/A'):,.0f}")
                ]),
                html.Tr([
                    html.Td("Economic Order Quantity (EOQ)"),
                    html.Td(f"{inventory_params['eoq'].get(product_id, 'N/A'):,.0f}")
                ])
            ])
        ], bordered=True, hover=True, striped=True)
        
        return inventory_table
    
    except Exception as e:
        return html.Div(f"Error retrieving data: {str(e)}")

# Callback for inventory levels graph
@app.callback(
    Output('inventory-levels-graph', 'figure'),
    [Input('apply-filters', 'n_clicks')],
    [State('date-filter', 'start_date'),
     State('date-filter', 'end_date'),
     State('category-filter', 'value'),
     State('store-filter', 'value'),
     State('region-filter', 'value'),
     State('product-filter', 'value')]
)
def update_inventory_graph(n_clicks, start_date, end_date, categories, stores, regions, products):
    """Update inventory levels graph based on filters."""
    filtered_df = df.copy()
    
    # Apply filters
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]
    
    if categories and len(categories) > 0:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    
    if stores and len(stores) > 0:
        filtered_df = filtered_df[filtered_df['Store ID'].isin(stores)]
    
    if regions and len(regions) > 0:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    if products and len(products) > 0:
        filtered_df = filtered_df[filtered_df['Product ID'].isin(products)]
    
    # Aggregate inventory levels by date
    daily_inventory = filtered_df.groupby('Date')['Inventory Level'].mean().reset_index()
    daily_sales = filtered_df.groupby('Date')['Units Sold'].sum().reset_index()
    
    # Merge the datasets
    daily_data = pd.merge(daily_inventory, daily_sales, on='Date')
    
    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=daily_data['Date'],
            y=daily_data['Inventory Level'],
            name="Avg Inventory Level",
            line=dict(color=colors['primary'], width=2)
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_data['Date'],
            y=daily_data['Units Sold'],
            name="Daily Sales",
            line=dict(color=colors['accent'], width=2, dash='dash')
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        title="Inventory Levels and Sales",
        title_font=dict(size=20),
        title_x=0.5,
        xaxis_title="Date",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text="Inventory Level", secondary_y=False)
    fig.update_yaxes(title_text="Units Sold", secondary_y=True)
    
    return fig

# Define function to run the server
def run_server():
    """Run the dashboard server."""
    print("Starting interactive dashboard server...")
    print("Navigate to http://127.0.0.1:8050/ in your web browser")
    print("Press Ctrl+C to stop the server")
    app.run_server(debug=False, host='0.0.0.0', port=8050)

# Main function to create and save dashboard
def main():
    """Main function to create the interactive dashboard."""
    print("Starting Interactive Dashboard Generation...")
    
    output_dir = "output/dashboard"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dashboard to HTML file (static version)
    app_string = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Retail Inventory Management Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #f9f9f9;
                    color: #333333;
                }
                .card {
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    transition: all 0.3s ease;
                }
                .card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
                }
                .nav-tabs .nav-link {
                    color: #2C3E50;
                    font-weight: 500;
                }
                .nav-tabs .nav-link.active {
                    color: #18BC9C;
                    font-weight: 600;
                }
            </style>
        </head>
        <body>
            <div class="container-fluid py-4">
                <h1 class="text-center text-primary mb-5">Retail Inventory Management Dashboard</h1>
                <p class="alert alert-info">
                    <b>Note:</b> This is a static preview of the dashboard. To interact with all features, please run the dashboard locally using:
                    <br><code>python scripts/06_interactive_dashboard.py</code>
                </p>
                
                <div class="row">
                    <div class="col-md-12">
                        <img src="../visualizations/dashboard_preview.png" class="img-fluid" alt="Dashboard Preview">
                    </div>
                </div>
                
                <div class="row mt-5">
                    <div class="col-md-12">
                        <h3>Interactive Features</h3>
                        <ul>
                            <li>Filter by date range, category, store, region, and product</li>
                            <li>Drill down into specific metrics and time periods</li>
                            <li>View detailed inventory optimization parameters</li>
                            <li>Compare forecast vs actual sales</li>
                            <li>Analyze price elasticity and competitive positioning</li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <footer class="container-fluid py-3 text-center text-muted">
                <hr>
                <p>Dashboard generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </footer>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "retail_dashboard.html"), "w") as f:
        f.write(app_string)
    
    print(f"Interactive dashboard created and saved to {output_dir}/retail_dashboard.html")
    print("To run the dashboard locally, execute: python scripts/06_interactive_dashboard.py")
    
    # Return the app for local testing
    return app

if __name__ == "__main__":
    app = main()
    
    # Check if command line argument was passed to run server
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--serve':
        run_server()
    else:
        print("\nTo launch the interactive dashboard, run:")
        print("python scripts/06_interactive_dashboard.py --serve") 