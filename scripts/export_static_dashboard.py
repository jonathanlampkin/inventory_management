import os
import importlib
interactive_dashboard = importlib.import_module("scripts.06_interactive_dashboard")
app = interactive_dashboard.app
load_data = interactive_dashboard.load_data
import plotly.io as pio
from datetime import datetime
import plotly.express as px

def generate_static_html(data):
    """Generate static HTML with pre-rendered visualizations."""
    # Create a basic template
    html_template = """
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
                <p>Dashboard generated on {generation_date}</p>
            </footer>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        </body>
    </html>
    """
    
    # Generate overview metrics
    df = data['df']
    overview_metrics = f"""
    <div class="col-md-3">
        <div class="card bg-light">
            <div class="card-body">
                <h5>Total Products</h5>
                <h3>{df['Product'].nunique()}</h3>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card bg-light">
            <div class="card-body">
                <h5>Stores</h5>
                <h3>{df['Store'].nunique()}</h3>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card bg-light">
            <div class="card-body">
                <h5>Total Sales</h5>
                <h3>{df['Units Sold'].sum():,.0f}</h3>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card bg-light">
            <div class="card-body">
                <h5>Current Inventory</h5>
                <h3>{df.loc[df['Date'] == df['Date'].max(), 'Inventory Level'].sum():,.0f}</h3>
            </div>
        </div>
    </div>
    """
    
    # Generate sales chart
    sales_fig = generate_sales_chart(df)
    sales_chart = pio.to_html(sales_fig, full_html=False, include_plotlyjs='cdn')
    
    # Generate inventory chart
    inventory_fig = generate_inventory_chart(df)
    inventory_chart = pio.to_html(inventory_fig, full_html=False, include_plotlyjs='cdn')
    
    # Fill template
    html_content = html_template.format(
        overview_metrics=overview_metrics,
        sales_chart=sales_chart,
        inventory_chart=inventory_chart,
        generation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    return html_content

def generate_sales_chart(df):
    """Generate a sales trend chart."""
    # Aggregate sales by date
    sales_by_date = df.groupby('Date')['Units Sold'].sum().reset_index()
    
    # Create figure
    fig = px.line(sales_by_date, x='Date', y='Units Sold', 
                 title='Sales Trend Over Time',
                 labels={'Units Sold': 'Units Sold', 'Date': 'Date'})
    
    fig.update_layout(
        template='plotly_white',
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

def generate_inventory_chart(df):
    """Generate an inventory status chart."""
    # Get supply status distribution
    supply_status = df.groupby('Supply_Status').size().reset_index(name='Count')
    
    # Create figure
    colors = {'Undersupplied': '#E74C3C', 'Optimal': '#18BC9C', 'Oversupplied': '#3498DB'}
    
    fig = px.pie(supply_status, values='Count', names='Supply_Status', 
                title='Inventory Supply Status Distribution',
                color='Supply_Status',
                color_discrete_map=colors)
    
    fig.update_layout(
        template='plotly_white',
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

def export_static_dashboard():
    """Export the dashboard as static HTML files."""
    try:
        data = load_data()
        
        # Create directory for static export
        output_dir = "static_dashboard"
        os.makedirs(output_dir, exist_ok=True)
        
        # Export main dashboard
        with open(f"{output_dir}/index.html", "w") as f:
            static_html = generate_static_html(data)
            f.write(static_html)
        
        print(f"Static dashboard exported to {output_dir}/index.html")
        print("You can now deploy this HTML file to GitHub Pages or any static hosting service.")
    except Exception as e:
        print(f"Error exporting static dashboard: {str(e)}")

if __name__ == "__main__":
    export_static_dashboard() 