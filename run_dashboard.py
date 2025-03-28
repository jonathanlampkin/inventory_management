"""Script to run the inventory management dashboard."""

import pandas as pd
from src.visualization.dashboard import Dashboard, DashboardConfig

def main():
    """Run the dashboard."""
    # Load the processed data
    df = pd.read_csv('data/processed_inventory.csv')
    
    # Create dashboard configuration
    config = DashboardConfig(
        refresh_interval=3600,  # 1 hour
        max_points=10000,
        cache_timeout=300,  # 5 minutes
        port=8050,
        debug=True
    )
    
    # Create dashboard instance
    dashboard = Dashboard(
        results={'data': df},
        config=config
    )
    
    # Run the dashboard
    print("Starting dashboard server at http://localhost:8050")
    dashboard.run(host="0.0.0.0", port=8050, debug=True)

if __name__ == "__main__":
    main() 