import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
from scipy.stats import norm

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_data(filepath):
    """Load the dataset and process dates."""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def create_output_dirs():
    """Create necessary output directories."""
    dirs = ['output/inventory_optimization/visualizations', 
            'output/inventory_optimization/recommendations']
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def calculate_safety_stock(demand_mean, demand_std, lead_time, service_level=0.95):
    """Calculate safety stock based on demand variability and service level."""
    # Convert service level to z-score
    z_score = norm.ppf(service_level)
    
    # Calculate safety stock
    safety_stock = z_score * demand_std * np.sqrt(lead_time)
    
    return safety_stock

def calculate_reorder_point(demand_mean, demand_std, lead_time, service_level=0.95):
    """Calculate reorder point based on lead time demand and safety stock."""
    # Expected demand during lead time
    lead_time_demand = demand_mean * lead_time
    
    # Safety stock
    safety_stock = calculate_safety_stock(demand_mean, demand_std, lead_time, service_level)
    
    # Reorder point = lead time demand + safety stock
    reorder_point = lead_time_demand + safety_stock
    
    return reorder_point, safety_stock

def calculate_eoq(annual_demand, ordering_cost, holding_cost_pct, unit_cost):
    """Calculate Economic Order Quantity (EOQ)."""
    # Annual holding cost per unit
    annual_holding_cost = unit_cost * holding_cost_pct
    
    # EOQ formula
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / annual_holding_cost)
    
    return eoq

def optimize_inventory(df, output_dir):
    """Optimize inventory levels using inventory models."""
    results = {}
    
    # Define parameters for optimization
    lead_time_days = 7  # Assuming 7 days lead time for all products
    service_level = 0.95  # 95% service level
    ordering_cost = 50  # Flat ordering cost
    holding_cost_pct = 0.25  # Annual holding cost as percentage of unit cost
    
    # Calculate historical demand statistics by product and store
    product_store_stats = df.groupby(['Product ID', 'Store ID']).agg({
        'Units Sold': ['mean', 'std', 'sum'],
        'Price': 'mean'
    })
    
    product_store_stats.columns = ['demand_mean', 'demand_std', 'annual_demand', 'unit_cost']
    product_store_stats.reset_index(inplace=True)
    
    # Adjust for time period if less than a year
    days_in_data = (df['Date'].max() - df['Date'].min()).days + 1
    if days_in_data < 365:
        scaling_factor = 365 / days_in_data
        product_store_stats['annual_demand'] = product_store_stats['annual_demand'] * scaling_factor
    
    # Calculate optimal inventory parameters
    product_store_stats['safety_stock'] = product_store_stats.apply(
        lambda x: calculate_safety_stock(x['demand_mean'], x['demand_std'], lead_time_days, service_level),
        axis=1
    )
    
    product_store_stats['reorder_point'], _ = zip(*product_store_stats.apply(
        lambda x: calculate_reorder_point(x['demand_mean'], x['demand_std'], lead_time_days, service_level),
        axis=1
    ))
    
    product_store_stats['eoq'] = product_store_stats.apply(
        lambda x: calculate_eoq(x['annual_demand'], ordering_cost, holding_cost_pct, x['unit_cost']),
        axis=1
    )
    
    # Round to nearest integer for inventory values
    for col in ['safety_stock', 'reorder_point', 'eoq']:
        product_store_stats[col] = product_store_stats[col].round().astype(int)
    
    # Get aggregated statistics for each product across all stores
    product_stats = product_store_stats.groupby('Product ID').agg({
        'demand_mean': 'sum',
        'demand_std': 'sum',
        'annual_demand': 'sum',
        'safety_stock': 'sum',
        'reorder_point': 'sum',
        'eoq': 'sum',
        'unit_cost': 'mean'
    })
    
    # Visualize safety stock by product
    top_products = product_stats.sort_values('annual_demand', ascending=False).head(10)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_products.index, y=top_products['safety_stock'])
    plt.title('Safety Stock for Top 10 Products by Demand', fontsize=16)
    plt.ylabel('Safety Stock (Units)')
    plt.xlabel('Product ID')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/visualizations/top_products_safety_stock.png")
    plt.close()
    
    # Visualize reorder points
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_products.index, y=top_products['reorder_point'])
    plt.title('Reorder Points for Top 10 Products by Demand', fontsize=16)
    plt.ylabel('Reorder Point (Units)')
    plt.xlabel('Product ID')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/visualizations/top_products_reorder_points.png")
    plt.close()
    
    # Visualize EOQ
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_products.index, y=top_products['eoq'])
    plt.title('Economic Order Quantity for Top 10 Products by Demand', fontsize=16)
    plt.ylabel('EOQ (Units)')
    plt.xlabel('Product ID')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/visualizations/top_products_eoq.png")
    plt.close()
    
    # Store results
    results = {
        'product_inventory_params': product_stats.to_dict(),
        'store_product_inventory_params': product_store_stats.to_dict(),
        'optimization_parameters': {
            'lead_time_days': lead_time_days,
            'service_level': service_level,
            'ordering_cost': ordering_cost,
            'holding_cost_pct': holding_cost_pct
        }
    }
    
    return results, product_store_stats

def generate_optimization_report(results, product_store_stats, output_dir):
    """Generate a report with inventory optimization recommendations."""
    with open(f"{output_dir}/recommendations/inventory_optimization_report.md", "w") as f:
        f.write("# Inventory Optimization Recommendations\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Optimization Parameters\n\n")
        params = results['optimization_parameters']
        f.write(f"- **Lead Time**: {params['lead_time_days']} days\n")
        f.write(f"- **Service Level**: {params['service_level'] * 100}%\n")
        f.write(f"- **Ordering Cost**: ${params['ordering_cost']}\n")
        f.write(f"- **Annual Holding Cost**: {params['holding_cost_pct'] * 100}% of unit cost\n\n")
        
        f.write("## Top 10 Products by Demand\n\n")
        top_products = product_store_stats.groupby('Product ID')['annual_demand'].sum().sort_values(ascending=False).head(10)
        
        f.write("| Product ID | Annual Demand | Safety Stock | Reorder Point | EOQ |\n")
        f.write("|------------|---------------|--------------|---------------|-----|\n")
        
        for product_id in top_products.index:
            # Get aggregated stats for this product
            product_data = product_store_stats[product_store_stats['Product ID'] == product_id]
            safety_stock = product_data['safety_stock'].sum()
            reorder_point = product_data['reorder_point'].sum()
            eoq = product_data['eoq'].sum()
            annual_demand = product_data['annual_demand'].sum()
            
            f.write(f"| {product_id} | {annual_demand:.0f} | {safety_stock:.0f} | {reorder_point:.0f} | {eoq:.0f} |\n")
        f.write("\n")
        
        # Store-specific recommendations
        f.write("## Store-Specific Recommendations\n\n")
        for store_id in product_store_stats['Store ID'].unique():
            f.write(f"### Store {store_id}\n\n")
            store_data = product_store_stats[product_store_stats['Store ID'] == store_id]
            top_store_products = store_data.sort_values('annual_demand', ascending=False).head(5)
            
            f.write("| Product ID | Annual Demand | Safety Stock | Reorder Point | EOQ |\n")
            f.write("|------------|---------------|--------------|---------------|-----|\n")
            
            for _, row in top_store_products.iterrows():
                f.write(f"| {row['Product ID']} | {row['annual_demand']:.0f} | {row['safety_stock']:.0f} | {row['reorder_point']:.0f} | {row['eoq']:.0f} |\n")
            f.write("\n")
        
        # General recommendations
        f.write("## Implementation Recommendations\n\n")
        f.write("1. **Implement Reorder Points**: Configure inventory systems to alert when stock reaches reorder points\n")
        f.write("2. **Order in EOQ Quantities**: Place orders in economic order quantities to minimize total costs\n")
        f.write("3. **Maintain Safety Stock**: Ensure safety stock levels are maintained to buffer against demand uncertainty\n")
        f.write("4. **Review Periodically**: Review and adjust these parameters quarterly based on updated demand patterns\n")
        f.write("5. **Monitor Service Levels**: Track stockouts and adjust safety stock if service levels are not being met\n")

def main():
    """Main function to run the inventory optimization."""
    # Set paths
    data_path = "data/retail_store_inventory.csv"
    output_dir = "output/inventory_optimization"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = load_data(data_path)
    
    # Create output directories
    print("Creating output directories...")
    create_output_dirs()
    
    # Optimize inventory
    print("Calculating optimal inventory parameters...")
    results, product_store_stats = optimize_inventory(df, output_dir)
    
    # Generate report
    print("Generating optimization report...")
    generate_optimization_report(results, product_store_stats, output_dir)
    
    # Save results as JSON
    print("Saving results...")
    with open(f"{output_dir}/inventory_optimization_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Optimization complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main() 