import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import os
import json
from datetime import datetime

def load_data(file_path):
    """Load the dataset and return a pandas DataFrame."""
    return pd.read_csv(file_path)

def generate_basic_info(df):
    """Generate basic information about the dataset."""
    basic_info = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "column_names": list(df.columns),
        "column_dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": df.isnull().sum().to_dict(),
        "duplicated_rows": int(df.duplicated().sum())
    }
    return basic_info

def generate_temporal_analysis(df):
    """Analyze temporal aspects of the data."""
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract time components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    # Time range
    time_info = {
        "min_date": df['Date'].min().strftime('%Y-%m-%d'),
        "max_date": df['Date'].max().strftime('%Y-%m-%d'),
        "date_range_days": (df['Date'].max() - df['Date'].min()).days + 1,
        "unique_dates": df['Date'].nunique()
    }
    
    # Create month-year summary - convert tuple keys to strings
    monthly_counts = df.groupby([df['Date'].dt.year, df['Date'].dt.month]).size()
    monthly_distribution = {}
    for (year, month), count in monthly_counts.items():
        monthly_distribution[f"{year}-{month:02d}"] = int(count)
    
    time_info["monthly_distribution"] = monthly_distribution
    
    return time_info, df

def generate_categorical_analysis(df):
    """Analyze categorical columns."""
    categorical_columns = ['Store ID', 'Product ID', 'Category', 'Region', 
                           'Weather Condition', 'Holiday/Promotion', 'Seasonality']
    
    categorical_info = {}
    for col in categorical_columns:
        value_counts = df[col].value_counts().to_dict()
        unique_count = df[col].nunique()
        
        categorical_info[col] = {
            "unique_values": unique_count,
            "top_values": dict(list(value_counts.items())[:10]) if unique_count > 10 else value_counts
        }
    
    return categorical_info

def generate_numerical_analysis(df):
    """Analyze numerical columns."""
    numerical_columns = ['Inventory Level', 'Units Sold', 'Units Ordered', 
                         'Demand Forecast', 'Price', 'Discount', 'Competitor Pricing']
    
    numerical_info = {}
    for col in numerical_columns:
        stats = df[col].describe().to_dict()
        numerical_info[col] = {
            "min": stats['min'],
            "max": stats['max'],
            "mean": stats['mean'],
            "median": stats['50%'],
            "std": stats['std'],
            "range": stats['max'] - stats['min']
        }
    
    return numerical_info

def analyze_relationships(df):
    """Analyze relationships between key variables."""
    relationship_info = {}
    
    # Correlation matrix for numerical variables
    numerical_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 
                      'Demand Forecast', 'Price', 'Discount', 'Competitor Pricing']
    correlation_matrix = df[numerical_cols].corr().round(2).to_dict()
    relationship_info["correlation_matrix"] = correlation_matrix
    
    # Category-based statistics
    category_stats = df.groupby('Category')[numerical_cols].mean().round(2).to_dict()
    relationship_info["category_stats"] = category_stats
    
    # Region-based statistics
    region_stats = df.groupby('Region')[numerical_cols].mean().round(2).to_dict()
    relationship_info["region_stats"] = region_stats
    
    # Seasonality statistics
    seasonality_stats = df.groupby('Seasonality')[numerical_cols].mean().round(2).to_dict()
    relationship_info["seasonality_stats"] = seasonality_stats
    
    # Weather impact
    weather_stats = df.groupby('Weather Condition')[numerical_cols].mean().round(2).to_dict()
    relationship_info["weather_stats"] = weather_stats
    
    # Holiday/Promotion impact
    holiday_stats = df.groupby('Holiday/Promotion')[numerical_cols].mean().round(2).to_dict()
    relationship_info["holiday_stats"] = holiday_stats
    
    return relationship_info

def generate_insights(df):
    """Generate key insights from the data."""
    insights = []
    
    # Product performance
    top_sold_products = df.groupby('Product ID')['Units Sold'].sum().nlargest(5).to_dict()
    insights.append(f"Top 5 products by units sold: {top_sold_products}")
    
    # Store performance
    top_stores = df.groupby('Store ID')['Units Sold'].sum().nlargest(5).to_dict()
    insights.append(f"Top 5 stores by units sold: {top_stores}")
    
    # Category performance
    category_performance = df.groupby('Category')['Units Sold'].sum().sort_values(ascending=False).to_dict()
    insights.append(f"Category sales performance: {category_performance}")
    
    # Discount effectiveness
    discount_effect = df.groupby('Discount')['Units Sold'].mean().round(2).to_dict()
    insights.append(f"Average units sold by discount level: {discount_effect}")
    
    # Price vs Units Sold
    price_correlation = df['Price'].corr(df['Units Sold']).round(3)
    insights.append(f"Correlation between Price and Units Sold: {price_correlation}")
    
    # Seasonal patterns
    seasonal_sales = df.groupby('Seasonality')['Units Sold'].mean().round(2).to_dict()
    insights.append(f"Average units sold by season: {seasonal_sales}")
    
    # Weather impact
    weather_impact = df.groupby('Weather Condition')['Units Sold'].mean().round(2).to_dict()
    insights.append(f"Average units sold by weather condition: {weather_impact}")
    
    # Inventory management effectiveness
    inventory_sales_ratio = (df['Inventory Level'] / df['Units Sold']).mean().round(2)
    insights.append(f"Average inventory to sales ratio: {inventory_sales_ratio}")
    
    # Forecast accuracy
    forecast_accuracy = (1 - abs(df['Demand Forecast'] - df['Units Sold']) / df['Demand Forecast']).mean().round(3)
    insights.append(f"Average forecast accuracy: {forecast_accuracy}")
    
    return insights

def save_profile_report(df, output_path):
    """Generate and save a profile report."""
    profile = ProfileReport(df, title="Retail Store Inventory Data Profile")
    profile.to_file(output_path)
    return f"Profile report saved to {output_path}"

def save_visualizations(df, output_dir):
    """Generate and save key visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Units Sold Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Units Sold'], kde=True)
    plt.title('Distribution of Units Sold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/units_sold_dist.png")
    plt.close()
    
    # Units Sold by Category
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Category', y='Units Sold', data=df, estimator=sum, ci=None)
    plt.title('Total Units Sold by Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/units_sold_by_category.png")
    plt.close()
    
    # Price vs Units Sold
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Price', y='Units Sold', data=df, alpha=0.5)
    plt.title('Price vs Units Sold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/price_vs_units_sold.png")
    plt.close()
    
    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    numerical_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 
                       'Demand Forecast', 'Price', 'Discount', 'Competitor Pricing']
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()
    
    # Sales Trend
    monthly_sales = df.groupby(pd.Grouper(key='Date', freq='M'))['Units Sold'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sales['Date'], monthly_sales['Units Sold'], marker='o')
    plt.title('Monthly Sales Trend')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/monthly_sales_trend.png")
    plt.close()
    
    # Inventory vs Sales
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Inventory Level', y='Units Sold', data=df, alpha=0.5)
    plt.title('Inventory Level vs Units Sold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/inventory_vs_sales.png")
    plt.close()
    
    # Forecast vs Actual
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Demand Forecast', y='Units Sold', data=df, alpha=0.5)
    plt.plot([df['Demand Forecast'].min(), df['Demand Forecast'].max()], 
             [df['Demand Forecast'].min(), df['Demand Forecast'].max()], 
             'r--', alpha=0.8)
    plt.title('Demand Forecast vs Actual Units Sold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/forecast_vs_actual.png")
    plt.close()
    
    return f"Visualizations saved to {output_dir}"

def main():
    # Set file paths
    data_path = "data/retail_store_inventory.csv"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = load_data(data_path)
    
    # Generate analysis
    print("Generating basic information...")
    basic_info = generate_basic_info(df)
    
    print("Analyzing temporal aspects...")
    temporal_info, df = generate_temporal_analysis(df)
    
    print("Analyzing categorical variables...")
    categorical_info = generate_categorical_analysis(df)
    
    print("Analyzing numerical variables...")
    numerical_info = generate_numerical_analysis(df)
    
    print("Analyzing relationships...")
    relationship_info = analyze_relationships(df)
    
    print("Generating insights...")
    insights = generate_insights(df)
    
    # Combine all analyses
    data_understanding = {
        "basic_info": basic_info,
        "temporal_info": temporal_info,
        "categorical_info": categorical_info,
        "numerical_info": numerical_info,
        "relationship_info": relationship_info,
        "insights": insights,
        "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save JSON report
    with open(f"{output_dir}/data_understanding.json", 'w') as f:
        json.dump(data_understanding, f, indent=4)
    
    print("Saving JSON report...")
    print(f"JSON report saved to {output_dir}/data_understanding.json")
    
    # Generate and save markdown report
    print("Generating markdown report...")
    with open(f"{output_dir}/data_understanding.md", 'w') as f:
        f.write("# Retail Store Inventory Data Understanding\n\n")
        f.write(f"Analysis Date: {data_understanding['analysis_date']}\n\n")
        
        f.write("## Basic Information\n\n")
        f.write(f"- Number of rows: {basic_info['num_rows']}\n")
        f.write(f"- Number of columns: {basic_info['num_columns']}\n")
        f.write(f"- Columns: {', '.join(basic_info['column_names'])}\n")
        f.write(f"- Duplicated rows: {basic_info['duplicated_rows']}\n\n")
        
        f.write("## Column Data Types\n\n")
        for col, dtype in basic_info['column_dtypes'].items():
            f.write(f"- {col}: {dtype}\n")
        f.write("\n")
        
        f.write("## Temporal Information\n\n")
        f.write(f"- Date range: {temporal_info['min_date']} to {temporal_info['max_date']}\n")
        f.write(f"- Total days: {temporal_info['date_range_days']}\n")
        f.write(f"- Unique dates: {temporal_info['unique_dates']}\n\n")
        
        f.write("## Categorical Variables\n\n")
        for col, info in categorical_info.items():
            f.write(f"### {col}\n\n")
            f.write(f"- Unique values: {info['unique_values']}\n")
            f.write("- Top values:\n")
            for val, count in info['top_values'].items():
                f.write(f"  - {val}: {count}\n")
            f.write("\n")
        
        f.write("## Numerical Variables\n\n")
        for col, info in numerical_info.items():
            f.write(f"### {col}\n\n")
            f.write(f"- Min: {info['min']:.2f}\n")
            f.write(f"- Max: {info['max']:.2f}\n")
            f.write(f"- Mean: {info['mean']:.2f}\n")
            f.write(f"- Median: {info['median']:.2f}\n")
            f.write(f"- Standard Deviation: {info['std']:.2f}\n")
            f.write(f"- Range: {info['range']:.2f}\n\n")
        
        f.write("## Key Insights\n\n")
        for insight in insights:
            f.write(f"- {insight}\n")
        
    print(f"Markdown report saved to {output_dir}/data_understanding.md")
    
    # Save profile report
    print("Generating profile report (this may take a while)...")
    profile_status = save_profile_report(df, f"{output_dir}/profile_report.html")
    print(profile_status)
    
    # Save visualizations
    print("Creating visualizations...")
    viz_status = save_visualizations(df, f"{output_dir}/visualizations")
    print(viz_status)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
