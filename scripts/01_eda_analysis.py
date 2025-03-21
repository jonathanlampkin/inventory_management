import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
from scipy.stats import pearsonr
import warnings

# Suppress non-critical warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class RetailEDA:
    """Comprehensive EDA class for retail inventory analysis."""
    
    def __init__(self, data_path, output_dir="output"):
        """Initialize the EDA class with data and output directory."""
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.output_paths = {}
        self.performance_metrics = {}
        self.start_time = datetime.now()
        
        # Create output directories
        self.create_output_dirs()
        
    def create_output_dirs(self):
        """Create necessary output directories for results."""
        # Main directories
        dirs = [
            f"{self.output_dir}/eda",
            f"{self.output_dir}/eda/visualizations",
            f"{self.output_dir}/eda/data_understanding",
            f"{self.output_dir}/eda/seasonality_analysis",
            f"{self.output_dir}/eda/supply_analysis",
            f"{self.output_dir}/eda/recommendations",
            f"{self.output_dir}/eda/performance"
        ]
        
        # Create all directories
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
        # Create subdirectories for visualizations
        viz_subdirs = ["category", "product", "region", "seasonal", "supply"]
        for subdir in viz_subdirs:
            os.makedirs(f"{self.output_dir}/eda/visualizations/{subdir}", exist_ok=True)
        
        # Store paths for later use
        self.output_paths = {
            "visualizations": f"{self.output_dir}/eda/visualizations",
            "data_understanding": f"{self.output_dir}/eda/data_understanding",
            "seasonality": f"{self.output_dir}/eda/seasonality_analysis",
            "supply": f"{self.output_dir}/eda/supply_analysis",
            "recommendations": f"{self.output_dir}/eda/recommendations"
        }
        
    def load_data(self):
        """Load and preprocess the data."""
        print("Loading data...")
        start = datetime.now()
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        # Process dates
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Day'] = self.df['Date'].dt.day
        self.df['Quarter'] = self.df['Date'].dt.quarter
        self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
        self.df['WeekOfYear'] = self.df['Date'].dt.isocalendar().week
        self.df['MonthName'] = self.df['Date'].dt.strftime('%b')
        
        # Ensure MonthName is ordered correctly
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        self.df['MonthName'] = pd.Categorical(self.df['MonthName'], categories=month_order, ordered=True)
        
        # Create additional metrics for supply analysis
        self.df['Inventory_Sales_Ratio'] = self.df['Inventory Level'] / self.df['Units Sold'].replace(0, np.nan)
        self.df['Sell_Through_Rate'] = self.df['Units Sold'] / (self.df['Inventory Level'] + self.df['Units Sold'])
        self.df['Forecast_Accuracy'] = 1 - abs(self.df['Demand Forecast'] - self.df['Units Sold']) / self.df['Demand Forecast'].replace(0, np.nan)
        self.df['Supply_Gap'] = self.df['Inventory Level'] - self.df['Units Sold']
        
        # Define optimal threshold ranges
        self.df['Optimal_Inventory'] = self.df['Units Sold'] * 1.5  # Example: 1.5x of sales as optimal inventory
        
        # Supply status categories
        conditions = [
            (self.df['Inventory Level'] < self.df['Units Sold']),
            (self.df['Inventory Level'] >= self.df['Units Sold']) & 
            (self.df['Inventory Level'] <= self.df['Optimal_Inventory']),
            (self.df['Inventory Level'] > self.df['Optimal_Inventory'])
        ]
        choices = ['Undersupplied', 'Optimal', 'Oversupplied']
        self.df['Supply_Status'] = np.select(conditions, choices, default='Unknown')
        
        # Calculate performance metrics
        end = datetime.now()
        self.performance_metrics['data_loading'] = {
            'duration_seconds': (end - start).total_seconds(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / (1024 * 1024),
            'row_count': len(self.df),
            'column_count': len(self.df.columns)
        }
        
        print(f"Data loaded: {len(self.df)} rows and {len(self.df.columns)} columns")
        
    def analyze_basic_stats(self):
        """Analyze basic statistics of the dataset."""
        print("Analyzing basic statistics...")
        start = datetime.now()
        
        # Summary statistics
        summary_stats = self.df.describe()
        
        # Correlation analysis
        correlation = self.df[['Price', 'Units Sold', 'Inventory Level', 'Demand Forecast', 
                               'Discount', 'Competitor Pricing']].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/correlation_heatmap.png")
        plt.close()
        
        # Units sold distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['Units Sold'], kde=True)
        plt.title('Distribution of Units Sold', fontsize=16)
        plt.xlabel('Units Sold')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/units_sold_dist.png")
        plt.close()
        
        # Units sold by category
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Category', y='Units Sold', data=self.df)
        plt.title('Units Sold by Category', fontsize=16)
        plt.xlabel('Category')
        plt.ylabel('Units Sold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/units_sold_by_category.png")
        plt.close()
        
        # Price vs Units Sold
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df.sample(5000), x='Price', y='Units Sold', alpha=0.5, hue='Category')
        plt.title('Price vs Units Sold (Sample of 5000 points)', fontsize=16)
        plt.xlabel('Price')
        plt.ylabel('Units Sold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/price_vs_units_sold.png")
        plt.close()
        
        # Inventory vs Sales
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df.sample(5000), x='Inventory Level', y='Units Sold', alpha=0.5, hue='Category')
        plt.title('Inventory Level vs Units Sold (Sample of 5000 points)', fontsize=16)
        plt.xlabel('Inventory Level')
        plt.ylabel('Units Sold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/inventory_vs_sales.png")
        plt.close()
        
        # Monthly sales trend
        monthly_sales = self.df.groupby([self.df['Date'].dt.year, self.df['Date'].dt.month])['Units Sold'].sum().reset_index()
        monthly_sales.columns = ['Year', 'Month', 'Units Sold']
        
        plt.figure(figsize=(14, 6))
        plt.plot(monthly_sales['YearMonth'], monthly_sales['Units Sold'], marker='o')
        plt.title('Monthly Sales Trend', fontsize=16)
        plt.xlabel('Year-Month')
        plt.ylabel('Total Units Sold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/monthly_sales_trend.png")
        plt.close()
        
        # Calculate performance metrics
        end = datetime.now()
        self.performance_metrics['basic_stats'] = {
            'duration_seconds': (end - start).total_seconds()
        }
        
        # Return basic statistics
        return {
            'summary_stats': summary_stats.to_dict(),
            'correlation': correlation.to_dict()
        }

    def analyze_seasonality(self):
        """Analyze seasonal patterns in the data."""
        print("Analyzing seasonality patterns...")
        start = datetime.now()
        
        # 1. Category seasonality
        monthly_cat = self.df.groupby(['MonthName', 'Category'], observed=False)['Units Sold'].mean().unstack()
        
        # Plot monthly category sales
        plt.figure(figsize=(14, 7))
        monthly_cat.plot(marker='o')
        plt.title('Monthly Sales by Category', fontsize=16)
        plt.ylabel('Average Units Sold')
        plt.xlabel('Month')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Category')
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/seasonal/monthly_category_sales.png")
        plt.close()
        
        # Find peak season for each category
        category_peak_season = {}
        for category in monthly_cat.columns:
            peak_month = monthly_cat[category].idxmax()
            category_peak_season[category] = {
                'peak_month': peak_month,
                'peak_value': float(monthly_cat.loc[peak_month, category])
            }
        
        # Calculate seasonality index for each category
        # Higher number means more seasonal variation
        category_seasonality_index = {}
        for category in monthly_cat.columns:
            max_val = monthly_cat[category].max()
            min_val = monthly_cat[category].min()
            mean_val = monthly_cat[category].mean()
            
            if mean_val > 0:
                seasonality_index = (max_val - min_val) / mean_val
                category_seasonality_index[category] = float(seasonality_index)
        
        # 2. Product seasonality
        # Get top 10 products by total sales
        top_products = self.df.groupby('Product ID')['Units Sold'].sum().nlargest(10)
        top_products_df = self.df[self.df['Product ID'].isin(top_products.index)]
        
        # Monthly sales for top products
        monthly_product = top_products_df.groupby(['MonthName', 'Product ID'], observed=False)['Units Sold'].mean().unstack()
        
        # Plot monthly product sales
        plt.figure(figsize=(14, 7))
        monthly_product.plot(marker='o')
        plt.title('Monthly Sales for Top 10 Products', fontsize=16)
        plt.ylabel('Average Units Sold')
        plt.xlabel('Month')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Product ID')
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/seasonal/monthly_product_sales.png")
        plt.close()
        
        # Find peak season for each top product
        product_peak_season = {}
        for product in monthly_product.columns:
            peak_month = monthly_product[product].idxmax()
            product_peak_season[product] = {
                'peak_month': peak_month,
                'peak_value': float(monthly_product.loc[peak_month, product])
            }
        
        # 3. Region seasonality
        monthly_region = self.df.groupby(['MonthName', 'Region'], observed=False)['Units Sold'].mean().unstack()
        
        # Plot monthly region sales
        plt.figure(figsize=(14, 7))
        monthly_region.plot(marker='o')
        plt.title('Monthly Sales by Region', fontsize=16)
        plt.ylabel('Average Units Sold')
        plt.xlabel('Month')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Region')
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/seasonal/monthly_region_sales.png")
        plt.close()
        
        # Find peak season for each region
        region_peak_season = {}
        for region in monthly_region.columns:
            peak_month = monthly_region[region].idxmax()
            region_peak_season[region] = {
                'peak_month': peak_month,
                'peak_value': float(monthly_region.loc[peak_month, region])
            }
        
        # 4. Overall seasonality
        monthly_sales = self.df.groupby('MonthName', observed=False)['Units Sold'].mean()
        
        # Plot overall monthly sales
        plt.figure(figsize=(14, 7))
        monthly_sales.plot(kind='bar', color='skyblue')
        plt.title('Overall Monthly Sales Pattern', fontsize=16)
        plt.ylabel('Average Units Sold')
        plt.xlabel('Month')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/seasonal/overall_monthly_sales.png")
        plt.close()
        
        # Weather impact analysis
        weather_impact = self.df.groupby('Weather Condition')['Units Sold'].mean().to_dict()
        
        # Holiday/Promotion impact
        holiday_impact = self.df.groupby('Holiday/Promotion')['Units Sold'].mean().to_dict()
        
        # Yearly comparison by month
        yearly_monthly_sales = self.df.groupby(['Year', 'MonthName'], observed=False)['Units Sold'].mean().unstack(0)
        
        # Plot yearly comparison
        plt.figure(figsize=(14, 7))
        yearly_monthly_sales.plot(marker='o')
        plt.title('Monthly Sales Comparison by Year', fontsize=16)
        plt.ylabel('Average Units Sold')
        plt.xlabel('Month')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Year')
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/seasonal/yearly_comparison.png")
        plt.close()
        
        # Compile results
        seasonality_results = {
            'category_seasonality': {
                'monthly_category_sales': monthly_cat.to_dict(),
                'category_peak_season': category_peak_season,
                'category_seasonality_index': category_seasonality_index
            },
            'product_seasonality': {
                'top_products': top_products.to_dict(),
                'monthly_product_sales': monthly_product.to_dict(),
                'product_peak_season': product_peak_season
            },
            'region_seasonality': {
                'monthly_region_sales': monthly_region.to_dict(),
                'region_peak_season': region_peak_season
            },
            'overall_seasonality': {
                'monthly_sales': monthly_sales.to_dict(),
                'yearly_monthly_sales': yearly_monthly_sales.to_dict(),
                'weather_impact': weather_impact,
                'holiday_impact': holiday_impact
            }
        }
        
        # Save seasonality analysis results
        with open(f"{self.output_paths['seasonality']}/seasonality_analysis_results.json", 'w') as f:
            json.dump(seasonality_results, f, indent=4)
        
        # Calculate performance metrics
        end = datetime.now()
        self.performance_metrics['seasonality_analysis'] = {
            'duration_seconds': (end - start).total_seconds()
        }
        
        # Generate seasonality report
        self.generate_seasonality_report(seasonality_results)
        
        return seasonality_results 

    def analyze_supply(self):
        """Analyze supply levels and identify issues."""
        print("Analyzing supply patterns...")
        start = datetime.now()
        
        # 1. Overall supply analysis
        supply_status_counts = self.df['Supply_Status'].value_counts()
        
        # Plot supply status distribution
        plt.figure(figsize=(10, 6))
        sns.barplot(x=supply_status_counts.index, y=supply_status_counts.values)
        plt.title('Distribution of Supply Status', fontsize=16)
        plt.ylabel('Count')
        plt.xlabel('Supply Status')
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/supply/supply_status_distribution.png")
        plt.close()
        
        # Inventory to sales ratio distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(self.df['Inventory_Sales_Ratio'].dropna().clip(0, 10), bins=50)
        plt.axvline(x=1.5, color='red', linestyle='--', label='Optimal Threshold (1.5)')
        plt.title('Distribution of Inventory to Sales Ratio (Capped at 10)', fontsize=16)
        plt.ylabel('Frequency')
        plt.xlabel('Inventory to Sales Ratio')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/supply/inventory_sales_ratio_distribution.png")
        plt.close()
        
        # 2. Category level supply analysis
        category_status = self.df.groupby(['Category', 'Supply_Status']).size().unstack(fill_value=0)
        category_status_pct = category_status.div(category_status.sum(axis=1), axis=0) * 100
        
        plt.figure(figsize=(12, 7))
        category_status_pct.plot(kind='bar', stacked=True)
        plt.title('Supply Status Distribution by Category', fontsize=16)
        plt.ylabel('Percentage')
        plt.xlabel('Category')
        plt.legend(title='Supply Status')
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/supply/category_supply_status.png")
        plt.close()
        
        # 3. Product level supply analysis
        product_metrics = self.df.groupby(['Product ID', 'Category']).agg({
            'Supply_Status': lambda x: x.value_counts().index[0],
            'Supply_Gap': 'mean',
            'Units Sold': 'sum',
            'Inventory Level': 'mean',
            'Forecast_Accuracy': 'mean'
        }).reset_index()
        
        # Top undersupplied products
        undersupplied = product_metrics[product_metrics['Supply_Status'] == 'Undersupplied']
        critical_undersupply = undersupplied[
            undersupplied['Units Sold'] > undersupplied['Units Sold'].median()
        ].sort_values('Supply_Gap').head(10)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Product ID', y='Supply_Gap', data=critical_undersupply)
        plt.title('Top 10 Critical Undersupplied Products', fontsize=16)
        plt.ylabel('Supply Gap (Inventory - Sales)')
        plt.xlabel('Product ID')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/supply/critical_undersupplied_products.png")
        plt.close()
        
        # Top oversupplied products
        oversupplied = product_metrics[product_metrics['Supply_Status'] == 'Oversupplied']
        excessive_oversupply = oversupplied[
            oversupplied['Supply_Gap'] > oversupplied['Supply_Gap'].quantile(0.9)
        ].sort_values('Supply_Gap', ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Product ID', y='Supply_Gap', data=excessive_oversupply)
        plt.title('Top 10 Excessive Oversupplied Products', fontsize=16)
        plt.ylabel('Supply Gap (Inventory - Sales)')
        plt.xlabel('Product ID')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/supply/excessive_oversupplied_products.png")
        plt.close()
        
        # 4. Store level supply analysis
        store_metrics = self.df.groupby('Store ID').agg({
            'Supply_Status': lambda x: x.value_counts().index[0],
            'Inventory_Sales_Ratio': 'mean',
            'Supply_Gap': 'mean',
            'Forecast_Accuracy': 'mean'
        }).reset_index()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Store ID', y='Inventory_Sales_Ratio', data=store_metrics)
        plt.title('Average Inventory-Sales Ratio by Store', fontsize=16)
        plt.ylabel('Ratio')
        plt.xlabel('Store ID')
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/supply/store_inventory_sales_ratio.png")
        plt.close()
        
        # 5. Forecast accuracy impact on supply
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.df.sample(5000), 
            x='Forecast_Accuracy', 
            y='Supply_Gap', 
            alpha=0.5, 
            hue='Supply_Status'
        )
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title('Forecast Accuracy vs. Supply Gap (Sample of 5000 points)', fontsize=16)
        plt.ylabel('Supply Gap')
        plt.xlabel('Forecast Accuracy')
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/supply/forecast_accuracy_vs_supply_gap.png")
        plt.close()
        
        # 6. Generate supply recommendations
        supply_results = {
            "overall_supply": {
                "supply_status_distribution": supply_status_counts.to_dict(),
                "avg_inventory_sales_ratio": float(self.df['Inventory_Sales_Ratio'].mean()),
                "avg_supply_gap": float(self.df['Supply_Gap'].mean()),
                "avg_forecast_accuracy": float(self.df['Forecast_Accuracy'].mean())
            },
            "category_supply": {
                "category_status": category_status.to_dict(),
                "category_status_pct": category_status_pct.to_dict()
            },
            "product_supply": {
                "critical_undersupply": critical_undersupply.to_dict(),
                "excessive_oversupply": excessive_oversupply.to_dict()
            },
            "store_supply": {
                "store_metrics": store_metrics.to_dict()
            }
        }
        
        # Save supply analysis results
        with open(f"{self.output_paths['supply']}/supply_analysis_results.json", 'w') as f:
            json.dump(supply_results, f, indent=4)
        
        # Calculate performance metrics
        end = datetime.now()
        self.performance_metrics['supply_analysis'] = {
            'duration_seconds': (end - start).total_seconds()
        }
        
        # Generate supply recommendations
        self.generate_supply_recommendations(critical_undersupply, excessive_oversupply, self.df[self.df['Forecast_Accuracy'] < self.df['Forecast_Accuracy'].quantile(0.1)])
        
        return supply_results 

    def generate_seasonality_report(self, seasonality_results):
        """Generate a report with seasonal analysis findings."""
        with open(f"{self.output_paths['recommendations']}/seasonality_report.md", "w") as f:
            f.write("# Retail Inventory: Seasonality Analysis\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Category seasonality
            f.write("## Category Seasonality\n\n")
            f.write("### Peak Seasons by Category\n\n")
            f.write("| Category | Peak Month | Peak Average Sales |\n")
            f.write("|----------|------------|-------------------|\n")
            
            for category, data in seasonality_results['category_seasonality']['category_peak_season'].items():
                f.write(f"| {category} | {data['peak_month']} | {data['peak_value']:.2f} |\n")
            f.write("\n")
            
            # Product seasonality
            f.write("## Product Seasonality\n\n")
            f.write("### Peak Seasons for Top Products\n\n")
            f.write("| Product ID | Peak Month | Peak Average Sales |\n")
            f.write("|------------|------------|-------------------|\n")
            
            for product, data in seasonality_results['product_seasonality']['product_peak_season'].items():
                f.write(f"| {product} | {data['peak_month']} | {data['peak_value']:.2f} |\n")
            f.write("\n")
            
            # Region seasonality
            f.write("## Regional Seasonality\n\n")
            f.write("### Peak Seasons by Region\n\n")
            f.write("| Region | Peak Month | Peak Average Sales |\n")
            f.write("|--------|------------|-------------------|\n")
            
            for region, data in seasonality_results['region_seasonality']['region_peak_season'].items():
                f.write(f"| {region} | {data['peak_month']} | {data['peak_value']:.2f} |\n")
            f.write("\n")
            
            # Weather impact
            f.write("## Impact Factors\n\n")
            f.write("### Weather Impact on Sales\n\n")
            f.write("| Weather Condition | Average Units Sold |\n")
            f.write("|-------------------|-------------------|\n")
            
            for weather, sales in seasonality_results['overall_seasonality']['weather_impact'].items():
                f.write(f"| {weather} | {sales:.2f} |\n")
            f.write("\n")
            
            # Holiday/Promotion impact
            f.write("### Holiday/Promotion Impact\n\n")
            f.write("| Holiday/Promotion | Average Units Sold |\n")
            f.write("|-------------------|-------------------|\n")
            
            for holiday, sales in seasonality_results['overall_seasonality']['holiday_impact'].items():
                holiday_str = "Yes" if holiday == 1 else "No"
                f.write(f"| {holiday_str} | {sales:.2f} |\n")
            f.write("\n")
            
            # Recommendations
            f.write("## Seasonal Planning Recommendations\n\n")
            f.write("1. **Inventory Buildup**: Start building inventory 1-2 months before peak seasons for each category\n")
            f.write("2. **Regional Focus**: Allocate inventory to regions based on their specific seasonal peaks\n")
            f.write("3. **Weather-Based Planning**: Adjust inventory based on weather forecasts, especially for weather-sensitive categories\n")
            f.write("4. **Promotion Timing**: Align major promotions with historically strong sales months\n")
            f.write("5. **Seasonal Clearance**: Plan clearance sales immediately after peak seasons to reduce excess inventory\n")
    
    def generate_supply_recommendations(self, critical_undersupply, excessive_oversupply, poor_forecast):
        """Generate specific product recommendations based on supply analysis."""
        print("Generating supply recommendations...")
        
        with open(f"{self.output_paths['recommendations']}/supply_recommendations.md", "w") as f:
            f.write("# Supply Analysis: Inventory Recommendations\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Critical Undersupplied Products\n\n")
            f.write("These high-volume products have significant supply shortages and should be restocked immediately:\n\n")
            
            f.write("| Product ID | Category | Avg Supply Gap | Total Units Sold | Avg Inventory | Forecast Accuracy |\n")
            f.write("|------------|----------|----------------|-----------------|---------------|-------------------|\n")
            for _, row in critical_undersupply.head(5).iterrows():
                f.write(f"| {row['Product ID']} | {row['Category']} | {row['Supply_Gap']:.2f} | {row['Units Sold']:.0f} | {row['Inventory Level']:.2f} | {row['Forecast_Accuracy']:.2f} |\n")
            f.write("\n")
            
            f.write("## Excessive Oversupplied Products\n\n")
            f.write("These products have excessive inventory relative to their sales and should be considered for markdown or promotion:\n\n")
            
            f.write("| Product ID | Category | Avg Supply Gap | Total Units Sold | Avg Inventory | Forecast Accuracy |\n")
            f.write("|------------|----------|----------------|-----------------|---------------|-------------------|\n")
            for _, row in excessive_oversupply.head(5).iterrows():
                f.write(f"| {row['Product ID']} | {row['Category']} | {row['Supply_Gap']:.2f} | {row['Units Sold']:.0f} | {row['Inventory Level']:.2f} | {row['Forecast_Accuracy']:.2f} |\n")
            f.write("\n")
            
            # General recommendations
            f.write("## General Inventory Recommendations\n\n")
            
            avg_metrics = self.df.groupby('Supply_Status').size() / len(self.df) * 100
            optimal_pct = avg_metrics.get('Optimal', 0)
            
            f.write(f"- **Current Optimal Inventory Rate**: {optimal_pct:.1f}% of products are optimally supplied\n")
            f.write(f"- **Target Optimal Inventory Rate**: 75% or higher\n\n")
            
            f.write("### Key Recommendations:\n\n")
            f.write("1. **Implement Just-in-Time Inventory**: For high-volume products with consistent demand\n")
            f.write("2. **Increase Safety Stock**: For products with high variability or critical to business\n")
            f.write("3. **Review Order Frequency**: Consider more frequent orders with smaller quantities\n")
            f.write("4. **Improve Forecast Models**: Especially for products with poor forecast accuracy\n")
            f.write("5. **Implement Cross-Store Balancing**: Redistribute inventory between stores to address local shortages\n")
    
    def generate_summary_report(self):
        """Generate a comprehensive EDA summary report."""
        with open(f"{self.output_paths['recommendations']}/eda_summary_report.md", "w") as f:
            f.write("# Retail Inventory Analysis: Executive Summary\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data overview
            f.write("## Data Overview\n\n")
            f.write(f"- **Period Analyzed**: {self.df['Date'].min().strftime('%Y-%m-%d')} to {self.df['Date'].max().strftime('%Y-%m-%d')}\n")
            f.write(f"- **Number of Stores**: {self.df['Store ID'].nunique()}\n")
            f.write(f"- **Number of Products**: {self.df['Product ID'].nunique()}\n")
            f.write(f"- **Product Categories**: {', '.join(self.df['Category'].unique())}\n")
            f.write(f"- **Regions**: {', '.join(self.df['Region'].unique())}\n\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            
            # Seasonality
            f.write("### Seasonality Patterns\n\n")
            
            # Top-selling month overall
            monthly_sales = self.df.groupby('MonthName', observed=False)['Units Sold'].mean()
            top_month = monthly_sales.idxmax()
            f.write(f"- **Peak Sales Month**: {top_month} (Average units sold: {monthly_sales[top_month]:.2f})\n")
            
            # Category with highest seasonality
            category_seasonality = {}
            for category in self.df['Category'].unique():
                category_data = self.df[self.df['Category'] == category]
                monthly_cat_sales = category_data.groupby('MonthName', observed=False)['Units Sold'].mean()
                max_val = monthly_cat_sales.max()
                min_val = monthly_cat_sales.min()
                mean_val = monthly_cat_sales.mean()
                
                if mean_val > 0:
                    seasonality_index = (max_val - min_val) / mean_val
                    category_seasonality[category] = float(seasonality_index)
            
            most_seasonal_category = max(category_seasonality.items(), key=lambda x: x[1])
            f.write(f"- **Most Seasonal Category**: {most_seasonal_category[0]} (Seasonality index: {most_seasonal_category[1]:.2f})\n")
            
            # Weather impact
            weather_impact = self.df.groupby('Weather Condition')['Units Sold'].mean()
            best_weather = weather_impact.idxmax()
            worst_weather = weather_impact.idxmin()
            f.write(f"- **Weather Impact**: Sales are highest during {best_weather} weather and lowest during {worst_weather} weather\n\n")
            
            # Supply Status
            f.write("### Inventory Status\n\n")
            
            supply_status_pct = self.df['Supply_Status'].value_counts(normalize=True) * 100
            f.write(f"- **Undersupplied**: {supply_status_pct.get('Undersupplied', 0):.1f}% of inventory records\n")
            f.write(f"- **Optimal Supply**: {supply_status_pct.get('Optimal', 0):.1f}% of inventory records\n")
            f.write(f"- **Oversupplied**: {supply_status_pct.get('Oversupplied', 0):.1f}% of inventory records\n")
            
            # Most undersupplied category
            category_supply_gap = self.df.groupby('Category')['Supply_Gap'].mean()
            most_undersupplied_cat = category_supply_gap.idxmin()
            most_oversupplied_cat = category_supply_gap.idxmax()
            
            f.write(f"- **Most Undersupplied Category**: {most_undersupplied_cat} (Avg. gap: {category_supply_gap[most_undersupplied_cat]:.2f} units)\n")
            f.write(f"- **Most Oversupplied Category**: {most_oversupplied_cat} (Avg. gap: {category_supply_gap[most_oversupplied_cat]:.2f} units)\n\n")
            
            # Performance section
            f.write("## Analysis Performance\n\n")
            
            for analysis, metrics in self.performance_metrics.items():
                f.write(f"### {analysis.replace('_', ' ').title()}\n\n")
                for metric, value in metrics.items():
                    f.write(f"- **{metric.replace('_', ' ').title()}**: {value}\n")
                f.write("\n")
            
            # Next steps
            f.write("## Recommended Next Steps\n\n")
            f.write("1. **Implement Seasonal Inventory Planning**: Adjust inventory levels based on identified seasonal patterns\n")
            f.write("2. **Address Critical Supply Gaps**: Prioritize restocking for undersupplied high-volume products\n")
            f.write("3. **Improve Forecasting Accuracy**: Review forecasting methods for products with poor accuracy\n")
            f.write("4. **Optimize by Category**: Apply category-specific inventory strategies based on seasonality and supply patterns\n")
            f.write("5. **Regional Adjustments**: Customize inventory plans by region to account for regional variations\n")

    def run_analysis(self):
        """Run the complete EDA analysis pipeline."""
        print("Starting comprehensive EDA analysis...")
        
        # Load data
        self.load_data()
        
        # Create necessary directories
        os.makedirs(f"{self.output_paths['visualizations']}/supply", exist_ok=True)
        
        # Run basic EDA
        self.analyze_basic_stats()
        
        # Run seasonality analysis
        seasonality_results = self.analyze_seasonality()
        
        # Run supply analysis
        supply_results = self.analyze_supply()
        
        # Generate recommendations
        self.generate_seasonality_report(seasonality_results)
        
        # Generate supply recommendations
        if hasattr(self, 'df') and 'Supply_Status' in self.df.columns:
            critical_undersupply = self.df[self.df['Supply_Status'] == 'Undersupplied'].head(10)
            excessive_oversupply = self.df[self.df['Supply_Status'] == 'Oversupplied'].head(10)
            poor_forecast = self.df[self.df['Forecast_Accuracy'] < 0.5].head(10)
            self.generate_supply_recommendations(critical_undersupply, excessive_oversupply, poor_forecast)
        
        # Generate summary report
        self.generate_summary_report()
        
        # Calculate total runtime
        total_runtime = (datetime.now() - self.start_time).total_seconds()
        self.performance_metrics['total_runtime'] = total_runtime
        
        # Save performance metrics
        with open(f"{self.output_dir}/eda/performance/performance_metrics.json", 'w') as f:
            json.dump(self.performance_metrics, f, indent=4)
        
        print(f"EDA completed in {total_runtime:.2f} seconds")
        print(f"Results saved to {self.output_dir}/eda/")


def main():
    """Main function to run the EDA analysis."""
    print("Starting Retail Inventory EDA...")
    
    data_path = "data/retail_store_inventory.csv"
    output_dir = "output"
    
    # Initialize and run EDA
    eda = RetailEDA(data_path, output_dir)
    eda.run_analysis()
    
    print("EDA analysis complete!")

if __name__ == "__main__":
    main()