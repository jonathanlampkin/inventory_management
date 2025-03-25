import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress non-critical warnings
warnings.filterwarnings('ignore')

class RetailEDA:
    """Comprehensive EDA class for retail inventory analysis."""
    
    def __init__(self, data_path, output_dir):
        """Initialize the EDA class."""
        self.data_path = data_path
        self.output_dir = output_dir
        self.start_time = datetime.now()
        
        # Define output paths
        self.output_paths = {
            'visualizations': f"{output_dir}/eda/visualizations",
            'seasonality': f"{output_dir}/eda/seasonality_analysis",
            'supply': f"{output_dir}/eda/supply_analysis",
            'recommendations': f"{output_dir}/eda/recommendations"
        }
        
        # Create output directories
        for path in self.output_paths.values():
            os.makedirs(path, exist_ok=True)
            os.makedirs(f"{path}/seasonal", exist_ok=True)
            os.makedirs(f"{path}/supply", exist_ok=True)
        
        # Initialize performance metrics
        self.performance_metrics = {}
        
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
        category_seasonality_index = {}
        for category in monthly_cat.columns:
            max_val = monthly_cat[category].max()
            min_val = monthly_cat[category].min()
            mean_val = monthly_cat[category].mean()
            
            if mean_val > 0:
                seasonality_index = (max_val - min_val) / mean_val
                category_seasonality_index[category] = float(seasonality_index)
        
        # 2. Product seasonality using statistical thresholds
        # Calculate sales quantiles
        sales_quantiles = self.df.groupby('Product ID')['Units Sold'].sum().quantile([0.95, 0.05])
        
        # High performers (top 5%)
        top_products = self.df[self.df['Product ID'].isin(
            self.df.groupby('Product ID')['Units Sold'].sum()
            .sort_values(ascending=False)
            .head(int(len(self.df['Product ID'].unique()) * 0.05))
            .index
        )]
        
        # Monthly sales for top products
        monthly_product = top_products.groupby(['MonthName', 'Product ID'], observed=False)['Units Sold'].mean().unstack()
        
        # Plot monthly product sales
        plt.figure(figsize=(14, 7))
        monthly_product.plot(marker='o')
        plt.title('Monthly Sales for Top 5% Products', fontsize=16)
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
                'top_products': top_products['Product ID'].unique().tolist(),
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
        
        # 3. Store level supply analysis
        store_metrics = self.df.groupby('Store ID').agg({
            'Supply_Status': lambda x: x.value_counts().index[0],
            'Inventory_Sales_Ratio': 'mean',
            'Supply_Gap': 'mean',
            'Forecast_Accuracy': 'mean'
        }).reset_index()
        
        # Plot store-level supply metrics
        plt.figure(figsize=(12, 6))
        sns.scatterplot(
            data=store_metrics,
            x='Inventory_Sales_Ratio',
            y='Supply_Gap',
            size='Forecast_Accuracy',
            hue='Supply_Status',
            alpha=0.6
        )
        plt.title('Store-Level Supply Analysis', fontsize=16)
        plt.ylabel('Average Supply Gap')
        plt.xlabel('Inventory-Sales Ratio')
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/supply/store_supply_analysis.png")
        plt.close()
        
        # 4. Critical supply issues
        # Identify critical undersupply and oversupply
        critical_undersupply = self.df[
            (self.df['Supply_Status'] == 'Undersupplied') &
            (self.df['Units Sold'] > self.df['Units Sold'].quantile(0.75))
        ].groupby('Product ID').agg({
            'Supply_Gap': 'mean',
            'Units Sold': 'sum',
            'Inventory Level': 'mean',
            'Forecast_Accuracy': 'mean'
        }).reset_index()
        
        excessive_oversupply = self.df[
            (self.df['Supply_Status'] == 'Oversupplied') &
            (self.df['Inventory_Sales_Ratio'] > self.df['Inventory_Sales_Ratio'].quantile(0.9))
        ].groupby('Product ID').agg({
            'Supply_Gap': 'mean',
            'Units Sold': 'sum',
            'Inventory Level': 'mean',
            'Forecast_Accuracy': 'mean'
        }).reset_index()
        
        # Plot critical supply issues
        plt.figure(figsize=(12, 6))
        plt.scatter(
            critical_undersupply['Units Sold'],
            critical_undersupply['Supply_Gap'],
            color='red',
            alpha=0.6,
            label='Critical Undersupply'
        )
        plt.scatter(
            excessive_oversupply['Units Sold'],
            excessive_oversupply['Supply_Gap'],
            color='blue',
            alpha=0.6,
            label='Excessive Oversupply'
        )
        plt.title('Critical Supply Issues', fontsize=16)
        plt.ylabel('Supply Gap')
        plt.xlabel('Total Units Sold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/supply/critical_supply_issues.png")
        plt.close()
        
        # Generate supply recommendations
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
            "store_supply": {
                "store_metrics": store_metrics.to_dict()
            },
            "critical_issues": {
                "undersupply": critical_undersupply.to_dict(),
                "oversupply": excessive_oversupply.to_dict()
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
        self.generate_supply_recommendations(
            critical_undersupply,
            excessive_oversupply,
            self.df[self.df['Forecast_Accuracy'] < self.df['Forecast_Accuracy'].quantile(0.1)]
        )
        
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
            f.write("### Peak Seasons for Top 5% Products\n\n")
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
            
            # Holiday/Promotion impact
            f.write("## Impact Factors\n\n")
            f.write("### Holiday/Promotion Impact\n\n")
            f.write("| Holiday/Promotion | Average Units Sold |\n")
            f.write("|-------------------|-------------------|\n")
            
            for holiday, sales in seasonality_results['overall_seasonality']['holiday_impact'].items():
                holiday_str = "Yes" if holiday == 1 else "No"
                f.write(f"| {holiday_str} | {sales:.2f} |\n")
            f.write("\n")
            
            # Key Findings
            f.write("## Key Findings\n\n")
            
            # Most seasonal category
            most_seasonal_cat = max(seasonality_results['category_seasonality']['category_seasonality_index'].items(), 
                                  key=lambda x: x[1])
            f.write(f"1. **Most Seasonal Category**: {most_seasonal_cat[0]} (Seasonality index: {most_seasonal_cat[1]:.2f})\n")
            
            # Peak sales month
            peak_month = max(seasonality_results['overall_seasonality']['monthly_sales'].items(), 
                            key=lambda x: x[1])
            f.write(f"2. **Peak Sales Month**: {peak_month[0]} (Average units sold: {peak_month[1]:.2f})\n")
            
            # Holiday impact
            holiday_impact = seasonality_results['overall_seasonality']['holiday_impact']
            holiday_diff = holiday_impact[1] - holiday_impact[0]
            f.write(f"3. **Holiday Impact**: Sales increase by {holiday_diff:.2f} units during holidays/promotions\n")
            
            # Regional variation
            region_peaks = seasonality_results['region_seasonality']['region_peak_season']
            best_region = max(region_peaks.items(), key=lambda x: x[1]['peak_value'])
            f.write(f"4. **Strongest Regional Performance**: {best_region[0]} region peaks at {best_region[1]['peak_value']:.2f} units in {best_region[1]['peak_month']}\n")
            
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
            
            f.write("## Products with Poor Forecast Accuracy\n\n")
            f.write("These products have low forecast accuracy and may need review of forecasting methods:\n\n")
            
            f.write("| Product ID | Category | Forecast Accuracy |\n")
            f.write("|------------|----------|-------------------|\n")
            for _, row in poor_forecast.head(5).iterrows():
                f.write(f"| {row['Product ID']} | {row['Category']} | {row['Forecast_Accuracy']:.2f} |\n")
            f.write("\n")
            
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
            
            # Holiday impact
            holiday_impact = self.df.groupby('Holiday/Promotion')['Units Sold'].mean()
            holiday_diff = holiday_impact[1] - holiday_impact[0]
            f.write(f"- **Holiday Impact**: Sales increase by {holiday_diff:.2f} units during holidays/promotions\n\n")
            
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
            
            # Store Performance
            f.write("### Store Performance\n\n")
            
            # Store with best supply management
            store_metrics = self.df.groupby('Store ID').agg({
                'Supply_Status': lambda x: (x == 'Optimal').mean(),
                'Forecast_Accuracy': 'mean'
            })
            best_store = store_metrics['Supply_Status'].idxmax()
            f.write(f"- **Best Supply Management**: Store {best_store} (Optimal supply rate: {store_metrics.loc[best_store, 'Supply_Status']*100:.1f}%)\n")
            
            # Store with worst forecast accuracy
            worst_store = store_metrics['Forecast_Accuracy'].idxmin()
            f.write(f"- **Worst Forecast Accuracy**: Store {worst_store} (Accuracy: {store_metrics.loc[worst_store, 'Forecast_Accuracy']:.2f})\n\n")
            
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

if __name__ == "__main__":
    # Initialize and run EDA
    eda = RetailEDA("data/retail_store_inventory.csv", "output")
    eda.run_analysis() 