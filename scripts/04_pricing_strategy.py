import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
from sklearn.linear_model import LinearRegression
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class PricingStrategy:
    """Class for developing pricing strategies."""
    
    def __init__(self, data_path, output_dir="output"):
        """Initialize the pricing strategy analyzer."""
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.output_paths = {}
        self.performance_metrics = {}
        self.start_time = datetime.now()
        
        # Create output directories
        self.create_output_dirs()
    
    def create_output_dirs(self):
        """Create necessary output directories."""
        dirs = [
            f"{self.output_dir}/pricing_strategy",
            f"{self.output_dir}/pricing_strategy/visualizations",
            f"{self.output_dir}/pricing_strategy/recommendations",
            f"{self.output_dir}/pricing_strategy/models",
            f"{self.output_dir}/pricing_strategy/performance"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        self.output_paths = {
            "visualizations": f"{self.output_dir}/pricing_strategy/visualizations",
            "recommendations": f"{self.output_dir}/pricing_strategy/recommendations",
            "models": f"{self.output_dir}/pricing_strategy/models",
            "performance": f"{self.output_dir}/pricing_strategy/performance"
        }
    
    def load_data(self):
        """Load data for pricing strategy analysis."""
        print("Loading data...")
        start = datetime.now()
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        # Process dates
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Calculate revenue
        self.df['Revenue'] = self.df['Price'] * self.df['Units Sold'] * (1 - self.df['Discount']/100)
        
        # Calculate performance metrics
        end = datetime.now()
        self.performance_metrics['data_loading'] = {
            'duration_seconds': (end - start).total_seconds(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / (1024 * 1024),
            'row_count': len(self.df),
            'column_count': len(self.df.columns)
        }
        
        print(f"Data loaded: {len(self.df)} rows and {len(self.df.columns)} columns")
        return self.df 

    def analyze_price_elasticity(self):
        """Analyze price elasticity of demand with robust error handling."""
        print("Analyzing price elasticity...")
        start = datetime.now()
        
        results = {}
        
        # Group by product and calculate average price and units sold
        product_price_data = []
        
        for product_id in self.df['Product ID'].unique():
            product_df = self.df[self.df['Product ID'] == product_id]
            
            # Group by price points
            price_data = product_df.groupby('Price')['Units Sold'].mean().reset_index()
            price_data['Product ID'] = product_id
            
            # Only include products with multiple price points and non-zero values
            if len(price_data) > 2:  # Need at least 3 points for meaningful regression
                # Filter out zero values to avoid division by zero
                price_data = price_data[(price_data['Price'] > 0) & (price_data['Units Sold'] > 0)]
                if len(price_data) > 2:  # Still need 3+ points after filtering
                    product_price_data.append(price_data)
        
        if not product_price_data:
            print("No products with sufficient price points found")
            
            # Return empty results with structure
            return {
                "category_elasticity": {},
                "product_elasticity": {},
                "analysis_summary": {
                    "products_analyzed": 0,
                    "elastic_products": 0,
                    "inelastic_products": 0
                }
            }
        
        # Combine all price data
        all_price_data = pd.concat(product_price_data)
        
        # Calculate elasticity for products with sufficient data
        elasticity_results = {}
        elastic_count = 0
        inelastic_count = 0
        
        for product_id in all_price_data['Product ID'].unique():
            product_data = all_price_data[all_price_data['Product ID'] == product_id]
            
            if len(product_data) < 3:
                continue
            
            try:
                # Take logarithms for elasticity calculation
                product_data['log_price'] = np.log(product_data['Price'])
                product_data['log_quantity'] = np.log(product_data['Units Sold'])
                
                # Fit linear regression to log-log data
                model = LinearRegression()
                X = product_data['log_price'].values.reshape(-1, 1)
                y = product_data['log_quantity'].values
                model.fit(X, y)
                
                # Coefficient is the elasticity
                elasticity = model.coef_[0]
                
                # Store result
                product_category = self.df[self.df['Product ID'] == product_id]['Category'].iloc[0]
                avg_price = product_data['Price'].mean()
                
                elasticity_results[product_id] = {
                    'elasticity': float(elasticity),
                    'elastic_status': 'Elastic' if elasticity < -1 else 'Inelastic',
                    'category': product_category,
                    'avg_price': float(avg_price)
                }
                
                # Count elastic and inelastic products
                if elasticity < -1:
                    elastic_count += 1
                else:
                    inelastic_count += 1
                    
                # Plot price vs quantity for this product
                plt.figure(figsize=(10, 6))
                plt.scatter(product_data['Price'], product_data['Units Sold'])
                
                # Plot fitted curve
                price_range = np.linspace(product_data['Price'].min(), product_data['Price'].max(), 100)
                log_price_range = np.log(price_range).reshape(-1, 1)
                log_quantity_pred = model.predict(log_price_range)
                quantity_pred = np.exp(log_quantity_pred)
                
                plt.plot(price_range, quantity_pred, 'r-')
                plt.title(f'Price Elasticity: Product {product_id} (e = {elasticity:.2f})', fontsize=16)
                plt.xlabel('Price ($)')
                plt.ylabel('Units Sold')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{self.output_paths['visualizations']}/elasticity_product_{product_id}.png")
                plt.close()
                
            except Exception as e:
                print(f"Error calculating elasticity for product {product_id}: {str(e)}")
        
        # Get category for each product
        product_categories = self.df.groupby('Product ID')['Category'].first().to_dict()
        
        # Calculate elasticity by category
        category_elasticity = {}
        for product_id, data in elasticity_results.items():
            category = product_categories.get(product_id)
            if category not in category_elasticity:
                category_elasticity[category] = []
            
            category_elasticity[category].append(data['elasticity'])
        
        # Calculate average elasticity by category
        category_avg_elasticity = {}
        for cat, values in category_elasticity.items():
            if values:  # Ensure there are values
                # Use median to be more robust against outliers
                category_avg_elasticity[cat] = {
                    'elasticity': float(np.median(values)),
                    'elastic_status': 'Elastic' if abs(np.median(values)) > 1 else 'Inelastic',
                    'products_analyzed': len(values)
                }
        
        # Create category elasticity visualization
        if category_avg_elasticity:
            categories = list(category_avg_elasticity.keys())
            elasticities = [abs(data['elasticity']) for data in category_avg_elasticity.values()]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(categories, elasticities)
            plt.axhline(y=1, color='r', linestyle='--', label='Elastic Threshold')
            plt.title('Price Elasticity by Category', fontsize=16)
            plt.ylabel('Absolute Elasticity Value')
            plt.xlabel('Category')
            
            # Color bars based on elasticity
            for i, bar in enumerate(bars):
                if elasticities[i] > 1:
                    bar.set_color('orange')
                else:
                    bar.set_color('skyblue')
            
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.output_paths['visualizations']}/category_elasticity.png")
            plt.close()
        
        # Calculate performance metrics
        end = datetime.now()
        self.performance_metrics['elasticity_analysis'] = {
            'duration_seconds': (end - start).total_seconds(),
            'products_analyzed': len(elasticity_results),
            'elastic_products': elastic_count,
            'inelastic_products': inelastic_count
        }
        
        # Store results
        results = {
            'product_elasticity': elasticity_results,
            'category_elasticity': category_avg_elasticity,
            'analysis_summary': {
                'products_analyzed': len(elasticity_results),
                'elastic_products': elastic_count,
                'inelastic_products': inelastic_count
            }
        }
        
        return results 

    def analyze_discount_effectiveness(self):
        """Analyze the effectiveness of discounts on sales and revenue."""
        print("Analyzing discount effectiveness...")
        start = datetime.now()
        
        results = {}
        
        # Calculate average units sold by discount level
        discount_impact = self.df.groupby('Discount')['Units Sold'].mean().reset_index()
        
        # Plot discount impact on sales
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Discount', y='Units Sold', data=discount_impact)
        plt.title('Average Units Sold by Discount Level', fontsize=16)
        plt.ylabel('Average Units Sold')
        plt.xlabel('Discount (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/discount_impact_sales.png")
        plt.close()
        
        # Revenue by discount level
        discount_revenue = self.df.groupby('Discount')['Revenue'].mean().reset_index()
        
        # Plot discount impact on revenue
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Discount', y='Revenue', data=discount_revenue)
        plt.title('Average Revenue by Discount Level', fontsize=16)
        plt.ylabel('Average Revenue ($)')
        plt.xlabel('Discount (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/discount_impact_revenue.png")
        plt.close()
        
        # Find optimal discount level (highest revenue)
        optimal_discount = discount_revenue.loc[discount_revenue['Revenue'].idxmax()]
        
        # Calculate discount effectiveness by category
        category_discount = {}
        for category in self.df['Category'].unique():
            category_data = self.df[self.df['Category'] == category]
            
            if len(category_data) > 0:
                # Calculate revenue by discount for this category
                cat_discount_revenue = category_data.groupby('Discount')['Revenue'].mean().reset_index()
                
                if not cat_discount_revenue.empty:
                    # Find optimal discount for this category
                    cat_optimal = cat_discount_revenue.loc[cat_discount_revenue['Revenue'].idxmax()]
                    
                    category_discount[category] = {
                        'optimal_discount': float(cat_optimal['Discount']),
                        'max_revenue': float(cat_optimal['Revenue']),
                        'revenue_by_discount': cat_discount_revenue.to_dict(orient='records')
                    }
        
        # Calculate performance metrics
        end = datetime.now()
        self.performance_metrics['discount_analysis'] = {
            'duration_seconds': (end - start).total_seconds()
        }
        
        # Store results
        results = {
            'discount_impact_sales': discount_impact.to_dict(orient='records'),
            'discount_impact_revenue': discount_revenue.to_dict(orient='records'),
            'optimal_discount': {
                'discount': float(optimal_discount['Discount']),
                'revenue': float(optimal_discount['Revenue'])
            },
            'category_discount': category_discount
        }
        
        return results 

    def analyze_competitive_pricing(self):
        """Analyze the impact of competitive pricing strategies."""
        print("Analyzing competitive pricing...")
        start = datetime.now()
        
        results = {}
        
        # Calculate price difference from competitor
        self.df['Price_Difference'] = self.df['Price'] - self.df['Competitor Pricing']
        self.df['Price_Diff_Pct'] = (self.df['Price'] - self.df['Competitor Pricing']) / self.df['Competitor Pricing'] * 100
        
        # Create bins for price difference percentage
        bins = [-100, -10, -5, -2, 0, 2, 5, 10, 100]
        labels = ['<-10%', '-10% to -5%', '-5% to -2%', '-2% to 0%', '0% to 2%', '2% to 5%', '5% to 10%', '>10%']
        self.df['Price_Diff_Bucket'] = pd.cut(self.df['Price_Diff_Pct'], bins=bins, labels=labels)
        
        # Calculate impact on sales
        price_diff_impact = self.df.groupby('Price_Diff_Bucket', observed=False)['Units Sold'].mean().reset_index()
        
        # Plot impact on sales
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Price_Diff_Bucket', y='Units Sold', data=price_diff_impact)
        plt.title('Impact of Price Difference on Sales', fontsize=16)
        plt.ylabel('Average Units Sold')
        plt.xlabel('Price Difference from Competitor')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/price_diff_impact.png")
        plt.close()
        
        # Calculate impact on revenue
        revenue_by_diff = self.df.groupby('Price_Diff_Bucket', observed=False)['Revenue'].mean().reset_index()
        
        # Plot impact on revenue
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Price_Diff_Bucket', y='Revenue', data=revenue_by_diff)
        plt.title('Impact of Price Difference on Revenue', fontsize=16)
        plt.ylabel('Average Revenue ($)')
        plt.xlabel('Price Difference from Competitor')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_paths['visualizations']}/price_diff_revenue.png")
        plt.close()
        
        # Find optimal price difference
        optimal_diff = revenue_by_diff.loc[revenue_by_diff['Revenue'].idxmax()]
        
        # Calculate category-specific competitive positioning
        category_competition = {}
        for category in self.df['Category'].unique():
            category_data = self.df[self.df['Category'] == category]
            
            if len(category_data) > 0:
                # Calculate revenue by price difference for this category
                cat_revenue_by_diff = category_data.groupby('Price_Diff_Bucket', observed=False)['Revenue'].mean().reset_index()
                
                if not cat_revenue_by_diff.empty:
                    # Find optimal positioning for this category
                    cat_optimal = cat_revenue_by_diff.loc[cat_revenue_by_diff['Revenue'].idxmax()]
                    
                    category_competition[category] = {
                        'optimal_diff': str(cat_optimal['Price_Diff_Bucket']),
                        'max_revenue': float(cat_optimal['Revenue']),
                        'revenue_by_diff': cat_revenue_by_diff.to_dict(orient='records')
                    }
        
        # Calculate performance metrics
        end = datetime.now()
        self.performance_metrics['competitive_analysis'] = {
            'duration_seconds': (end - start).total_seconds()
        }
        
        # Store results
        results = {
            'price_diff_impact': price_diff_impact.to_dict(orient='records'),
            'revenue_by_diff': revenue_by_diff.to_dict(orient='records'),
            'optimal_diff': {
                'price_diff': str(optimal_diff['Price_Diff_Bucket']),
                'revenue': float(optimal_diff['Revenue'])
            },
            'category_competition': category_competition
        }
        
        return results 

    def generate_pricing_recommendations(self, elasticity_results, discount_results, competition_results):
        """Generate comprehensive pricing recommendations."""
        print("Generating pricing recommendations...")
        
        with open(f"{self.output_paths['recommendations']}/pricing_recommendations.md", "w") as f:
            f.write("# Pricing Strategy Recommendations\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Price Elasticity Recommendations
            f.write("## Price Elasticity Analysis\n\n")
            f.write("Price elasticity measures how sensitive demand is to changes in price.\n")
            f.write("- Elasticity < -1: Elastic (demand is highly sensitive to price)\n")
            f.write("- Elasticity > -1: Inelastic (demand is less sensitive to price)\n\n")
            
            f.write("### Category Elasticity\n\n")
            f.write("| Category | Price Elasticity | Recommendation |\n")
            f.write("|----------|------------------|----------------|\n")
            
            category_elasticity = elasticity_results.get('category_elasticity', {})
            for category, data in category_elasticity.items():
                elasticity = data['elasticity']
                if elasticity > -0.5:
                    recommendation = "Consider price increase, demand is highly inelastic"
                elif elasticity > -1:
                    recommendation = "Modest price increase possible, demand is inelastic"
                elif elasticity > -1.5:
                    recommendation = "Maintain current pricing, moderate elasticity"
                else:
                    recommendation = "Consider price decrease, demand is highly elastic"
                
                f.write(f"| {category} | {elasticity:.2f} | {recommendation} |\n")
            f.write("\n")
            
            # Product-specific recommendations
            f.write("### Product-Specific Pricing Recommendations\n\n")
            f.write("| Product ID | Elasticity | Current Avg Price | Recommendation |\n")
            f.write("|------------|------------|-------------------|----------------|\n")
            
            product_elasticity = elasticity_results.get('product_elasticity', {})
            sorted_products = sorted(product_elasticity.items(), key=lambda x: x[1]['elasticity'])
            
            # Show top elastic and inelastic products
            top_products = sorted_products[:5] + sorted_products[-5:]
            
            for product_id, data in top_products:
                elasticity = data['elasticity']
                avg_price = data['avg_price']
                
                if elasticity > -0.5:
                    recommendation = f"Increase price by 5-10%, current ${avg_price:.2f}"
                elif elasticity > -1:
                    recommendation = f"Small increase (2-5%), current ${avg_price:.2f}"
                elif elasticity > -1.5:
                    recommendation = f"Maintain near ${avg_price:.2f}, monitor competition"
                else:
                    recommendation = f"Decrease price or offer promotions, current ${avg_price:.2f}"
                
                f.write(f"| {product_id} | {elasticity:.2f} | ${avg_price:.2f} | {recommendation} |\n")
            f.write("\n")
            
            # Discount Strategy Recommendations
            f.write("## Discount Strategy\n\n")
            
            if 'optimal_discount' in discount_results:
                optimal = discount_results['optimal_discount']
                f.write(f"The optimal discount level across all products is **{optimal['discount']}%**, ")
                f.write(f"which generates an average revenue of **${optimal['revenue']:.2f}** per transaction.\n\n")
            
            f.write("### Category-Specific Discount Recommendations\n\n")
            f.write("| Category | Optimal Discount | Max Revenue |\n")
            f.write("|----------|------------------|-------------|\n")
            
            category_discount = discount_results.get('category_discount', {})
            for category, data in category_discount.items():
                f.write(f"| {category} | {data['optimal_discount']}% | ${data['max_revenue']:.2f} |\n")
            f.write("\n")
            
            # Competitive Pricing Recommendations
            f.write("## Competitive Pricing Strategy\n\n")
            
            if 'optimal_diff' in competition_results:
                optimal_diff = competition_results['optimal_diff']
                f.write(f"The optimal price differential compared to competitors is **{optimal_diff['price_diff']}**, ")
                f.write(f"which generates an average revenue of **${optimal_diff['revenue']:.2f}** per transaction.\n\n")
            
            f.write("### Category-Specific Competitive Positioning\n\n")
            f.write("| Category | Optimal Price Position | Max Revenue |\n")
            f.write("|----------|------------------------|-------------|\n")
            
            category_competition = competition_results.get('category_competition', {})
            for category, data in category_competition.items():
                f.write(f"| {category} | {data['optimal_diff']} | ${data['max_revenue']:.2f} |\n")
            f.write("\n")
            
            # Overall Pricing Recommendations
            f.write("## Strategic Pricing Initiatives\n\n")
            f.write("1. **Dynamic Pricing**: Implement dynamic pricing for highly elastic products\n")
            f.write("2. **Price Skimming**: For inelastic categories, gradually decrease prices from initial high points\n")
            f.write("3. **Penetration Pricing**: For elastic categories, start with lower prices to gain market share\n")
            f.write("4. **Psychological Pricing**: Use price points ending in .99 for elastic products\n")
            f.write("5. **Bundle Pricing**: Create bundles for complementary products, especially with elastic items\n")
            f.write("6. **Price Anchoring**: Display premium products alongside standard offerings to increase perception of value\n")
            f.write("7. **Seasonal Adjustments**: Adjust pricing strategies by season based on the seasonality analysis\n")
    
    def run_pricing_analysis(self):
        """Run the complete pricing strategy analysis pipeline."""
        print("Starting pricing strategy analysis...")
        
        # Load data
        self.load_data()
        
        # Create output directories
        self.create_output_dirs()
        
        # Run price elasticity analysis
        elasticity_results = self.analyze_price_elasticity()
        
        # Analyze discount effectiveness
        discount_results = self.analyze_discount_effectiveness()
        
        # Analyze competitive pricing
        competition_results = self.analyze_competitive_pricing()
        
        # Generate pricing recommendations
        self.generate_pricing_recommendations(elasticity_results, discount_results, competition_results)
        
        # Calculate total runtime
        total_runtime = (datetime.now() - self.start_time).total_seconds()
        self.performance_metrics['total_runtime'] = total_runtime
        
        # Save performance metrics
        with open(f"{self.output_paths['performance']}/performance_metrics.json", 'w') as f:
            json.dump(self.performance_metrics, f, indent=4)
        
        # Save results
        results = {
            'elasticity_analysis': elasticity_results,
            'discount_analysis': discount_results,
            'competitive_pricing': competition_results,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(f"{self.output_dir}/pricing_strategy/pricing_strategy_results.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Pricing analysis completed in {total_runtime:.2f} seconds")
        print(f"Results saved to {self.output_dir}/pricing_strategy/")


def main():
    """Main function to run the pricing strategy analysis."""
    print("Starting Retail Pricing Strategy Analysis...")
    
    data_path = "data/retail_store_inventory.csv"
    output_dir = "output"
    
    # Initialize and run pricing analysis
    pricing = PricingStrategy(data_path, output_dir)
    pricing.run_pricing_analysis()
    
    print("Pricing strategy analysis complete!")

if __name__ == "__main__":
    main()