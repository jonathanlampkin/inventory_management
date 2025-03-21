import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_data(filepath):
    """Load the dataset and process dates."""
    print("Loading data...")
    start_time = datetime.now()
    
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Day'] = df['Date'].dt.day
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['MonthYear'] = df['Date'].dt.strftime('%Y-%m')
    
    end_time = datetime.now()
    print(f"Data loaded in {(end_time-start_time).total_seconds():.2f} seconds")
    
    return df

def create_output_dirs():
    """Create necessary output directories."""
    print("Creating output directories...")
    
    dirs = [
        'output/forecasting/models', 
        'output/forecasting/visualizations',
        'output/forecasting/predictions',
        'output/forecasting/evaluation'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def aggregate_data(df):
    """Aggregate data for forecasting."""
    print("Aggregating data for forecasting...")
    start_time = datetime.now()
    
    # Total daily sales
    daily_sales = df.groupby('Date')['Units Sold'].sum().reset_index()
    daily_sales.set_index('Date', inplace=True)
    
    # Sales by category
    category_sales = df.groupby(['Date', 'Category'])['Units Sold'].sum().unstack(fill_value=0)
    
    # Sales by store
    store_sales = df.groupby(['Date', 'Store ID'])['Units Sold'].sum().unstack(fill_value=0)
    
    # Get top products by total sales
    top_products = df.groupby('Product ID')['Units Sold'].sum().nlargest(5).index
    product_sales = df[df['Product ID'].isin(top_products)].groupby(['Date', 'Product ID'])['Units Sold'].sum().unstack(fill_value=0)
    
    end_time = datetime.now()
    print(f"Data aggregation completed in {(end_time-start_time).total_seconds():.2f} seconds")
    
    return {
        'daily_sales': daily_sales,
        'category_sales': category_sales,
        'store_sales': store_sales,
        'product_sales': product_sales
    }

def plot_time_series(agg_data, output_dir):
    """Plot time series data for visualization."""
    print("Plotting time series data...")
    start_time = datetime.now()
    
    # Plot daily sales
    plt.figure(figsize=(14, 7))
    agg_data['daily_sales'].plot()
    plt.title('Total Daily Sales', fontsize=16)
    plt.ylabel('Units Sold')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/visualizations/daily_sales_time_series.png")
    plt.close()
    
    # Plot category sales
    plt.figure(figsize=(14, 7))
    agg_data['category_sales'].plot()
    plt.title('Daily Sales by Category', fontsize=16)
    plt.ylabel('Units Sold')
    plt.xlabel('Date')
    plt.legend(title='Category')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/visualizations/category_sales_time_series.png")
    plt.close()
    
    # Plot store sales
    plt.figure(figsize=(16, 8))
    agg_data['store_sales'].plot()
    plt.title('Daily Sales by Store', fontsize=16)
    plt.ylabel('Units Sold')
    plt.xlabel('Date')
    plt.legend(title='Store ID')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/visualizations/store_sales_time_series.png")
    plt.close()
    
    end_time = datetime.now()
    print(f"Time series plotting completed in {(end_time-start_time).total_seconds():.2f} seconds")

def train_sarima_models(agg_data, output_dir):
    """Train SARIMA models for different time series."""
    print("Training SARIMA models (with fixed parameters, not using auto_arima)...")
    start_time = datetime.now()
    
    forecast_days = 30  # Forecast for next 30 days
    results = {}
    
    # Helper function to train and evaluate SARIMA model
    def train_evaluate_sarima(series, name):
        print(f"Training SARIMA model for {name}...")
        
        # Split data into train and test
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]
        
        # Use fixed SARIMA parameters
        # Order: (p,d,q), Seasonal Order: (P,D,Q,s)
        sarima_params = {
            'order': (1, 1, 1),
            'seasonal_order': (1, 1, 0, 7)  # Weekly seasonality
        }
        
        # Train SARIMA model
        model = SARIMAX(
            train,
            order=sarima_params['order'],
            seasonal_order=sarima_params['seasonal_order'],
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        try:
            model_fit = model.fit(disp=False)
            
            # Make predictions on test set
            predictions = model_fit.get_forecast(steps=len(test))
            predicted_mean = predictions.predicted_mean
            
            # Calculate metrics
            mae = mean_absolute_error(test, predicted_mean)
            rmse = np.sqrt(mean_squared_error(test, predicted_mean))
            r2 = r2_score(test, predicted_mean)
            
            # Make future predictions
            future_predictions = model_fit.get_forecast(steps=forecast_days)
            forecast = future_predictions.predicted_mean
            conf_int = future_predictions.conf_int()
            
            # Save the model
            joblib.dump(model_fit, f"{output_dir}/models/sarima_{name.replace(' ', '_').lower()}.pkl")
            
            # Plot actual vs predicted
            plt.figure(figsize=(14, 7))
            plt.plot(test.index, test, label='Actual')
            plt.plot(test.index, predicted_mean, label='Predicted')
            plt.title(f'SARIMA Model Forecast vs Actual: {name}', fontsize=16)
            plt.ylabel('Units Sold')
            plt.xlabel('Date')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/evaluation/sarima_{name.replace(' ', '_').lower()}_validation.png")
            plt.close()
            
            # Plot future forecast
            future_index = pd.date_range(
                start=series.index[-1] + pd.Timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            plt.figure(figsize=(14, 7))
            plt.plot(series[-90:].index, series[-90:], label='Historical')
            plt.plot(future_index, forecast, label='Forecast')
            plt.fill_between(
                future_index,
                conf_int.iloc[:, 0],
                conf_int.iloc[:, 1],
                alpha=0.2,
                label='95% Confidence Interval'
            )
            plt.title(f'SARIMA Forecast: {name}', fontsize=16)
            plt.ylabel('Units Sold')
            plt.xlabel('Date')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/predictions/sarima_{name.replace(' ', '_').lower()}_forecast.png")
            plt.close()
            
            # Store results
            model_result = {
                'metrics': {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2': float(r2)
                },
                'forecast': {
                    'dates': [date.strftime('%Y-%m-%d') for date in future_index],
                    'values': [float(val) for val in forecast],
                    'lower_bound': [float(val) for val in conf_int.iloc[:, 0]],
                    'upper_bound': [float(val) for val in conf_int.iloc[:, 1]]
                }
            }
            
            return model_result
            
        except Exception as e:
            print(f"Error training SARIMA model for {name}: {str(e)}")
            return {
                'metrics': {
                    'mae': float('nan'),
                    'rmse': float('nan'),
                    'r2': float('nan')
                },
                'error': str(e)
            }
    
    # Train model for overall daily sales
    print("Training models for overall daily sales...")
    results['overall_daily'] = train_evaluate_sarima(agg_data['daily_sales']['Units Sold'], 'Overall Sales')
    
    # Train models for each category
    print("Training models for category sales...")
    for category in agg_data['category_sales'].columns:
        results[f'category_{category}'] = train_evaluate_sarima(
            agg_data['category_sales'][category],
            f'{category} Sales'
        )
    
    # Train models for top products
    print("Training models for product sales...")
    for product in agg_data['product_sales'].columns:
        results[f'product_{product}'] = train_evaluate_sarima(
            agg_data['product_sales'][product],
            f'{product} Sales'
        )
    
    end_time = datetime.now()
    print(f"SARIMA modeling completed in {(end_time-start_time).total_seconds():.2f} seconds")
    
    return results

def train_ml_forecast_model(df, output_dir):
    """Train a machine learning model for forecasting."""
    print("Training machine learning forecasting model...")
    start_time = datetime.now()
    
    # Prepare data for machine learning forecast model
    print("Preparing data for machine learning forecast model...")
    
    # Aggregate by day and store
    daily_store_df = df.groupby(['Date', 'Store ID']).agg({
        'Units Sold': 'sum',
        'Inventory Level': 'mean',
        'Price': 'mean',
        'Discount': 'mean',
        'Weather Condition': 'first',
        'Holiday/Promotion': 'first',
        'Competitor Pricing': 'mean',
        'Month': 'first',
        'Year': 'first',
        'Day': 'first',
        'DayOfWeek': 'first',
        'WeekOfYear': 'first'
    }).reset_index()
    
    # Create lag features (previous days' sales)
    for store_id in daily_store_df['Store ID'].unique():
        store_data = daily_store_df[daily_store_df['Store ID'] == store_id].sort_values('Date')
        
        # Create lag features (1, 7, 14, 28 days)
        for lag in [1, 7, 14, 28]:
            col_name = f'Sales_Lag_{lag}'
            store_data[col_name] = store_data['Units Sold'].shift(lag)
        
        # Create rolling mean features
        for window in [7, 14, 28]:
            col_name = f'Sales_RollingMean_{window}'
            store_data[col_name] = store_data['Units Sold'].rolling(window=window).mean()
        
        # Update the main dataframe
        daily_store_df.loc[daily_store_df['Store ID'] == store_id] = store_data
    
    # Drop NaN values (from lag/rolling features)
    daily_store_df.dropna(inplace=True)
    
    # Convert categorical features
    daily_store_df = pd.get_dummies(daily_store_df, columns=['Weather Condition', 'Store ID'])
    
    # Define features and target
    features = [col for col in daily_store_df.columns if col not in ['Date', 'Units Sold']]
    X = daily_store_df[features]
    y = daily_store_df['Units Sold']
    
    # Split data into train and test
    train_size = int(len(daily_store_df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
    
    # Save the model
    joblib.dump(rf_model, f"{output_dir}/models/random_forest_forecast.pkl")
    joblib.dump(scaler, f"{output_dir}/models/feature_scaler.pkl")
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title('Random Forest: Actual vs Predicted Sales', fontsize=16)
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/evaluation/rf_actual_vs_predicted.png")
    plt.close()
    
    # Plot feature importances
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(14, 10))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Random Forest Feature Importance', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/evaluation/feature_importance.png")
    plt.close()
    
    end_time = datetime.now()
    print(f"ML forecasting completed in {(end_time-start_time).total_seconds():.2f} seconds")
    
    # Store results
    return {
        'metrics': {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        },
        'feature_importance': feature_importance.head(20).to_dict(orient='records'),
        'model_type': 'random_forest'
    }

def generate_forecast_report(sarima_results, ml_results, output_dir):
    """Generate a comprehensive forecast report."""
    print("Generating forecast report...")
    
    with open(f"{output_dir}/forecast_report.md", "w") as f:
        f.write("# Demand Forecasting Report\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall Forecast Summary
        f.write("## Forecast Summary\n\n")
        f.write("This report provides demand forecasts at multiple levels (overall, category, product) using SARIMA and machine learning models.\n\n")
        
        # SARIMA Models Results
        f.write("## Time Series (SARIMA) Models\n\n")
        
        # Overall sales results
        f.write("### Overall Sales Forecast\n\n")
        if 'overall_daily' in sarima_results and 'metrics' in sarima_results['overall_daily']:
            metrics = sarima_results['overall_daily']['metrics']
            f.write(f"- MAE: {metrics['mae']:.2f}\n")
            f.write(f"- RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"- R²: {metrics['r2']:.2f}\n\n")
        
        f.write("![Overall Sales Forecast](./predictions/sarima_overall_sales_forecast.png)\n\n")
        
        # Category results
        f.write("### Category Forecasts\n\n")
        category_metrics = {}
        for key, result in sarima_results.items():
            if key.startswith('category_') and 'metrics' in result:
                category = key.replace('category_', '')
                category_metrics[category] = result['metrics']
        
        # Sort categories by RMSE (lower is better)
        sorted_categories = sorted(category_metrics.items(), key=lambda x: x[1]['rmse'])
        
        f.write("| Category | RMSE | MAE | R² |\n")
        f.write("|----------|------|-----|----|\n")
        for category, metrics in sorted_categories:
            f.write(f"| {category} | {metrics['rmse']:.2f} | {metrics['mae']:.2f} | {metrics['r2']:.2f} |\n")
        f.write("\n")
        
        # Product results
        f.write("### Product Forecasts\n\n")
        product_metrics = {}
        for key, result in sarima_results.items():
            if key.startswith('product_') and 'metrics' in result:
                product = key.replace('product_', '')
                product_metrics[product] = result['metrics']
        
        # Sort products by RMSE (lower is better)
        sorted_products = sorted(product_metrics.items(), key=lambda x: x[1]['rmse'])
        
        f.write("| Product ID | RMSE | MAE | R² |\n")
        f.write("|------------|------|-----|----|\n")
        for product, metrics in sorted_products:
            f.write(f"| {product} | {metrics['rmse']:.2f} | {metrics['mae']:.2f} | {metrics['r2']:.2f} |\n")
        f.write("\n")
        
        # Machine Learning Model Results
        f.write("## Machine Learning Forecast Model\n\n")
        
        if 'metrics' in ml_results:
            metrics = ml_results['metrics']
            f.write(f"- Model Type: {ml_results['model_type']}\n")
            f.write(f"- MAE: {metrics['mae']:.2f}\n")
            f.write(f"- RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"- R²: {metrics['r2']:.2f}\n\n")
        
        # Feature importance
        f.write("### Top 20 Features by Importance\n\n")
        f.write("| Feature | Importance |\n")
        f.write("|---------|------------|\n")
        
        if 'feature_importance' in ml_results:
            for feature in ml_results['feature_importance']:
                f.write(f"| {feature['Feature']} | {feature['Importance']:.4f} |\n")
            f.write("\n")
        
        f.write("![Feature Importance](./evaluation/feature_importance.png)\n\n")
        f.write("![Actual vs Predicted](./evaluation/rf_actual_vs_predicted.png)\n\n")
        
        # Forecast recommendations
        f.write("## Recommendations\n\n")
        f.write("Based on the forecasting models, we recommend:\n\n")
        f.write("1. **High Priority Inventory Planning**: Focus on products with strong upward sales trends\n")
        f.write("2. **Seasonal Adjustments**: Adjust inventory levels based on identified seasonal patterns\n")
        f.write("3. **Store-Specific Strategies**: Implement store-specific inventory levels based on forecast models\n")
        f.write("4. **Forecast Accuracy Improvements**: Consider adding external factors to improve forecast accuracy\n")
        f.write("5. **Early Warning System**: Monitor forecast vs. actual sales daily to catch deviations early\n")

def train_prophet_model(self, agg_data, output_dir):
    """Train Facebook Prophet models for more accurate forecasting."""
    try:
        from prophet import Prophet
        
        print("Training Prophet models...")
        start_time = datetime.now()
        
        forecast_days = 30
        results = {}
        
        # Function to train and evaluate Prophet model
        def train_evaluate_prophet(series, name):
            # Prepare data for Prophet
            df_prophet = pd.DataFrame({'ds': series.index, 'y': series.values})
            
            # Create and fit model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            model.fit(df_prophet)
            
            # Make future dataframe
            future = model.make_future_dataframe(periods=forecast_days)
            
            # Forecast
            forecast = model.predict(future)
            
            # Extract results
            train_size = int(len(series) * 0.8)
            y_true = series[train_size:]
            dates_test = y_true.index
            
            # Get predictions for test period
            y_pred = forecast.loc[forecast['ds'].isin(dates_test), 'yhat'].values
            
            # Calculate metrics
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            # Plot forecast
            fig = model.plot(forecast)
            plt.title(f'Prophet Forecast: {name}')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/predictions/prophet_{name}_forecast.png")
            plt.close()
            
            # Plot components
            fig = model.plot_components(forecast)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/predictions/prophet_{name}_components.png")
            plt.close()
            
            return {
                'metrics': {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2': float(r2)
                },
                'forecast': forecast.to_dict(orient='records')
            }
        
        # Train Prophet model for overall daily sales
        results['overall_daily'] = train_evaluate_prophet(
            agg_data['daily_sales']['Units Sold'], 
            'overall_sales'
        )
        
        # Train models for top categories
        for category in agg_data['category_sales'].columns[:3]:
            results[f'category_{category}'] = train_evaluate_prophet(
                agg_data['category_sales'][category],
                f'category_{category}'
            )
        
        end_time = datetime.now()
        print(f"Prophet forecasting completed in {(end_time-start_time).total_seconds():.2f} seconds")
        
        return results
    
    except ImportError:
        print("Prophet is not installed. Using only SARIMA and ML models.")
        return {}

def compare_models(self):
    """Compare different forecasting models on the same data."""
    print("Comparing multiple forecasting models...")
    
    # Prepare test data - last 30 days
    test_size = 30
    train_df = self.df[:-test_size].copy()
    test_df = self.df[-test_size:].copy()
    
    # Models to compare
    models = {
        'SARIMA': self.train_sarima_model,
        'RandomForest': self.train_random_forest_model,
        'Prophet': self.train_prophet_model,
        'LSTM': self.train_lstm_model,
        'XGBoost': self.train_xgboost_model
    }
    
    # Store results
    results = {}
    
    # Compare each model
    for name, model_func in models.items():
        print(f"Evaluating {name} model...")
        try:
            model, predictions, metrics = model_func(train_df, test_df)
            results[name] = {
                'rmse': metrics['rmse'],
                'mape': metrics['mape'],
                'mae': metrics['mae'],
                'r2': metrics['r2']
            }
            print(f"  {name} RMSE: {metrics['rmse']:.2f}, MAPE: {metrics['mape']:.2f}%")
        except Exception as e:
            print(f"  Error with {name} model: {str(e)}")
            results[name] = {'error': str(e)}
    
    # Save comparison results
    output_dir = f"{self.output_dir}/forecasting/model_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/model_comparison_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create comparison visualization
    self.visualize_model_comparison(results, output_dir)
    
    # Return best model name
    best_model = min(results.items(), key=lambda x: x[1].get('rmse', float('inf')))[0]
    print(f"Best model based on RMSE: {best_model}")
    
    return best_model, results

def main():
    """Main function to run the forecasting pipeline."""
    start_time = datetime.now()
    print("Starting demand forecasting analysis...")
    
    # Set paths
    data_path = "data/retail_store_inventory.csv"
    output_dir = "output/forecasting"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_data(data_path)
    
    # Create output directories
    create_output_dirs()
    
    # Aggregate data for forecasting
    agg_data = aggregate_data(df)
    
    # Plot time series
    plot_time_series(agg_data, output_dir)
    
    # Train SARIMA models
    sarima_results = train_sarima_models(agg_data, output_dir)
    
    # Train ML forecasting model
    ml_results = train_ml_forecast_model(df, output_dir)
    
    # Generate forecast report
    generate_forecast_report(sarima_results, ml_results, output_dir)
    
    # Save results
    results = {
        'sarima_models': sarima_results,
        'ml_model': ml_results,
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(f"{output_dir}/forecast_results.json", 'w') as f:
        # Convert non-serializable objects to strings
        json_str = json.dumps(results, default=str, indent=4)
        f.write(json_str)
    
    end_time = datetime.now()
    print(f"Forecasting completed in {(end_time-start_time).total_seconds():.2f} seconds")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main() 