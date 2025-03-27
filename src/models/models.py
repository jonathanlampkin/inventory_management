import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import os

class TimeSeriesForecaster:
    """Handles time series forecasting using various models."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/results", exist_ok=True)
    
    def prepare_ts_data(self, cluster_data):
        """Prepare time series data for modeling."""
        # Create sequences for LSTM
        def create_sequences(data, seq_length=30):
            sequences = []
            targets = []
            for i in range(len(data) - seq_length):
                sequences.append(data[i:(i + seq_length)])
                targets.append(data[i + seq_length])
            return np.array(sequences), np.array(targets)
        
        # Prepare data for each model type
        data = cluster_data['Units Sold'].values
        
        # Prophet format
        prophet_data = pd.DataFrame({
            'ds': cluster_data['Date'],
            'y': data
        })
        
        # SARIMA format
        sarima_data = data
        
        # LSTM format
        X_lstm, y_lstm = create_sequences(data)
        
        # XGBoost format
        X_xgb = cluster_data[['Month', 'Year', 'DayOfWeek', 'DayOfMonth', 'WeekOfYear']].values
        y_xgb = data
        
        return {
            'prophet': prophet_data,
            'sarima': sarima_data,
            'lstm': (X_lstm, y_lstm),
            'xgboost': (X_xgb, y_xgb)
        }
    
    def evaluate_model(self, model, data, model_type):
        """Evaluate a forecasting model."""
        if model_type == 'prophet':
            # Split data
            train_size = int(len(data) * 0.8)
            train_data = data[:train_size]
            test_data = data[train_size:]
            
            # Fit and predict
            model.fit(train_data)
            future = model.make_future_dataframe(periods=len(test_data))
            forecast = model.predict(future)
            
            # Calculate metrics
            y_true = test_data['y'].values
            y_pred = forecast['yhat'].values[-len(test_data):]
        
        elif model_type == 'sarima':
            # Split data
            train_size = int(len(data) * 0.8)
            train_data = data[:train_size]
            test_data = data[train_size:]
            
            # Fit and predict
            model.fit(train_data)
            forecast = model.forecast(len(test_data))
            
            # Calculate metrics
            y_true = test_data
            y_pred = forecast
        
        elif model_type == 'lstm':
            X, y = data
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Fit and predict
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            y_true = y_test
            y_pred = y_pred.flatten()
        
        else:  # xgboost
            X, y = data
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            y_true = y_test
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape)
        }
    
    def select_forecast_model(self, df, features, clusters):
        """Select best forecasting model for each cluster."""
        print("Selecting best forecasting model...")
        
        # Select diverse clusters for model comparison
        selected_clusters = self._select_diverse_clusters(features, n_samples=3)
        
        # Define models to compare
        models = {
            'prophet': Prophet(),
            'sarima': SARIMAX(order=(1, 1, 1), seasonal_order=(1, 1, 0, 7)),
            'lstm': Sequential([
                LSTM(50, input_shape=(30, 1)),
                Dense(1)
            ]),
            'xgboost': xgb.XGBRegressor()
        }
        
        # Compare models on selected clusters
        cluster_scores = {}
        for cluster_id in selected_clusters:
            print(f"\nEvaluating models for cluster {cluster_id}")
            cluster_data = df[features['Cluster'] == cluster_id]
            prepared_data = self.prepare_ts_data(cluster_data)
            
            cluster_scores[cluster_id] = {}
            for name, model in models.items():
                try:
                    metrics = self.evaluate_model(model, prepared_data[name], name)
                    cluster_scores[cluster_id][name] = metrics
                    print(f"{name} metrics for cluster {cluster_id}:")
                    print(f"RMSE: {metrics['rmse']:.2f}")
                    print(f"MAE: {metrics['mae']:.2f}")
                    print(f"MAPE: {metrics['mape']:.2f}%")
                except Exception as e:
                    print(f"Error with {name} on cluster {cluster_id}: {str(e)}")
        
        # Select best model based on weighted average of metrics
        best_model = None
        best_score = -float('inf')
        
        for name, model in models.items():
            cluster_metrics = []
            for cluster_id in selected_clusters:
                if name in cluster_scores[cluster_id]:
                    metrics = cluster_scores[cluster_id][name]
                    # Weighted score (favoring MAPE)
                    score = 0.5 * (1 / (1 + metrics['mape'])) + \
                           0.3 * (1 / (1 + metrics['rmse'])) + \
                           0.2 * (1 / (1 + metrics['mae']))
                    cluster_metrics.append(score)
            
            if cluster_metrics:
                avg_score = np.mean(cluster_metrics)
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = (name, model)
        
        # Save model selection results
        model_results = {
            'best_model': best_model[0],
            'best_score': float(best_score),
            'cluster_scores': cluster_scores
        }
        
        with open(f"{self.output_dir}/models/model_selection_results.json", 'w') as f:
            json.dump(model_results, f, indent=4)
        
        return best_model, best_score
    
    def generate_forecasts(self, df, features, clusters, best_model):
        """Generate forecasts for each cluster using the best model."""
        print("Generating forecasts...")
        
        forecasts = {}
        model_name, model = best_model
        
        for cluster_id in np.unique(clusters):
            cluster_data = df[features['Cluster'] == cluster_id]
            prepared_data = self.prepare_ts_data(cluster_data)
            
            try:
                if model_name == 'prophet':
                    # Fit on all data
                    model.fit(prepared_data['prophet'])
                    future = model.make_future_dataframe(periods=30)  # 30-day forecast
                    forecast = model.predict(future)
                    forecasts[cluster_id] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_dict()
                
                elif model_name == 'sarima':
                    # Fit on all data
                    model.fit(prepared_data['sarima'])
                    forecast = model.forecast(30)  # 30-day forecast
                    forecasts[cluster_id] = {
                        'forecast': forecast.tolist(),
                        'lower_bound': (forecast - 1.96 * np.std(forecast)).tolist(),
                        'upper_bound': (forecast + 1.96 * np.std(forecast)).tolist()
                    }
                
                elif model_name == 'lstm':
                    # Prepare sequence for prediction
                    last_sequence = prepared_data['lstm'][0][-1:]
                    predictions = []
                    for _ in range(30):
                        pred = model.predict(last_sequence.reshape(1, 30, 1))
                        predictions.append(pred[0][0])
                        last_sequence = np.roll(last_sequence, -1)
                        last_sequence[-1] = pred[0][0]
                    
                    forecasts[cluster_id] = {
                        'forecast': predictions,
                        'lower_bound': [p - 1.96 * np.std(predictions) for p in predictions],
                        'upper_bound': [p + 1.96 * np.std(predictions) for p in predictions]
                    }
                
                else:  # xgboost
                    # Prepare future dates
                    future_dates = pd.date_range(
                        start=cluster_data['Date'].max() + pd.Timedelta(days=1),
                        periods=30,
                        freq='D'
                    )
                    future_features = pd.DataFrame({
                        'Month': future_dates.month,
                        'Year': future_dates.year,
                        'DayOfWeek': future_dates.dayofweek,
                        'DayOfMonth': future_dates.day,
                        'WeekOfYear': future_dates.isocalendar().week
                    })
                    
                    # Generate predictions
                    predictions = model.predict(future_features)
                    forecasts[cluster_id] = {
                        'forecast': predictions.tolist(),
                        'lower_bound': (predictions - 1.96 * np.std(predictions)).tolist(),
                        'upper_bound': (predictions + 1.96 * np.std(predictions)).tolist()
                    }
            
            except Exception as e:
                print(f"Error generating forecast for cluster {cluster_id}: {str(e)}")
                forecasts[cluster_id] = None
        
        # Save forecasts
        with open(f"{self.output_dir}/results/forecasts.json", 'w') as f:
            json.dump(forecasts, f, indent=4)
        
        return forecasts
    
    def _select_diverse_clusters(self, features, n_samples=3):
        """Select diverse clusters for model comparison."""
        # Calculate cluster centroids
        centroids = features.groupby('Cluster').mean()
        
        # Use k-means to select diverse clusters
        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        diversity_groups = kmeans.fit_predict(centroids)
        
        # Select one cluster from each diversity group
        selected_clusters = []
        for i in range(n_samples):
            group_clusters = centroids[diversity_groups == i].index
            # Select the largest cluster in each group
            selected_cluster = max(group_clusters, 
                                 key=lambda x: len(features[features['Cluster'] == x]))
            selected_clusters.append(selected_cluster)
        
        return selected_clusters 