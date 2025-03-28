import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from typing import Dict, List, Optional, Tuple, Union
from joblib import dump, load
import os
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class HierarchicalForecasting:
    """
    Hierarchical Forecasting System
    
    Implements a three-level hierarchical forecasting approach:
    1. Global model for market-wide trends
    2. Cluster-specific models for segment patterns
    3. Product-level adjustments for individual characteristics
    
    Parameters:
    -----------
    base_model_type : str
        Type of model to use as base ('rf' for Random Forest, 'xgb' for XGBoost)
    min_cluster_size : int
        Minimum number of samples required to create a cluster-specific model
    """
    
    def __init__(self, base_model_type: str = 'xgb', min_cluster_size: int = 10):
        self.base_model_type = base_model_type
        self.min_cluster_size = min_cluster_size
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize models
        self.base_model = None
        self.cluster_models = {}
        self.product_adjustments = {}
        
        # Model performance metrics
        self.metrics = {
            'base_model': {},
            'cluster_models': {},
            'overall': {}
        }
        
        self.models: Dict[int, RandomForestRegressor] = {}
        self.scalers: Dict[int, StandardScaler] = {}
    
    def _get_model_path(self, model_type: str) -> str:
        """Get path for saved model."""
        return os.path.join(self.model_dir, f"{model_type}_model.joblib")
    
    def _get_scaler_path(self, cluster_id: int) -> str:
        """Get path for saved scaler."""
        return os.path.join(self.model_dir, f"scaler_cluster_{cluster_id}.joblib")
    
    def save_models(self) -> None:
        """Save trained models to disk."""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Save base model
            if self.base_model is not None:
                dump(self.base_model, self._get_model_path('base'))
            
            # Save cluster models
            for cluster_id, model in self.cluster_models.items():
                dump(model, self._get_model_path(f'cluster_{cluster_id}'))
            
            # Save product adjustments
            if self.product_adjustments:
                dump(self.product_adjustments, self._get_model_path('adjustments'))
            
            # Save models and scalers
            for cluster_id in self.models:
                dump(self.models[cluster_id], self._get_model_path(f"model_cluster_{cluster_id}"))
                dump(self.scalers[cluster_id], self._get_scaler_path(cluster_id))
            
            print("Successfully saved all models to disk")
            
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            raise
    
    def load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Try to load models
            for cluster_id in range(5):  # Assuming maximum of 5 clusters
                model_path = os.path.join(self.model_dir, f'model_cluster_{cluster_id}.joblib')
                scaler_path = os.path.join(self.model_dir, f'scaler_cluster_{cluster_id}.joblib')
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    try:
                        self.models[cluster_id] = load(model_path)
                        self.scalers[cluster_id] = load(scaler_path)
                        print(f"Successfully loaded model and scaler for cluster {cluster_id}")
                    except Exception as e:
                        print(f"Error loading model for cluster {cluster_id}: {str(e)}")
                        continue
            
            # Check if any models were loaded
            if not self.models:
                print("No valid models found on disk")
                return False
            
            print("Successfully loaded all models from disk")
            return True
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
    
    def _create_model(self) -> object:
        """Create a new model instance based on specified type."""
        if self.base_model_type == 'rf':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            return XGBRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature set for modeling."""
        # Handle empty data case
        if len(data) == 0:
            return pd.DataFrame()
        
        features = data.copy()
        
        # Ensure we have required columns
        if 'Units_Sold' not in features.columns:
            raise ValueError("Missing required column 'Units_Sold'")
        
        # Handle date column
        if 'date' in features.columns:
            if not pd.api.types.is_datetime64_any_dtype(features['date']):
                features['date'] = pd.to_datetime(features['date'])
        else:
            # Create a date range index if missing
            features['date'] = pd.date_range(
                start='2024-01-01',  # Use a default start date
                periods=len(features),
                freq='D'
            )
        
        # Ensure Product_ID exists for grouping
        if 'Product_ID' not in features.columns:
            features['Product_ID'] = 'default_product'
        
        # Sort by Product_ID and date
        features = features.sort_values(['Product_ID', 'date']).reset_index(drop=True)
        
        # Create time-based features
        features['day_of_week'] = features['date'].dt.dayofweek
        features['month'] = features['date'].dt.month
        features['year'] = features['date'].dt.year
        features['day_of_month'] = features['date'].dt.day
        
        # Create lag features
        for product_id in features['Product_ID'].unique():
            mask = features['Product_ID'] == product_id
            product_data = features[mask].copy()
            
            # Calculate lags
            features.loc[mask, 'lag_1'] = product_data['Units_Sold'].shift(1)
            features.loc[mask, 'lag_7'] = product_data['Units_Sold'].shift(7)
            
            # Calculate rolling means
            features.loc[mask, 'rolling_mean_7'] = product_data['Units_Sold'].rolling(7, min_periods=1).mean()
            features.loc[mask, 'rolling_mean_30'] = product_data['Units_Sold'].rolling(30, min_periods=1).mean()
        
        # Handle missing values
        features = features.ffill().bfill()
        
        # Ensure all numeric columns have finite values
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan)
        features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].mean())
        
        return features
    
    def _find_optimal_clusters(self, data: pd.DataFrame, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using elbow method and silhouette analysis."""
        # Prepare features
        features = self._prepare_features(data)
        
        # Calculate silhouette scores for different numbers of clusters
        silhouette_scores = []
        inertias = []
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            
            # Calculate silhouette score
            silhouette_scores.append(silhouette_score(features, cluster_labels))
            
            # Calculate inertia (within-cluster sum of squares)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        inertias = np.array(inertias)
        inertias_diff = np.diff(inertias)
        inertias_diff2 = np.diff(inertias_diff)
        elbow_idx = np.argmin(inertias_diff2) + 2
        
        # Find maximum silhouette score
        silhouette_idx = np.argmax(silhouette_scores) + 2
        
        # Use the average of elbow and silhouette methods
        optimal_clusters = int(np.round((elbow_idx + silhouette_idx) / 2))
        
        return optimal_clusters
    
    def train(self, data: pd.DataFrame, clusters: np.ndarray) -> None:
        """Train the hierarchical forecasting model."""
        # Define core and optional columns
        core_cols = ['Units_Sold']
        optional_cols = ['Price', 'Discount', 'Inventory_Level', 'Demand_Forecast', 'Competitor_Pricing']
        
        # Validate core columns
        missing_core_cols = [col for col in core_cols if col not in data.columns]
        if missing_core_cols:
            raise ValueError(f"Missing required core columns: {missing_core_cols}")
        
        # Check which optional columns are available
        available_cols = core_cols + [col for col in optional_cols if col in data.columns]
        print(f"Training with available columns: {available_cols}")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Try to load existing models first
        if self.load_models():
            print("Successfully loaded existing models from disk")
            return
        
        # Handle different cluster input types
        if isinstance(clusters, dict):
            if 'clusters' in clusters:
                cluster_labels = clusters['clusters']
            else:
                raise ValueError("Dictionary input must contain 'clusters' key with numpy array")
        else:
            cluster_labels = clusters
        
        # Train model for each cluster
        for cluster_id in np.unique(cluster_labels):
            print(f"Training model for cluster {cluster_id}...")
            
            # Get data for this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_data = data[cluster_mask].copy()
            
            if len(cluster_data) < self.min_cluster_size:
                print(f"Skipping cluster {cluster_id} due to insufficient data (needs {self.min_cluster_size}, has {len(cluster_data)})")
                continue
            
            try:
                # Prepare features and target
                X = cluster_data[available_cols].copy()
                y = X.pop('Units_Sold')  # Remove target from features
                
                # Add time-based features if date column exists
                if 'date' in cluster_data.columns:
                    if not pd.api.types.is_datetime64_any_dtype(cluster_data['date']):
                        cluster_data['date'] = pd.to_datetime(cluster_data['date'])
                    X['day_of_week'] = cluster_data['date'].dt.dayofweek
                    X['month'] = cluster_data['date'].dt.month
                    X['year'] = cluster_data['date'].dt.year
                    X['day_of_month'] = cluster_data['date'].dt.day
                
                # Check for constant columns
                constant_cols = X.columns[X.nunique() == 1]
                if len(constant_cols) > 0:
                    print(f"Warning: Found constant columns in cluster {cluster_id}: {constant_cols}")
                    X = X.drop(columns=constant_cols)
                
                # Check if we have any features left
                if len(X.columns) == 0:
                    print(f"Warning: No valid features for cluster {cluster_id} after preprocessing")
                    continue
                
                # Handle missing values
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())
                
                # Ensure all numeric columns have finite values
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train model
                model = self._create_model()
                model.fit(X_scaled, y)
                
                # Save model and scaler
                self.models[cluster_id] = model
                self.scalers[cluster_id] = scaler
                print(f"Successfully trained model for cluster {cluster_id}")
                
            except Exception as e:
                print(f"Error training model for cluster {cluster_id}: {str(e)}")
                continue
        
        # Check if any models were trained
        if not self.models:
            raise ValueError("No models were trained due to insufficient data in all clusters")
        
        # Save all models
        try:
            self.save_models()
            print("Successfully saved all models to disk")
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            raise
    
    def _train_base_model(self, features: pd.DataFrame) -> object:
        """Train the base model on complete dataset."""
        model = self._create_model()
        X = features.drop(['Units_Sold', 'date'], axis=1)
        y = features['Units_Sold']
        
        model.fit(X, y)
        return model
    
    def _train_cluster_model(self, cluster_data: pd.DataFrame, base_predictions: np.ndarray) -> object:
        """Train cluster-specific model."""
        model = self._create_model()
        features = self._prepare_features(cluster_data)
        
        # Use residuals as target
        residuals = cluster_data['Units_Sold'] - base_predictions
        
        X = features.drop(['Units_Sold', 'date'], axis=1)
        model.fit(X, residuals)
        
        return model
    
    def _calculate_product_adjustments(self, data: pd.DataFrame, base_predictions: np.ndarray) -> Dict:
        """Calculate product-specific adjustment factors."""
        adjustments = {}
        
        for product_id in data['Product_ID'].unique():
            product_mask = data['Product_ID'] == product_id
            actual = data.loc[product_mask, 'Units_Sold']
            predicted = base_predictions[product_mask]
            
            # Calculate multiplicative adjustment factor
            if len(actual) > 0 and np.mean(predicted) > 0:
                adjustment = np.mean(actual) / np.mean(predicted)
                adjustments[product_id] = adjustment
            else:
                adjustments[product_id] = 1.0
        
        return adjustments
    
    def predict(self, data: pd.DataFrame, clusters: np.ndarray) -> np.ndarray:
        """Make predictions using the trained models."""
        # Validate core columns
        core_cols = ['Units_Sold']
        optional_cols = ['Price', 'Discount', 'Inventory_Level', 'Demand_Forecast', 'Competitor_Pricing']
        
        # Check which columns are available
        available_cols = core_cols + [col for col in optional_cols if col in data.columns]
        
        # Check if models are loaded
        if not self.models:
            raise ValueError("No trained models found. Please train models first.")
        
        # Initialize predictions array
        predictions = np.zeros(len(data))
        
        # Handle empty data case
        if len(data) == 0:
            return predictions
        
        # Ensure clusters array matches data length
        if len(clusters) != len(data):
            print(f"Warning: Clusters array length ({len(clusters)}) does not match data length ({len(data)})")
            clusters = np.full(len(data), clusters[0])
        
        # Process each cluster
        for cluster_id in self.models:
            cluster_mask = clusters == cluster_id
            if not np.any(cluster_mask):
                continue
            
            # Get data for this cluster
            cluster_data = data[cluster_mask].copy()
            
            # Skip if cluster data is empty
            if len(cluster_data) == 0:
                continue
            
            try:
                # Prepare features
                X = cluster_data[available_cols].copy()
                target = X.pop('Units_Sold')  # Remove target from features
                
                # Add time-based features if date column exists
                if 'date' in cluster_data.columns:
                    if not pd.api.types.is_datetime64_any_dtype(cluster_data['date']):
                        cluster_data['date'] = pd.to_datetime(cluster_data['date'])
                    X['day_of_week'] = cluster_data['date'].dt.dayofweek
                    X['month'] = cluster_data['date'].dt.month
                    X['year'] = cluster_data['date'].dt.year
                    X['day_of_month'] = cluster_data['date'].dt.day
                
                # Check for constant columns that were dropped during training
                constant_cols = X.columns[X.nunique() == 1]
                if len(constant_cols) > 0:
                    X = X.drop(columns=constant_cols)
                
                # Check if we have any features left
                if len(X.columns) == 0:
                    print(f"Warning: No valid features for cluster {cluster_id}")
                    predictions[cluster_mask] = target.mean()
                    continue
                
                # Handle missing values
                X = X.fillna(X.mean())
                
                # Make predictions
                cluster_predictions = self.models[cluster_id].predict(X)
                predictions[cluster_mask] = cluster_predictions
                
            except Exception as e:
                print(f"Warning: Error making predictions for cluster {cluster_id}: {str(e)}")
                # Use historical mean as fallback
                predictions[cluster_mask] = target.mean() if 'target' in locals() else cluster_data['Units_Sold'].mean()
        
        return predictions
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict:
        """
        Evaluate model performance at all levels.
        
        Parameters:
        -----------
        test_data : pd.DataFrame
            Test dataset
        
        Returns:
        --------
        Dict
            Performance metrics at each level
        """
        features = self._prepare_features(test_data)
        predictions = self.predict(features)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        metrics = {
            'overall': {
                'rmse': np.sqrt(mean_squared_error(test_data['units_sold'], predictions)),
                'mae': mean_absolute_error(test_data['units_sold'], predictions)
            }
        }
        
        # Calculate cluster-specific metrics
        for cluster_id in test_data['cluster_id'].unique():
            cluster_mask = test_data['cluster_id'] == cluster_id
            if cluster_mask.any():
                metrics[f'cluster_{cluster_id}'] = {
                    'rmse': np.sqrt(mean_squared_error(
                        test_data.loc[cluster_mask, 'units_sold'],
                        predictions[cluster_mask]
                    )),
                    'mae': mean_absolute_error(
                        test_data.loc[cluster_mask, 'units_sold'],
                        predictions[cluster_mask]
                    )
                }
        
        return metrics 