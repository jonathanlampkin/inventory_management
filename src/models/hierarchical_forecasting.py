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
    
    def __init__(self, base_model_type: str = 'xgb', min_cluster_size: int = 100):
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
        if self.base_model is not None:
            dump(self.base_model, self._get_model_path('base'))
        
        for cluster_id, model in self.cluster_models.items():
            dump(model, self._get_model_path(f'cluster_{cluster_id}'))
        
        # Save product adjustments
        dump(self.product_adjustments, self._get_model_path('adjustments'))
        
        for cluster_id in self.models:
            dump(self.models[cluster_id], self._get_model_path(f"model_cluster_{cluster_id}"))
            dump(self.scalers[cluster_id], self._get_scaler_path(cluster_id))
    
    def load_models(self) -> bool:
        """Load trained models from disk if they exist."""
        try:
            # Load base model
            base_path = self._get_model_path('base')
            if os.path.exists(base_path):
                self.base_model = load(base_path)
            
            # Load cluster models
            for file in os.listdir(self.model_dir):
                if file.startswith('cluster_') and file.endswith('.joblib'):
                    cluster_id = int(file.split('_')[1].split('.')[0])
                    self.cluster_models[cluster_id] = load(os.path.join(self.model_dir, file))
            
            # Load product adjustments
            adj_path = self._get_model_path('adjustments')
            if os.path.exists(adj_path):
                self.product_adjustments = load(adj_path)
            
            # Get all model files
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith("model_cluster_")]
            if not model_files:
                return False
            
            # Load each model and its corresponding scaler
            for model_file in model_files:
                cluster_id = int(model_file.split("_")[-1].split(".")[0])
                self.models[cluster_id] = load(self._get_model_path(f"model_cluster_{cluster_id}"))
                self.scalers[cluster_id] = load(self._get_scaler_path(cluster_id))
            
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
        features = data.copy()
        
        # Time-based features
        features['day_of_week'] = features['date'].dt.dayofweek
        features['month'] = features['date'].dt.month
        features['year'] = features['date'].dt.year
        features['day_of_month'] = features['date'].dt.day
        
        # Lag features
        features['lag_1'] = features.groupby('product_id')['units_sold'].shift(1)
        features['lag_7'] = features.groupby('product_id')['units_sold'].shift(7)
        
        # Rolling means
        features['rolling_mean_7'] = features.groupby('product_id')['units_sold'].rolling(7).mean().reset_index(0, drop=True)
        features['rolling_mean_30'] = features.groupby('product_id')['units_sold'].rolling(30).mean().reset_index(0, drop=True)
        
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
    
    def train(self, data: pd.DataFrame, clusters: Union[np.ndarray, Dict]) -> None:
        """Train models for each cluster."""
        print("Training hierarchical forecasting models...")
        
        # Validate input data
        required_cols = ['Units_Sold', 'Price', 'Discount', 'Inventory_Level', 
                        'Demand_Forecast', 'Competitor_Pricing']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for missing values
        if data[required_cols].isnull().any().any():
            print("Warning: Found missing values in required columns. Filling with appropriate defaults.")
            data = data.copy()
            data['Units_Sold'] = data['Units_Sold'].fillna(0)
            data['Price'] = data['Price'].fillna(data['Price'].mean())
            data['Discount'] = data['Discount'].fillna(0)
            data['Inventory_Level'] = data['Inventory_Level'].fillna(0)
            data['Demand_Forecast'] = data['Demand_Forecast'].fillna(data['Units_Sold'].mean())
            data['Competitor_Pricing'] = data['Competitor_Pricing'].fillna(data['Price'].mean())
        
        # Try to load existing models first
        if self.load_models():
            print("Loaded existing models from disk")
            return
        
        # Prepare features
        feature_cols = required_cols
        target_col = 'Units_Sold'
        
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
            cluster_data = data[cluster_mask]
            
            if len(cluster_data) < 100:  # Skip clusters with too few samples
                print(f"Skipping cluster {cluster_id} due to insufficient data")
                continue
            
            # Prepare features and target
            X = cluster_data[feature_cols]
            y = cluster_data[target_col]
            
            # Check for constant columns
            constant_cols = X.columns[X.nunique() == 1]
            if len(constant_cols) > 0:
                print(f"Warning: Found constant columns in cluster {cluster_id}: {constant_cols}")
                X = X.drop(columns=constant_cols)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            # Save model and scaler
            self.models[cluster_id] = model
            self.scalers[cluster_id] = scaler
        
        # Check if any models were trained
        if not self.models:
            raise ValueError("No models were trained due to insufficient data in all clusters")
        
        # Save all models
        self.save_models()
    
    def _train_base_model(self, features: pd.DataFrame) -> object:
        """Train the base model on complete dataset."""
        model = self._create_model()
        X = features.drop(['units_sold', 'date'], axis=1)
        y = features['units_sold']
        
        model.fit(X, y)
        return model
    
    def _train_cluster_model(self, cluster_data: pd.DataFrame, base_predictions: np.ndarray) -> object:
        """Train cluster-specific model."""
        model = self._create_model()
        features = self._prepare_features(cluster_data)
        
        # Use residuals as target
        residuals = cluster_data['units_sold'] - base_predictions
        
        X = features.drop(['units_sold', 'date'], axis=1)
        model.fit(X, residuals)
        
        return model
    
    def _calculate_product_adjustments(self, data: pd.DataFrame, base_predictions: np.ndarray) -> Dict:
        """Calculate product-specific adjustment factors."""
        adjustments = {}
        
        for product_id in data['product_id'].unique():
            product_mask = data['product_id'] == product_id
            actual = data.loc[product_mask, 'units_sold']
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
        # Validate input data
        required_cols = ['Units_Sold', 'Price', 'Discount', 'Inventory_Level', 
                        'Demand_Forecast', 'Competitor_Pricing']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check if models are loaded
        if not self.models:
            raise ValueError("No trained models found. Please train models first.")
        
        predictions = np.zeros(len(data))
        
        for cluster_id in self.models:
            cluster_mask = clusters == cluster_id
            if not np.any(cluster_mask):
                continue
            
            # Get data for this cluster
            cluster_data = data[cluster_mask]
            
            # Prepare features
            feature_cols = required_cols
            X = cluster_data[feature_cols]
            
            # Check for constant columns
            constant_cols = X.columns[X.nunique() == 1]
            if len(constant_cols) > 0:
                X = X.drop(columns=constant_cols)
            
            # Scale features and predict
            X_scaled = self.scalers[cluster_id].transform(X)
            predictions[cluster_mask] = self.models[cluster_id].predict(X_scaled)
        
        # Check if any predictions were made
        if np.all(predictions == 0):
            raise ValueError("No predictions were made. Check if clusters match the training data.")
        
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