import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans, KShape
from joblib import dump, load, Parallel, delayed
import json
import os
from typing import Optional, Dict, Any, Tuple, List
from sklearn.preprocessing import StandardScaler
import logging
import hashlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from optuna import create_study, Trial
import optuna

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """Dataset for deep time series clustering."""
    def __init__(self, data: np.ndarray):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class DeepTimeSeriesEncoder(nn.Module):
    """Deep encoder for time series clustering."""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
    
    def forward(self, x):
        # x shape: (batch_size, features)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x

class DeepTimeSeriesClusterer:
    """Deep learning based time series clustering."""
    def __init__(self, input_size: int, n_clusters: int, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = DeepTimeSeriesEncoder(input_size).to(device)
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5)
    
    def fit(self, data: np.ndarray, batch_size: int = 32, epochs: int = 50):
        """Train the deep clustering model."""
        # Ensure data is 2D (samples, features)
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        
        dataset = TimeSeriesDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = criterion(output, output)  # Reconstruction loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                # Load best model
                self.model.load_state_dict(torch.load('best_model.pt'))
                break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict clusters using the trained model."""
        # Ensure data is 2D (samples, features)
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
            
        self.model.eval()
        dataset = TimeSeriesDataset(data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                output = self.model(batch)
                embeddings.append(output.cpu().numpy())
        
        embeddings = np.concatenate(embeddings)
        return self.kmeans.fit_predict(embeddings)

class TimeSeriesClusterer:
    """Handles time series clustering using various methods."""
    
    def __init__(self, output_dir: str, config: Optional[Dict] = None):
        self.output_dir = output_dir
        self.cluster_dir = f"{output_dir}/clustering"
        self.config = config or {}
        os.makedirs(self.cluster_dir, exist_ok=True)
        
        # Default hyperparameters if not in config
        self.default_params = {
            'dtw_window': 0.2,  # As fraction of series length
            'softdtw_gamma': 1.0,
            'gak_sigma': 'auto',
            'min_cluster_size': 10,
            'n_init': 5,
            'random_state': 42,
            'deep_clustering': {
                'hidden_size': 64,
                'num_layers': 2,
                'batch_size': 32,
                'epochs': 50
            }
        }
        
        # Update with config values if provided
        self.params = {**self.default_params, **self.config.get('clustering', {})}
        
        # Initialize device for deep learning
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _get_cluster_path(self, ext: str = '.joblib') -> str:
        """Get path for saved clustering results."""
        return os.path.join(self.cluster_dir, f"clustering_results{ext}")
    
    def _hash_config(self) -> str:
        """Create a hash of the current configuration."""
        config_str = json.dumps(self.params, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _validate_hyperparameters(self, saved_params: Dict) -> bool:
        """Check if saved hyperparameters match current configuration."""
        current_hash = self._hash_config()
        return saved_params.get('config_hash') == current_hash
    
    def _convert_to_json_serializable(self, results: Dict) -> Dict:
        """Convert numpy arrays and other non-JSON serializable types."""
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, pd.DataFrame):
                json_results[key] = value.to_dict(orient='records')
            elif isinstance(value, (np.int64, np.float64)):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        return json_results
    
    def _save_clustering_results(self, results: Dict) -> None:
        """Save clustering results with hyperparameters."""
        try:
            # Add hyperparameters to results
            results['hyperparameters'] = {
                **self.params,
                'config_hash': self._hash_config()
            }
            
            # Save joblib for fast loading
            dump(results, self._get_cluster_path())
            
            # Save JSON for human inspection
            json_results = self._convert_to_json_serializable(results)
            with open(self._get_cluster_path(ext='.json'), 'w') as f:
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Saved clustering results to {self._get_cluster_path()}")
            
        except Exception as e:
            logger.error(f"Error saving clustering results: {str(e)}")
    
    def _load_clustering_results(self) -> Optional[Dict]:
        """Load cached clustering results with validation."""
        try:
            joblib_path = self._get_cluster_path()
            if os.path.exists(joblib_path):
                results = load(joblib_path)
                
                # Validate cached results
                required_keys = [
                    'features', 'clusters', 'best_method', 'best_score',
                    'hyperparameters', 'method_scores', 'cluster_characteristics'
                ]
                
                if all(key in results for key in required_keys):
                    # Validate hyperparameters match current config
                    if self._validate_hyperparameters(results['hyperparameters']):
                        logger.info("Using cached clustering results")
                        return results
                    else:
                        logger.warning("Cached hyperparameters don't match current config")
                else:
                    logger.warning("Cached results are incomplete")
                    
        except Exception as e:
            logger.error(f"Error loading clustering results: {str(e)}")
        return None
    
    def _configure_methods(self, n_clusters: int, input_size: int) -> Dict[str, str]:
        """Configure available clustering methods."""
        return {
            'dtw_kmeans': 'dtw_kmeans',
            'softdtw_kmeans': 'softdtw_kmeans',
            'kshape': 'kshape',
            'deep_clustering': 'deep_clustering'
        }
    
    def _score_clustering(self, clusters: np.ndarray, data: np.ndarray) -> float:
        """Score clustering results using multiple metrics."""
        try:
            # Reshape data if needed
            if len(data.shape) > 2:
                data_2d = data.reshape(data.shape[0], -1)
            else:
                data_2d = data
            
            # Calculate individual scores
            sil_score = silhouette_score(data_2d, clusters)
            ch_score = calinski_harabasz_score(data_2d, clusters)
            db_score = davies_bouldin_score(data_2d, clusters)
            
            # Weighted combination (higher is better)
            weights = {'silhouette': 0.4, 'calinski': 0.4, 'davies': 0.2}
            score = (
                weights['silhouette'] * sil_score +
                weights['calinski'] * (ch_score / 1000) +  # Normalize CH score
                weights['davies'] * (1 - db_score)  # Invert DB score
            )
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating clustering score: {str(e)}")
            return float('-inf')
    
    def _find_optimal_clusters(self, data: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using multiple metrics."""
        scores = []
        n_samples = len(data)
        
        # Use a smaller range of clusters for better balance
        cluster_range = [2, 3, 4, 5]
        
        for n_clusters in cluster_range:
            logger.info(f"Evaluating {n_clusters} clusters...")
            
            # Try each method in parallel
            methods = self._configure_methods(n_clusters, data.shape[-1])
            results = Parallel(n_jobs=-1)(
                delayed(self._parallel_cluster)(name, data, n_clusters)
                for name, method in methods.items()
            )
            
            # Calculate mean score for this n_clusters
            valid_scores = [score for _, score in results if score != float('-inf')]
            if valid_scores:
                scores.append((n_clusters, np.mean(valid_scores)))
        
        # Find best number of clusters
        if scores:
            best_n_clusters = max(scores, key=lambda x: x[1])[0]
            logger.info(f"Selected optimal number of clusters: {best_n_clusters}")
            return best_n_clusters
        else:
            logger.warning("Could not determine optimal clusters, using default")
            return 3  # Conservative default
    
    def _optimize_hyperparameters(self, data: np.ndarray, method: str) -> Dict:
        """Optimize hyperparameters using Optuna with efficient search strategy and early stopping."""
        # Check for cached hyperparameters
        cache_key = f"{method}_{hashlib.md5(data.tobytes()).hexdigest()}"
        cache_path = os.path.join(self.cluster_dir, f"hyperparams_{cache_key}.json")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_params = json.load(f)
                logger.info(f"Using cached hyperparameters for {method}")
                return cached_params
            except Exception as e:
                logger.warning(f"Error loading cached hyperparameters: {str(e)}")
        
        study = create_study(direction='maximize')
        
        # Calculate adaptive batch size based on data size
        n_samples = len(data)
        base_batch_size = min(32, max(16, n_samples // 100))
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.cluster_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if method == 'dtw_kmeans':
            # Efficient search with early stopping and adaptive pruning
            study.optimize(
                lambda trial: self._objective(trial, data, method, checkpoint_dir),
                n_trials=10,
                n_jobs=-1,
                callbacks=[
                    optuna.callbacks.EarlyStopping(
                        stopping_rounds=3,
                        min_trials=5,
                        grace_period=2
                    ),
                    optuna.callbacks.MedianPruner(
                        n_startup_trials=3,
                        n_warmup_steps=2,
                        interval_steps=1
                    )
                ]
            )
        elif method == 'deep_clustering':
            # Efficient search with early stopping, pruning, and warm-up
            study.optimize(
                lambda trial: self._objective(trial, data, method, checkpoint_dir),
                n_trials=10,
                n_jobs=-1,
                callbacks=[
                    optuna.callbacks.EarlyStopping(
                        stopping_rounds=3,
                        min_trials=5,
                        grace_period=2
                    ),
                    optuna.callbacks.MedianPruner(
                        n_startup_trials=3,
                        n_warmup_steps=2,
                        interval_steps=1
                    )
                ]
            )
        else:
            # For other methods, use default parameters
            return self.params.get(method, {})
        
        # Cache the best parameters
        best_params = study.best_params
        try:
            with open(cache_path, 'w') as f:
                json.dump(best_params, f)
        except Exception as e:
            logger.warning(f"Error caching hyperparameters: {str(e)}")
        
        return best_params
    
    def _objective(self, trial: Trial, data: np.ndarray, method: str, checkpoint_dir: str) -> float:
        """Objective function for hyperparameter optimization with efficient search space."""
        if method == 'dtw_kmeans':
            # Focused search space for DTW parameters with adaptive ranges
            dtw_window = trial.suggest_float('dtw_window', 0.1, 0.3)
            softdtw_gamma = trial.suggest_float('softdtw_gamma', 0.5, 1.5)
            
            # Use fixed cluster range but with adaptive selection
            n_clusters = trial.suggest_categorical('n_clusters', [2, 4, 6, 8, 10])
            
            model = TimeSeriesKMeans(
                n_clusters=n_clusters,
                metric="dtw",
                metric_params={"sakoe_chiba_radius": int(dtw_window * len(data))},
                random_state=self.params['random_state']
            )
            
        elif method == 'deep_clustering':
            # Calculate adaptive batch size based on data size
            n_samples = len(data)
            base_batch_size = min(32, max(16, n_samples // 100))
            
            # Efficient deep learning parameter search with warm-up
            hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 96])
            num_layers = trial.suggest_categorical('num_layers', [1, 2])
            batch_size = trial.suggest_categorical('batch_size', [base_batch_size, base_batch_size * 2])
            
            # Adaptive learning rate based on batch size and hidden size
            lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            
            # Learning rate schedule parameters
            warmup_epochs = trial.suggest_int('warmup_epochs', 3, 7)
            decay_rate = trial.suggest_float('decay_rate', 0.8, 0.99)
            decay_steps = trial.suggest_int('decay_steps', 3, 5)
            
            model = DeepTimeSeriesClusterer(
                input_size=data.shape[-1],
                n_clusters=trial.suggest_categorical('n_clusters', [2, 4, 6, 8, 10]),
                device=self.device
            )
            
            # Sophisticated warm-up and training strategy
            # Phase 1: Warm-up with linear learning rate increase
            for epoch in range(warmup_epochs):
                current_lr = lr * (epoch + 1) / warmup_epochs
                model.fit(data, batch_size=batch_size, epochs=1, learning_rate=current_lr)
                trial.report(0, step=epoch)  # Report intermediate value for pruning
            
            # Phase 2: Main training with learning rate decay
            best_score = float('-inf')
            patience = 3
            no_improvement = 0
            
            for epoch in range(15):
                # Decay learning rate
                current_lr = lr * (decay_rate ** (epoch // decay_steps))
                
                # Train for one epoch
                model.fit(data, batch_size=batch_size, epochs=1, learning_rate=current_lr)
                
                # Evaluate and checkpoint
                clusters = model.predict(data)
                score = self._score_clustering(clusters, data)
                
                # Save checkpoint if best score
                if score > best_score:
                    best_score = score
                    checkpoint_path = os.path.join(checkpoint_dir, f"trial_{trial.number}_best.pt")
                    torch.save(model.model.state_dict(), checkpoint_path)
                    no_improvement = 0
                else:
                    no_improvement += 1
                
                # Early stopping if no improvement
                if no_improvement >= patience:
                    break
                
                trial.report(score, step=epoch + warmup_epochs)
            
            # Load best model
            if os.path.exists(checkpoint_path):
                model.model.load_state_dict(torch.load(checkpoint_path))
            
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        clusters = model.predict(data)
        score = self._score_clustering(clusters, data)
        
        # Sophisticated pruning strategy
        if score < -0.5:  # Basic threshold
            trial.report(-1, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Additional pruning based on cluster quality
        if method == 'deep_clustering':
            unique_clusters = np.unique(clusters)
            if len(unique_clusters) < 2:  # All samples in one cluster
                trial.report(-1, step=2)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Check cluster balance
            cluster_sizes = [np.sum(clusters == c) for c in unique_clusters]
            min_size = min(cluster_sizes)
            max_size = max(cluster_sizes)
            if max_size / min_size > 10:  # Highly imbalanced clusters
                trial.report(-1, step=3)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        return score
    
    def _parallel_cluster(self, method: str, data: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, float]:
        """Run clustering method in parallel."""
        try:
            if method == 'dtw_kmeans':
                model = TimeSeriesKMeans(
                    n_clusters=n_clusters,
                    metric="dtw",
                    metric_params={"sakoe_chiba_radius": int(self.params['dtw_window'] * len(data))},
                    random_state=self.params['random_state']
                )
                clusters = model.fit_predict(data)
            elif method == 'softdtw_kmeans':
                model = TimeSeriesKMeans(
                    n_clusters=n_clusters,
                    metric="softdtw",
                    metric_params={"gamma": self.params['softdtw_gamma']},
                    random_state=self.params['random_state']
                )
                clusters = model.fit_predict(data)
            elif method == 'kshape':
                model = KShape(
                    n_clusters=n_clusters,
                    n_init=self.params['n_init'],
                    random_state=self.params['random_state']
                )
                clusters = model.fit_predict(data)
            elif method == 'deep_clustering':
                # Use GPU if available
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = DeepTimeSeriesClusterer(
                    input_size=data.shape[-1],
                    n_clusters=n_clusters,
                    device=device
                )
                # Reduced epochs for faster training
                model.fit(data, batch_size=32, epochs=20)
                clusters = model.predict(data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Validate cluster sizes
            unique_clusters = np.unique(clusters)
            cluster_sizes = [np.sum(clusters == c) for c in unique_clusters]
            min_size = min(cluster_sizes)
            max_size = max(cluster_sizes)
            
            # Skip if clusters are too imbalanced
            if min_size < 5 or max_size / min_size > 10:
                logger.warning(f"Clusters too imbalanced for {method}: min_size={min_size}, max_size={max_size}")
                return None, float('-inf')
            
            score = self._score_clustering(clusters, data)
            return clusters, score
            
        except Exception as e:
            logger.error(f"Error in {method} clustering: {str(e)}")
            return None, float('-inf')
    
    def cluster_time_series(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, str, float]:
        """Cluster products based on their sales patterns and features."""
        logger.info("Starting time series clustering...")
        
        # Check for cached results
        cached_results = self._load_clustering_results()
        if cached_results is not None:
            return (
                cached_results['features'],
                cached_results['clusters'],
                cached_results['best_method'],
                cached_results['best_score']
            )
        
        # Prepare data
        data, features = self.prepare_time_series_data(df)
        
        # Find optimal number of clusters
        n_clusters = self._find_optimal_clusters(data)
        
        # Run clustering methods in parallel
        methods = self._configure_methods(n_clusters, data.shape[-1])
        results = []
        for name, method in methods.items():
            clusters, score = self._parallel_cluster(method, data, n_clusters)
            results.append((name, clusters, score))
        
        # Find best method
        best_result = max(results, key=lambda x: x[2] if x[2] != float('-inf') else float('-inf'))
        if best_result[2] == float('-inf'):
            raise ValueError("All clustering methods failed")
        
        best_method, clusters, best_score = best_result
        
        # Save results
        results_dict = {
            'features': features,
            'clusters': clusters,
            'best_method': best_method,
            'best_score': best_score,
            'method_scores': {name: score for name, _, score in results},
            'cluster_characteristics': self._analyze_clusters(features, clusters)
        }
        self._save_clustering_results(results_dict)
        
        return features, clusters, best_method, best_score
    
    def prepare_time_series_data(self, df):
        """Prepare time series data for clustering."""
        # Validate input data
        required_cols = ['date', 'Product_ID', 'Units_Sold']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for missing values
        if df[required_cols].isnull().any().any():
            print("Warning: Found missing values in required columns. Filling with 0.")
            df = df[required_cols].fillna(0)
        
        # Create features for each product
        features = []
        
        for product_id in df['Product_ID'].unique():
            product_data = df[df['Product_ID'] == product_id]
            
            # Calculate time series features
            sales_stats = {
                'Product_ID': product_id,
                'mean_sales': product_data['Units_Sold'].mean(),
                'std_sales': product_data['Units_Sold'].std(),
                'max_sales': product_data['Units_Sold'].max(),
                'min_sales': product_data['Units_Sold'].min(),
                'total_sales': product_data['Units_Sold'].sum(),
                'sales_days': len(product_data),
                'zero_sales_days': len(product_data[product_data['Units_Sold'] == 0]),
            }
            
            # Calculate sales patterns
            if 'date' in product_data.columns:
                product_data['date'] = pd.to_datetime(product_data['date'])
                product_data['day_of_week'] = product_data['date'].dt.dayofweek
                product_data['month'] = product_data['date'].dt.month
                
                # Add day of week patterns
                for day in range(7):
                    day_sales = product_data[product_data['day_of_week'] == day]['Units_Sold'].mean()
                    sales_stats[f'dow_{day}_sales'] = day_sales if not pd.isna(day_sales) else 0
                
                # Add monthly patterns
                for month in range(1, 13):
                    month_sales = product_data[product_data['month'] == month]['Units_Sold'].mean()
                    sales_stats[f'month_{month}_sales'] = month_sales if not pd.isna(month_sales) else 0
            
            features.append(sales_stats)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features)
        
        # Fill any remaining NaN values with 0
        features_df = features_df.fillna(0)
        
        # Scale the features
        scaler = StandardScaler()
        feature_cols = [col for col in features_df.columns if col != 'Product_ID']
        features_scaled = scaler.fit_transform(features_df[feature_cols])
        
        return features_scaled, features_df
    
    def _analyze_clusters(self, features_df, clusters):
        """Analyze characteristics of each cluster."""
        cluster_analysis = {}
        
        # Get feature names (excluding Product_ID and Cluster)
        feature_names = [col for col in features_df.columns if col not in ['Product_ID', 'Cluster']]
        
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_features = features_df[cluster_mask]
            
            # Calculate cluster statistics
            stats = {
                'size': int(len(cluster_features)),
                'products': cluster_features['Product_ID'].tolist(),
                'mean_features': {},
                'std_features': {},
                'min_features': {},
                'max_features': {}
            }
            
            # Calculate statistics for each feature
            for feature in feature_names:
                stats['mean_features'][feature] = float(cluster_features[feature].mean())
                stats['std_features'][feature] = float(cluster_features[feature].std())
                stats['min_features'][feature] = float(cluster_features[feature].min())
                stats['max_features'][feature] = float(cluster_features[feature].max())
            
            # Add cluster characteristics
            stats['characteristics'] = {
                'avg_daily_sales': float(cluster_features['mean_sales'].mean()),
                'total_sales': float(cluster_features['total_sales'].sum()),
                'sales_variability': float(cluster_features['std_sales'].mean()),
                'zero_sales_ratio': float(cluster_features['zero_sales_days'].sum() / cluster_features['sales_days'].sum())
            }
            
            # Add day of week patterns
            dow_cols = [col for col in feature_names if col.startswith('dow_')]
            if dow_cols:
                stats['day_of_week_pattern'] = {
                    day: float(cluster_features[f'dow_{day}_sales'].mean())
                    for day in range(7)
                }
            
            # Add monthly patterns
            month_cols = [col for col in feature_names if col.startswith('month_')]
            if month_cols:
                stats['monthly_pattern'] = {
                    month: float(cluster_features[f'month_{month}_sales'].mean())
                    for month in range(1, 13)
                }
            
            cluster_analysis[str(cluster_id)] = stats
        
        return cluster_analysis 