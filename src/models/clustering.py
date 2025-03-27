import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans, KShape
from joblib import dump, load
import json
import os
from typing import Optional

class TimeSeriesClusterer:
    """Handles time series clustering using various methods."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.cluster_dir = f"{output_dir}/clustering"
        os.makedirs(self.cluster_dir, exist_ok=True)
    
    def _get_cluster_path(self) -> str:
        """Get path for saved clustering results."""
        return os.path.join(self.cluster_dir, "clustering_results.joblib")
    
    def _save_clustering_results(self, results: dict) -> None:
        """Save clustering results to disk."""
        dump(results, self._get_cluster_path())
    
    def _load_clustering_results(self) -> Optional[dict]:
        """Load clustering results from disk if they exist."""
        try:
            if os.path.exists(self._get_cluster_path()):
                return load(self._get_cluster_path())
        except Exception as e:
            print(f"Error loading clustering results: {str(e)}")
        return None
    
    def _find_optimal_clusters(self, ts_data: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using multiple metrics with custom weights."""
        print("Finding optimal number of clusters...")
        
        # Initialize metrics storage
        metrics = {
            'n_clusters': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': [],
            'wcss': [],
            'wcss_ratio': []
        }
        
        # Calculate metrics for different numbers of clusters
        for n_clusters in range(2, max_clusters + 1):
            print(f"Calculating metrics for {n_clusters} clusters...")
            
            # Fit KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(ts_data)
            
            # Calculate metrics
            metrics['n_clusters'].append(n_clusters)
            metrics['silhouette'].append(silhouette_score(ts_data, cluster_labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(ts_data, cluster_labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(ts_data, cluster_labels))
            metrics['wcss'].append(kmeans.inertia_)
        
        # Calculate WCSS ratio (rate of change in WCSS)
        wcss = np.array(metrics['wcss'])
        wcss_diff = np.diff(wcss)
        wcss_ratio = np.abs(wcss_diff / wcss[:-1])
        metrics['wcss_ratio'] = [0] + wcss_ratio.tolist()
        
        # Normalize metrics to [0,1] range
        normalized_metrics = {}
        for metric_name, values in metrics.items():
            if metric_name == 'n_clusters':
                normalized_metrics[metric_name] = values
                continue
                
            values = np.array(values)
            if metric_name in ['davies_bouldin', 'wcss', 'wcss_ratio']:
                # Lower is better for these metrics
                normalized_metrics[metric_name] = (values - values.min()) / (values.max() - values.min())
            else:
                # Higher is better for these metrics
                normalized_metrics[metric_name] = (values - values.min()) / (values.max() - values.min())
        
        # Calculate weighted score for each number of clusters
        weights = {
            'silhouette': 0.35,      # Higher weight for silhouette as it's generally reliable
            'calinski_harabasz': 0.25,  # Good for well-separated clusters
            'davies_bouldin': 0.15,     # Penalizes clusters that are too close
            'wcss_ratio': 0.25          # Helps identify elbow point
        }
        
        scores = []
        for i in range(len(metrics['n_clusters'])):
            score = (
                weights['silhouette'] * normalized_metrics['silhouette'][i] +
                weights['calinski_harabasz'] * normalized_metrics['calinski_harabasz'][i] +
                weights['davies_bouldin'] * (1 - normalized_metrics['davies_bouldin'][i]) +  # Invert since lower is better
                weights['wcss_ratio'] * (1 - normalized_metrics['wcss_ratio'][i])  # Invert since lower is better
            )
            scores.append(score)
        
        # Find optimal number of clusters
        optimal_idx = np.argmax(scores)
        optimal_clusters = metrics['n_clusters'][optimal_idx]
        
        # Print detailed metrics for the optimal number of clusters
        print(f"\nOptimal number of clusters: {optimal_clusters}")
        print(f"Metrics for optimal clusters:")
        print(f"  Silhouette Score: {metrics['silhouette'][optimal_idx]:.4f}")
        print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz'][optimal_idx]:.4f}")
        print(f"  Davies-Bouldin Score: {metrics['davies_bouldin'][optimal_idx]:.4f}")
        print(f"  WCSS Ratio: {metrics['wcss_ratio'][optimal_idx]:.4f}")
        print(f"  Combined Score: {scores[optimal_idx]:.4f}")
        
        return optimal_clusters
    
    def prepare_time_series_data(self, df):
        """Prepare time series data for clustering."""
        # Validate input data
        required_cols = ['Date', 'Product_ID', 'Units_Sold']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for missing values
        if df[required_cols].isnull().any().any():
            print("Warning: Found missing values in required columns. Filling with 0.")
            df = df[required_cols].fillna(0)
        
        # Aggregate data by date and product
        agg_data = df.groupby(['Date', 'Product_ID'])['Units_Sold'].sum().reset_index()
        
        # Check if we have enough data points
        if len(agg_data) < 2:
            raise ValueError("Insufficient data points for clustering")
        
        # Pivot data to get time series for each product
        ts_data = agg_data.pivot(index='Date', columns='Product_ID', values='Units_Sold')
        
        # Check if we have enough products
        if ts_data.shape[1] < 2:
            raise ValueError("Insufficient number of products for clustering")
        
        # Scale the time series
        scaler = TimeSeriesScalerMeanVariance()
        ts_scaled = scaler.fit_transform(ts_data.T)
        
        # Reshape to 2D array for standard clustering methods
        ts_2d = ts_scaled.reshape(ts_scaled.shape[0], -1)
        
        return ts_2d, ts_data
    
    def cluster_time_series(self, df):
        """Cluster products based on their time series patterns using multiple methods."""
        print("Clustering time series...")
        
        # Try to load existing clustering results
        existing_results = self._load_clustering_results()
        if existing_results is not None:
            print("Loaded existing clustering results from disk")
            return (
                existing_results['features'],
                existing_results['clusters'],
                existing_results['best_method'],
                existing_results['best_score']
            )
        
        # Prepare time series data
        try:
            ts_data, ts_pivot = self.prepare_time_series_data(df)
        except Exception as e:
            print(f"Error preparing time series data: {str(e)}")
            raise
        
        # Find optimal number of clusters
        try:
            n_clusters = self._find_optimal_clusters(ts_data)
            print(f"Optimal number of clusters: {n_clusters}")
        except Exception as e:
            print(f"Error finding optimal clusters: {str(e)}")
            raise
        
        # Compare clustering methods
        methods = {
            'kmeans': KMeans(n_clusters=n_clusters, random_state=42),
            'hierarchical': AgglomerativeClustering(n_clusters=n_clusters),
            'dbscan': DBSCAN(eps=0.5, min_samples=5),
            'dtw_kmeans': TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42),
            'softdtw_kmeans': TimeSeriesKMeans(n_clusters=n_clusters, metric="softdtw", random_state=42),
            'kshape': KShape(n_clusters=n_clusters, random_state=42)
        }
        
        # Compare methods
        best_method = None
        best_score = -float('inf')
        best_clusters = None
        method_scores = {}
        
        for name, method in methods.items():
            try:
                print(f"Trying {name} clustering...")
                clusters = method.fit_predict(ts_data)
                
                # Check if clustering was successful
                if len(np.unique(clusters)) < 2:
                    print(f"Warning: {name} clustering produced only one cluster")
                    continue
                
                score = self._score_clustering(clusters, ts_data)
                method_scores[name] = score
                
                if score > best_score:
                    best_score = score
                    best_method = name
                    best_clusters = clusters
                
                print(f"{name} clustering score: {score:.4f}")
            except Exception as e:
                print(f"Error with {name} clustering: {str(e)}")
                method_scores[name] = None
        
        # Check if any clustering method was successful
        if best_method is None:
            raise ValueError("No clustering method produced valid results")
        
        # Create features DataFrame
        features = pd.DataFrame({
            'Product_ID': df['Product_ID'].unique(),
            'Cluster': best_clusters
        })
        
        # Analyze clusters
        try:
            cluster_analysis = self._analyze_clusters(ts_data, best_clusters, features)
        except Exception as e:
            print(f"Warning: Error analyzing clusters: {str(e)}")
            cluster_analysis = {}
        
        # Save clustering results
        clustering_results = {
            'best_method': best_method,
            'best_score': float(best_score),
            'method_scores': method_scores,
            'cluster_sizes': {str(k): int(v) for k, v in pd.Series(best_clusters).value_counts().to_dict().items()},
            'cluster_characteristics': cluster_analysis,
            'features': features,
            'clusters': best_clusters
        }
        
        # Save to both JSON and joblib for different use cases
        try:
            with open(f"{self.cluster_dir}/clustering_results.json", 'w') as f:
                json.dump(clustering_results, f, indent=4)
            
            self._save_clustering_results(clustering_results)
        except Exception as e:
            print(f"Warning: Error saving clustering results: {str(e)}")
        
        return features, best_clusters, best_method, best_score
    
    def _score_clustering(self, clusters, ts_data):
        """Score clustering results using multiple metrics."""
        # Silhouette score
        sil_score = silhouette_score(ts_data.reshape(ts_data.shape[0], -1), clusters)
        
        # Calinski-Harabasz score
        ch_score = calinski_harabasz_score(ts_data.reshape(ts_data.shape[0], -1), clusters)
        
        # Davies-Bouldin score
        db_score = davies_bouldin_score(ts_data.reshape(ts_data.shape[0], -1), clusters)
        
        # Weighted average (favoring silhouette score)
        return 0.5 * sil_score + 0.3 * ch_score + 0.2 * (1 - db_score)
    
    def _analyze_clusters(self, ts_data, clusters, features):
        """Analyze characteristics of each cluster."""
        cluster_analysis = {}
        
        for cluster_id in np.unique(clusters):
            cluster_ts = ts_data[clusters == cluster_id]
            
            # Calculate cluster statistics
            cluster_analysis[str(cluster_id)] = {
                'size': int(len(cluster_ts)),
                'mean_pattern': cluster_ts.mean(axis=0).tolist(),
                'std_pattern': cluster_ts.std(axis=0).tolist(),
                'products': features[features['Cluster'] == cluster_id]['Product_ID'].tolist()
            }
        
        return cluster_analysis 