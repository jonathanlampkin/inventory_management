import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans, KShape
import json
import os

class TimeSeriesClusterer:
    """Handles time series clustering using various methods."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(f"{output_dir}/clustering", exist_ok=True)
    
    def prepare_time_series_data(self, df):
        """Prepare time series data for clustering."""
        # Pivot data to get time series for each product
        ts_data = df.pivot(index='Date', columns='Product ID', values='Units Sold')
        
        # Scale the time series
        scaler = TimeSeriesScalerMeanVariance()
        ts_scaled = scaler.fit_transform(ts_data.T)
        
        return ts_scaled, ts_data
    
    def cluster_time_series(self, df):
        """Cluster products based on their time series patterns using multiple methods."""
        print("Clustering time series...")
        
        # Prepare time series data
        ts_data, ts_pivot = self.prepare_time_series_data(df)
        
        # Compare clustering methods
        methods = {
            'kmeans': KMeans(n_clusters=5, random_state=42),
            'hierarchical': AgglomerativeClustering(n_clusters=5),
            'dbscan': DBSCAN(eps=0.5, min_samples=5),
            'dtw_kmeans': TimeSeriesKMeans(n_clusters=5, metric="dtw", random_state=42),
            'softdtw_kmeans': TimeSeriesKMeans(n_clusters=5, metric="softdtw", random_state=42),
            'kshape': KShape(n_clusters=5, random_state=42)
        }
        
        # Compare methods
        best_method = None
        best_score = -float('inf')
        best_clusters = None
        
        for name, method in methods.items():
            try:
                print(f"Trying {name} clustering...")
                clusters = method.fit_predict(ts_data)
                score = self._score_clustering(clusters, ts_data)
                
                if score > best_score:
                    best_score = score
                    best_method = name
                    best_clusters = clusters
                
                print(f"{name} clustering score: {score:.4f}")
            except Exception as e:
                print(f"Error with {name} clustering: {str(e)}")
        
        # Create features DataFrame
        features = pd.DataFrame({
            'Product ID': df['Product ID'].unique(),
            'Cluster': best_clusters
        })
        
        # Save clustering results
        clustering_results = {
            'best_method': best_method,
            'best_score': float(best_score),
            'cluster_sizes': pd.Series(best_clusters).value_counts().to_dict(),
            'cluster_characteristics': self._analyze_clusters(ts_data, best_clusters, features)
        }
        
        with open(f"{self.output_dir}/clustering/clustering_results.json", 'w') as f:
            json.dump(clustering_results, f, indent=4)
        
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
            cluster_analysis[cluster_id] = {
                'size': len(cluster_ts),
                'mean_pattern': cluster_ts.mean(axis=0).tolist(),
                'std_pattern': cluster_ts.std(axis=0).tolist(),
                'products': features[features['Cluster'] == cluster_id]['Product ID'].tolist()
            }
        
        return cluster_analysis 