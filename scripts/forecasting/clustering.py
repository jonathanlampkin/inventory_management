# Prepare time series data
ts_data = df.pivot(index='Date', columns='Product_ID', values='Units_Sold')

# Create features DataFrame
features = pd.DataFrame({
    'Product_ID': df['Product_ID'].unique(),
    'Cluster': best_clusters
})

cluster_analysis[cluster_id] = {
    'size': len(cluster_ts),
    'mean_pattern': cluster_ts.mean(axis=0).tolist(),
    'std_pattern': cluster_ts.std(axis=0).tolist(),
    'products': features[features['Cluster'] == cluster_id]['Product_ID'].tolist()
} 