"""Main pipeline for inventory management system."""

from typing import Dict, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, date
import logging
import json
import yaml
import os

from .data.preprocessor import DataPreprocessor, PreprocessingError
from .data.validator import DataValidator, ValidationError
from .models.hierarchical_forecasting import HierarchicalForecasting
from .optimization.inventory_optimizer import InventoryOptimizer
from .optimization.pricing_optimizer import MLPricingOptimizer
from .visualization.dashboard import Dashboard, DashboardConfig
from .utils.config import Config
from .utils.logging_config import setup_logging
from .models.clustering import TimeSeriesClusterer

logger = logging.getLogger(__name__)

class PipelineError(Exception):
    """Base class for pipeline exceptions."""
    pass

class InventoryPipeline:
    """Main pipeline for inventory management system."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize pipeline.
        
        Args:
            config: Optional configuration object
        """
        self.config = config if config else Config()
        self._setup_logging()
        
        logger.info("Initializing pipeline...")
        
        # Initialize components
        self.preprocessor = DataPreprocessor(self.config.data)
        self.validator = DataValidator(self.config.data)
        
        self.forecaster = None
        self.inventory_optimizer = None
        self.pricing_optimizer = None
        self.dashboard = None
        
        # Performance metrics
        self.metrics = {
            'start_time': datetime.now(),
            'component_times': {}
        }
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_config = self.config.logging
        setup_logging(
            log_dir=Path(log_config.file).parent,
            log_level=log_config.level
        )
    
    def _time_component(self, name: str) -> None:
        """Record component execution time."""
        if name in self.metrics['component_times']:
            start_time = self.metrics['component_times'][name]['start']
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics['component_times'][name]['duration'] = duration
            logger.info(f"Component {name} completed in {duration:.2f} seconds")
        else:
            self.metrics['component_times'][name] = {
                'start': datetime.now(),
                'duration': None
            }
            logger.info(f"Starting component {name}")
    
    def process_data(self, data_path: str) -> pd.DataFrame:
        """Process and validate data.
        
        Args:
            data_path: Path to input data
            
        Returns:
            Processed DataFrame
            
        Raises:
            PipelineError: If processing fails
        """
        self._time_component('data_processing')
        logger.info("Starting data processing...")
        
        try:
            # Preprocess data
            df = self.preprocessor.process(data_path)
            
            # Validate data
            validation_results = self.validator.validate(df)
            
            # Check validation results
            if not all(r.is_valid for r in validation_results.values()):
                for category, result in validation_results.items():
                    if not result.is_valid:
                        logger.error(f"{category} validation failed:")
                        for error in result.errors:
                            logger.error(f"  - {error}")
                raise PipelineError("Data validation failed")
            
            self._time_component('data_processing')
            return df
            
        except (PreprocessingError, ValidationError) as e:
            logger.error(f"Data processing failed: {str(e)}")
            raise PipelineError(f"Data processing failed: {str(e)}")
    
    def train_models(self, df: pd.DataFrame):
        """Train forecasting models for each cluster."""
        try:
            # Initialize clusterer
            clusterer = TimeSeriesClusterer(self.config.data.output_dir)
            
            # Perform clustering
            features, clusters, best_method, best_score = clusterer.cluster_time_series(df)
            
            # Map cluster assignments back to original DataFrame
            cluster_map = dict(zip(features['Product_ID'], clusters))
            df['Cluster'] = df['Product_ID'].map(cluster_map)
            
            # Validate cluster assignments
            unique_clusters = np.unique(clusters)
            unique_clusters = unique_clusters[~pd.isna(unique_clusters)]
            
            if len(unique_clusters) < 2:
                raise ValueError("Clustering produced less than 2 valid clusters")
            
            # Initialize forecaster
            self.forecaster = HierarchicalForecasting(
                base_model_type=self.config.model.base_model_type,
                min_cluster_size=self.config.model.min_cluster_size
            )
            
            # Train forecaster with progress tracking
            logger.info("Training forecasting models...")
            self.forecaster.train(df, df['Cluster'].values)
            
            # Save models
            logger.info("Saving trained models...")
            self.forecaster.save_models()
            
            # Initialize optimizers with cluster-specific parameters
            logger.info("Initializing optimizers...")
            self.inventory_optimizer = InventoryOptimizer(
                self.forecaster,
                holding_cost_rate=self.config.inventory.holding_cost_rate,
                ordering_cost=self.config.inventory.ordering_cost,
                stockout_cost=self.config.inventory.stockout_cost,
                lead_time_days=self.config.inventory.lead_time_days
            )
            
            self.pricing_optimizer = MLPricingOptimizer(
                self.forecaster,
                min_margin=self.config.pricing.min_margin,
                competitor_weight=self.config.pricing.competitor_weight
            )
            
            # Log clustering results
            logger.info(f"Clustering completed with method: {best_method}")
            logger.info(f"Clustering score: {best_score:.4f}")
            logger.info(f"Number of clusters: {len(unique_clusters)}")
            
            # Log cluster sizes
            cluster_sizes = {str(int(c)): int(np.sum(clusters == c)) for c in unique_clusters}
            logger.info("Cluster sizes: " + str(cluster_sizes))
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise PipelineError(f"Model training failed: {str(e)}")
    
    def optimize(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run optimization for inventory and pricing policies."""
        self._time_component('optimization')
        logger.info("Running optimization...")
        
        try:
            # Validate cluster assignments
            if 'Cluster' not in df.columns:
                raise ValueError("No cluster assignments found in data")
            
            unique_clusters = np.unique(df['Cluster'])
            unique_clusters = unique_clusters[~pd.isna(unique_clusters)]
            
            if len(unique_clusters) < 2:
                raise ValueError("Less than 2 valid clusters found in data")
            
            # Get inventory policies
            logger.info("Optimizing inventory policies...")
            inventory_policies = {}
            for cluster_id in unique_clusters:
                try:
                    cluster_data = df[df['Cluster'] == cluster_id].copy()
                    cluster_data = cluster_data.reset_index(drop=True)  # Reset index to avoid issues
                    if len(cluster_data) < 5:  # Skip clusters with too few samples
                        logger.warning(f"Skipping inventory optimization for cluster {cluster_id} due to insufficient data")
                        continue
                    
                    # Ensure required columns exist
                    required_columns = ['Units_Sold', 'Price', 'Inventory_Level', 'Product_ID']
                    missing_cols = [col for col in required_columns if col not in cluster_data.columns]
                    if missing_cols:
                        logger.warning(f"Missing required columns {missing_cols} for cluster {cluster_id}")
                        continue
                    
                    # Ensure we have enough data points
                    if len(cluster_data) < 30:  # Minimum required for meaningful optimization
                        logger.warning(f"Insufficient data points for cluster {cluster_id}")
                        continue
                    
                    # Convert date column if it exists
                    if 'date' in cluster_data.columns:
                        cluster_data['date'] = pd.to_datetime(cluster_data['date'])
                    
                    policy = self.inventory_optimizer.optimize_policy(
                        cluster_data,
                        f"cluster_{cluster_id}",
                        int(cluster_id)
                    )
                    inventory_policies[f"cluster_{cluster_id}"] = policy
                except Exception as e:
                    logger.warning(f"Error optimizing inventory for cluster {cluster_id}: {str(e)}")
                    continue
            
            # Get pricing policies
            logger.info("Optimizing pricing policies...")
            pricing_policies = {}
            for cluster_id in unique_clusters:
                try:
                    cluster_data = df[df['Cluster'] == cluster_id].copy()
                    cluster_data = cluster_data.reset_index(drop=True)  # Reset index to avoid issues
                    if len(cluster_data) < 5:  # Skip clusters with too few samples
                        logger.warning(f"Skipping pricing optimization for cluster {cluster_id} due to insufficient data")
                        continue
                    
                    # Ensure required columns exist
                    required_columns = ['Units_Sold', 'Price', 'Discount', 'Product_ID']
                    missing_cols = [col for col in required_columns if col not in cluster_data.columns]
                    if missing_cols:
                        logger.warning(f"Missing required columns {missing_cols} for cluster {cluster_id}")
                        continue
                    
                    # Ensure we have enough data points
                    if len(cluster_data) < 30:  # Minimum required for meaningful optimization
                        logger.warning(f"Insufficient data points for cluster {cluster_id}")
                        continue
                    
                    # Convert date column if it exists
                    if 'date' in cluster_data.columns:
                        cluster_data['date'] = pd.to_datetime(cluster_data['date'])
                    
                    policy = self.pricing_optimizer.optimize_price(
                        cluster_data,
                        f"cluster_{cluster_id}",
                        int(cluster_id)
                    )
                    pricing_policies[f"cluster_{cluster_id}"] = policy
                except Exception as e:
                    logger.warning(f"Error optimizing price for cluster {cluster_id}: {str(e)}")
                    continue
            
            # Validate optimization results
            if not inventory_policies and not pricing_policies:
                raise ValueError("No valid policies were generated for any cluster")
            
            # Add optimization summary
            optimization_summary = {
                'total_clusters': len(unique_clusters),
                'inventory_policies': len(inventory_policies),
                'pricing_policies': len(pricing_policies),
                'average_price_change': np.mean([
                    (policy.base_price - df[df['Cluster'] == int(cluster_id.split('_')[1])]['Price'].mean()) / 
                    df[df['Cluster'] == int(cluster_id.split('_')[1])]['Price'].mean() * 100
                    for cluster_id, policy in pricing_policies.items()
                ]) if pricing_policies else 0
            }
            
            self._time_component('optimization')
            logger.info("Optimization completed")
            
            return {
                'inventory_policies': inventory_policies,
                'pricing_policies': pricing_policies,
                'optimization_summary': optimization_summary
            }
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise PipelineError(f"Optimization failed: {str(e)}")
    
    def create_dashboard(self, results: Dict[str, Any]) -> Dashboard:
        """Create and configure dashboard."""
        self._time_component('dashboard')
        logger.info("Creating dashboard...")
        
        try:
            # Prepare data for dashboard
            df = results.get('data', pd.DataFrame())
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            
            # Ensure required columns exist
            required_cols = ['Date', 'Category', 'Product_ID', 'Units_Sold', 'Inventory_Level', 'Price', 'Discount']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            # Try to fix common column name issues
            if 'date' in df.columns and 'Date' not in df.columns:
                df = df.rename(columns={'date': 'Date'})
                if 'Date' in missing_cols:
                    missing_cols.remove('Date')
            
            if missing_cols:
                logger.warning(f"Missing required columns for dashboard: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Add derived columns
            df['Revenue'] = df['Units_Sold'] * df['Price'] * (1 - df['Discount']/100)
            
            # Create dashboard instance
            dashboard_results = {
                'data': df,
                'inventory_policies': results.get('inventory_policies', {}),
                'pricing_policies': results.get('pricing_policies', {}),
                'optimization_summary': results.get('optimization_summary', {})
            }
            
            self.dashboard = Dashboard(
                results=dashboard_results,
                config=DashboardConfig(
                    refresh_interval=self.config.dashboard.refresh_interval,
                    max_points=self.config.dashboard.max_points,
                    cache_timeout=self.config.dashboard.cache_timeout,
                    port=self.config.dashboard.port,
                    debug=self.config.dashboard.debug
                )
            )
            
            self._time_component('dashboard')
            logger.info("Dashboard created successfully")
            
            return self.dashboard
            
        except Exception as e:
            logger.error(f"Dashboard creation failed: {str(e)}")
            raise PipelineError(f"Dashboard creation failed: {str(e)}")
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save pipeline results to disk."""
        try:
            # Create results directory if it doesn't exist
            os.makedirs(self.config.data.output_dir, exist_ok=True)
            
            # Convert datetime objects to strings and policy objects to dictionaries
            def datetime_handler(obj):
                if isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            # Save metrics
            metrics_path = os.path.join(self.config.data.output_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=4, default=datetime_handler)
            
            # Save results
            results_path = os.path.join(self.config.data.output_dir, 'results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4, default=datetime_handler)
            
            logger.info(f"Saved results to {self.config.data.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise PipelineError(f"Error saving results: {str(e)}")
    
    def run(self, data_path: str) -> Dict[str, Any]:
        """Run the complete pipeline.
        
        Args:
            data_path: Path to input data
            
        Returns:
            Dictionary of pipeline results
            
        Raises:
            PipelineError: If pipeline fails
        """
        logger.info("Starting pipeline execution...")
        
        try:
            # Process data
            df = self.process_data(data_path)
            
            # Train models
            self.train_models(df)
            
            # Run optimization
            optimization_results = self.optimize(df)
            
            # Create dashboard
            dashboard = self.create_dashboard({
                **optimization_results,
                'data': df
            })
            
            # Calculate total execution time
            total_time = (datetime.now() - self.metrics['start_time']).total_seconds()
            self.metrics['total_execution_time'] = total_time
            
            # Prepare results for saving (without dashboard)
            results_to_save = {
                **optimization_results,
                'metrics': self.metrics
            }
            
            # Save results
            self.save_results(results_to_save)
            
            # Prepare results for return (with dashboard)
            results = {
                **results_to_save,
                'dashboard': dashboard
            }
            
            logger.info(f"Pipeline completed in {total_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise PipelineError(f"Pipeline failed: {str(e)}")

def main():
    """Main function to run the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inventory management pipeline")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--data", help="Path to input data")
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = InventoryPipeline(args.config)
    results = pipeline.run(args.data)
    
    # Run dashboard if created
    if results['dashboard']:
        results['dashboard'].run()

if __name__ == "__main__":
    main() 