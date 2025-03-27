"""Main pipeline for inventory management system."""

from typing import Dict, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import yaml

from .data.preprocessor import DataPreprocessor, PreprocessingError
from .data.validator import DataValidator, ValidationError
from .models.hierarchical_forecasting import HierarchicalForecasting
from .optimization.inventory_optimizer import InventoryOptimizer
from .optimization.pricing_optimizer import MLPricingOptimizer
from .visualization.dashboard import Dashboard
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
    
    def train_models(self, df: pd.DataFrame) -> None:
        """Train all models.
        
        Args:
            df: Processed DataFrame
            
        Raises:
            PipelineError: If training fails
        """
        self._time_component('model_training')
        logger.info("Training models...")
        
        try:
            # Perform time series clustering
            logger.info("Performing time series clustering...")
            clusterer = TimeSeriesClusterer(self.config.data.output_dir)
            features, clusters, best_method, best_score = clusterer.cluster_time_series(df)
            logger.info(f"Selected clustering method: {best_method} with score {best_score:.4f}")
            
            # Map cluster labels back to original DataFrame
            cluster_map = dict(zip(features['Product_ID'], features['Cluster']))
            df['Cluster'] = df['Product_ID'].map(cluster_map)
            
            # Initialize models
            self.forecaster = HierarchicalForecasting(
                base_model_type=self.config.model.base_model_type,
                min_cluster_size=self.config.model.min_cluster_size
            )
            
            # Train forecaster with clusters
            self.forecaster.train(df, df['Cluster'].values)
            
            # Initialize optimizers with supported parameters
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
            
            self._time_component('model_training')
            logger.info("Model training completed")
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise PipelineError(f"Model training failed: {str(e)}")
    
    def optimize(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run optimization processes.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary of optimization results
            
        Raises:
            PipelineError: If optimization fails
        """
        self._time_component('optimization')
        logger.info("Running optimization...")
        
        try:
            # Get inventory policies
            inventory_policies = self.inventory_optimizer.optimize_all(
                df,
                self.forecaster.cluster_models
            )
            
            # Get pricing policies
            pricing_policies = self.pricing_optimizer.optimize_all(
                df,
                self.forecaster.cluster_models
            )
            
            self._time_component('optimization')
            logger.info("Optimization completed")
            
            return {
                'inventory_policies': inventory_policies,
                'pricing_policies': pricing_policies
            }
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise PipelineError(f"Optimization failed: {str(e)}")
    
    def create_dashboard(self, results: Dict[str, Any]) -> Dashboard:
        """Create and configure dashboard.
        
        Args:
            results: Pipeline results
            
        Returns:
            Configured dashboard instance
            
        Raises:
            PipelineError: If dashboard creation fails
        """
        self._time_component('dashboard')
        logger.info("Creating dashboard...")
        
        try:
            self.dashboard = Dashboard(
                results,
                config=self.config.dashboard
            )
            
            self._time_component('dashboard')
            logger.info("Dashboard created")
            
            return self.dashboard
            
        except Exception as e:
            logger.error(f"Dashboard creation failed: {str(e)}")
            raise PipelineError(f"Dashboard creation failed: {str(e)}")
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save pipeline results.
        
        Args:
            results: Pipeline results to save
            
        Raises:
            PipelineError: If saving fails
        """
        try:
            output_dir = Path(self.config.data.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save policies
            with open(output_dir / "inventory_policies.json", 'w') as f:
                json.dump(results['inventory_policies'], f, indent=4)
            
            with open(output_dir / "pricing_policies.json", 'w') as f:
                json.dump(results['pricing_policies'], f, indent=4)
            
            # Save metrics
            with open(output_dir / "metrics.json", 'w') as f:
                json.dump(self.metrics, f, indent=4)
            
            logger.info(f"Results saved to {output_dir}")
            
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
            dashboard = self.create_dashboard(optimization_results)
            
            # Calculate total execution time
            total_time = (datetime.now() - self.metrics['start_time']).total_seconds()
            self.metrics['total_execution_time'] = total_time
            
            # Prepare results
            results = {
                **optimization_results,
                'dashboard': dashboard,
                'metrics': self.metrics
            }
            
            # Save results
            self.save_results(results)
            
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