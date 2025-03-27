"""Integration tests for inventory management pipeline."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

from inventory_management.pipeline import InventoryPipeline, PipelineError
from inventory_management.utils.config import Config

@pytest.fixture
def sample_data(tmp_path):
    """Create sample data for testing."""
    # Create dates
    dates = pd.date_range(start='2024-01-01', periods=100)
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Store_ID': ['S001'] * 100,
        'Product_ID': ['P001'] * 100,
        'Category': ['Electronics'] * 100,
        'Region': ['North'] * 100,
        'Inventory_Level': np.random.randint(50, 200, 100),
        'Units_Sold': np.random.randint(1, 50, 100),
        'Units_Ordered': np.random.randint(10, 100, 100),
        'Demand_Forecast': np.random.randint(10, 60, 100),
        'Price': np.random.uniform(10, 100, 100),
        'Discount': np.random.randint(0, 30, 100),
        'Weather_Condition': ['Sunny'] * 100,
        'Holiday_Promotion': np.random.randint(0, 2, 100),
        'Competitor_Pricing': np.random.uniform(8, 120, 100),
        'Seasonality': np.random.choice(['Spring', 'Summer', 'Autumn', 'Winter'], 100)
    })
    
    # Save to CSV
    data_path = tmp_path / "test_data.csv"
    df.to_csv(data_path, index=False)
    
    return data_path

@pytest.fixture
def config():
    """Create test configuration."""
    return {
        'data': {
            'output_dir': 'test_output',
            'validation': {
                'min_date_range_days': 30,
                'max_missing_pct': 0.1
            }
        },
        'model': {
            'base_model_type': 'xgb',
            'min_cluster_size': 50,
            'n_estimators': 50,
            'max_depth': 5,
            'learning_rate': 0.1
        },
        'inventory': {
            'lead_time_days': 7,
            'service_level': 0.95,
            'holding_cost_rate': 0.25,
            'ordering_cost': 50.0,
            'stockout_cost': 100.0
        },
        'pricing': {
            'min_margin': 0.2,
            'competitor_weight': 0.3,
            'max_discount': 0.3,
            'price_elasticity_window': 90
        },
        'dashboard': {
            'refresh_interval': 3600,
            'max_points': 10000,
            'cache_timeout': 300,
            'port': 8050,
            'debug': False
        },
        'logging': {
            'level': 'INFO',
            'log_dir': 'test_logs'
        }
    }

def test_pipeline_initialization(config):
    """Test pipeline initialization."""
    pipeline = InventoryPipeline(config)
    assert pipeline.config is not None
    assert pipeline.preprocessor is not None
    assert pipeline.validator is not None

def test_data_processing(sample_data, config):
    """Test data processing stage."""
    pipeline = InventoryPipeline(config)
    df = pipeline.process_data(sample_data)
    
    # Check DataFrame
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert all(col in df.columns for col in [
        'Date', 'Store_ID', 'Product_ID', 'Category',
        'Inventory_Level', 'Units_Sold', 'Price', 'Discount'
    ])
    
    # Check derived features
    assert 'Year' in df.columns
    assert 'Month' in df.columns
    assert 'Revenue' in df.columns

def test_model_training(sample_data, config):
    """Test model training stage."""
    pipeline = InventoryPipeline(config)
    df = pipeline.process_data(sample_data)
    pipeline.train_models(df)
    
    # Check models
    assert pipeline.forecaster is not None
    assert pipeline.inventory_optimizer is not None
    assert pipeline.pricing_optimizer is not None

def test_optimization(sample_data, config):
    """Test optimization stage."""
    pipeline = InventoryPipeline(config)
    df = pipeline.process_data(sample_data)
    pipeline.train_models(df)
    results = pipeline.optimize(df)
    
    # Check results
    assert 'inventory_policies' in results
    assert 'pricing_policies' in results
    assert isinstance(results['inventory_policies'], dict)
    assert isinstance(results['pricing_policies'], dict)

def test_dashboard_creation(sample_data, config):
    """Test dashboard creation."""
    pipeline = InventoryPipeline(config)
    df = pipeline.process_data(sample_data)
    pipeline.train_models(df)
    results = pipeline.optimize(df)
    dashboard = pipeline.create_dashboard(results)
    
    assert dashboard is not None
    assert dashboard.app is not None

def test_complete_pipeline(sample_data, config, tmp_path):
    """Test complete pipeline execution."""
    # Set output directory
    config['data']['output_dir'] = str(tmp_path)
    
    # Run pipeline
    pipeline = InventoryPipeline(config)
    results = pipeline.run(sample_data)
    
    # Check results
    assert results is not None
    assert 'inventory_policies' in results
    assert 'pricing_policies' in results
    assert 'dashboard' in results
    assert 'metrics' in results
    
    # Check output files
    output_dir = Path(config['data']['output_dir'])
    assert (output_dir / "inventory_policies.json").exists()
    assert (output_dir / "pricing_policies.json").exists()
    assert (output_dir / "metrics.json").exists()

def test_pipeline_error_handling(tmp_path):
    """Test pipeline error handling."""
    # Test with non-existent file
    pipeline = InventoryPipeline()
    with pytest.raises(PipelineError):
        pipeline.run(tmp_path / "non_existent.csv")
    
    # Test with invalid data
    invalid_data = pd.DataFrame({
        'Invalid Column': [1, 2, 3]
    })
    invalid_path = tmp_path / "invalid_data.csv"
    invalid_data.to_csv(invalid_path)
    
    with pytest.raises(PipelineError):
        pipeline.run(invalid_path)

def test_pipeline_metrics(sample_data, config):
    """Test pipeline performance metrics."""
    pipeline = InventoryPipeline(config)
    results = pipeline.run(sample_data)
    
    # Check metrics
    metrics = results['metrics']
    assert 'start_time' in metrics
    assert 'total_execution_time' in metrics
    assert 'component_times' in metrics
    
    # Check component times
    component_times = metrics['component_times']
    assert 'data_processing' in component_times
    assert 'model_training' in component_times
    assert 'optimization' in component_times 