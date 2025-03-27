"""Unit tests for data preprocessor module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.preprocessor import DataPreprocessor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=10)
    return pd.DataFrame({
        'Date': dates,
        'Store_ID': ['S001'] * 10,
        'Product_ID': ['P001'] * 10,
        'Category': ['Electronics'] * 10,
        'Inventory_Level': np.random.randint(50, 200, 10),
        'Units_Sold': np.random.randint(1, 50, 10),
        'Price': np.random.uniform(10, 100, 10),
        'Discount': np.random.uniform(0, 30, 10)
    })

@pytest.fixture
def preprocessor():
    """Create preprocessor instance."""
    return DataPreprocessor()

def test_load_data(preprocessor, tmp_path):
    """Test data loading functionality."""
    # Create test data file
    df = pd.DataFrame({
        'Date': ['2023-01-01'],
        'Store_ID': ['S001'],
        'Product_ID': ['P001'],
        'Category': ['Electronics'],
        'Inventory_Level': [100],
        'Units_Sold': [10],
        'Price': [50.0],
        'Discount': [0.0]
    })
    
    data_path = tmp_path / "test_data.csv"
    df.to_csv(data_path, index=False)
    
    # Test loading
    loaded_df = preprocessor.load_data(data_path)
    assert not loaded_df.empty
    assert list(loaded_df.columns) == list(df.columns)

def test_process_dates(preprocessor, sample_data):
    """Test date processing functionality."""
    preprocessor.df = sample_data
    processed_df = preprocessor.process_dates()
    
    # Check date-based features
    assert 'Year' in processed_df.columns
    assert 'Month' in processed_df.columns
    assert 'Day' in processed_df.columns
    assert 'DayOfWeek' in processed_df.columns
    assert 'Quarter' in processed_df.columns
    
    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(processed_df['Date'])
    assert pd.api.types.is_integer_dtype(processed_df['Year'])
    assert pd.api.types.is_integer_dtype(processed_df['Month'])

def test_create_features(preprocessor, sample_data):
    """Test feature creation functionality."""
    preprocessor.df = sample_data
    processed_df = preprocessor.create_features()
    
    # Check derived features
    assert 'Inventory_Sales_Ratio' in processed_df.columns
    assert 'Sell_Through_Rate' in processed_df.columns
    assert 'Supply_Gap' in processed_df.columns
    assert 'Revenue' in processed_df.columns
    
    # Check calculations
    assert all(processed_df['Revenue'] == 
              processed_df['Price'] * processed_df['Units_Sold'] * (1 - processed_df['Discount']/100))

def test_handle_missing_values(preprocessor, sample_data):
    """Test missing value handling functionality."""
    # Add some missing values
    sample_data.loc[0, 'Price'] = np.nan
    sample_data.loc[1, 'Units_Sold'] = np.nan
    
    preprocessor.df = sample_data
    strategy = {
        'Price': 'mean',
        'Units_Sold': 'zero'
    }
    
    processed_df = preprocessor.handle_missing_values(strategy)
    
    # Check if missing values are handled
    assert not processed_df['Price'].isna().any()
    assert not processed_df['Units_Sold'].isna().any()
    assert processed_df.loc[1, 'Units_Sold'] == 0

def test_preprocess_pipeline(preprocessor, sample_data, tmp_path):
    """Test complete preprocessing pipeline."""
    # Save sample data
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path, index=False)
    
    # Run pipeline
    processed_df = preprocessor.preprocess_pipeline(
        data_path,
        missing_value_strategy={
            'Units_Sold': 'zero',
            'Price': 'mean'
        }
    )
    
    # Check results
    assert not processed_df.empty
    assert not processed_df.isna().any().any()
    assert 'Year' in processed_df.columns
    assert 'Revenue' in processed_df.columns

def test_error_handling(preprocessor):
    """Test error handling in preprocessor."""
    # Test loading non-existent file
    with pytest.raises(FileNotFoundError):
        preprocessor.load_data("non_existent.csv")
    
    # Test processing dates without loading data
    with pytest.raises(ValueError):
        preprocessor.process_dates()
    
    # Test creating features without loading data
    with pytest.raises(ValueError):
        preprocessor.create_features() 