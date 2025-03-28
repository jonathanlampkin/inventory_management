from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from dataclasses import dataclass, field
from ..utils.config import DataConfig
from logging import Logger

logger = logging.getLogger(__name__)

class PreprocessingError(Exception):
    """Base class for preprocessing exceptions."""
    pass

@dataclass
class DataPreprocessor:
    """Data preprocessing component."""
    config: DataConfig
    logger: Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def process(self, data_path: str) -> pd.DataFrame:
        """Process the input data.
        
        Args:
            data_path: Path to the input data file
            
        Returns:
            Processed DataFrame
            
        Raises:
            PreprocessingError: If preprocessing fails
        """
        try:
            self.logger.info(f"Starting data preprocessing for {data_path}")
            
            # Load data
            self.logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Clean column names
            self.logger.info("Cleaning column names")
            df = self.clean_column_names(df)
            self.logger.info("Column names cleaned")
            
            # Convert data types
            self.logger.info("Converting data types")
            df = self.convert_data_types(df)
            self.logger.info("Data types converted")
            
            # Process dates
            self.logger.info("Processing dates")
            df = self.process_dates(df)
            self.logger.info("Date processing completed")
            
            # Create derived features
            self.logger.info("Creating derived features")
            df = self.create_derived_features(df)
            self.logger.info("Feature creation completed")
            
            # Handle missing values
            self.logger.info("Handling missing values")
            df = self.handle_missing_values(df)
            self.logger.info("Missing value handling completed")
            
            return df
            
        except Exception as e:
            raise PreprocessingError(f"Preprocessing failed: {str(e)}")

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names by replacing spaces with underscores and removing special characters.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned column names
        """
        try:
            # Create a mapping of old to new column names
            column_mapping = {
                'Store ID': 'Store_ID',
                'Product ID': 'Product_ID',
                'Inventory Level': 'Inventory_Level',
                'Units Sold': 'Units_Sold',
                'Units Ordered': 'Units_Ordered',
                'Demand Forecast': 'Demand_Forecast',
                'Weather Condition': 'Weather_Condition',
                'Holiday/Promotion': 'Holiday_Promotion',
                'Competitor Pricing': 'Competitor_Pricing'
            }
            
            # Rename columns that exist in the DataFrame
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            return df
            
        except Exception as e:
            raise PreprocessingError(f"Column name cleaning failed: {str(e)}")

    def convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with converted data types
        """
        try:
            # Numeric columns
            numeric_columns = [
                'Inventory_Level',
                'Units_Sold',
                'Units_Ordered',
                'Demand_Forecast',
                'Price',
                'Discount',
                'Competitor_Pricing'
            ]
            
            # Convert numeric columns
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert boolean columns
            if 'Holiday_Promotion' in df.columns:
                df['Holiday_Promotion'] = df['Holiday_Promotion'].astype(bool)
            
            # Convert categorical columns
            categorical_columns = [
                'Store_ID',
                'Product_ID',
                'Category',
                'Region',
                'Weather_Condition',
                'Seasonality'
            ]
            
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = df[col].astype('category')
            
            return df
            
        except Exception as e:
            raise PreprocessingError(f"Data type conversion failed: {str(e)}")

    def process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process date columns."""
        try:
            # Convert date columns to datetime
            date_columns = ['Date', 'date']  # Handle both cases
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            # Ensure we have a 'Date' column
            if 'date' in df.columns and 'Date' not in df.columns:
                df = df.rename(columns={'date': 'Date'})
            elif 'Date' not in df.columns:
                logger.warning("No date column found")
                return df
            
            # Create time-based features
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['Quarter'] = df['Date'].dt.quarter
            
            return df
            
        except Exception as e:
            logger.error(f"Date processing failed: {str(e)}")
            raise PreprocessingError(f"Date processing failed: {str(e)}")

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with derived features
        """
        try:
            # Add derived features based on configuration
            if hasattr(self.config, 'derived_features'):
                for feature in self.config.derived_features:
                    if feature['type'] == 'rolling_mean':
                        df[feature['name']] = df.groupby(feature['group_by'], observed=True)[feature['column']].transform(
                            lambda x: x.rolling(window=feature['window'], min_periods=1).mean()
                        )
                    elif feature['type'] == 'lag':
                        df[feature['name']] = df.groupby(feature['group_by'])[feature['column']].shift(feature['lag'])
            
            return df
            
        except Exception as e:
            raise PreprocessingError(f"Feature creation failed: {str(e)}")

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
            
        Raises:
            PreprocessingError: If missing value handling fails
        """
        try:
            # Default strategies
            strategies = {
                'numeric': 'mean',
                'categorical': 'mode',
                'date': 'ffill'
            }
            
            # Update with custom strategies if provided
            if hasattr(self.config, 'missing_value_strategies'):
                strategies.update(self.config.missing_value_strategies)
            
            # Apply strategies based on column types
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    strategy = strategies.get('numeric', 'mean')
                elif df[col].dtype == 'datetime64[ns]':
                    strategy = strategies.get('date', 'ffill')
                else:
                    strategy = strategies.get('categorical', 'mode')
                
                if df[col].isnull().any():
                    if strategy == 'mean':
                        df[col] = df[col].fillna(df[col].mean())
                    elif strategy == 'median':
                        df[col] = df[col].fillna(df[col].median())
                    elif strategy == 'mode':
                        df[col] = df[col].fillna(df[col].mode()[0])
                    elif strategy == 'ffill':
                        df[col] = df[col].fillna(method='ffill')
                    elif strategy == 'bfill':
                        df[col] = df[col].fillna(method='bfill')
                    elif strategy == 'zero':
                        df[col] = df[col].fillna(0)
            
            return df
            
        except Exception as e:
            raise PreprocessingError(f"Missing value handling failed: {str(e)}") 