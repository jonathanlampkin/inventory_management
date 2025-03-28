"""Data validation module for inventory management system."""

from typing import Dict, Optional, List, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Data validation result."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]

class ValidationError(Exception):
    """Base class for validation exceptions."""
    pass

class DataValidator:
    """Data validation for inventory management."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize validator with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.required_columns = [
            'date',  # Use lowercase date
            'Store_ID',
            'Product_ID',
            'Category',
            'Inventory_Level',
            'Units_Sold',
            'Price',
            'Discount'
        ]
        
        self.numeric_columns = [
            'Inventory_Level',
            'Units_Sold',
            'Price',
            'Discount',
            'Demand_Forecast'
        ]
        
        self.categorical_columns = [
            'Store_ID',
            'Product_ID',
            'Category',
            'Region'
        ]
    
    def validate(self, df: pd.DataFrame) -> Dict[str, ValidationResult]:
        """Run complete validation suite.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of validation results by category
            
        Raises:
            ValidationError: If validation setup fails
        """
        try:
            logger.info("Starting data validation")
            
            # Handle date column case sensitivity
            if 'Date' in df.columns and 'date' not in df.columns:
                df = df.rename(columns={'Date': 'date'})
            
            results = {
                'schema': self.validate_schema(df),
                'values': self.validate_values(df),
                'consistency': self.validate_consistency(df)
            }
            
            # Log validation results
            for category, result in results.items():
                if not result.is_valid:
                    logger.error(f"{category} validation failed:")
                    for error in result.errors:
                        logger.error(f"  - {error}")
                if result.warnings:
                    logger.warning(f"{category} validation warnings:")
                    for warning in result.warnings:
                        logger.warning(f"  - {warning}")
            
            return results
            
        except Exception as e:
            logger.error(f"Validation setup failed: {str(e)}")
            raise ValidationError(f"Validation setup failed: {str(e)}")
    
    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """Validate DataFrame schema.
        
        Args:
            df: Input DataFrame
            
        Returns:
            ValidationResult with schema validation results
        """
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Check required columns
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                errors.append(f"Missing required columns: {missing_cols}")
            
            # Check data types
            for col in df.columns:
                if col in self.numeric_columns and not np.issubdtype(df[col].dtype, np.number):
                    errors.append(f"Column {col} should be numeric")
                elif col in self.categorical_columns and not pd.api.types.is_object_dtype(df[col]):
                    warnings.append(f"Column {col} should be categorical")
            
            # Record metrics
            metrics['total_columns'] = len(df.columns)
            metrics['total_rows'] = len(df)
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Schema validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Schema validation failed: {str(e)}"],
                warnings=[],
                metrics={}
            )
    
    def validate_values(self, df: pd.DataFrame) -> ValidationResult:
        """Validate data values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            ValidationResult with value validation results
        """
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Check for negative values
            for col in ['Inventory_Level', 'Units_Sold', 'Price']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    neg_count = (df[col] < 0).sum()
                    if neg_count > 0:
                        errors.append(f"Found {neg_count} negative values in {col}")
            
            # Check discount range
            if 'Discount' in df.columns:
                df['Discount'] = pd.to_numeric(df['Discount'], errors='coerce')
                invalid_discounts = ((df['Discount'] < 0) | (df['Discount'] > 100)).sum()
                if invalid_discounts > 0:
                    errors.append(f"Found {invalid_discounts} invalid discount values")
            
            # Check date range
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                date_range = (df['date'].max() - df['date'].min()).days
                if date_range < 30:
                    warnings.append(f"Date range is only {date_range} days")
                metrics['date_range_days'] = date_range
            
            # Record metrics
            metrics['null_counts'] = df.isnull().sum().to_dict()
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Value validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Value validation failed: {str(e)}"],
                warnings=[],
                metrics={}
            )
    
    def validate_consistency(self, df: pd.DataFrame) -> ValidationResult:
        """Validate data consistency.
        
        Args:
            df: Input DataFrame
            
        Returns:
            ValidationResult with consistency validation results
        """
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Check for duplicate records
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                warnings.append(f"Found {duplicates} duplicate records")
            
            # Check inventory consistency
            if 'Units_Sold' in df.columns and 'Inventory_Level' in df.columns:
                df['Units_Sold'] = pd.to_numeric(df['Units_Sold'], errors='coerce')
                df['Inventory_Level'] = pd.to_numeric(df['Inventory_Level'], errors='coerce')
                invalid_inventory = (df['Units_Sold'] > df['Inventory_Level']).sum()
                if invalid_inventory > 0:
                    errors.append(f"Found {invalid_inventory} records where sales exceed inventory")
            
            # Check forecast consistency
            if 'Demand_Forecast' in df.columns and 'Units_Sold' in df.columns:
                df['Demand_Forecast'] = pd.to_numeric(df['Demand_Forecast'], errors='coerce')
                df['Units_Sold'] = pd.to_numeric(df['Units_Sold'], errors='coerce')
                forecast_accuracy = 1 - abs(df['Demand_Forecast'] - df['Units_Sold']).mean() / df['Units_Sold'].mean()
                if forecast_accuracy < 0.5:
                    warnings.append(f"Low forecast accuracy: {forecast_accuracy:.2%}")
            
            # Record metrics
            metrics['duplicate_count'] = duplicates
            metrics['invalid_inventory_count'] = invalid_inventory if 'invalid_inventory' in locals() else 0
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Consistency validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Consistency validation failed: {str(e)}"],
                warnings=[],
                metrics={}
            ) 