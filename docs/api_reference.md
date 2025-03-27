# API Reference

## Data Module

### DataPreprocessor

The `DataPreprocessor` class handles all data preprocessing tasks.

```python
from inventory_management.data import DataPreprocessor

preprocessor = DataPreprocessor(config=None)
```

#### Methods

- `load_data(data_path: str) -> pd.DataFrame`
  - Loads data from CSV file
  - Args:
    - data_path: Path to CSV file
  - Returns: Loaded DataFrame
  - Raises:
    - FileNotFoundError: If file doesn't exist
    - ValueError: If file is empty or corrupted

- `process_dates() -> pd.DataFrame`
  - Processes date columns and creates time-based features
  - Returns: DataFrame with processed dates
  - Raises:
    - ValueError: If date processing fails

- `create_features() -> pd.DataFrame`
  - Creates derived features for analysis
  - Returns: DataFrame with new features
  - Raises:
    - ValueError: If feature creation fails

- `handle_missing_values(strategy: Dict[str, str]) -> pd.DataFrame`
  - Handles missing values using specified strategies
  - Args:
    - strategy: Dictionary mapping column names to imputation strategies
  - Returns: DataFrame with handled missing values
  - Raises:
    - ValueError: If handling missing values fails

### DataValidator

The `DataValidator` class handles data validation.

```python
from inventory_management.data import DataValidator

validator = DataValidator(config=None)
```

#### Methods

- `validate_schema(df: pd.DataFrame) -> ValidationResult`
  - Validates DataFrame schema
  - Args:
    - df: Input DataFrame
  - Returns: ValidationResult with schema validation results

- `validate_values(df: pd.DataFrame) -> ValidationResult`
  - Validates data values
  - Args:
    - df: Input DataFrame
  - Returns: ValidationResult with value validation results

- `validate_consistency(df: pd.DataFrame) -> ValidationResult`
  - Validates data consistency
  - Args:
    - df: Input DataFrame
  - Returns: ValidationResult with consistency validation results

## Models Module

### HierarchicalForecasting

The `HierarchicalForecasting` class implements hierarchical forecasting.

```python
from inventory_management.models import HierarchicalForecasting

forecaster = HierarchicalForecasting(
    base_model_type='xgb',
    min_cluster_size=100
)
```

#### Methods

- `train(data: pd.DataFrame) -> None`
  - Trains the complete hierarchical model
  - Args:
    - data: Complete dataset

- `predict(features: pd.DataFrame) -> np.ndarray`
  - Generates hierarchical predictions
  - Args:
    - features: Features for prediction
  - Returns: Final predictions combining all hierarchical levels

## Optimization Module

### InventoryOptimizer

The `InventoryOptimizer` class handles inventory optimization.

```python
from inventory_management.optimization import InventoryOptimizer

optimizer = InventoryOptimizer(
    forecaster,
    holding_cost_rate=0.25,
    ordering_cost=50.0,
    stockout_cost=100.0,
    lead_time_days=7
)
```

#### Methods

- `optimize_policy(data: pd.DataFrame, product_id: str) -> InventoryPolicy`
  - Optimizes inventory policy for a specific product
  - Args:
    - data: Historical data
    - product_id: Product identifier
  - Returns: Optimized inventory policy

### MLPricingOptimizer

The `MLPricingOptimizer` class handles pricing optimization.

```python
from inventory_management.optimization import MLPricingOptimizer

optimizer = MLPricingOptimizer(
    forecaster,
    min_margin=0.2,
    competitor_weight=0.3
)
```

#### Methods

- `optimize_price(data: pd.DataFrame, product_id: str) -> PricingPolicy`
  - Optimizes pricing for a specific product
  - Args:
    - data: Historical data
    - product_id: Product identifier
  - Returns: Optimized pricing policy

## Pipeline

The main pipeline class that orchestrates all components.

```python
from inventory_management import Pipeline

pipeline = Pipeline(config_path="config/config.json")
results = pipeline.run("data/inventory_data.csv")
```

#### Methods

- `run(data_path: str) -> Dict[str, Any]`
  - Runs the complete pipeline
  - Args:
    - data_path: Path to input data
  - Returns: Dictionary of pipeline results
  - Raises:
    - Various exceptions based on component failures

## Configuration

The system is configured through a JSON or YAML file:

```json
{
    "data": {
        "input_path": "data/inventory_data.csv",
        "output_dir": "output",
        "validation": {
            "min_date_range_days": 30,
            "max_missing_pct": 0.1
        }
    },
    "model": {
        "base_model_type": "xgb",
        "min_cluster_size": 100
    },
    "inventory": {
        "lead_time_days": 7,
        "service_level": 0.95
    },
    "pricing": {
        "min_margin": 0.2,
        "competitor_weight": 0.3
    }
}
```

## Error Handling

All components use structured error handling:

```python
try:
    # Component operation
    pass
except Exception as e:
    logger.error(f"Operation failed: {str(e)}")
    raise
```

## Logging

The system uses hierarchical logging:

```python
logger = logging.getLogger(__name__)
logger.info("Operation started")
try:
    # Operation
    logger.info("Operation completed")
except Exception as e:
    logger.error(f"Operation failed: {str(e)}")
``` 