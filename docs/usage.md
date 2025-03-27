# Usage Guide

## Basic Usage

### Running the Pipeline

The simplest way to use the system is through the main pipeline:

```python
from inventory_management import Pipeline

# Initialize pipeline
pipeline = Pipeline(config_path="config/config.json")

# Run pipeline
results = pipeline.run("data/inventory_data.csv")

# Access results
inventory_policies = results['inventory_policies']
pricing_policies = results['pricing_policies']
dashboard = results['dashboard']

# Launch dashboard
dashboard.run_server(debug=False)
```

### Configuration

The system is highly configurable through a JSON or YAML file:

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
        "min_cluster_size": 100,
        "n_estimators": 100,
        "max_depth": 10,
        "learning_rate": 0.1
    },
    "inventory": {
        "lead_time_days": 7,
        "service_level": 0.95,
        "holding_cost_rate": 0.25,
        "ordering_cost": 50.0,
        "stockout_cost": 100.0
    },
    "pricing": {
        "min_margin": 0.2,
        "competitor_weight": 0.3,
        "max_discount": 0.3
    }
}
```

## Advanced Usage

### Data Processing

For custom data preprocessing:

```python
from inventory_management.data import DataPreprocessor, DataValidator

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Load and process data
df = preprocessor.load_data("data/inventory_data.csv")
df = preprocessor.process_dates()
df = preprocessor.create_features()

# Handle missing values
strategy = {
    'Units Sold': 'zero',
    'Price': 'mean',
    'Discount': 'zero'
}
df = preprocessor.handle_missing_values(strategy)

# Validate data
validator = DataValidator()
validation_results = validator.validate_dataset(df)

if not all(r.is_valid for r in validation_results.values()):
    print("Validation failed:")
    for category, result in validation_results.items():
        if not result.is_valid:
            print(f"{category}:")
            for error in result.errors:
                print(f"  - {error}")
```

### Forecasting

For custom forecasting:

```python
from inventory_management.models import HierarchicalForecasting

# Initialize forecaster
forecaster = HierarchicalForecasting(
    base_model_type='xgb',
    min_cluster_size=100
)

# Train model
forecaster.train(df)

# Make predictions
features = forecaster._prepare_features(df)
predictions = forecaster.predict(features)
```

### Optimization

For custom optimization:

```python
from inventory_management.optimization import InventoryOptimizer, MLPricingOptimizer

# Initialize optimizers
inventory_optimizer = InventoryOptimizer(
    forecaster,
    lead_time_days=7,
    service_level=0.95
)

pricing_optimizer = MLPricingOptimizer(
    forecaster,
    min_margin=0.2,
    competitor_weight=0.3
)

# Get policies
inventory_policies = inventory_optimizer.optimize_all(df)
pricing_policies = pricing_optimizer.optimize_all(df)

# Get specific policies
product_inventory = inventory_optimizer.optimize_policy(
    df,
    product_id="P001",
    cluster_id=0
)

product_pricing = pricing_optimizer.optimize_price(
    df,
    product_id="P001",
    cluster_id=0
)
```

### Dashboard

For custom dashboard:

```python
from inventory_management.visualization import InventoryDashboard

# Initialize dashboard
dashboard = InventoryDashboard(
    forecaster,
    inventory_optimizer,
    pricing_optimizer,
    config={
        'refresh_interval': 3600,
        'max_points': 10000,
        'cache_timeout': 300
    }
)

# Run dashboard
dashboard.run_server(debug=False, port=8050)
```

## Examples

### Basic Inventory Management

```python
from inventory_management import Pipeline

# Initialize and run pipeline
pipeline = Pipeline("config/config.json")
results = pipeline.run("data/inventory_data.csv")

# Get inventory policies
inventory_policies = results['inventory_policies']

# Print policies for each product
for product_id, policy in inventory_policies.items():
    print(f"\nProduct {product_id}:")
    print(f"  Reorder Point: {policy.reorder_point}")
    print(f"  Order Quantity: {policy.order_quantity}")
    print(f"  Safety Stock: {policy.safety_stock}")
```

### Price Optimization

```python
from inventory_management import Pipeline

# Initialize and run pipeline
pipeline = Pipeline("config/config.json")
results = pipeline.run("data/inventory_data.csv")

# Get pricing policies
pricing_policies = results['pricing_policies']

# Print policies for each product
for product_id, policy in pricing_policies.items():
    print(f"\nProduct {product_id}:")
    print(f"  Base Price: ${policy.base_price:.2f}")
    print(f"  Optimal Discount: {policy.optimal_discount:.1%}")
    print(f"  Expected Demand: {policy.expected_demand:.0f}")
```

### Custom Analysis

```python
from inventory_management.data import DataPreprocessor
from inventory_management.models import HierarchicalForecasting
import matplotlib.pyplot as plt

# Load and process data
preprocessor = DataPreprocessor()
df = preprocessor.preprocess_pipeline("data/inventory_data.csv")

# Train forecaster
forecaster = HierarchicalForecasting()
forecaster.train(df)

# Make predictions
features = forecaster._prepare_features(df)
predictions = forecaster.predict(features)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(df['Units Sold'], predictions, alpha=0.5)
plt.plot([0, max(df['Units Sold'])], [0, max(df['Units Sold'])], 'r--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()
```

## Best Practices

1. **Data Quality**
   - Validate input data
   - Handle missing values appropriately
   - Check for data consistency

2. **Model Training**
   - Use cross-validation
   - Monitor for overfitting
   - Regularly retrain models

3. **Optimization**
   - Start with conservative parameters
   - Monitor policy effectiveness
   - Adjust gradually

4. **Dashboard**
   - Keep data fresh
   - Monitor performance
   - Handle errors gracefully

## Troubleshooting

1. **Poor Forecast Accuracy**
   - Check data quality
   - Increase training data
   - Adjust model parameters

2. **Suboptimal Inventory Policies**
   - Verify cost parameters
   - Check lead time accuracy
   - Adjust service level

3. **Dashboard Performance**
   - Reduce data points
   - Increase cache timeout
   - Enable debug mode

## Next Steps

1. Read the [API Reference](api_reference.md)
2. Explore example notebooks
3. Join the community 