# Inventory Management System

A comprehensive inventory management system with ML-based optimization, forecasting, and interactive visualization.

## Features

- **Hierarchical Forecasting**: Multi-level demand forecasting combining global trends, cluster patterns, and product-specific adjustments
- **Inventory Optimization**: Smart inventory policies with safety stock and reorder point optimization
- **ML-based Pricing**: Dynamic pricing optimization using machine learning
- **Interactive Dashboard**: Real-time monitoring and control of inventory metrics
- **Automated Pipeline**: End-to-end data processing and model training pipeline

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/inventory_management.git
cd inventory_management

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# For development
pip install -e ".[dev]"
```

## Quick Start

```python
from inventory_management.pipeline import Pipeline

# Initialize pipeline
pipeline = Pipeline(config_path="config/config.json")

# Run pipeline
results = pipeline.run("data/retail_store_inventory.csv")

# Launch dashboard
results['dashboard'].run_server(debug=False)
```

## Configuration

The system is highly configurable through `config/config.json`:

```json
{
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

## Architecture

```
inventory_management/
├── src/
│   ├── data/           # Data processing
│   ├── models/         # ML models
│   ├── optimization/   # Optimization logic
│   ├── visualization/  # Dashboard
│   └── utils/         # Utilities
├── tests/
│   ├── unit/
│   └── integration/
├── config/
├── docs/
└── notebooks/
```

## Development

```bash
# Run tests
pytest

# Run linting
flake8 src tests
black src tests
isort src tests

# Type checking
mypy src
```

## Dashboard

The interactive dashboard provides:

1. **Executive View**
   - Key performance metrics
   - Sales trends
   - Inventory health
   - Automated recommendations

2. **Technical Analysis**
   - Model performance metrics
   - Forecast analysis
   - Cluster analysis

3. **Optimization Controls**
   - Inventory parameters
   - Pricing parameters
   - Real-time updates

## Documentation

Detailed documentation is available in the `docs/` directory:

- [User Guide](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [Development Guide](docs/development.md)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors
- Built with Python, scikit-learn, and Dash
- Inspired by modern inventory management practices 