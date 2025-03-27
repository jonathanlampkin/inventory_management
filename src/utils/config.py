import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import os

@dataclass
class DataConfig:
    """Data configuration parameters."""
    input_path: str = "data/retail_store_inventory.csv"
    output_dir: str = "output"
    validation: Dict[str, Any] = field(default_factory=lambda: {
        'min_date_range_days': 30,
        'max_missing_pct': 0.1,
        'required_columns': [
            'Date',
            'Store_ID',
            'Product_ID',
            'Category',
            'Region',
            'Inventory_Level',
            'Units_Sold',
            'Units_Ordered',
            'Demand_Forecast',
            'Price',
            'Discount',
            'Weather_Condition',
            'Holiday_Promotion',
            'Competitor_Pricing',
            'Seasonality'
        ],
        'date_columns': ['Date']
    })
    missing_value_strategies: Dict[str, str] = field(default_factory=lambda: {
        'Units_Sold': 'zero',
        'Inventory_Level': 'zero',
        'Price': 'mean',
        'Demand_Forecast': 'mean'
    })
    derived_features: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            'name': 'Inventory_Sales_Ratio',
            'type': 'rolling_mean',
            'column': 'Inventory_Level',
            'group_by': 'Product_ID',
            'window': 7
        },
        {
            'name': 'Sell_Through_Rate',
            'type': 'rolling_mean',
            'column': 'Units_Sold',
            'group_by': 'Product_ID',
            'window': 7
        },
        {
            'name': 'Forecast_Accuracy',
            'type': 'rolling_mean',
            'column': 'Demand_Forecast',
            'group_by': 'Product_ID',
            'window': 7
        }
    ])

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    base_model_type: str = 'xgb'
    min_cluster_size: int = 100
    n_estimators: int = 100
    max_depth: int = 10
    learning_rate: float = 0.1
    random_state: int = 42

@dataclass
class InventoryConfig:
    """Inventory optimization parameters."""
    lead_time_days: int = 7
    service_level: float = 0.95
    holding_cost_rate: float = 0.25
    ordering_cost: float = 50.0
    stockout_cost: float = 100.0
    min_reorder_point: int = 0
    max_order_quantity: int = 10000

@dataclass
class PricingConfig:
    """Pricing optimization parameters."""
    min_margin: float = 0.2
    competitor_weight: float = 0.3
    max_discount: float = 0.3
    price_elasticity_window: int = 90
    min_price_multiplier: float = 0.7
    max_price_multiplier: float = 1.3

@dataclass
class DashboardConfig:
    """Dashboard configuration parameters."""
    refresh_interval: int = 3600
    max_points: int = 10000
    cache_timeout: int = 300
    port: int = 8050
    debug: bool = False

@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/inventory_management.log"
    max_bytes: int = 10485760
    backup_count: int = 5

class Config:
    """
    Configuration management system with validation.
    
    Features:
    - JSON configuration loading
    - Environment variable override
    - Type validation
    - Default values
    - Nested configuration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to JSON configuration file
        """
        # Set default values
        self.data = DataConfig()
        self.model = ModelConfig()
        self.inventory = InventoryConfig()
        self.pricing = PricingConfig()
        self.dashboard = DashboardConfig()
        self.logging = LoggingConfig()
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
        
        # Override with environment variables
        self._load_env_vars()
        
        # Validate configuration
        self.validate()
    
    def load_config(self, config_path: str):
        """Load configuration from JSON file."""
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            # Update data config
            if 'data' in config:
                self.data = DataConfig(**config['data'])
            
            # Update model config
            if 'model' in config:
                self.model = ModelConfig(**config['model'])
            
            # Update inventory config
            if 'inventory' in config:
                self.inventory = InventoryConfig(**config['inventory'])
            
            # Update pricing config
            if 'pricing' in config:
                self.pricing = PricingConfig(**config['pricing'])
            
            # Update dashboard config
            if 'dashboard' in config:
                self.dashboard = DashboardConfig(**config['dashboard'])
            
            # Update logging config
            if 'logging' in config:
                self.logging = LoggingConfig(**config['logging'])
        
        except Exception as e:
            raise ValueError(f"Error loading configuration: {str(e)}")
    
    def _load_env_vars(self):
        """Override configuration with environment variables."""
        # Data configuration
        if 'INPUT_PATH' in os.environ:
            self.data.input_path = os.environ['INPUT_PATH']
        if 'OUTPUT_DIR' in os.environ:
            self.data.output_dir = os.environ['OUTPUT_DIR']
        
        # Model configuration
        if 'MODEL_TYPE' in os.environ:
            self.model.base_model_type = os.environ['MODEL_TYPE']
        if 'MIN_CLUSTER_SIZE' in os.environ:
            self.model.min_cluster_size = int(os.environ['MIN_CLUSTER_SIZE'])
        
        # Inventory configuration
        if 'LEAD_TIME_DAYS' in os.environ:
            self.inventory.lead_time_days = int(os.environ['LEAD_TIME_DAYS'])
        if 'SERVICE_LEVEL' in os.environ:
            self.inventory.service_level = float(os.environ['SERVICE_LEVEL'])
        
        # Pricing configuration
        if 'MIN_MARGIN' in os.environ:
            self.pricing.min_margin = float(os.environ['MIN_MARGIN'])
        if 'COMPETITOR_WEIGHT' in os.environ:
            self.pricing.competitor_weight = float(os.environ['COMPETITOR_WEIGHT'])
        
        # Dashboard configuration
        if 'DASHBOARD_PORT' in os.environ:
            self.dashboard.port = int(os.environ['DASHBOARD_PORT'])
        if 'DEBUG' in os.environ:
            self.dashboard.debug = os.environ['DEBUG'].lower() == 'true'
        
        # Logging configuration
        if 'LOG_LEVEL' in os.environ:
            self.logging.level = os.environ['LOG_LEVEL']
        if 'LOG_FILE' in os.environ:
            self.logging.file = os.environ['LOG_FILE']
    
    def validate(self):
        """Validate configuration values."""
        # Data validation
        assert os.path.exists(self.data.input_path), f"Input file not found: {self.data.input_path}"
        assert self.data.validation['min_date_range_days'] > 0, "min_date_range_days must be positive"
        assert 0 < self.data.validation['max_missing_pct'] < 1, "max_missing_pct must be between 0 and 1"
        assert len(self.data.validation['required_columns']) > 0, "required_columns cannot be empty"
        
        # Model validation
        assert self.model.base_model_type in ['xgb', 'rf'], "Invalid model type"
        assert self.model.min_cluster_size > 0, "min_cluster_size must be positive"
        assert self.model.n_estimators > 0, "n_estimators must be positive"
        assert self.model.max_depth > 0, "max_depth must be positive"
        assert self.model.learning_rate > 0, "learning_rate must be positive"
        
        # Inventory validation
        assert self.inventory.lead_time_days > 0, "lead_time_days must be positive"
        assert 0 < self.inventory.service_level < 1, "service_level must be between 0 and 1"
        assert self.inventory.holding_cost_rate > 0, "holding_cost_rate must be positive"
        assert self.inventory.ordering_cost > 0, "ordering_cost must be positive"
        assert self.inventory.stockout_cost > 0, "stockout_cost must be positive"
        
        # Pricing validation
        assert 0 < self.pricing.min_margin < 1, "min_margin must be between 0 and 1"
        assert 0 <= self.pricing.competitor_weight <= 1, "competitor_weight must be between 0 and 1"
        assert 0 < self.pricing.max_discount < 1, "max_discount must be between 0 and 1"
        assert self.pricing.price_elasticity_window > 0, "price_elasticity_window must be positive"
        
        # Dashboard validation
        assert self.dashboard.refresh_interval > 0, "refresh_interval must be positive"
        assert self.dashboard.max_points > 0, "max_points must be positive"
        assert self.dashboard.cache_timeout > 0, "cache_timeout must be positive"
        assert 0 < self.dashboard.port < 65536, "port must be between 1 and 65535"
        
        # Logging validation
        assert self.logging.level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], "Invalid log level"
        assert self.logging.max_bytes > 0, "max_bytes must be positive"
        assert self.logging.backup_count > 0, "backup_count must be positive"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': {
                'input_path': self.data.input_path,
                'output_dir': self.data.output_dir,
                'validation': self.data.validation
            },
            'model': {
                'base_model_type': self.model.base_model_type,
                'min_cluster_size': self.model.min_cluster_size,
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'learning_rate': self.model.learning_rate,
                'random_state': self.model.random_state
            },
            'inventory': {
                'lead_time_days': self.inventory.lead_time_days,
                'service_level': self.inventory.service_level,
                'holding_cost_rate': self.inventory.holding_cost_rate,
                'ordering_cost': self.inventory.ordering_cost,
                'stockout_cost': self.inventory.stockout_cost,
                'min_reorder_point': self.inventory.min_reorder_point,
                'max_order_quantity': self.inventory.max_order_quantity
            },
            'pricing': {
                'min_margin': self.pricing.min_margin,
                'competitor_weight': self.pricing.competitor_weight,
                'max_discount': self.pricing.max_discount,
                'price_elasticity_window': self.pricing.price_elasticity_window,
                'min_price_multiplier': self.pricing.min_price_multiplier,
                'max_price_multiplier': self.pricing.max_price_multiplier
            },
            'dashboard': {
                'refresh_interval': self.dashboard.refresh_interval,
                'max_points': self.dashboard.max_points,
                'cache_timeout': self.dashboard.cache_timeout,
                'port': self.dashboard.port,
                'debug': self.dashboard.debug
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'file': self.logging.file,
                'max_bytes': self.logging.max_bytes,
                'backup_count': self.logging.backup_count
            }
        }
    
    def save(self, config_path: str):
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file."""
        config = cls()
        config.load_config(config_path)
        return config 