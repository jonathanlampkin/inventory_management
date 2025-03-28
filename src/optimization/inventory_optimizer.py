import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from scipy.stats import norm
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class InventoryPolicy:
    """Data class for inventory policy parameters."""
    reorder_point: float
    order_quantity: float
    safety_stock: float
    review_period: int
    service_level: float

class InventoryOptimizer:
    """
    Inventory Optimization System
    
    Uses hierarchical forecasting to optimize inventory policies at multiple levels:
    - Global inventory strategy
    - Cluster-specific adjustments
    - Product-specific fine-tuning
    
    Parameters:
    -----------
    forecaster : HierarchicalForecasting
        Trained hierarchical forecasting model
    holding_cost_rate : float
        Annual holding cost as percentage of item value
    ordering_cost : float
        Fixed cost per order
    stockout_cost : float
        Cost of stockout per unit
    lead_time_days : int
        Standard lead time for orders
    """
    
    def __init__(
        self,
        forecaster,
        holding_cost_rate: float = 0.25,
        ordering_cost: float = 50.0,
        stockout_cost: float = 100.0,
        lead_time_days: int = 7
    ):
        self.forecaster = forecaster
        self.holding_cost_rate = holding_cost_rate
        self.ordering_cost = ordering_cost
        self.stockout_cost = stockout_cost
        self.lead_time_days = lead_time_days
        
        self.policies = {}
        self.metrics = {}
    
    def calculate_safety_stock(
        self,
        demand_std: float,
        service_level: float,
        lead_time: Optional[int] = None
    ) -> float:
        """Calculate safety stock level."""
        if lead_time is None:
            lead_time = self.lead_time_days
            
        z_score = norm.ppf(service_level)
        return z_score * demand_std * np.sqrt(lead_time)
    
    def calculate_reorder_point(
        self,
        demand_mean: float,
        demand_std: float,
        service_level: float,
        lead_time: Optional[int] = None
    ) -> float:
        """Calculate reorder point."""
        if lead_time is None:
            lead_time = self.lead_time_days
            
        safety_stock = self.calculate_safety_stock(demand_std, service_level, lead_time)
        lead_time_demand = demand_mean * lead_time
        
        return lead_time_demand + safety_stock
    
    def calculate_eoq(
        self,
        annual_demand: float,
        unit_cost: float,
        holding_cost_rate: Optional[float] = None
    ) -> float:
        """Calculate Economic Order Quantity."""
        if holding_cost_rate is None:
            holding_cost_rate = self.holding_cost_rate
            
        holding_cost = unit_cost * holding_cost_rate
        
        return np.sqrt(
            (2 * annual_demand * self.ordering_cost) / holding_cost
        )
    
    def optimize_policy(
        self,
        historical_data: pd.DataFrame,
        product_id: str,
        cluster_id: int,
        constraints: Optional[Dict] = None
    ) -> InventoryPolicy:
        """
        Optimize inventory policy for a specific product.
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical data for the product
        product_id : str
            Product identifier
        cluster_id : int
            Cluster identifier
        constraints : Dict, optional
            Additional constraints for optimization
            
        Returns:
        --------
        InventoryPolicy
            Optimized inventory policy
        """
        try:
            # Prepare data
            data = historical_data.copy()
            
            # Ensure required columns exist
            required_columns = ['Units_Sold', 'Price', 'Inventory_Level']
            for col in required_columns:
                if col not in data.columns:
                    logger.warning(f"Missing required column {col}, using defaults")
                    if col == 'Units_Sold':
                        data[col] = 1
                    elif col == 'Price':
                        data[col] = 100.0
                    elif col == 'Inventory_Level':
                        data[col] = 10.0
            
            # Ensure product_id column exists
            if 'Product_ID' not in data.columns:
                logger.warning("Missing Product_ID column, using provided product_id")
                data['Product_ID'] = product_id
            
            # Ensure date column exists and is datetime
            if 'date' not in data.columns:
                logger.warning("Missing date column, using index")
                data['date'] = pd.date_range(start='2024-01-01', periods=len(data), freq='D')
            elif not pd.api.types.is_datetime64_any_dtype(data['date']):
                data['date'] = pd.to_datetime(data['date'])
            
            # Sort data by date and reset index
            data = data.sort_values('date').reset_index(drop=True)
            
            # Get forecasts
            try:
                features = self.forecaster._prepare_features(data)
                # Create cluster assignments array
                clusters = np.full(len(data), cluster_id)
                demand_forecast = self.forecaster.predict(features, clusters)
            except Exception as e:
                logger.warning(f"Error generating forecasts: {str(e)}")
                # Use historical data as fallback
                demand_forecast = data['Units_Sold'].values
            
            # Calculate demand statistics with robust methods
            demand_mean = np.median(demand_forecast)  # Use median for robustness
            demand_std = np.std(demand_forecast)
            
            # Handle zero or negative demand
            if demand_mean <= 0:
                demand_mean = data['Units_Sold'].median()
            if demand_std <= 0:
                demand_std = data['Units_Sold'].std()
            
            # Calculate annual demand with robust scaling
            days_in_data = len(data)
            if days_in_data < 30:  # If less than 30 days of data
                annual_demand = demand_mean * 365
            else:
                annual_demand = np.sum(demand_forecast) * (365 / days_in_data)
            
            # Get product cost with fallback
            unit_cost = data['unit_cost'].iloc[0] if 'unit_cost' in data.columns else data['Price'].mean() * 0.7
            
            # Apply cluster-specific adjustments if available
            if cluster_id in self.forecaster.cluster_models:
                try:
                    cluster_adjustment = self.forecaster.cluster_models[cluster_id].predict(features)
                    demand_mean += np.median(cluster_adjustment)  # Use median for robustness
                    demand_std = np.std(demand_forecast + cluster_adjustment)
                except Exception as e:
                    logger.warning(f"Error applying cluster adjustments: {str(e)}")
            
            # Calculate base service level based on product value and demand variability
            base_service_level = 0.95  # Default
            if constraints and 'service_level' in constraints:
                base_service_level = constraints['service_level']
            elif unit_cost > 100:  # High-value items get higher service level
                base_service_level = 0.98
            elif demand_std / demand_mean > 0.5:  # High variability items get higher service level
                base_service_level = 0.97
            
            # Calculate policy parameters
            eoq = self.calculate_eoq(annual_demand, unit_cost)
            reorder_point = self.calculate_reorder_point(demand_mean, demand_std, base_service_level)
            safety_stock = self.calculate_safety_stock(demand_std, base_service_level)
            
            # Validate policy parameters
            if not all(np.isfinite([eoq, reorder_point, safety_stock])):
                logger.warning("Invalid policy parameters, using defaults")
                eoq = max(annual_demand / 12, 1)  # At least 1 unit
                reorder_point = max(demand_mean * self.lead_time_days, 1)
                safety_stock = max(demand_std * np.sqrt(self.lead_time_days), 1)
            
            # Ensure minimum order quantity
            order_quantity = max(eoq, 1)
            
            # Calculate review period with bounds
            review_period = max(min(int(order_quantity / demand_mean), 30), 1)
            
            return InventoryPolicy(
                reorder_point=float(reorder_point),
                order_quantity=float(order_quantity),
                safety_stock=float(safety_stock),
                review_period=int(review_period),
                service_level=float(base_service_level)
            )
            
        except Exception as e:
            logger.error(f"Error optimizing inventory policy: {str(e)}")
            # Return a conservative default policy
            return InventoryPolicy(
                reorder_point=10.0,
                order_quantity=10.0,
                safety_stock=5.0,
                review_period=7,
                service_level=0.95
            )
    
    def optimize_all(self, data: pd.DataFrame, cluster_models: Dict[int, Any]) -> Dict[str, Any]:
        """Optimize inventory policies for all clusters.
        
        Args:
            data: Training data
            cluster_models: Dictionary of trained cluster models
            
        Returns:
            Dictionary of optimization results
        """
        results = {}
        
        for cluster_id, model in cluster_models.items():
            try:
                # Get cluster data
                cluster_data = data[data['Product_ID'].isin(model.product_ids)]
                
                # Skip if no data
                if cluster_data.empty:
                    logger.warning(f"No data found for cluster {cluster_id}")
                    continue
                
                # Ensure required columns exist
                required_cols = ['Date', 'Units_Sold', 'Inventory_Level', 'Price']
                if not all(col in cluster_data.columns for col in required_cols):
                    logger.error(f"Missing required columns for cluster {cluster_id}")
                    continue
                
                # Generate forecasts
                forecasts = model.predict(cluster_data)
                
                # Calculate optimal inventory levels
                optimal_levels = self._calculate_optimal_levels(
                    forecasts,
                    cluster_data['Inventory_Level'].mean(),
                    cluster_data['Units_Sold'].std()
                )
                
                results[str(cluster_id)] = {
                    'optimal_level': float(optimal_levels),
                    'reorder_point': float(self._calculate_reorder_point(optimal_levels)),
                    'order_quantity': float(self._calculate_order_quantity(optimal_levels))
                }
                
            except Exception as e:
                logger.error(f"Error optimizing cluster {cluster_id}: {str(e)}")
                continue
        
        return results
    
    def evaluate_policy(
        self,
        policy: InventoryPolicy,
        historical_data: pd.DataFrame
    ) -> Dict:
        """
        Evaluate an inventory policy using historical data.
        
        Parameters:
        -----------
        policy : InventoryPolicy
            Inventory policy to evaluate
        historical_data : pd.DataFrame
            Historical data to use for evaluation
            
        Returns:
        --------
        Dict
            Performance metrics for the policy
        """
        inventory_level = policy.safety_stock
        stockouts = 0
        holding_cost = 0
        ordering_cost = 0
        
        for _, row in historical_data.iterrows():
            # Simulate demand
            demand = row['units_sold']
            
            # Check for stockout
            if demand > inventory_level:
                stockouts += demand - inventory_level
                inventory_level = 0
            else:
                inventory_level -= demand
            
            # Place order if needed
            if inventory_level <= policy.reorder_point:
                inventory_level += policy.order_quantity
                ordering_cost += self.ordering_cost
            
            # Calculate holding cost
            holding_cost += (
                inventory_level * 
                row['unit_cost'] * 
                (self.holding_cost_rate / 365)
            )
        
        return {
            'stockout_rate': stockouts / historical_data['units_sold'].sum(),
            'average_inventory': inventory_level / len(historical_data),
            'holding_cost': holding_cost,
            'ordering_cost': ordering_cost,
            'total_cost': holding_cost + ordering_cost + (stockouts * self.stockout_cost)
        } 