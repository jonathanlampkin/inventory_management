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
        # Get forecasts
        features = self.forecaster._prepare_features(historical_data)
        demand_forecast = self.forecaster.predict(features)
        
        # Calculate demand statistics
        demand_mean = np.mean(demand_forecast)
        demand_std = np.std(demand_forecast)
        annual_demand = np.sum(demand_forecast) * (365 / len(demand_forecast))
        
        # Get product cost
        unit_cost = historical_data['unit_cost'].iloc[0]
        
        # Apply cluster-specific adjustments if available
        if cluster_id in self.forecaster.cluster_models:
            cluster_adjustment = self.forecaster.cluster_models[cluster_id].predict(features)
            demand_mean += np.mean(cluster_adjustment)
            demand_std = np.std(demand_forecast + cluster_adjustment)
        
        # Calculate base service level based on product value
        base_service_level = 0.95  # Default
        if constraints and 'service_level' in constraints:
            base_service_level = constraints['service_level']
        elif unit_cost > 100:  # High-value items get higher service level
            base_service_level = 0.98
        
        # Calculate policy parameters
        safety_stock = self.calculate_safety_stock(
            demand_std,
            base_service_level
        )
        
        reorder_point = self.calculate_reorder_point(
            demand_mean,
            demand_std,
            base_service_level
        )
        
        eoq = self.calculate_eoq(
            annual_demand,
            unit_cost
        )
        
        # Adjust for constraints
        if constraints:
            if 'max_order_quantity' in constraints:
                eoq = min(eoq, constraints['max_order_quantity'])
            if 'min_order_quantity' in constraints:
                eoq = max(eoq, constraints['min_order_quantity'])
        
        # Calculate review period (in days)
        review_period = int(np.ceil(eoq / demand_mean))
        
        return InventoryPolicy(
            reorder_point=reorder_point,
            order_quantity=eoq,
            safety_stock=safety_stock,
            review_period=review_period,
            service_level=base_service_level
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