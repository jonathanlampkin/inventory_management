import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PricingPolicy:
    """Data class for pricing policy parameters."""
    base_price: float
    min_price: float
    max_price: float
    optimal_discount: float
    price_elasticity: float
    expected_demand: float

class MLPricingOptimizer:
    """
    ML-based Pricing Optimization System
    
    Uses machine learning to:
    1. Predict demand at different price points
    2. Calculate price elasticity
    3. Optimize pricing and discounts
    4. Consider competitive positioning
    
    Parameters:
    -----------
    forecaster : HierarchicalForecasting
        Trained hierarchical forecasting model
    min_margin : float
        Minimum required profit margin
    competitor_weight : float
        Weight given to competitor prices (0-1)
    """
    
    def __init__(
        self,
        forecaster,
        min_margin: float = 0.2,
        competitor_weight: float = 0.3
    ):
        self.forecaster = forecaster
        self.min_margin = min_margin
        self.competitor_weight = competitor_weight
        
        self.demand_model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1
        )
        
        self.scaler = StandardScaler()
        self.policies = {}
        self.metrics = {}
    
    def _prepare_price_features(
        self,
        data: pd.DataFrame,
        price: float
    ) -> pd.DataFrame:
        """Prepare features for price-demand modeling."""
        features = self.forecaster._prepare_features(data).copy()
        
        # Add price-related features
        features['price'] = price
        features['price_diff'] = features['price'] - features['competitor_price']
        features['price_ratio'] = features['price'] / features['competitor_price']
        features['discount_depth'] = features['original_price'] - features['price']
        features['discount_percentage'] = (
            features['discount_depth'] / features['original_price']
        )
        
        return features
    
    def calculate_elasticity(
        self,
        data: pd.DataFrame,
        price_range: Optional[List[float]] = None
    ) -> Tuple[float, List[float], List[float]]:
        """
        Calculate price elasticity using ML predictions.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical data
        price_range : List[float], optional
            Range of prices to test
            
        Returns:
        --------
        Tuple[float, List[float], List[float]]
            Elasticity, prices tested, corresponding demands
        """
        if price_range is None:
            base_price = data['price'].mean()
            price_range = np.linspace(
                base_price * 0.7,
                base_price * 1.3,
                20
            )
        
        demands = []
        for price in price_range:
            features = self._prepare_price_features(data, price)
            demand = self.demand_model.predict(features)
            demands.append(np.mean(demand))
        
        # Calculate elasticity using midpoint method
        mid_price_idx = len(price_range) // 2
        price_diff = price_range[mid_price_idx + 1] - price_range[mid_price_idx - 1]
        demand_diff = demands[mid_price_idx + 1] - demands[mid_price_idx - 1]
        
        mid_price = price_range[mid_price_idx]
        mid_demand = demands[mid_price_idx]
        
        elasticity = (demand_diff / price_diff) * (mid_price / mid_demand)
        
        return elasticity, price_range.tolist(), demands
    
    def optimize_price(
        self,
        data: pd.DataFrame,
        product_id: str,
        cluster_id: int,
        constraints: Optional[Dict] = None
    ) -> PricingPolicy:
        """
        Optimize pricing for a specific product.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical data for the product
        product_id : str
            Product identifier
        cluster_id : int
            Cluster identifier
        constraints : Dict, optional
            Additional constraints for optimization
            
        Returns:
        --------
        PricingPolicy
            Optimized pricing policy
        """
        # Get base price and cost
        base_price = data['price'].mean()
        unit_cost = data['unit_cost'].iloc[0]
        
        # Define price range for testing
        min_price = max(unit_cost * (1 + self.min_margin), base_price * 0.7)
        max_price = base_price * 1.3
        
        if constraints:
            if 'min_price' in constraints:
                min_price = max(min_price, constraints['min_price'])
            if 'max_price' in constraints:
                max_price = min(max_price, constraints['max_price'])
        
        price_range = np.linspace(min_price, max_price, 20)
        
        # Calculate elasticity and demand curve
        elasticity, prices, demands = self.calculate_elasticity(
            data,
            price_range
        )
        
        # Find optimal base price (maximum revenue point)
        revenue = np.array(prices) * np.array(demands)
        optimal_idx = np.argmax(revenue)
        optimal_base_price = prices[optimal_idx]
        
        # Calculate optimal discount
        optimal_discount = self._optimize_discount(
            data,
            optimal_base_price,
            unit_cost
        )
        
        # Adjust for competitor pricing if available
        if 'competitor_price' in data.columns:
            competitor_price = data['competitor_price'].mean()
            optimal_base_price = (
                optimal_base_price * (1 - self.competitor_weight) +
                competitor_price * self.competitor_weight
            )
        
        # Get expected demand at optimal price
        features = self._prepare_price_features(
            data,
            optimal_base_price * (1 - optimal_discount)
        )
        expected_demand = np.mean(self.demand_model.predict(features))
        
        return PricingPolicy(
            base_price=optimal_base_price,
            min_price=min_price,
            max_price=max_price,
            optimal_discount=optimal_discount,
            price_elasticity=elasticity,
            expected_demand=expected_demand
        )
    
    def _optimize_discount(
        self,
        data: pd.DataFrame,
        base_price: float,
        unit_cost: float
    ) -> float:
        """Optimize discount level."""
        discount_range = np.linspace(0, 0.3, 10)  # 0% to 30%
        best_profit = float('-inf')
        optimal_discount = 0
        
        for discount in discount_range:
            price = base_price * (1 - discount)
            if price < unit_cost * (1 + self.min_margin):
                continue
                
            features = self._prepare_price_features(data, price)
            demand = np.mean(self.demand_model.predict(features))
            
            revenue = price * demand
            cost = unit_cost * demand
            profit = revenue - cost
            
            if profit > best_profit:
                best_profit = profit
                optimal_discount = discount
        
        return optimal_discount
    
    def optimize_all(self, data: pd.DataFrame, clusters: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Optimize pricing for all clusters."""
        print("Optimizing pricing for all clusters...")
        
        # Validate input data
        required_cols = ['Date', 'Units_Sold', 'Price', 'Competitor_Pricing']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for missing values
        if data[required_cols].isnull().any().any():
            print("Warning: Found missing values in required columns. Filling with appropriate defaults.")
            data = data.copy()
            data['Units_Sold'] = data['Units_Sold'].fillna(0)
            data['Price'] = data['Price'].fillna(data['Price'].mean())
            data['Competitor_Pricing'] = data['Competitor_Pricing'].fillna(data['Price'].mean())
        
        # Initialize results dictionary
        results = {}
        
        # Optimize for each cluster
        for cluster_id in np.unique(clusters):
            print(f"Optimizing pricing for cluster {cluster_id}...")
            
            # Get data for this cluster
            cluster_mask = clusters == cluster_id
            cluster_data = data[cluster_mask]
            
            if len(cluster_data) < 50:  # Skip clusters with too few samples
                print(f"Skipping cluster {cluster_id} due to insufficient data")
                continue
            
            try:
                # Calculate price elasticity
                elasticity = self._calculate_price_elasticity(cluster_data)
                
                # Optimize price
                optimal_price = self._optimize_price(cluster_data, elasticity)
                
                # Calculate expected margin
                expected_margin = self._calculate_expected_margin(cluster_data, optimal_price)
                
                # Store results
                results[str(cluster_id)] = {
                    'optimal_price': float(optimal_price),
                    'price_elasticity': float(elasticity),
                    'expected_margin': float(expected_margin),
                    'sample_size': int(len(cluster_data))
                }
            except Exception as e:
                print(f"Warning: Error optimizing cluster {cluster_id}: {str(e)}")
                results[str(cluster_id)] = {
                    'error': str(e),
                    'sample_size': int(len(cluster_data))
                }
        
        # Check if any clusters were optimized
        if not results:
            raise ValueError("No clusters were optimized due to insufficient data or errors")
        
        return results
    
    def _calculate_price_elasticity(self, data: pd.DataFrame) -> float:
        """Calculate price elasticity of demand."""
        try:
            # Calculate log differences
            log_price = np.log(data['Price'])
            log_quantity = np.log(data['Units_Sold'])
            
            # Calculate elasticity using linear regression
            X = np.column_stack([np.ones(len(log_price)), log_price])
            beta = np.linalg.inv(X.T @ X) @ X.T @ log_quantity
            elasticity = beta[1]  # Price coefficient
            
            # Validate elasticity
            if not np.isfinite(elasticity):
                raise ValueError("Invalid elasticity value")
            
            return float(elasticity)
        except Exception as e:
            print(f"Error calculating price elasticity: {str(e)}")
            raise
    
    def _optimize_price(self, data: pd.DataFrame, elasticity: float) -> float:
        """Optimize price based on elasticity and cost."""
        try:
            # Calculate average cost
            avg_cost = data['Price'].mean() * 0.7  # Assuming 30% margin
            
            # Calculate optimal price using elasticity formula
            optimal_price = avg_cost / (1 + 1/elasticity)
            
            # Validate optimal price
            if not np.isfinite(optimal_price) or optimal_price <= 0:
                raise ValueError("Invalid optimal price")
            
            # Ensure price is within reasonable bounds
            min_price = data['Price'].min() * 0.5
            max_price = data['Price'].max() * 1.5
            optimal_price = np.clip(optimal_price, min_price, max_price)
            
            return float(optimal_price)
        except Exception as e:
            print(f"Error optimizing price: {str(e)}")
            raise
    
    def _calculate_expected_margin(self, data: pd.DataFrame, optimal_price: float) -> float:
        """Calculate expected margin at optimal price."""
        try:
            # Calculate current margin
            current_margin = (data['Price'] - data['Price'] * 0.7) / data['Price']
            
            # Calculate expected margin at optimal price
            expected_margin = (optimal_price - optimal_price * 0.7) / optimal_price
            
            # Validate margin
            if not np.isfinite(expected_margin) or expected_margin < 0:
                raise ValueError("Invalid expected margin")
            
            return float(expected_margin)
        except Exception as e:
            print(f"Error calculating expected margin: {str(e)}")
            raise
    
    def evaluate_policy(
        self,
        policy: PricingPolicy,
        historical_data: pd.DataFrame
    ) -> Dict:
        """
        Evaluate a pricing policy using historical data.
        
        Parameters:
        -----------
        policy : PricingPolicy
            Pricing policy to evaluate
        historical_data : pd.DataFrame
            Historical data to use for evaluation
            
        Returns:
        --------
        Dict
            Performance metrics for the policy
        """
        # Calculate metrics at optimal price
        optimal_price = policy.base_price * (1 - policy.optimal_discount)
        features = self._prepare_price_features(historical_data, optimal_price)
        predicted_demand = self.demand_model.predict(features)
        
        # Calculate revenue and profit
        revenue = optimal_price * predicted_demand
        cost = historical_data['unit_cost'].iloc[0] * predicted_demand
        profit = revenue - cost
        
        # Calculate market share if competitor data available
        market_share = None
        if 'competitor_sales' in historical_data.columns:
            total_market = predicted_demand + historical_data['competitor_sales']
            market_share = predicted_demand / total_market
        
        return {
            'predicted_demand': float(np.mean(predicted_demand)),
            'expected_revenue': float(np.mean(revenue)),
            'expected_profit': float(np.mean(profit)),
            'price_position': optimal_price / historical_data['competitor_price'].mean(),
            'market_share': float(np.mean(market_share)) if market_share is not None else None
        } 