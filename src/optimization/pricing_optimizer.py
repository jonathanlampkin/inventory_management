import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from dataclasses import dataclass
import logging
from sklearn.linear_model import HuberRegressor

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
        
        # Calculate elasticity using robust regression
        prices = np.array(price_range)
        demands = np.array(demands)
        
        # Add small constant to avoid log(0)
        prices = prices + 1e-6
        demands = demands + 1e-6
        
        # Use log-log regression for more robust elasticity
        log_prices = np.log(prices)
        log_demands = np.log(demands)
        
        # Remove outliers using IQR method
        q1 = np.percentile(log_demands, 25)
        q3 = np.percentile(log_demands, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        mask = (log_demands >= lower_bound) & (log_demands <= upper_bound)
        log_prices = log_prices[mask]
        log_demands = log_demands[mask]
        
        # Fit robust regression
        model = HuberRegressor(epsilon=1.35)
        model.fit(log_prices.reshape(-1, 1), log_demands)
        elasticity = -model.coef_[0]  # Negative because we want positive elasticity
        
        # Ensure elasticity is within reasonable bounds
        elasticity = max(0.1, min(5.0, elasticity))
        
        return elasticity, price_range.tolist(), demands.tolist()
    
    def _optimize_base_price(
        self,
        data: pd.DataFrame,
        demand_forecast: np.ndarray,
        unit_cost: float,
        base_price: float
    ) -> float:
        """Optimize base price using demand forecast."""
        try:
            # Ensure we have valid base price and unit cost
            if not np.isfinite(base_price) or base_price <= 0:
                base_price = data['Price'].mean() if 'Price' in data.columns else 100.0
            if not np.isfinite(unit_cost) or unit_cost <= 0:
                unit_cost = base_price * 0.7
            
            # Define price range for testing
            min_price = max(unit_cost * (1 + self.min_margin), base_price * 0.7)
            max_price = base_price * 1.3
            
            # Create price points to test
            price_range = np.linspace(min_price, max_price, 20)
            
            # Calculate revenue at each price point
            revenues = []
            for price in price_range:
                # Ensure valid price ratio
                price_ratio = max(0.1, min(10.0, price / base_price))
                estimated_demand = demand_forecast * np.power(price_ratio, -1.5)  # Assume price elasticity of -1.5
                revenue = price * estimated_demand
                revenues.append(np.mean(revenue))
            
            # Find price that maximizes revenue
            optimal_idx = np.argmax(revenues)
            optimal_price = price_range[optimal_idx]
            
            # Adjust for competitor pricing if available
            if 'Competitor_Pricing' in data.columns:
                competitor_price = data['Competitor_Pricing'].mean()
                if np.isfinite(competitor_price) and competitor_price > 0:
                    optimal_price = (
                        optimal_price * (1 - self.competitor_weight) +
                        competitor_price * self.competitor_weight
                    )
            
            return float(optimal_price)
            
        except Exception as e:
            logger.warning(f"Error optimizing base price: {str(e)}")
            return float(base_price)
    
    def _optimize_discount(
        self,
        data: pd.DataFrame,
        demand_forecast: np.ndarray,
        base_price: float,
        unit_cost: float
    ) -> float:
        """Optimize discount percentage."""
        try:
            # Ensure we have valid base price and unit cost
            if not np.isfinite(base_price) or base_price <= 0:
                base_price = data['Price'].mean() if 'Price' in data.columns else 100.0
            if not np.isfinite(unit_cost) or unit_cost <= 0:
                unit_cost = base_price * 0.7
            
            # Define discount range
            discount_range = np.linspace(0, 0.5, 20)  # 0% to 50% discount
            
            # Calculate profit at each discount level
            profits = []
            for discount in discount_range:
                # Calculate effective price
                price = base_price * (1 - discount)
                
                # Skip if price is below cost
                if price <= unit_cost:
                    profits.append(-np.inf)
                    continue
                
                # Ensure valid price ratio
                price_ratio = max(0.1, min(10.0, price / base_price))
                estimated_demand = demand_forecast * np.power(price_ratio, -1.5)  # Assume price elasticity of -1.5
                
                # Calculate profit
                profit = (price - unit_cost) * estimated_demand
                profits.append(np.mean(profit))
            
            # Find discount that maximizes profit
            optimal_idx = np.argmax(profits)
            optimal_discount = discount_range[optimal_idx]
            
            return float(optimal_discount)
            
        except Exception as e:
            logger.warning(f"Error optimizing discount: {str(e)}")
            return 0.1  # Conservative default
    
    def _estimate_demand(
        self,
        data: pd.DataFrame,
        price: float,
        demand_forecast: np.ndarray
    ) -> float:
        """Estimate demand at a given price."""
        try:
            # Calculate base price
            base_price = data['Price'].mean() / (1 - data['Discount'].mean()) if 'Price' in data.columns and 'Discount' in data.columns else 100.0
            
            # Ensure valid price ratio
            price_ratio = max(0.1, min(10.0, price / base_price))
            
            # Estimate demand using price elasticity
            estimated_demand = demand_forecast * np.power(price_ratio, -1.5)  # Assume price elasticity of -1.5
            
            return float(np.mean(estimated_demand))
            
        except Exception as e:
            logger.warning(f"Error estimating demand: {str(e)}")
            return float(data['Units_Sold'].mean() if 'Units_Sold' in data.columns else 10.0)
    
    def optimize_price(
        self,
        historical_data: pd.DataFrame,
        product_id: str,
        cluster_id: int,
        constraints: Optional[Dict] = None
    ) -> PricingPolicy:
        """Optimize pricing policy for a product."""
        try:
            # Ensure required columns exist
            required_cols = ['Units_Sold', 'Price', 'Discount', 'Product_ID', 'date']
            for col in required_cols:
                if col not in historical_data.columns:
                    historical_data[col] = 0.0 if col in ['Price', 'Discount'] else 'default'
            
            # Handle missing values
            historical_data = historical_data.fillna({
                'Price': historical_data['Price'].mean(),
                'Discount': 0.0,
                'Units_Sold': 0.0
            })
            
            # Ensure date is datetime
            if 'date' in historical_data.columns:
                historical_data['date'] = pd.to_datetime(historical_data['date'])
            
            # Get product-specific data
            product_data = historical_data[historical_data['Product_ID'] == product_id].copy()
            if len(product_data) == 0:
                product_data = historical_data.copy()
            
            # Calculate base price and unit cost
            base_price = product_data['Price'].mean()
            unit_cost = base_price * (1 - self.min_margin)  # Use minimum margin to estimate cost
            
            # Generate demand forecast
            try:
                demand_forecast = self.forecaster.predict(product_data, np.full(len(product_data), cluster_id))
            except Exception as e:
                logger.warning(f"Error generating demand forecast: {str(e)}")
                demand_forecast = product_data['Units_Sold'].values
            
            # Optimize base price
            try:
                optimal_base_price = self._optimize_base_price(
                    product_data,
                    demand_forecast,
                    unit_cost,
                    base_price
                )
            except Exception as e:
                logger.warning(f"Error optimizing base price: {str(e)}")
                optimal_base_price = base_price
            
            # Optimize discount
            try:
                optimal_discount = self._optimize_discount(
                    product_data,
                    demand_forecast,
                    optimal_base_price,
                    unit_cost
                )
            except Exception as e:
                logger.warning(f"Error optimizing discount: {str(e)}")
                optimal_discount = 0.0
            
            # Calculate price elasticity
            try:
                elasticity = self._calculate_price_elasticity(product_data)
            except Exception as e:
                logger.warning(f"Error calculating price elasticity: {str(e)}")
                elasticity = -1.0  # Default to unit elasticity
            
            # Calculate expected demand
            try:
                expected_demand = self._estimate_demand(
                    product_data,
                    optimal_base_price * (1 - optimal_discount),
                    demand_forecast
                )
            except Exception as e:
                logger.warning(f"Error estimating demand: {str(e)}")
                expected_demand = product_data['Units_Sold'].mean()
            
            # Calculate price bounds
            min_price = max(unit_cost, optimal_base_price * 0.5)
            max_price = optimal_base_price * 2.0
            
            # Apply constraints if provided
            if constraints:
                min_price = max(min_price, constraints.get('min_price', min_price))
                max_price = min(max_price, constraints.get('max_price', max_price))
                optimal_discount = min(optimal_discount, constraints.get('max_discount', optimal_discount))
            
            return PricingPolicy(
                base_price=optimal_base_price,
                min_price=min_price,
                max_price=max_price,
                optimal_discount=optimal_discount,
                price_elasticity=elasticity,
                expected_demand=expected_demand
            )
            
        except Exception as e:
            logger.error(f"Error optimizing pricing policy: {str(e)}")
            # Return conservative default policy
            return PricingPolicy(
                base_price=base_price if 'base_price' in locals() else 100.0,
                min_price=base_price * 0.5 if 'base_price' in locals() else 50.0,
                max_price=base_price * 2.0 if 'base_price' in locals() else 200.0,
                optimal_discount=0.0,
                price_elasticity=-1.0,
                expected_demand=product_data['Units_Sold'].mean() if 'product_data' in locals() else 0.0
            )
    
    def optimize_all(self, data: pd.DataFrame, clusters: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Optimize pricing for all clusters."""
        print("Optimizing pricing for all clusters...")
        
        # Validate input data
        required_cols = ['Date', 'Units_Sold', 'Price', 'Competitor_Pricing']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for missing values and handle them appropriately
        data = data.copy()
        data['Units_Sold'] = data['Units_Sold'].fillna(0)
        data['Price'] = data['Price'].fillna(data['Price'].mean())
        data['Competitor_Pricing'] = data['Competitor_Pricing'].fillna(data['Price'].mean())
        
        # Add cluster column to data
        data['Cluster'] = clusters
        
        # Initialize results dictionary
        results = {}
        
        # Get unique clusters and sort them
        unique_clusters = np.unique(clusters)
        unique_clusters = unique_clusters[~pd.isna(unique_clusters)]
        
        # Optimize for each cluster
        for cluster_id in unique_clusters:
            print(f"Optimizing pricing for cluster {int(cluster_id)}...")
            
            # Get data for this cluster
            cluster_data = data[data['Cluster'] == cluster_id]
            
            if len(cluster_data) < 10:  # Skip clusters with too few samples
                print(f"Skipping cluster {int(cluster_id)} due to insufficient data")
                continue
            
            try:
                # Calculate price elasticity with error handling
                elasticity = self._calculate_price_elasticity(cluster_data)
                
                # Optimize price with constraints
                optimal_price = self._optimize_price(cluster_data, elasticity)
                
                # Calculate expected margin
                expected_margin = self._calculate_expected_margin(cluster_data, optimal_price)
                
                # Calculate additional metrics
                current_price = cluster_data['Price'].mean()
                price_change = (optimal_price - current_price) / current_price
                
                # Store results with additional metrics
                results[str(int(cluster_id))] = {
                    'optimal_price': float(optimal_price),
                    'current_price': float(current_price),
                    'price_change_percentage': float(price_change * 100),
                    'price_elasticity': float(elasticity),
                    'expected_margin': float(expected_margin),
                    'sample_size': int(len(cluster_data)),
                    'status': 'success'
                }
                
            except Exception as e:
                print(f"Warning: Error optimizing cluster {int(cluster_id)}: {str(e)}")
                results[str(int(cluster_id))] = {
                    'error': str(e),
                    'sample_size': int(len(cluster_data)),
                    'status': 'error'
                }
        
        # Check if any clusters were optimized
        successful_clusters = [k for k, v in results.items() if v.get('status') == 'success']
        if not successful_clusters:
            raise ValueError("No clusters were optimized due to insufficient data or errors")
        
        # Add summary statistics
        results['summary'] = {
            'total_clusters': len(unique_clusters),
            'successful_clusters': len(successful_clusters),
            'failed_clusters': len(unique_clusters) - len(successful_clusters),
            'average_price_change': np.mean([v.get('price_change_percentage', 0) for v in results.values() if v.get('status') == 'success'])
        }
        
        return results
    
    def _calculate_price_elasticity(self, data: pd.DataFrame) -> float:
        """Calculate price elasticity of demand with improved error handling."""
        try:
            # Add small constants to avoid log(0)
            epsilon = 1e-10
            log_price = np.log(data['Price'] + epsilon)
            log_quantity = np.log(data['Units_Sold'] + epsilon)
            
            # Remove outliers using IQR method
            price_q1 = np.percentile(log_price, 25)
            price_q3 = np.percentile(log_price, 75)
            price_iqr = price_q3 - price_q1
            # Calculate elasticity using linear regression
            X = np.column_stack([np.ones(len(log_price)), log_price])
            beta = np.linalg.inv(X.T @ X) @ X.T @ log_quantity
            elasticity = beta[1]  # Price coefficient
            
            # Validate elasticity
            if not np.isfinite(elasticity):
                # If elasticity calculation fails, use a reasonable default
                elasticity = -1.5  # Typical price elasticity for retail products
            
            return float(elasticity)
        except Exception as e:
            print(f"Error calculating price elasticity: {str(e)}")
            # Return a reasonable default elasticity
            return -1.5
    
    def _optimize_price(self, data: pd.DataFrame, elasticity: float) -> float:
        """Optimize price based on elasticity and cost."""
        try:
            # Calculate average cost
            avg_cost = data['Price'].mean() * 0.7  # Assuming 30% margin
            
            # Handle edge cases for elasticity
            if elasticity >= 0 or abs(elasticity) < 1e-10:
                # If elasticity is positive or very close to zero, use a reasonable default
                elasticity = -1.5
            
            # Calculate optimal price using elasticity formula
            optimal_price = avg_cost / (1 + 1/elasticity)
            
            # Validate optimal price
            if not np.isfinite(optimal_price) or optimal_price <= 0:
                # If optimal price is invalid, use a reasonable default based on current price
                optimal_price = data['Price'].mean()
            
            # Ensure price is within reasonable bounds
            min_price = data['Price'].min() * 0.5
            max_price = data['Price'].max() * 1.5
            optimal_price = np.clip(optimal_price, min_price, max_price)
            
            return float(optimal_price)
        except Exception as e:
            print(f"Error optimizing price: {str(e)}")
            # Return a reasonable default price
            return float(data['Price'].mean())
    
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