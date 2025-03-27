import numpy as np
import pandas as pd

class DataPreprocessor:
    def __init__(self, df):
        self.df = df
        self.rename_columns()

    def rename_columns(self):
        """Rename columns to use underscores instead of spaces."""
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
        self.df = self.df.rename(columns=column_mapping)

    def calculate_derived_features(self):
        # Calculate derived features
        self.df['Inventory_Sales_Ratio'] = self.df['Inventory_Level'] / self.df['Units_Sold'].replace(0, np.nan)
        self.df['Sell_Through_Rate'] = self.df['Units_Sold'] / (self.df['Inventory_Level'] + self.df['Units_Sold'])
        
        # Calculate forecast accuracy (normalized between -1 and 1)
        error = self.df['Demand_Forecast'] - self.df['Units_Sold']
        max_val = np.maximum(np.abs(self.df['Demand_Forecast']), np.abs(self.df['Units_Sold']))
        accuracy = -error / max_val.replace(0, np.nan)  # Negative because positive error means overforecast
        self.df['Forecast_Accuracy'] = np.clip(accuracy, -1, 1)  # Clip values to [-1, 1] range
        
        self.df['Supply_Gap'] = self.df['Inventory_Level'] - self.df['Units_Sold']

        # Calculate inventory status
        conditions = [
            (self.df['Inventory_Level'] < self.df['Units_Sold']),
            (self.df['Inventory_Level'] >= self.df['Units_Sold']) & (self.df['Inventory_Level'] <= self.df['Units_Sold'] * 1.5),
            (self.df['Inventory_Level'] > self.df['Units_Sold'] * 1.5)
        ]
        choices = np.array(['Understocked', 'In stock', 'Overstocked'], dtype=str)
        self.df['Inventory_Status'] = np.select(conditions, choices, default='Unknown')
        return self.df

def main():
    """Main function to preprocess the data."""
    try:
        # Load the data
        print("Loading data...")
        df = pd.read_csv('data/retail_store_inventory.csv')
        
        # Create preprocessor instance and process data
        print("Processing data...")
        preprocessor = DataPreprocessor(df)
        processed_df = preprocessor.calculate_derived_features()
        
        # Save processed data
        output_path = 'data/processed_inventory.csv'
        processed_df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        return 0
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main()) 