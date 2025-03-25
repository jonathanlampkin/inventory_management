import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings

# Suppress non-critical warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handles all data preprocessing tasks for the retail inventory analysis."""
    
    def __init__(self, data_path, output_dir="data/processed"):
        """Initialize the preprocessor with data path and output directory."""
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load the raw data."""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        return self
        
    def process_dates(self):
        """Process and create date-related features."""
        print("Processing dates...")
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Day'] = self.df['Date'].dt.day
        self.df['Quarter'] = self.df['Date'].dt.quarter
        self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
        self.df['WeekOfYear'] = self.df['Date'].dt.isocalendar().week
        self.df['MonthName'] = self.df['Date'].dt.strftime('%b')
        
        # Ensure MonthName is ordered correctly
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        self.df['MonthName'] = pd.Categorical(self.df['MonthName'], categories=month_order, ordered=True)
        
        return self
        
    def create_metrics(self):
        """Create derived metrics and ratios."""
        print("Creating derived metrics...")
        
        # Inventory and sales metrics
        self.df['Inventory_Sales_Ratio'] = self.df['Inventory Level'] / self.df['Units Sold'].replace(0, np.nan)
        self.df['Sell_Through_Rate'] = self.df['Units Sold'] / (self.df['Inventory Level'] + self.df['Units Sold'])
        self.df['Forecast_Accuracy'] = 1 - abs(self.df['Demand Forecast'] - self.df['Units Sold']) / self.df['Demand Forecast'].replace(0, np.nan)
        self.df['Supply_Gap'] = self.df['Inventory Level'] - self.df['Units Sold']
        
        # Supply status categories
        conditions = [
            (self.df['Inventory Level'] < self.df['Units Sold']),
            (self.df['Inventory Level'] >= self.df['Units Sold']) & 
            (self.df['Inventory Level'] <= self.df['Units Sold'] * 1.5),
            (self.df['Inventory Level'] > self.df['Units Sold'] * 1.5)
        ]
        choices = ['Undersupplied', 'Optimal', 'Oversupplied']
        self.df['Supply_Status'] = np.select(conditions, choices, default='Unknown')
        
        return self
        
    def handle_missing_values(self):
        """Handle missing values in the dataset."""
        print("Handling missing values...")
        
        # Fill missing values with appropriate methods
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].mean())
        
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        self.df[categorical_columns] = self.df[categorical_columns].fillna('Unknown')
        
        return self
        
    def save_processed_data(self):
        """Save the processed dataset."""
        print("Saving processed data...")
        output_path = os.path.join(self.output_dir, "processed_data.csv")
        self.df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        return self
        
    def run_preprocessing(self):
        """Run the complete preprocessing pipeline."""
        print("Starting data preprocessing...")
        start_time = datetime.now()
        
        self.load_data()
        self.process_dates()
        self.create_metrics()
        self.handle_missing_values()
        self.save_processed_data()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Preprocessing completed in {duration:.2f} seconds")
        
        return self.df

def main():
    """Main function to run the preprocessing pipeline."""
    print("Starting Retail Inventory Data Preprocessing...")
    
    data_path = "data/retail_store_inventory.csv"
    output_dir = "data/processed"
    
    # Initialize and run preprocessing
    preprocessor = DataPreprocessor(data_path, output_dir)
    processed_df = preprocessor.run_preprocessing()
    
    print("Data preprocessing complete!")

if __name__ == "__main__":
    main() 