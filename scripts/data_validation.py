import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# Import pandas_schema with error handling
try:
    import pandas_schema
    from pandas_schema import Column
    from pandas_schema.validation import CustomElementValidation, InRangeValidation, IsDistinctValidation
    PANDAS_SCHEMA_AVAILABLE = True
except ImportError:
    PANDAS_SCHEMA_AVAILABLE = False
    print("WARNING: pandas_schema not available. Using simplified validation instead.")

def load_config():
    """Load configuration file."""
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return {
            "data": {
                "path": "data/retail_store_inventory.csv"
            }
        }

def validate_data(data_path, output_dir="output/data_validation"):
    """Validate the data based on business rules and schema."""
    print(f"Loading data from {data_path} for validation...")
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Create validation report directory
    os.makedirs(os.path.join(output_dir, "validation"), exist_ok=True)
    
    # Basic validation that doesn't require pandas_schema
    validation_issues = []
    
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        validation_issues.append(f"Found {missing_values.sum()} missing values")
        for col in missing_values[missing_values > 0].index:
            validation_issues.append(f"  - {col}: {missing_values[col]} missing values")
    
    # Check for negative inventory values
    if 'Inventory Level' in data.columns and (data['Inventory Level'] < 0).any():
        neg_count = (data['Inventory Level'] < 0).sum()
        validation_issues.append(f"Found {neg_count} negative inventory levels")
    
    # Check date ranges
    if 'Date' in data.columns:
        try:
            data['Date'] = pd.to_datetime(data['Date'])
            date_range = (data['Date'].max() - data['Date'].min()).days
            validation_issues.append(f"Data spans {date_range} days from {data['Date'].min().date()} to {data['Date'].max().date()}")
        except:
            validation_issues.append("Error parsing Date column")
    
    # Advanced validation with pandas_schema if available
    if PANDAS_SCHEMA_AVAILABLE:
        # Define schema
        schema = pandas_schema.Schema([
            Column("Date", [CustomElementValidation(lambda d: pd.to_datetime(d, errors='coerce') is not pd.NaT, "Invalid date")]),
            Column("Store ID", [CustomElementValidation(lambda x: pd.to_numeric(x, errors='coerce') is not pd.NA, "Invalid Store ID")]),
            Column("Product ID", [CustomElementValidation(lambda x: pd.to_numeric(x, errors='coerce') is not pd.NA, "Invalid Product ID")]),
            Column("Category", [CustomElementValidation(lambda x: isinstance(x, str), "Category must be string")]),
            Column("Region", [CustomElementValidation(lambda x: isinstance(x, str), "Region must be string")]),
            Column("Price", [CustomElementValidation(lambda x: x >= 0, "Price must be positive")]),
            Column("Units Sold", [CustomElementValidation(lambda x: x >= 0, "Units Sold must be positive")]),
            Column("Inventory Level", [CustomElementValidation(lambda x: x >= 0, "Inventory Level must be positive")]),
            Column("Demand Forecast", [CustomElementValidation(lambda x: x >= 0, "Demand Forecast must be positive")]),
            Column("Discount", [CustomElementValidation(lambda x: 0 <= x <= 100, "Discount must be between 0-100%")]),
            Column("Weather Condition", [CustomElementValidation(lambda x: isinstance(x, str), "Weather must be string")]),
            Column("Holiday/Promotion", [CustomElementValidation(lambda x: x in [0, 1], "Holiday/Promotion must be 0 or 1")]),
            Column("Competitor Pricing", [CustomElementValidation(lambda x: x >= 0, "Competitor Pricing must be positive")])
        ])
        
        # Validate schema
        errors = schema.validate(data)
        schema_errors = []
        for error in errors:
            schema_errors.append({
                "column": error.column,
                "row": error.row,
                "message": str(error)
            })
        
        # Check for duplicates
        duplicates = data.duplicated().sum()
        
        # Check for outliers using IQR method
        outliers = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            if outlier_count > 0:
                outliers[col] = int(outlier_count)
        
        # Check date consistency
        date_issues = []
        try:
            date_range = (data['Date'].max() - data['Date'].min()).days + 1
            unique_dates = data['Date'].nunique()
            if date_range != unique_dates:
                date_issues.append(f"Missing dates in sequence. Range: {date_range} days, Unique dates: {unique_dates}")
        except Exception as e:
            date_issues.append(f"Error processing dates: {str(e)}")
        
        # Check for negative values in numerical columns that should be positive
        negative_values = {}
        for col in ["Price", "Units Sold", "Inventory Level", "Demand Forecast"]:
            if col in data.columns:
                neg_count = (data[col] < 0).sum()
                if neg_count > 0:
                    negative_values[col] = int(neg_count)
        
        # Validation summary
        validation_results = {
            "status": "passed" if not (schema_errors or missing_values or duplicates or outliers or date_issues or negative_values) else "failed",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": data_path,
            "rows": len(data),
            "columns": len(data.columns),
            "schema_validation": {
                "passed": len(schema_errors) == 0,
                "errors": schema_errors
            },
            "missing_values": {
                "has_missing": missing_values.sum() > 0,
                "counts": missing_values.to_dict()
            },
            "duplicates": {
                "has_duplicates": duplicates > 0,
                "count": int(duplicates)
            },
            "outliers": {
                "has_outliers": len(outliers) > 0,
                "counts": outliers
            },
            "date_validation": {
                "passed": len(date_issues) == 0,
                "issues": date_issues
            },
            "negative_values": {
                "has_negatives": len(negative_values) > 0,
                "counts": negative_values
            }
        }
        
        # Save validation results
        with open(os.path.join(output_dir, "validation", "validation_results.json"), "w") as f:
            json.dump(validation_results, f, indent=4)
        
        # Generate validation report
        with open(os.path.join(output_dir, "validation", "validation_report.md"), "w") as f:
            f.write("# Data Validation Report\n\n")
            f.write(f"**File:** {data_path}\n")
            f.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Status:** {'PASSED' if validation_results['status'] == 'passed' else 'FAILED'}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Rows:** {len(data)}\n")
            f.write(f"- **Columns:** {len(data.columns)}\n")
            f.write(f"- **Schema Validation:** {'PASSED' if not schema_errors else 'FAILED'}\n")
            f.write(f"- **Missing Values:** {'YES' if missing_values.sum() > 0 else 'NO'}\n")
            f.write(f"- **Duplicates:** {'YES' if duplicates > 0 else 'NO'}\n")
            f.write(f"- **Outliers:** {'YES' if outliers else 'NO'}\n")
            f.write(f"- **Date Issues:** {'YES' if date_issues else 'NO'}\n")
            f.write(f"- **Negative Values:** {'YES' if negative_values else 'NO'}\n\n")
            
            if schema_errors:
                f.write("## Schema Errors\n\n")
                for i, error in enumerate(schema_errors[:10]):  # Show first 10 errors
                    f.write(f"{i+1}. **{error['column']}** (Row {error['row']}): {error['message']}\n")
                if len(schema_errors) > 10:
                    f.write(f"... and {len(schema_errors) - 10} more errors\n")
                f.write("\n")
            
            if missing_values.sum() > 0:
                f.write("## Missing Values\n\n")
                f.write("| Column | Missing Count | Percentage |\n")
                f.write("|--------|--------------|------------|\n")
                for col, count in missing_values.items():
                    if count > 0:
                        percentage = (count / len(data)) * 100
                        f.write(f"| {col} | {count} | {percentage:.2f}% |\n")
                f.write("\n")
            
            if outliers:
                f.write("## Outliers\n\n")
                f.write("| Column | Outlier Count | Percentage |\n")
                f.write("|--------|--------------|------------|\n")
                for col, count in outliers.items():
                    percentage = (count / len(data)) * 100
                    f.write(f"| {col} | {count} | {percentage:.2f}% |\n")
                f.write("\n")
                
            if negative_values:
                f.write("## Negative Values\n\n")
                f.write("| Column | Count | Percentage |\n")
                f.write("|--------|-------|------------|\n")
                for col, count in negative_values.items():
                    percentage = (count / len(data)) * 100
                    f.write(f"| {col} | {count} | {percentage:.2f}% |\n")
                f.write("\n")
            
            f.write("## Recommendations\n\n")
            
            if schema_errors:
                f.write("- Fix schema violations in the data\n")
            if missing_values.sum() > 0:
                f.write("- Handle missing values through imputation or removal\n")
            if duplicates > 0:
                f.write("- Review and remove duplicated records\n")
            if outliers:
                f.write("- Investigate outliers to determine if they are errors or valid extreme values\n")
            if date_issues:
                f.write("- Address date continuity issues\n")
            if negative_values:
                f.write("- Correct negative values in columns that should be positive\n")
        
        # Return validation result
        return validation_results['status'] == 'passed'
    else:
        # Return validation result
        return len(validation_issues) == 0 or all(issue.startswith("Data spans") for issue in validation_issues)

def main():
    """Main function to run data validation."""
    print("Starting data validation...")
    
    # Load configuration
    config = load_config()
    data_path = config.get("data", {}).get("path", "data/retail_store_inventory.csv")
    output_dir = "output/data_validation"
    
    # Validate data
    is_valid = validate_data(data_path, output_dir)
    
    if is_valid:
        print("Data validation PASSED! Data is ready for analysis.")
    else:
        print("Data validation FAILED! Please review validation report and address issues.")

if __name__ == "__main__":
    main() 