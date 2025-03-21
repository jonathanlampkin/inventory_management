# Retail Store Inventory Data Understanding

Analysis Date: 2025-03-21 13:34:17

## Basic Information

- Number of rows: 73100
- Number of columns: 15
- Columns: Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold, Units Ordered, Demand Forecast, Price, Discount, Weather Condition, Holiday/Promotion, Competitor Pricing, Seasonality
- Duplicated rows: 0

## Column Data Types

- Date: object
- Store ID: object
- Product ID: object
- Category: object
- Region: object
- Inventory Level: int64
- Units Sold: int64
- Units Ordered: int64
- Demand Forecast: float64
- Price: float64
- Discount: int64
- Weather Condition: object
- Holiday/Promotion: int64
- Competitor Pricing: float64
- Seasonality: object

## Temporal Information

- Date range: 2022-01-01 to 2024-01-01
- Total days: 731
- Unique dates: 731

## Categorical Variables

### Store ID

- Unique values: 5
- Top values:
  - S001: 14620
  - S002: 14620
  - S003: 14620
  - S004: 14620
  - S005: 14620

### Product ID

- Unique values: 20
- Top values:
  - P0001: 3655
  - P0002: 3655
  - P0003: 3655
  - P0004: 3655
  - P0005: 3655
  - P0006: 3655
  - P0007: 3655
  - P0008: 3655
  - P0009: 3655
  - P0010: 3655

### Category

- Unique values: 5
- Top values:
  - Furniture: 14699
  - Toys: 14643
  - Clothing: 14626
  - Groceries: 14611
  - Electronics: 14521

### Region

- Unique values: 4
- Top values:
  - East: 18349
  - South: 18297
  - North: 18228
  - West: 18226

### Weather Condition

- Unique values: 4
- Top values:
  - Sunny: 18290
  - Rainy: 18278
  - Snowy: 18272
  - Cloudy: 18260

### Holiday/Promotion

- Unique values: 2
- Top values:
  - 0: 36747
  - 1: 36353

### Seasonality

- Unique values: 4
- Top values:
  - Spring: 18317
  - Summer: 18305
  - Winter: 18285
  - Autumn: 18193

## Numerical Variables

### Inventory Level

- Min: 50.00
- Max: 500.00
- Mean: 274.47
- Median: 273.00
- Standard Deviation: 129.95
- Range: 450.00

### Units Sold

- Min: 0.00
- Max: 499.00
- Mean: 136.46
- Median: 107.00
- Standard Deviation: 108.92
- Range: 499.00

### Units Ordered

- Min: 20.00
- Max: 200.00
- Mean: 110.00
- Median: 110.00
- Standard Deviation: 52.28
- Range: 180.00

### Demand Forecast

- Min: -9.99
- Max: 518.55
- Mean: 141.49
- Median: 113.02
- Standard Deviation: 109.25
- Range: 528.54

### Price

- Min: 10.00
- Max: 100.00
- Mean: 55.14
- Median: 55.05
- Standard Deviation: 26.02
- Range: 90.00

### Discount

- Min: 0.00
- Max: 20.00
- Mean: 10.01
- Median: 10.00
- Standard Deviation: 7.08
- Range: 20.00

### Competitor Pricing

- Min: 5.03
- Max: 104.94
- Mean: 55.15
- Median: 55.01
- Standard Deviation: 26.19
- Range: 99.91

## Key Insights

- Top 5 products by units sold: {'P0016': 508472, 'P0020': 507708, 'P0014': 507622, 'P0015': 507283, 'P0005': 503648}
- Top 5 stores by units sold: {'S003': 2022696, 'S005': 2010176, 'S002': 1987715, 'S004': 1979245, 'S001': 1975750}
- Category sales performance: {'Furniture': 2025017, 'Groceries': 2000482, 'Clothing': 1999166, 'Toys': 1990485, 'Electronics': 1960432}
- Average units sold by discount level: {0: 135.69, 5: 136.57, 10: 136.77, 15: 136.66, 20: 136.64}
- Correlation between Price and Units Sold: 0.001
- Average units sold by season: {'Autumn': 137.78, 'Spring': 135.83, 'Summer': 135.43, 'Winter': 136.83}
- Average units sold by weather condition: {'Cloudy': 136.76, 'Rainy': 135.16, 'Snowy': 135.91, 'Sunny': 138.03}
- Average inventory to sales ratio: inf
- Average forecast accuracy: 0.854
