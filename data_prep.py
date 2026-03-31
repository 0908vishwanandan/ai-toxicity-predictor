import pandas as pd

# 1. Load the dataset 
df = pd.read_csv('tox21.csv')

# 2. Print the shape (Number of Rows, Number of Columns)
print(f"Dataset Shape: {df.shape}")

# 3. Print the column names so we know what data we have
print("\nColumns in the dataset:")
print(df.columns.tolist())

# 4. Print the first 3 rows just to look at the raw data
print("\nFirst 3 rows:")
print(df.head(3).to_string())