import pandas as pd
import numpy as np

# --------------------------------------------------
# CODE PREPARATION
# Loading enriched datasets from Weeks 2-3
# --------------------------------------------------
print("=" * 50)
print("CODE PREPARATION: LOADING WEEKS 2-3 ENRICHED DATASETS")
print("=" * 50)

sold_df = pd.read_csv("week2-3_sold_with_rates.csv", low_memory=False)
listings_df = pd.read_csv("week2-3_listings_with_rates.csv", low_memory=False)

initial_sold_rows = len(sold_df)
initial_listings_rows = len(listings_df)

print(f"Sold dataset loaded: {initial_sold_rows} rows x {sold_df.shape[1]} columns")
print(f"Listings dataset loaded: {initial_listings_rows} rows x {listings_df.shape[1]} columns")

# --------------------------------------------------
# PART 1: TASKS
# --------------------------------------------------
print("=" * 50)
print("PART 1:TASKS")
print("=" * 50)

# 1. Convert date fields to datetime format (CloseDate, PurchaseContractDate, ListingContractDate, ContractStatusChangeDate)
print ("- " * 25)
print("Task 1: Convert date fields to datetime format.")
print ("- " * 25)

date_fields = ['CloseDate', 'PurchaseContractDate', 'ListingContractDate', 'ContractStatusChangeDate']

# Change this:
# for df in [sold_df, "Sold"], (listings_df, "Listings"):

# To this:
for df, name in [(sold_df, "Sold"), (listings_df, "Listings")]:
    print(f"\nProcessing {name} Datetime Fields...")
    for col in date_fields:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', format='mixed')
            valid_dates = df[col].notnull().sum()
            print(f"  -> {col:26s}: {valid_dates:,} valid dates parsed ({df[col].isnull().sum():,} NaT/nulls)")

# 2. Remove unnecessary or redundant columns

# 3. Handle missing values appropriately
# 4. Ensure numeric fields are properly typed
# 5. Remove or flag invalid numeric values: ClosePrice <= 0, LivingArea <= 0, DaysOnMarket < 0, negative Bedrooms or Bathrooms

# --------------------------------------------------
# PART 2: DATE CONSISTENCY CHECKS
# --------------------------------------------------

# --------------------------------------------------
# PART 3: GEOGRAPHIC DATA CHECKS
# --------------------------------------------------