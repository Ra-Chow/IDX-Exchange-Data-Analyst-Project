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
print("PART 1: TASKS")
print("=" * 50)

# Task 1: Convert date fields to datetime format (CloseDate, PurchaseContractDate, ListingContractDate, ContractStatusChangeDate)
print ("- " * 25)
print("Task 1: Convert date fields to datetime format.")
print ("- " * 25)

# List of fields to convert to datetime format.
date_fields = ['CloseDate', 'PurchaseContractDate', 'ListingContractDate', 'ContractStatusChangeDate']

# Conversion to datetime format for both sold and listings datasets, with error handling for invalid formats.
for df, name in [(sold_df, "Sold"), (listings_df, "Listings")]:
    print(f"\nProcessing {name} Datetime Fields...")
    for col in date_fields:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', format='mixed')
            valid_dates = df[col].notnull().sum()
            print(f"  -> {col:26s}: {valid_dates:,} valid dates parsed ({df[col].isnull().sum():,} NaT/nulls)")

# Task 2: Remove unnecessary or redundant columns
print ("- " * 25)
print("Task 2: Remove unnecessary or redundant columns.")
print ("- " * 25)

def apply_date_flags(df, name):
    print(f"\nAnalyzing Timeline Logic for {name} Dataset:")
    
    # Flag 1: ListingContractDate happens AFTER CloseDate
    if 'ListingContractDate' in df.columns and 'CloseDate' in df.columns:
        df['listing_after_close_flag'] = (df['ListingContractDate'] > df['CloseDate'])
    else:
        df['listing_after_close_flag'] = False

    # Flag 2: PurchaseContractDate happens AFTER CloseDate
    if 'PurchaseContractDate' in df.columns and 'CloseDate' in df.columns:
        df['purchase_after_close_flag'] = (df['PurchaseContractDate'] > df['CloseDate'])
    else:
        df['purchase_after_close_flag'] = False

    # Flag 3: ListingContractDate happens AFTER PurchaseContractDate (Negative timeline)
    if 'ListingContractDate' in df.columns and 'PurchaseContractDate' in df.columns:
        df['negative_timeline_flag'] = (df['ListingContractDate'] > df['PurchaseContractDate'])
    else:
        df['negative_timeline_flag'] = False

    print(f"  -> listing_after_close_flag  : {df['listing_after_close_flag'].sum():,} records flagged")
    print(f"  -> purchase_after_close_flag : {df['purchase_after_close_flag'].sum():,} records flagged")
    print(f"  -> negative_timeline_flag    : {df['negative_timeline_flag'].sum():,} records flagged")
    
    return df

sold_df = apply_date_flags(sold_df, "Sold")
listings_df = apply_date_flags(listings_df, "Listings")

# Task 3: Handle missing values appropriately
print ("- " * 25)
print("Task 3: Handle missing values appropriately.")
print ("- " * 25)

def clean_numeric_boundaries(df, name):
    initial_count = len(df)
    
    # 1. Price check: ClosePrice <= 0 (or ListPrice <= 0 for active listings)
    if 'ClosePrice' in df.columns:
        invalid_price = (df['ClosePrice'] <= 0) | (df['ClosePrice'].isnull())
    elif 'ListPrice' in df.columns:
        invalid_price = (df['ListPrice'] <= 0) | (df['ListPrice'].isnull())
    else:
        invalid_price = pd.Series(False, index=df.index)

    # 2. Living area check: LivingArea <= 0
    if 'LivingArea' in df.columns:
        invalid_area = (df['LivingArea'] <= 0) | (df['LivingArea'].isnull())
    else:
        invalid_area = pd.Series(False, index=df.index)

    # 3. Days on Market check: DaysOnMarket < 0
    if 'DaysOnMarket' in df.columns:
        invalid_dom = (df['DaysOnMarket'] < 0)
    else:
        invalid_dom = pd.Series(False, index=df.index)

    # 4. Room count check: BedroomsTotal < 0 or BathroomsTotalInteger < 0
    invalid_beds = (df['BedroomsTotal'] < 0) if 'BedroomsTotal' in df.columns else pd.Series(False, index=df.index)
    invalid_baths = (df['BathroomsTotalInteger'] < 0) if 'BathroomsTotalInteger' in df.columns else pd.Series(False, index=df.index)

    total_invalid_mask = invalid_price | invalid_area | invalid_dom | invalid_beds | invalid_baths
    invalid_records_count = total_invalid_mask.sum()

    print(f"\n{name} Dataset Invalid Values Found:")
    print(f"  -> Invalid Prices (<=0 or Null)   : {invalid_price.sum():,}")
    print(f"  -> Invalid Living Area (<=0/Null) : {invalid_area.sum():,}")
    print(f"  -> Invalid Days on Market (<0)   : {invalid_dom.sum():,}")
    print(f"  -> Invalid Bedrooms/Bathrooms (<0): {(invalid_beds | invalid_baths).sum():,}")
    print(f"  -> Total Unique Rows Excluded    : {invalid_records_count:,}")

    df_cleaned = df[~total_invalid_mask].copy()
    print(f"  -> Remaining Clean Rows          : {len(df_cleaned):,} ({(len(df_cleaned)/initial_count)*100:.2f}% retained)")
    
    return df_cleaned

sold_cleaned = clean_numeric_boundaries(sold_df, "Sold")
listings_cleaned = clean_numeric_boundaries(listings_df, "Listings")

# Task 4: Ensure numeric fields are properly typed
print ("- " * 25)
print("Task 4: Ensure numeric fields are properly typed.")
print ("- " * 25)

# Task 5: Remove or flag invalid numeric values: ClosePrice <= 0, LivingArea <= 0, DaysOnMarket < 0, negative Bedrooms or Bathrooms
print ("- " * 25)
print("Task 5: Remove or flag invalid numeric values.")
print ("- " * 25)

# --------------------------------------------------
# PART 2: DATE CONSISTENCY CHECKS
# --------------------------------------------------
print("=" * 50)
print("PART 2: DATE CONSISTENCY CHECKS")
print("=" * 50)

def apply_date(df, name):
    print(f"Analyzing Timeline Logic for {name} Dataset:")
    
    # Flag 1: ListingContractDate happens AFTER CloseDate
    if 'ListingContractDate' in df.columns and 'CloseDate' in df.columns:
        df['listing_after_close_flag'] = (df['ListingContractDate'] > df['CloseDate'])
    else:
        df['listing_after_close_flag'] = False

    # Flag 2: PurchaseContractDate happens AFTER CloseDate
    if 'PurchaseContractDate' in df.columns and 'CloseDate' in df.columns:
        df['purchase_after_close_flag'] = (df['PurchaseContractDate'] > df['CloseDate'])
    else:
        df['purchase_after_close_flag'] = False

    # Flag 3: ListingContractDate happens AFTER PurchaseContractDate (Negative timeline)
    if 'ListingContractDate' in df.columns and 'PurchaseContractDate' in df.columns:
        df['negative_timeline_flag'] = (df['ListingContractDate'] > df['PurchaseContractDate'])
    else:
        df['negative_timeline_flag'] = False

    print(f"  -> listing_after_close_flag  : {df['listing_after_close_flag'].sum():,} records flagged")
    print(f"  -> purchase_after_close_flag : {df['purchase_after_close_flag'].sum():,} records flagged")
    print(f"  -> negative_timeline_flag    : {df['negative_timeline_flag'].sum():,} records flagged")
    
    return df

sold_df = apply_date(sold_cleaned, "Sold")
listings_df = apply_date(listings_cleaned, "Listings")


def clean_numeric_boundaries(df, name):
    initial_count = len(df)
    
    # 1. Price check: ClosePrice <= 0 (or ListPrice <= 0 for active listings)
    if 'ClosePrice' in df.columns:
        invalid_price = (df['ClosePrice'] <= 0) | (df['ClosePrice'].isnull())
    elif 'ListPrice' in df.columns:
        invalid_price = (df['ListPrice'] <= 0) | (df['ListPrice'].isnull())
    else:
        invalid_price = pd.Series(False, index=df.index)

    # 2. Living area check: LivingArea <= 0
    if 'LivingArea' in df.columns:
        invalid_area = (df['LivingArea'] <= 0) | (df['LivingArea'].isnull())
    else:
        invalid_area = pd.Series(False, index=df.index)

    # 3. Days on Market check: DaysOnMarket < 0
    if 'DaysOnMarket' in df.columns:
        invalid_dom = (df['DaysOnMarket'] < 0)
    else:
        invalid_dom = pd.Series(False, index=df.index)

    # 4. Room count check: BedroomsTotal < 0 or BathroomsTotalInteger < 0
    invalid_beds = (df['BedroomsTotal'] < 0) if 'BedroomsTotal' in df.columns else pd.Series(False, index=df.index)
    invalid_baths = (df['BathroomsTotalInteger'] < 0) if 'BathroomsTotalInteger' in df.columns else pd.Series(False, index=df.index)

    total_invalid_mask = invalid_price | invalid_area | invalid_dom | invalid_beds | invalid_baths
    invalid_records_count = total_invalid_mask.sum()

    print(f"\n{name} Dataset Invalid Values Found:")
    print(f"  -> Invalid Prices (<=0 or Null)   : {invalid_price.sum():,}")
    print(f"  -> Invalid Living Area (<=0/Null) : {invalid_area.sum():,}")
    print(f"  -> Invalid Days on Market (<0)   : {invalid_dom.sum():,}")
    print(f"  -> Invalid Bedrooms/Bathrooms (<0): {(invalid_beds | invalid_baths).sum():,}")
    print(f"  -> Total Unique Rows Excluded    : {invalid_records_count:,}")

    df_cleaned = df[~total_invalid_mask].copy()
    print(f"  -> Remaining Clean Rows          : {len(df_cleaned):,} ({(len(df_cleaned)/initial_count)*100:.2f}% retained)")

    return df_cleaned

sold_cleaned = clean_numeric_boundaries(sold_df, "Sold")
listings_cleaned = clean_numeric_boundaries(listings_df, "Listings")


# --------------------------------------------------
# PART 3: GEOGRAPHIC DATA CHECKS
# --------------------------------------------------
print("=" * 50)
print("PART 3: GEOGRAPHIC DATA CHECKS")
print("=" * 50)

def apply_geographic_flags(df, name):
    print(f"\nAuditing Geographic Coordinates for {name} Dataset:")
    
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        lat = pd.to_numeric(df['Latitude'], errors='coerce')
        lon = pd.to_numeric(df['Longitude'], errors='coerce')

        # Rule 1: Missing coordinates
        df['geo_missing_flag'] = lat.isnull() | lon.isnull()

        # Rule 2: Sentinel zero values (Latitude = 0 or Longitude = 0)
        df['geo_zero_flag'] = (lat == 0) | (lon == 0)

        # Rule 3: Longitude > 0 error (California longitudes must be negative)
        df['geo_positive_longitude_flag'] = (lon > 0)

        # Rule 4: Out-of-state / Implausible California boundaries
        # CA Bounding Box approx: Lat [32.5, 42.0], Lon [-124.5, -114.1]
        out_of_bounds_lat = (lat < 32.5) | (lat > 42.0)
        out_of_bounds_lon = (lon < -124.5) | (lon > -114.1)
        df['geo_out_of_bounds_flag'] = out_of_bounds_lat | out_of_bounds_lon

        print(f"  -> Missing Coordinates (Null)   : {df['geo_missing_flag'].sum():,}")
        print(f"  -> Sentinel Zero Coordinates    : {df['geo_zero_flag'].sum():,}")
        print(f"  -> Positive Longitude Errors    : {df['geo_positive_longitude_flag'].sum():,}")
        print(f"  -> Out of California Boundaries : {df['geo_out_of_bounds_flag'].sum():,}")
    else:
        print("  -> Geographic columns ('Latitude'/'Longitude') not found.")
        
    return df

sold_cleaned = apply_geographic_flags(sold_cleaned, "Sold")
listings_cleaned = apply_geographic_flags(listings_cleaned, "Listings")

# --------------------------------------------------
# PART 4: EXPORT CLEANED DATASETS
# --------------------------------------------------
print("=" * 50)
print("PART 4: EXPORT CLEANED DATASETS")
print("=" * 50)

print(f"Final Cleaned Sold Row Count    : {len(sold_cleaned):,} (Before: {initial_sold_rows:,})")
print(f"Final Cleaned Listings Row Count: {len(listings_cleaned):,} (Before: {initial_listings_rows:,})")

sold_cleaned.to_csv("week4-5_sold_cleaned.csv", index=False)
listings_cleaned.to_csv("week4-5_listings_cleaned.csv", index=False)

print("\nSuccessfully generated output deliverables:")
print("  -> 'week4-5_sold_cleaned.csv'")
print("  -> 'week4-5_listings_cleaned.csv'")
print("\nWeeks 4-5 Data Cleaning Pipeline Execution Complete!")
