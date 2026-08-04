import pandas as pd
import numpy as np

# --------------------------------------------------
# CODE PREPARATION
# Loading cleaned datasets from Weeks 4-5
# --------------------------------------------------
print("=" * 50)
print("CODE PREPARATION: LOADING WEEKS 4-5 CLEANED DATASETS")
print("=" * 50)

sold_df = pd.read_csv("week4-5_sold_cleaned.csv", low_memory=False)
listings_df = pd.read_csv("week4-5_listings_cleaned.csv", low_memory=False)

print(f"Sold dataset loaded: {sold_df.shape[0]} rows x {sold_df.shape[1]} columns")
print(f"Listings dataset loaded: {listings_df.shape[0]} rows x {listings_df.shape[1]} columns")

# Ensure date fields are properly formatted as datetime
date_fields = ['CloseDate', 'PurchaseContractDate', 'ListingContractDate']
for df in [sold_df, listings_df]:
    for col in date_fields:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

# --------------------------------------------------
# PART 1: FEATURE ENGINEERING & MARKET METRICS
# --------------------------------------------------
print("=" * 50)
print("PART 1: FEATURE ENGINEERING & MARKET METRICS")
print("=" * 50)

def engineer_market_metrics(df, name):
    print(f"\nEngineering key market metrics for {name} dataset...")
    
    # Price Ratio: Close Price / Original List Price
    if 'ClosePrice' in df.columns and 'OriginalListPrice' in df.columns:
        df['price_ratio'] = df['ClosePrice'] / df['OriginalListPrice']
        df['close_to_original_list_ratio'] = df['ClosePrice'] / df['OriginalListPrice']
    elif 'ListPrice' in df.columns and 'OriginalListPrice' in df.columns:
        df['price_ratio'] = df['ListPrice'] / df['OriginalListPrice']
        df['close_to_original_list_ratio'] = df['ListPrice'] / df['OriginalListPrice']

    # Price Per Sq Ft: Close Price / Living Area
    if 'ClosePrice' in df.columns and 'LivingArea' in df.columns:
        df['price_per_sq_ft'] = df['ClosePrice'] / df['LivingArea']
    elif 'ListPrice' in df.columns and 'LivingArea' in df.columns:
        df['price_per_sq_ft'] = df['ListPrice'] / df['LivingArea']

    # Days on Market (verify presence of raw field)
    if 'DaysOnMarket' not in df.columns:
        df['DaysOnMarket'] = np.nan

    # Time-Series Dimensions: Year, Month, YrMo derived from CloseDate or ListingContractDate
    target_date_col = 'CloseDate' if 'CloseDate' in df.columns else 'ListingContractDate'
    if target_date_col in df.columns:
        df['Year'] = df[target_date_col].dt.year
        df['Month'] = df[target_date_col].dt.month
        df['YrMo'] = df[target_date_col].dt.to_period('M').astype(str)

    # Listing to Contract Days: Purchase Contract Date - Listing Contract Date
    if 'PurchaseContractDate' in df.columns and 'ListingContractDate' in df.columns:
        df['listing_to_contract_days'] = (df['PurchaseContractDate'] - df['ListingContractDate']).dt.days

    # Contract to Close Days: Close Date - Purchase Contract Date
    if 'CloseDate' in df.columns and 'PurchaseContractDate' in df.columns:
        df['contract_to_close_days'] = (df['CloseDate'] - df['PurchaseContractDate']).dt.days

    return df

sold_engineered = engineer_market_metrics(sold_df, "Sold")
listings_engineered = engineer_market_metrics(listings_df, "Listings")

# Display sample output of engineered columns
sample_cols = [col for col in ['ClosePrice', 'OriginalListPrice', 'price_ratio', 'price_per_sq_ft', 
                               'DaysOnMarket', 'YrMo', 'listing_to_contract_days', 'contract_to_close_days'] 
               if col in sold_engineered.columns]

print("\nSample Output Table (Engineered Metrics - First 5 Rows):")
print(sold_engineered[sample_cols].head().to_string())

# --------------------------------------------------
# PART 2: SEGMENT ANALYSIS
# --------------------------------------------------
print("\n" + "=" * 50)
print("PART 2: SEGMENT ANALYSIS")
print("=" * 50)

# Segment 1: Summary Statistics by Property Type
if 'PropertyType' in sold_engineered.columns or 'PropertySubType' in sold_engineered.columns:
    group_col = 'PropertySubType' if 'PropertySubType' in sold_engineered.columns else 'PropertyType'
    print(f"\nSummary Statistics Grouped by {group_col}:")
    
    type_segment = sold_engineered.groupby(group_col).agg(
        Total_Sales=('ClosePrice', 'count'),
        Median_Close_Price=('ClosePrice', 'median'),
        Avg_Price_Per_SqFt=('price_per_sq_ft', 'mean'),
        Avg_Days_On_Market=('DaysOnMarket', 'mean'),
        Avg_Close_To_List_Ratio=('price_ratio', 'mean')
    ).reset_index()
    
    print(type_segment.round(2).to_string(index=False))

# Segment 2: Summary Statistics by County
if 'CountyOrParish' in sold_engineered.columns:
    print("\nSummary Statistics Grouped by County (Top 5 by Volume):")
    
    county_segment = sold_engineered.groupby('CountyOrParish').agg(
        Total_Sales=('ClosePrice', 'count'),
        Median_Close_Price=('ClosePrice', 'median'),
        Avg_Price_Per_SqFt=('price_per_sq_ft', 'mean'),
        Avg_Days_On_Market=('DaysOnMarket', 'mean')
    ).sort_values(by='Total_Sales', ascending=False).head(5).reset_index()
    
    print(county_segment.round(2).to_string(index=False))

# Segment 3: Competitive Intelligence (Top Offices by Sales Volume)
if 'ListOfficeName' in sold_engineered.columns:
    print("\nTop 5 List Offices by Total Sales Volume:")
    
    office_segment = sold_engineered.groupby('ListOfficeName').agg(
        Total_Volume=('ClosePrice', 'sum'),
        Units_Sold=('ClosePrice', 'count'),
        Median_Close_Price=('ClosePrice', 'median')
    ).sort_values(by='Total_Volume', ascending=False).head(5).reset_index()
    
    # Format volume as currency string for presentation
    office_segment_display = office_segment.copy()
    office_segment_display['Total_Volume'] = office_segment_display['Total_Volume'].apply(lambda x: f"${x:,.2f}")
    office_segment_display['Median_Close_Price'] = office_segment_display['Median_Close_Price'].apply(lambda x: f"${x:,.2f}")
    
    print(office_segment_display.to_string(index=False))

# --------------------------------------------------
# PART 3: EXPORT FEATURE-ENGINEERED DATASETS
# --------------------------------------------------
print("\n" + "=" * 50)
print("PART 3: EXPORT ENGINEERED DATASETS")
print("=" * 50)

sold_engineered.to_csv("week6_sold_engineered.csv", index=False)
listings_engineered.to_csv("week6_listings_engineered.csv", index=False)

print("Successfully exported feature-engineered datasets:")
print("  -> 'week6_sold_engineered.csv'")
print("  -> 'week6_listings_engineered.csv'")
print("\nWeek 6 Feature Engineering & Market Metrics Execution Complete!")