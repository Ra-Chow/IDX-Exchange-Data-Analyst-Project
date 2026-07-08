import pandas as pd
import numpy as np

print("Loading filtered datasets from Week 1...")
# Load the cleaned datasets generated in Week 1
# Using low_memory=False prevents historical mixed-data warning prompts
listings_df = pd.read_csv("week1_residential_listings.csv", low_memory=False)
sold_df = pd.read_csv("week1_residential_sold.csv", low_memory=False)

# --------------------------------------------------
# PART 1: DATASET UNDERSTANDING & MISSING VALUE ANALYSIS
# --------------------------------------------------

# 1. Identify the number of rows and columns in each dataset.
print("DATASET DIMENSIONS")
print(f"Listings Dataset: {listings_df.shape[0]} rows, {listings_df.shape[1]} columns")
print(f"Sold Dataset: {sold_df.shape[0]} rows, {sold_df.shape[1]} columns")

# 2. Review column data types.
print("LISTINGS DATA TYPE SUMMARY COUNTS")
print(listings_df.dtypes.value_counts())
print("SOLD DATA TYPE SUMMARY COUNTS")
print(sold_df.dtypes.value_counts())

# 3. Missing Value Analysis
# 3a. Identify high-missing columns (specific columns where more than 90% of the rows are empty).
print(f"HIGH-MISSING COLUMNS (>90% rows empty)")

# 3b. The function identifyHighMissingColumns takes a DataFrame and its name as input, calculates the number and percentage of missing values for each column, and prints a report of columns with more than 90% missing values. It also returns the report DataFrame for further analysis if needed.
def identifyHighMissingColumns(df, name):
    # 3ba. Calculate and print the number and percentage of missing values for each column in the DataFrame.
    nullCounts = df.isnull().sum()
    nullPercentages = (df.isnull().sum() / len(df)) * 100
    reportDf = pd.DataFrame({
        'Null Count': nullCounts,
        'Null Percentage': nullPercentages
    })
    
    # 3bb. Flag columns with >90% missing values.
    highMissingColumns = reportDf[reportDf['Null Percentage'] > 90.0]

    # 3bc. Print the high-missing columns report by calculating the missing counts and percentages per column.
    print(f"{name} Dataset - High-Missing Columns: ({len(highMissingColumns)})")
    # 3bd. If there are high-missing columns, round the percentages to 2 decimal places and print them; otherwise, indicate that no high-missing columns were found.
    if len(highMissingColumns) > 0:
        print(highMissingColumns.round(2).to_string())
    # 3be. If there are not any high-missing columns, print a message indicating that.
    else:
        print("No high-missing columns found.")
    
    # 3bf. Return the report DataFrame for further analysis if needed.
    return reportDf

# 3c. Calculate and save complete null summary tables to workspace for auditing if needed.
listingsNullSummary = identifyHighMissingColumns(listings_df, "Listings")
soldNullSummary = identifyHighMissingColumns(sold_df, "Sold")
listingsNullSummary.to_csv("listings_null_summary.csv")
soldNullSummary.to_csv("sold_null_summary.csv")

# 3d. Decide which columns to drop vs. retain (keep core fields even if partially missing)
print("DROPPED VS. RETAINED COLUMNS")

# 3da. Define a list of high-missing columnns that are too valuable to drop.
criticalMarketDrivers = ['WaterfrontYN', 'BasementYN', 'PoolFeatures', 'ViewYN']
highMissingColumnNames = soldNullSummary[soldNullSummary['Null Percentage'] > 90.0].index.tolist()

columnsToDrop = []
columnsToRetain = []

# 3db. Iterate through the high-missing columns to decide which to drop vs. retain based on whether they are critical market drivers or not.
for col in highMissingColumnNames:
    # 3db-1. If the column is a known real estate driver, defined by the list criticalMarketDrivers, the 90% rules does not apply and the column is retained.
    if col in criticalMarketDrivers:
        columnsToRetain.append(col)
    # 3db-2. If the column is not a known real estate driver, it is dropped from the dataset.
    else:
        columnsToDrop.append(col)

# 3dc. Print the lists of columns to drop and retain for auditing purposes.
print(f"Columns to Drop: {columnsToDrop}")
print(f"Columns to Retain: {columnsToRetain}")

# 4. Separate market analysis fields from metadata fields.
targetNumericFields = ['ClosePrice', 'LivingArea', 'DaysOnMarket']
distributionSummary = sold_df[targetNumericFields].describe()
# 4a. Log results.
print("MARKET FIELDS NUMERIC DISTRIBUTION SUMMARY")
print(distributionSummary.round(2).to_string())

# --------------------------------------------------
# PART 2: NUMERIC DISTRIBUTION REVIEW
# --------------------------------------------------

# Analyze the distribution of key numeric fields: ClosePrice, ListPrice, OriginalListPrice, LivingArea, LotSizeAcres, BedroomsTotal, BathroomsTotalInteger, DaysOnMarket, and YearBuilt.
# For each field, generate histograms, boxplots, and percentile summaries, and identify extreme outliers for later handling.

# --------------------------------------------------
# PART 3: MORTGAGE RATE ENRICHMENT
# --------------------------------------------------

# Enrich both the combined sold and listings datasets by merging in the national 30-year fixed mortgage rate from the St. Louis Federal Reserve (FRED). This introduces interns to fetching live economic data from a public API and performing a time-series join on a monthly key.