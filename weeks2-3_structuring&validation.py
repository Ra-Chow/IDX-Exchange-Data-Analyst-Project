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
print("NUMERIC DISTRIBUTION REVIEW")

# Analyze the distribution of key numeric fields: ClosePrice, ListPrice, OriginalListPrice, LivingArea, LotSizeAcres, BedroomsTotal, BathroomsTotalInteger, DaysOnMarket, and YearBuilt.
requiredNumericFields = ['ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea', 'LotSizeAcres', 'BedroomsTotal', 'BathroomsTotalInteger', 'DaysOnMarket', 'YearBuilt']

# Check that the required numeric fields are present in the sold dataset before proceeding with analysis.
availableFields = [field for field in requiredNumericFields if field in sold_df.columns]

# For each field, generate histograms, boxplots, and percentile summaries, and identify extreme outliers for later handling.
distributionSummary = sold_df[availableFields].describe(percentiles=[0.25, 0.5, 0.75])
print("Sold Dataset Numeric Distribution Summary")
print(distributionSummary.round(2).to_string())

# Applying filter separation
print("Applying filter separation for outlier analysis...")

# Determine columns to drop that safely exist inside each dataframe's boundaries
listingsDropTargets = [col for col in columnsToDrop if col in listings_df.columns]
soldDropTargets = [col for col in columnsToDrop if col in sold_df.columns]

listingsFiltered = listings_df.drop(columns=listingsDropTargets)
soldFiltered = sold_df.drop(columns=soldDropTargets)

# Save the primary structural filtered datasets as new baseline CSV files.
listingsFiltered.to_csv("week2-3_listings_filtered.csv", index=False)
soldFiltered.to_csv("week2-3_sold_filtered.csv", index=False)
print("Filtered datasets saved as week2-3_listings_filtered.csv and week2-3_sold_filtered.csv")

# Property type analysis
print("Property Type Analysis (Deliverable Documentation)")

# Unique property types orginally present in raw data
uniquePropertyTypes = ['Residential', 'Commerical', 'Land', 'Multi-Family', 'Industial']
print(f"Unique Property Types in Raw Data: {uniquePropertyTypes}")
print("Filtering logic: sold = sold[sold.Propertytype == 'Residential']")

# --------------------------------------------------
# PART 3: SUGGESTED INTERN QUESTIONS (EDA ANSWERS)
# --------------------------------------------------
print("SUGGESTED INTERN QUESTIONS (EDA ANSWERS)")

# 1. What are the medium and average close prices?
avgClosePrice = sold_df['ClosePrice'].mean()
medianClosePrice = sold_df['ClosePrice'].median()
print(f"Average Close Price: ${avgClosePrice:,.2f}")
print(f"Median Close Price: ${medianClosePrice:,.2f}")

# 2. What percentage of homes sold above vs. below list price?
validPrices = sold_df[(sold_df['ListPrice'] > 0) & (sold_df['ClosePrice'] > 0)]
soldAbove = (validPrices['ClosePrice'] > validPrices['ListPrice']).sum() / len(validPrices) * 100
soldBelow = (validPrices['ClosePrice'] < validPrices['ListPrice']).sum() / len(validPrices) * 100
soldAtList = (validPrices['ClosePrice'] == validPrices['ListPrice']).sum() / len(validPrices) * 100
print(f"Homes Sold Above List Price: {soldAbove:.2f}%")
print(f"Homes Sold Below List Price: {soldBelow:.2f}%")
print(f"Homes Sold Exactly At List Price: {soldAtList:.2f}%")

# 3. Are there any apparent date consistency issues?
closeDates = pd.to_datetime(sold_df['CloseDate'], errors='coerce')
listDates = pd.to_datetime(sold_df['ListingContractDate'], errors='coerce')
dateConsistencyIssues = (closeDates < listDates).sum()
print(f"Number of records with CloseDate earlier than ListDate: {dateConsistencyIssues}")

# 4. Which countries have the highest median prices?
if 'CountyOrParish' in sold_df.columns:
    county_prices = sold_df.groupby('CountyOrParish')['ClosePrice'].median().sort_values(ascending=False)
    print("\nTop 5 Counties by Median Close Price:")
    print(county_prices.head(5).apply(lambda x: f"${x:,.2f}"))

# --------------------------------------------------
# PART 4: MORTGAGE RATE ENRICHMENT
# --------------------------------------------------
# Bypass macOS local SSL certificate requirement
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print("MORTGAGE RATE ENRICHMENT")
# Step 1: Fetch the mortgage rate data from FRED
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
# Read CSV from FRED with SSL context
mortgage = pd.read_csv(url, parse_dates=['observation_date'])
mortgage.columns = ['date', 'rate_30yr_fixed']

# Step 2: Resample weekly rates to monthly averages.
mortgage['year_month'] = mortgage['date'].dt.to_period('M')
mortgage_monthly = (
    mortgage.groupby('year_month')['rate_30yr_fixed']
    .mean()
    .reset_index()
)
print("Successfully resampled weekly economic interest data points to histroical monthly averages")

# Step 3: Create a matching year_month key on the MLS datasets.
soldFiltered['year_month'] = pd.to_datetime(soldFiltered['CloseDate'], errors='coerce').dt.to_period('M')
listingsFiltered['year_month'] = pd.to_datetime(listingsFiltered['ListingContractDate'], errors='coerce').dt.to_period('M')

# Step 4: Merge
soldWithRates = soldFiltered.merge(mortgage_monthly, on='year_month', how='left')
listingsWithRates = listingsFiltered.merge(mortgage_monthly, on='year_month', how='left')

# Step 5: Validate the merge
print("TIME-SERIES JOIN VALIDATION CHECK")
soldNullRates = soldWithRates['rate_30yr_fixed'].isnull().sum()
listingsNullRates = listingsWithRates['rate_30yr_fixed'].isnull().sum()
print(f"Sold Dataset - Missing Mortgage Rates: {soldNullRates}")
print(f"Listings Dataset - Missing Mortgage Rates: {listingsNullRates}")

# Saving final enriched datasets as new workspace CSVs
soldWithRates.to_csv("week2-3_sold_with_rates.csv", index=False)
listingsWithRates.to_csv("week2-3_listings_with_rates.csv", index=False)
print("Enriched datasets saved as week2-3_sold_with_rates.csv and week2-3_listings_with_rates.csv")

# --------------------------------------------------
# PART 5: VISUAL PLOTS
# --------------------------------------------------
print("VISUAL PLOTS")
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Core fields requested by deliverable guidelines
    for field in ['ClosePrice', 'LivingArea', 'DaysOnMarket']:
        # Generating histogram
        plt.figure(figsize=(8, 4))
        sns.histplot(soldWithRates[field].dropna(), bins=30, kde=True)
        plt.title(f"Histogram of {field} - Distribution Histogram")
        plt.savefig(f"histogram_{field}.png")
        plt.close()

        # Generating boxplot
        plt.figure(figsize=(8, 2))
        sns.boxplot(x=soldWithRates[field].dropna())
        plt.title(f"Boxplot of {field} - Distribution Boxplot")
        plt.savefig(f"boxplot_{field}.png")
        plt.close()
    print("Visual plots generated and saved as PNG files.")
except ImportError:
    print("Matplotlib or Seaborn not installed. Skipping visual plot generation.")
