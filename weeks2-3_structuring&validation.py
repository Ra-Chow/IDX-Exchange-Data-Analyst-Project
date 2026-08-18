import os
from pathlib import Path
import ssl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# PART 0: LOAD CLEANED DATASETS FROM WEEK 1 AND SETTING UP WORKSPACE
# ---------------------------------------------------------------------
# Load the cleaned datasets generated in Week 1
print("Loading filtered datasets from Week 1...")
listings_df = pd.read_csv(
    "/Users/raineechow/Documents/IDX Exchange/Week 1/week1_residential_listings.csv",
    low_memory=False,
)
sold_df = pd.read_csv(
    "/Users/raineechow/Documents/IDX Exchange/Week 1/week1_residential_sold.csv",
    low_memory=False,
)

# Weeks 2-3 subfolder path shortcut to save filtered datasets and null summaries
save_path = Path("/Users/raineechow/Documents/IDX Exchange/Weeks 2-3/")
save_path.mkdir(parents=True, exist_ok=True)

# Giving user the choice to save the files this run or not to avoid cluttering the workspace with multiple versions of the same files.
user_choice = (
    input("Do you want to save generated files and charts this run? (Y/N): ")
    .strip()
    .upper()
)
if user_choice != "Y":
  print("Running code WITHOUT saving files to workspace.")
else:
  print("Running code WITH saving files to workspace.")


def save_csv(df, file_path_str: str, index: bool = True):
  if user_choice == "Y":
    # Check if the file already exists and remove it before saving the new version to avoid overw
    target_path = Path(file_path_str)
    # Deletes original file if it exists before saving the new one.
    if target_path.exists():
      try:
        os.remove(target_path)
      except OSError as e:
        print(f"Error removing file {file_path_str}: {e}")

    df.to_csv(target_path, index=index)
    print(f"Saved file: {target_path.name}")


def save_plot(fig, file_path_str: str):
  if user_choice == "Y":
    # Check if the file already exists and remove it before saving the new version to avoid overwriting.
    target_path = Path(file_path_str)
    # Deletes original file if it exists before saving the new one.
    if target_path.exists():
      try:
        os.remove(target_path)
      except OSError as e:
        print(f"Error removing file {file_path_str}: {e}")

    fig.savefig(target_path)
    print(f"Saved plot: {target_path.name}")


# -------------------------------------------------------
# PART 1: DATASET UNDERSTANDING & MISSING VALUE ANALYSIS
# -------------------------------------------------------

# 1. Identify the number of rows and columns in each dataset.
print("DATASET DIMENSIONS")
print(
    f"Listings Dataset: {listings_df.shape[0]} rows, {listings_df.shape[1]}"
    " columns"
)
print(f"Sold Dataset: {sold_df.shape[0]} rows, {sold_df.shape[1]} columns")

# 2. Review column data types.
print("LISTINGS DATA TYPE SUMMARY COUNTS")
print(listings_df.dtypes.value_counts())
print("SOLD DATA TYPE SUMMARY COUNTS")
print(sold_df.dtypes.value_counts())

# 3. Missing Value Analysis
# Identify high-missing columns (specific columns where more than 90% of the rows are empty).
print("HIGH-MISSING COLUMNS (>90% rows empty)")


# The function identifyHighMissingColumns takes a DataFrame and its name as input, calculates the number and percentage of missing values for each column, and prints a report of columns with more than 90% missing values. It also returns the report DataFrame for further analysis if needed.
def identifyHighMissingColumns(df, name):
  # 3ba. Calculate and print the number and percentage of missing values for each column in the DataFrame.
  nullCounts = df.isnull().sum()
  nullPercentages = (df.isnull().sum() / len(df)) * 100
  reportDf = pd.DataFrame(
      {"Null Count": nullCounts, "Null Percentage": nullPercentages}
  )

  # Flag columns with >90% missing values.
  highMissingColumns = reportDf[reportDf["Null Percentage"] > 90.0]

  # Print the high-missing columns report by calculating the missing counts and percentages per column.
  print(f"{name} Dataset - High-Missing Columns: ({len(highMissingColumns)})")
  # If there are high-missing columns, round the percentages to 2 decimal places and print them; otherwise, indicate that no high-missing columns were found.
  if len(highMissingColumns) > 0:
    print(highMissingColumns.round(2).to_string())
  # If there are not any high-missing columns, print a message indicating that.
  else:
    print("No high-missing columns found.")

  # Return the report DataFrame for further analysis if needed.
  return reportDf


# Calculate and save complete null summary tables to workspace for auditing if needed.
listingsNullSummary = identifyHighMissingColumns(listings_df, "Listings")
soldNullSummary = identifyHighMissingColumns(sold_df, "Sold")
save_csv(listingsNullSummary, save_path / "weeks2-3_listings_null_summary.csv")
save_csv(soldNullSummary, save_path / "weeks2-3_sold_null_summary.csv")

# Decide which columns to drop vs. retain (keep core fields even if partially missing)
print("DROPPED VS. RETAINED COLUMNS")

# Define a list of high-missing columns that are too valuable to drop.
criticalMarketDrivers = ["WaterfrontYN", "BasementYN", "PoolFeatures", "ViewYN"]

# Evaluate drop targets independently for listings vs sold datasets to prevent premature data loss
listingsHighMissing = listingsNullSummary[
    listingsNullSummary["Null Percentage"] > 90.0
].index.tolist()
listingsDropTargets = [
    col for col in listingsHighMissing if col not in criticalMarketDrivers
]

highMissingColumnNames = soldNullSummary[
    soldNullSummary["Null Percentage"] > 90.0
].index.tolist()
soldDropTargets = [
    col for col in highMissingColumnNames if col not in criticalMarketDrivers
]

# Print the lists of columns to drop for auditing purposes.
print(f"Listings Columns to Drop: {listingsDropTargets}")
print(f"Sold Columns to Drop: {soldDropTargets}")

# Apply drops safely to their respective datasets
listingsFiltered = listings_df.drop(
    columns=[col for col in listingsDropTargets if col in listings_df.columns]
)
soldFiltered = sold_df.drop(
    columns=[col for col in soldDropTargets if col in sold_df.columns]
)

# Save the primary structural filtered datasets as new baseline CSV files.
save_csv(
    listingsFiltered, save_path / "weeks2-3_listings_filtered.csv", index=False
)
save_csv(soldFiltered, save_path / "weeks2-3_sold_filtered.csv", index=False)
print(
    "Filtered datasets saved as weeks2-3_listings_filtered.csv and"
    " weeks2-3_sold_filtered.csv"
)

# Property type analysis documentation string
print("Property Type Analysis (Deliverable Documentation)")
uniquePropertyTypes = [
    "Residential",
    "Commerical",
    "Land",
    "Multi-Family",
    "Industial",
]
print(f"Unique Property Types in Raw Data: {uniquePropertyTypes}")
print("Filtering logic: sold = sold[sold['PropertyType'] == 'Residential']")

# --------------------------------------------------
# PART 2: NUMERIC DISTRIBUTION REVIEW
# --------------------------------------------------
print("NUMERIC DISTRIBUTION REVIEW")

# Analyze the distribution of key numeric fields: ClosePrice, ListPrice, OriginalListPrice, LivingArea, LotSizeAcres, BedroomsTotal, BathroomsTotalInteger, DaysOnMarket, and YearBuilt.
requiredNumericFields = [
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "LivingArea",
    "LotSizeAcres",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "DaysOnMarket",
    "YearBuilt",
]

# Check that the required numeric fields are present in the sold dataset before proceeding with analysis.
availableFields = [
    field for field in requiredNumericFields if field in soldFiltered.columns
]

# For each field, generate percentile summaries including extreme tails to satisfy handbook criteria.
distributionSummary = soldFiltered[availableFields].describe(
    percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
)
print("Sold Dataset Numeric Distribution Summary")
print(distributionSummary.round(2).to_string())

# --------------------------------------------------
# PART 3: SUGGESTED INTERN QUESTIONS (EDA ANSWERS)
# --------------------------------------------------
print("SUGGESTED INTERN QUESTIONS (EDA ANSWERS)")

# 1. What are the median and average close prices calculated from the clean baseline?
avgClosePrice = soldFiltered["ClosePrice"].mean()
medianClosePrice = soldFiltered["ClosePrice"].median()
print(f"Average Close Price: ${avgClosePrice:,.2f}")
print(f"Median Close Price: ${medianClosePrice:,.2f}")

# 2. What percentage of homes sold above vs. below list price?
validPrices = soldFiltered[
    (soldFiltered["ListPrice"] > 0) & (soldFiltered["ClosePrice"] > 0)
]
soldAbove = (
    (validPrices["ClosePrice"] > validPrices["ListPrice"]).sum()
    / len(validPrices)
    * 100
)
soldBelow = (
    (validPrices["ClosePrice"] < validPrices["ListPrice"]).sum()
    / len(validPrices)
    * 100
)
soldAtList = (
    (validPrices["ClosePrice"] == validPrices["ListPrice"]).sum()
    / len(validPrices)
    * 100
)
print(f"Homes Sold Above List Price: {soldAbove:.2f}%")
print(f"Homes Sold Below List Price: {soldBelow:.2f}%")
print(f"Homes Sold Exactly At List Price: {soldAtList:.2f}%")

# 3. Are there any apparent date consistency issues?
closeDates = pd.to_datetime(soldFiltered["CloseDate"], errors="coerce")
listDates = pd.to_datetime(soldFiltered["ListingContractDate"], errors="coerce")
dateConsistencyIssues = (closeDates < listDates).sum()
print(
    "Number of records with CloseDate earlier than ListDate:"
    f" {dateConsistencyIssues}"
)

# 4. Which countries have the highest median prices?
if "CountyOrParish" in soldFiltered.columns:
  county_prices = (
      soldFiltered.groupby("CountyOrParish")["ClosePrice"]
      .median()
      .sort_values(ascending=False)
  )
  print("\nTop 5 Counties by Median Close Price:")
  print(county_prices.head(5).apply(lambda x: f"${x:,.2f}"))

# --------------------------------------------------
# PART 4: MORTGAGE RATE ENRICHMENT
# --------------------------------------------------
# Bypass macOS local SSL certificate requirement
ssl._create_default_https_context = ssl._create_unverified_context

print("MORTGAGE RATE ENRICHMENT")
# Step 1: Fetch the mortgage rate data from FRED
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
# Read CSV from FRED with SSL context
mortgage = pd.read_csv(url, parse_dates=["observation_date"])
mortgage.columns = ["date", "rate_30yr_fixed"]

# Step 2: Resample weekly rates to monthly averages.
mortgage["year_month"] = mortgage["date"].dt.to_period("M")
mortgage_monthly = (
    mortgage.groupby("year_month")["rate_30yr_fixed"].mean().reset_index()
)
print(
    "Successfully resampled weekly economic interest data points to historical"
    " monthly averages"
)

# Step 3: Create a matching year_month key on the MLS datasets.
soldFiltered["year_month"] = pd.to_datetime(
    soldFiltered["CloseDate"], errors="coerce"
).dt.to_period("M")
listingsFiltered["year_month"] = pd.to_datetime(
    listingsFiltered["ListingContractDate"], errors="coerce"
).dt.to_period("M")

# Step 4: Merge
soldWithRates = soldFiltered.merge(
    mortgage_monthly, on="year_month", how="left"
)
listingsWithRates = listingsFiltered.merge(
    mortgage_monthly, on="year_month", how="left"
)

# Step 5: Validate the merge
print("TIME-SERIES JOIN VALIDATION CHECK")
soldNullRates = soldWithRates["rate_30yr_fixed"].isnull().sum()
listingsNullRates = listingsWithRates["rate_30yr_fixed"].isnull().sum()
print(f"Sold Dataset - Missing Mortgage Rates: {soldNullRates}")
print(f"Listings Dataset - Missing Mortgage Rates: {listingsNullRates}")

# Saving final enriched datasets as new workspace CSVs
save_csv(soldWithRates, save_path / "weeks2-3_sold_with_rates.csv", index=False)
save_csv(
    listingsWithRates,
    save_path / "weeks2-3_listings_with_rates.csv",
    index=False,
)
print(
    "Enriched datasets saved as weeks2-3_sold_with_rates.csv and"
    " weeks2-3_listings_with_rates.csv"
)

# --------------------------------------------------
# PART 5: VISUAL PLOTS
# --------------------------------------------------
print("VISUAL PLOTS")
# Weeks 2-3 chart subfolder path shortcut to save visual plots
chart_save_path = save_path / "charts"
chart_save_path.mkdir(parents=True, exist_ok=True)

try:
  import matplotlib.pyplot as plt
  import seaborn as sns

  # Dynamic loop running across all 9 required numeric fields
  for field in availableFields:
    # Generating histogram
    fig1 = plt.figure(figsize=(8, 4))
    sns.histplot(soldWithRates[field].dropna(), bins=30, kde=True)
    plt.title(f"Histogram of {field} - Distribution Histogram")
    save_plot(fig1, chart_save_path / f"weeks2-3_histogram_{field}.png")
    plt.close(fig1)

    # Generating boxplot
    fig2 = plt.figure(figsize=(8, 2))
    sns.boxplot(x=soldWithRates[field].dropna())
    plt.title(f"Boxplot of {field} - Distribution Boxplot")
    save_plot(fig2, chart_save_path / f"weeks2-3_boxplot_{field}.png")
    plt.close(fig2)

  print(
      f"Visual plots generated for all {len(availableFields)} fields and saved"
      " as PNG files."
  )
except ImportError:
  print("Matplotlib or Seaborn not installed. Skipping visual plot generation.")
