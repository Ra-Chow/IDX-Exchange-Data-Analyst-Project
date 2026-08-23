import os
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

# --------------------------------------------------
# CODE PREPARATION
# Loading cleaned datasets from Weeks 4-5
# --------------------------------------------------
print("=" * 50)
print("CODE PREPARATION: LOADING WEEKS 4-5 CLEANED DATASETS")
print("=" * 50)

# Week 6 subfolder path shortcut to save engineered datasets
save_path = Path("Week 6/")
save_path.mkdir(parents=True, exist_ok=True)

# Giving user the choice to save the files this run or not to avoid cluttering the workspace with multiple versions of the same files.
user_choice = (
    input("Do you want to save generated files this run? (Y/N): ")
    .strip()
    .upper()
)
if user_choice != "Y":
  print(f"Running code WITHOUT saving files to workspace.")
else:
  print(f"Running code WITH saving files to workspace.")


def save_csv(df, file_path_str: str, index: bool = True):
  if user_choice == "Y":
    # Check if the file already exists and remove it before saving the new version to avoid overwriting issues.
    target_path = Path(file_path_str)
    # Deletes original file if it exists before saving the new one.
    if target_path.exists():
      try:
        os.remove(target_path)
      except OSError as e:
        print(f"Error removing file {file_path_str}: {e}")

    df.to_csv(target_path, index=index)
    print(f"Saved file: {target_path.name}")


sold_df = pd.read_csv("Weeks 4-5/weeks4-5_sold_cleaned.csv", low_memory=False)
listings_df = pd.read_csv(
    "Weeks 4-5/weeks4-5_listings_cleaned.csv", low_memory=False
)

print(
    f"Sold dataset loaded: {sold_df.shape[0]} rows x {sold_df.shape[1]} columns"
)
print(
    f"Listings dataset loaded: {listings_df.shape[0]} rows x"
    f" {listings_df.shape[1]} columns"
)

# Ensure date fields are properly formatted as datetime
date_fields = ["CloseDate", "PurchaseContractDate", "ListingContractDate"]
for df in [sold_df, listings_df]:
  for col in date_fields:
    if col in df.columns:
      df[col] = pd.to_datetime(df[col], errors="coerce")

# --------------------------------------------------
# PART 1: FEATURE ENGINEERING & MARKET METRICS
# --------------------------------------------------
print("=" * 50)
print("PART 1: FEATURE ENGINEERING & MARKET METRICS")
print("=" * 50)


def engineer_market_metrics(df, name):
  print(f"\nEngineering key market metrics for {name} dataset...")

  # Price Ratio & Close to Original List Ratio: Close Price / Original List Price
  if "ClosePrice" in df.columns and "OriginalListPrice" in df.columns:
    df["price_ratio"] = df["ClosePrice"] / df["OriginalListPrice"]
    df["close_to_original_list_ratio"] = (
        df["ClosePrice"] / df["OriginalListPrice"]
    )
  elif "ListPrice" in df.columns and "OriginalListPrice" in df.columns:
    df["price_ratio"] = df["ListPrice"] / df["OriginalListPrice"]
    df["close_to_original_list_ratio"] = (
        df["ListPrice"] / df["OriginalListPrice"]
    )

  # Price Per Sq Ft: Close Price / Living Area
  if "ClosePrice" in df.columns and "LivingArea" in df.columns:
    df["price_per_sq_ft"] = df["ClosePrice"] / df["LivingArea"]
  elif "ListPrice" in df.columns and "LivingArea" in df.columns:
    df["price_per_sq_ft"] = df["ListPrice"] / df["LivingArea"]

  # Days on Market (verify presence of raw field)
  if "DaysOnMarket" not in df.columns:
    df["DaysOnMarket"] = np.nan

  # Time-Series Dimensions: Year, Month, YrMo derived from CloseDate or ListingContractDate
  target_date_col = (
      "CloseDate" if "CloseDate" in df.columns else "ListingContractDate"
  )
  if target_date_col in df.columns:
    df["Year"] = df[target_date_col].dt.year
    df["Month"] = df[target_date_col].dt.month
    df["YrMo"] = df[target_date_col].dt.to_period("M").astype(str)

  # Listing to Contract Days: Purchase Contract Date - Listing Contract Date
  if (
      "PurchaseContractDate" in df.columns
      and "ListingContractDate" in df.columns
  ):
    df["listing_to_contract_days"] = (
        df["PurchaseContractDate"] - df["ListingContractDate"]
    ).dt.days

  # Contract to Close Days: Close Date - Purchase Contract Date
  if "CloseDate" in df.columns and "PurchaseContractDate" in df.columns:
    df["contract_to_close_days"] = (
        df["CloseDate"] - df["PurchaseContractDate"]
    ).dt.days

  return df


sold_engineered = engineer_market_metrics(sold_df, "Sold")
listings_engineered = engineer_market_metrics(listings_df, "Listings")

# --------------------------------------------------
# PART 2: SCHOOL DISTRICT GEOSPATIAL JOIN
# --------------------------------------------------
print("\n" + "=" * 50)
print("PART 2: SCHOOL DISTRICT GEOSPATIAL JOIN")
print("=" * 50)


def join_school_districts(df, name):
  print(f"\nMerging California School District data for {name} dataset...")
  # California School District GeoJSON/Shapefile endpoint
  school_district_url = "https://data.ca.gov/dataset/california-school-district-areas-2024-25/resource/7dfaf005-58eb-45db-93b1-7aff091b2172"

  if "Latitude" in df.columns and "Longitude" in df.columns:
    # Filter valid coordinates for point conversion
    valid_coords = df["Latitude"].notnull() & df["Longitude"].notnull()
    try:
      # Attempt reading school district spatial boundary file
      school_districts_gdf = gpd.read_file(school_district_url)
      geometry = [
          Point(xy)
          for xy in zip(
              df.loc[valid_coords, "Longitude"],
              df.loc[valid_coords, "Latitude"],
          )
      ]
      points_gdf = gpd.GeoDataFrame(
          df[valid_coords], geometry=geometry, crs="EPSG:4326"
      )

      # Ensure coordinate reference systems match
      if school_districts_gdf.crs != points_gdf.crs:
        school_districts_gdf = school_districts_gdf.to_crs(points_gdf.crs)

      # Perform spatial join to tag school districts
      joined_gdf = gpd.sjoin(
          points_gdf, school_districts_gdf, how="left", predicate="within"
      )
      df["SchoolDistrict"] = np.nan
      district_col_name = (
          "DistrictName"
          if "DistrictName" in joined_gdf.columns
          else "NAME"
          if "NAME" in joined_gdf.columns
          else None
      )
      if district_col_name:
        df.loc[valid_coords, "SchoolDistrict"] = joined_gdf[district_col_name]
        print(
            f"  -> Successfully enriched {df['SchoolDistrict'].notnull().sum():,} records with school districts."
        )
      else:
        print("  -> School district column not found in boundary layer.")
    except Exception as e:
      print(
          f"  -> School district spatial join skipped or could not fetch boundary file: {e}"
      )
      df["SchoolDistrict"] = np.nan
  else:
    df["SchoolDistrict"] = np.nan

  return df


sold_engineered = join_school_districts(sold_engineered, "Sold")
listings_engineered = join_school_districts(listings_engineered, "Listings")

# Display sample output of engineered columns
sample_cols = [
    col
    for col in [
        "ClosePrice",
        "OriginalListPrice",
        "price_ratio",
        "close_to_original_list_ratio",
        "price_per_sq_ft",
        "DaysOnMarket",
        "YrMo",
        "listing_to_contract_days",
        "contract_to_close_days",
        "SchoolDistrict",
    ]
    if col in sold_engineered.columns
]

print("\nSample Output Table (Engineered Metrics - First 5 Rows):")
print(sold_engineered[sample_cols].head().to_string())

# --------------------------------------------------
# PART 3: SEGMENT ANALYSIS
# --------------------------------------------------
print("\n" + "=" * 50)
print("PART 3: SEGMENT ANALYSIS")
print("=" * 50)

# Segment 1: Summary Statistics by Property Type & SubType
for group_col in ["PropertyType", "PropertySubType"]:
  if group_col in sold_engineered.columns:
    print(f"\nSummary Statistics Grouped by {group_col}:")
    type_segment = (
        sold_engineered.groupby(group_col)
        .agg(
            Total_Sales=("ClosePrice", "count"),
            Median_Close_Price=("ClosePrice", "median"),
            Avg_Price_Per_SqFt=("price_per_sq_ft", "mean"),
            Avg_Days_On_Market=("DaysOnMarket", "mean"),
            Avg_Close_To_List_Ratio=("price_ratio", "mean"),
        )
        .reset_index()
    )
    print(type_segment.round(2).to_string(index=False))

# Segment 2: Summary Statistics by County & MLSAreaMajor
for geo_col in ["CountyOrParish", "MLSAreaMajor"]:
  if geo_col in sold_engineered.columns:
    print(f"\nSummary Statistics Grouped by {geo_col} (Top 5 by Volume):")
    geo_segment = (
        sold_engineered.groupby(geo_col)
        .agg(
            Total_Sales=("ClosePrice", "count"),
            Median_Close_Price=("ClosePrice", "median"),
            Avg_Price_Per_SqFt=("price_per_sq_ft", "mean"),
            Avg_Days_On_Market=("DaysOnMarket", "mean"),
        )
        .sort_values(by="Total_Sales", ascending=False)
        .head(5)
        .reset_index()
    )
    print(geo_segment.round(2).to_string(index=False))

# Segment 3: Competitive Intelligence (Top Listing and Buyer Offices)
for office_col in ["ListOfficeName", "BuyerOfficeName"]:
  if office_col in sold_engineered.columns:
    print(f"\nTop 5 {office_col} by Total Sales Volume:")
    office_segment = (
        sold_engineered.groupby(office_col)
        .agg(
            Total_Volume=("ClosePrice", "sum"),
            Units_Sold=("ClosePrice", "count"),
            Median_Close_Price=("ClosePrice", "median"),
        )
        .sort_values(by="Total_Volume", ascending=False)
        .head(5)
        .reset_index()
    )
    office_segment_display = office_segment.copy()
    office_segment_display["Total_Volume"] = office_segment_display[
        "Total_Volume"
    ].apply(lambda x: f"${x:,.2f}")
    office_segment_display["Median_Close_Price"] = office_segment_display[
        "Median_Close_Price"
    ].apply(lambda x: f"${x:,.2f}")
    print(office_segment_display.to_string(index=False))

# --------------------------------------------------
# PART 4: EXPORT FEATURE-ENGINEERED DATASETS
# --------------------------------------------------
print("\n" + "=" * 50)
print("PART 4: EXPORT ENGINEERED DATASETS")
print("=" * 50)

save_csv(sold_engineered, save_path / "week6_sold_engineered.csv", index=False)
save_csv(
    listings_engineered,
    save_path / "week6_listings_engineered.csv",
    index=False,
)

if user_choice == "Y":
  print("Successfully exported feature-engineered datasets:")
  print("  -> 'Week 6/week6_sold_engineered.csv'")
  print("  -> 'Week 6/week6_listings_engineered.csv'")
else:
  print("Run completed in dry-run mode. No CSV files were written to disk.")

print("\nWeek 6 Feature Engineering & Market Metrics Execution Complete!")
