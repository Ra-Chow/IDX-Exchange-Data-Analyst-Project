import glob
import os
from pathlib import Path
import pandas as pd

# Define the file path patterns for the listings and sold CSV files.
filePathPatternListings = (
    "/Users/raineechow/Documents/IDX Exchange/csv/CRMLSListing*.csv"
)
filePathPatternSold = (
    "/Users/raineechow/Documents/IDX Exchange/csv/CRMLSSold*.csv"
)

# Week 1 subfolder path shortcut to save combined and filtered datasets
save_path = Path("Week 1/")
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


# --------------------------------------------------
# PART 1: COMBINE AND FILTER RESIDENTIAL LISTINGS
# --------------------------------------------------

# Read and combine all the listing CSV files into a single DataFrame.
print("Processing listing files...")
listingFiles = sorted(glob.glob(filePathPatternListings))

processedListings = []
# Removing the extra two columns from CSV files that have "_filled" in their names.
for file in listingFiles:
  dataFile = pd.read_csv(file)
  if "_filled" in file or "filled" in file.lower():
    dataFile = dataFile.iloc[:, :-2]  # Remove the last two columns
  processedListings.append(dataFile)

# Row counts before and after concatenation:
# Number of individual listing files concatenated: len(listingFiles)
combinedListings = pd.concat(processedListings, ignore_index=True)
initialListingsCount = len(combinedListings)
print("Initial Listings Count:", initialListingsCount)

# Row counts before and after PropertyType == 'Residential' filter:
# Initial Listings Count: initialListingsCount
# Filtering out only "Residential" property types from the combined listings.
filteredListings = combinedListings[
    combinedListings["PropertyType"] == "Residential"
]
finalListingsCount = len(filteredListings)
print("Final Listings Count:", finalListingsCount)
# Save the filtered listings to a new CSV file.
save_csv(
    filteredListings,
    save_path / "week1_residential_listings.csv",
    index=False,
)

# --------------------------------------------------
# PART 2: COMBINE AND FILTER SOLD PROPERTIES
# --------------------------------------------------

# Read and combine all the sold CSV files into a single DataFrame.
print("Processing sold files...")
soldFiles = sorted(glob.glob(filePathPatternSold))

processedSold = []
# Removing the extra two columns from CSV files that have "_filled" in their names.
for file in soldFiles:
  dataFile = pd.read_csv(file)
  if "_filled" in file or "filled" in file.lower():
    dataFile = dataFile.iloc[:, :-2]  # Remove the last two columns
  processedSold.append(dataFile)

# Row counts before and after concatenation:
# Number of individual sold files concatenated: len(soldFiles)
combinedSoldDF = pd.concat(processedSold, ignore_index=True)
initialSoldCount = len(combinedSoldDF)

# Row counts before and after PropertyType == 'Residential' filter:
# Initial Sold Count: initialSoldCount
# Filtering out only "Residential" property types from the combined sold properties.
filteredSold = combinedSoldDF[combinedSoldDF["PropertyType"] == "Residential"]
finalSoldCount = len(filteredSold)

# Save the filtered sold properties to a new CSV file.
save_csv(filteredSold, save_path / "week1_residential_sold.csv", index=False)

# --------------------------------------------------
# PART 3: PRINT SUMMARY OF RESULTS
# --------------------------------------------------

print("Summary of Results:")
print(f"Listing files concatenated: {len(listingFiles)}")
print(f"Initial number of listings: {initialListingsCount}")
print(f"Number of residential listings after filtering: {finalListingsCount}")
print(f"Sold files concatenated: {len(soldFiles)}")
print(f"Initial number of sold properties: {initialSoldCount}")
print(f"Number of residential sold properties after filtering: {finalSoldCount}")

if user_choice == "Y":
  print("\nSuccessfully generated output deliverables:")
  print(f"  -> '{save_path / 'week1_residential_listings.csv'}'")
  print(f"  -> '{save_path / 'week1_residential_sold.csv'}'")
else:
  print("\nRun completed in dry-run mode. No CSV files were written to disk.")
