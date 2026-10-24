
import pandas as pd

# Load VOC-ALS dataset
voc = pd.read_excel(
    "VOC-ALS.xlsx",
    sheet_name="VOC-ALS_Data",
    header=1
)

# Load the cleaned Minsk2020 dataset
minsk = pd.read_csv(
    "Minsk2020_ALS_clean_for_training.csv"
)

print("=" * 55)
print("DATASET INFORMATION")
print("=" * 55)

print("\nVOC-ALS shape:", voc.shape)
print("Minsk2020 shape:", minsk.shape)

print("\nVOC-ALS columns:")
print(voc.columns.tolist())

print("\nMinsk2020 columns:")
print(minsk.columns.tolist())

# Find columns with exactly matching names
voc_columns = set(voc.columns)
minsk_columns = set(minsk.columns)

common = sorted(voc_columns.intersection(minsk_columns))

print("\n" + "=" * 55)
print("COMMON COLUMN NAMES")
print("=" * 55)

for col in common:
    print(col)

print("\nTotal common columns:", len(common))

# Show numeric columns that could contain acoustic features
voc_numeric = set(
    voc.select_dtypes(include="number").columns
)

minsk_numeric = set(
    minsk.select_dtypes(include="number").columns
)

common_numeric = sorted(voc_numeric.intersection(minsk_numeric))

print("\n" + "=" * 55)
print("COMMON NUMERIC COLUMNS")
print("=" * 55)

for col in common_numeric:
    print(col)

print("\nTotal common numeric columns:", len(common_numeric))

print("\nIMPORTANT:")
print("Matching names do not guarantee matching measurements.")
print("Check feature meanings, units, recording tasks and labels")
print("before training a combined model.")
