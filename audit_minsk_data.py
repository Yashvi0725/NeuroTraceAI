
import pandas as pd
from pathlib import Path

PROJECT = Path(r"C:\NeuroTraceAIFinal")

raw_path = (
    PROJECT / "Minsk2020" /
    "Minsk2020_ALS_database-main" /
    "Minsk2020_ALS_dataset.csv"
)

clean_path = PROJECT / "Minsk2020_ALS_clean_for_training.csv"

raw = pd.read_csv(raw_path)
clean = pd.read_csv(clean_path)

print("RAW DATA")
print("Rows:", len(raw))
print("Columns:", len(raw.columns))
print("Duplicate IDs:", raw["ID"].duplicated().sum())
print("Diagnosis counts:")
print(raw["Diagnosis (ALS)"].value_counts(dropna=False))

print("\nCLEANED DATA")
print("Rows:", len(clean))
print("Columns:", len(clean.columns))
print("Duplicate IDs:", clean["ID"].duplicated().sum())
print("Diagnosis counts:")
print(clean["Diagnosis (ALS)"].value_counts(dropna=False))

print("\nCOLUMNS IN CLEANED DATA")
for column in clean.columns:
    print(column)

print("\nNUMERIC CANDIDATE FEATURES")
excluded = {"Diagnosis (ALS)", "ID", "Age", "Sex"}

candidates = [
    column for column in clean.columns
    if column not in excluded
    and pd.api.types.is_numeric_dtype(clean[column])
]

print("Count:", len(candidates))
for column in candidates:
    print(column)

print("\nROWS PRESENT IN RAW BUT NOT CLEANED")
raw_ids = set(raw["ID"].dropna())
clean_ids = set(clean["ID"].dropna())

print("Removed IDs:", sorted(raw_ids - clean_ids))
print("New IDs in cleaned file:", sorted(clean_ids - raw_ids))

print("\nAudit complete. No files were modified.")
