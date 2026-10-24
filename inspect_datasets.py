
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\NeuroTraceAIFinal")

voc = pd.read_excel(
    BASE / "VOC-ALS.xlsx",
    sheet_name="VOC-ALS_Data",
    header=1
)

jps = pd.read_csv(BASE / "JPS2024_ALS_dataset.csv")

minsk = pd.read_csv(
    BASE / "Minsk2020" / "Minsk2020_ALS_database-main"
    / "Minsk2020_ALS_dataset.csv"
)

datasets = {
    "VOC-ALS": voc,
    "JPS2024": jps,
    "Minsk2020": minsk
}

for name, df in datasets.items():
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print("Shape:", df.shape)
    print("Columns:", len(df.columns))
    print("Missing values:", df.isna().sum().sum())

    if "Category" in df.columns:
        print("\nCategory labels:")
        print(df["Category"].value_counts(dropna=False))

    if "Diagnosis (ALS)" in df.columns:
        print("\nDiagnosis labels:")
        print(df["Diagnosis (ALS)"].value_counts(dropna=False))

print("\n" + "=" * 50)
print("FEATURE COMPARISON")
print("=" * 50)

jps_cols = set(jps.columns)
minsk_cols = set(minsk.columns)

print("JPS2024 shared with Minsk2020:",
      len(jps_cols & minsk_cols))
print("JPS2024-only columns:",
      sorted(jps_cols - minsk_cols))
print("Minsk2020-only columns:",
      sorted(minsk_cols - jps_cols))

print("\n" + "=" * 50)
print("PARTICIPANT ID CHECK")
print("=" * 50)

for name, df in datasets.items():
    if "ID" in df.columns:
        ids = df["ID"].dropna().astype(str)
        print(name, "- unique IDs:", ids.nunique())

print("\nInspection complete.")
