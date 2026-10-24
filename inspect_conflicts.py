
import pandas as pd

df = pd.read_csv("Minsk2020_ALS_dataset.csv")

for col in ["ID", "Sex", "Age", "Diagnosis (ALS)"]:
    if col not in df.columns:
        print("Missing column:", col)

available = [
    col for col in ["ID", "Sex", "Age", "Diagnosis (ALS)"]
    if col in df.columns
]

print("\nAll records for IDs 24, 28 and 42:")
print(
    df[df["ID"].astype(str).str.strip().isin(["24", "28", "42"])]
    [available]
    .sort_values("ID")
    .to_string(index=False)
)

print("\nTotal rows:", len(df))
print("Unique IDs:", df["ID"].nunique())
print("\nDiagnosis counts:")
print(df["Diagnosis (ALS)"].value_counts(dropna=False))
