
import pandas as pd

# Load both datasets
voc = pd.read_excel(
    "VOC-ALS.xlsx",
    sheet_name="VOC-ALS_Data",
    header=1
)

minsk = pd.read_csv(
    "Minsk2020_ALS_clean_for_training.csv"
)

# Clean spaces around column names
voc.columns = voc.columns.astype(str).str.strip()
minsk.columns = minsk.columns.astype(str).str.strip()

# Candidate HNR features
features = [
    ("VOC-ALS", voc, "HNR_A", "Category"),
    ("Minsk2020", minsk, "HNR_a", "Diagnosis (ALS)")
]

for dataset_name, df, feature, label in features:
    print("\n" + "=" * 50)
    print(dataset_name)
    print("=" * 50)

    if feature not in df.columns:
        print("Feature not found:", feature)
        continue

    values = pd.to_numeric(df[feature], errors="coerce")

    print("Feature:", feature)
    print("Rows:", len(df))
    print("Missing values:", values.isna().sum())
    print("Minimum:", values.min())
    print("Median:", values.median())
    print("Maximum:", values.max())
    print("Standard deviation:", values.std())

    if label in df.columns:
        print("\nFeature summary by diagnosis:")
        print(
            pd.DataFrame({
                "Diagnosis": df[label],
                "Feature": values
            }).groupby("Diagnosis")["Feature"].describe()
        )

print("\nAudit completed.")
print("Similar ranges alone do not prove feature equivalence.")
