
import pandas as pd

df = pd.read_csv("Minsk2020_ALS_dataset.csv")

df["ID"] = df["ID"].astype(str).str.strip()
label_col = "Diagnosis (ALS)"

# Find IDs with conflicting diagnosis labels
label_counts = df.groupby("ID")[label_col].nunique()
conflicting_ids = label_counts[label_counts > 1].index.tolist()

print("Conflicting IDs:", conflicting_ids)

# Keep only participants with one consistent diagnosis
clean = df[~df["ID"].isin(conflicting_ids)].copy()

# Save a separate file; original remains unchanged
clean.to_csv("Minsk2020_ALS_clean_for_training.csv", index=False)

print("\nOriginal rows:", len(df))
print("Rows retained:", len(clean))
print("Unique participants retained:", clean["ID"].nunique())
print("\nDiagnosis counts:")
print(clean[label_col].value_counts())
