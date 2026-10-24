
import pandas as pd

jps = pd.read_csv("JPS2024_ALS_dataset.csv")
minsk = pd.read_csv("Minsk2020_ALS_dataset.csv")

jps["ID"] = jps["ID"].astype(str).str.strip()
minsk["ID"] = minsk["ID"].astype(str).str.strip()

label_col = "Diagnosis (ALS)"

print("JPS2024 unique participants:", jps["ID"].nunique())
print("Minsk2020 unique participants:", minsk["ID"].nunique())

shared = set(jps["ID"]) & set(minsk["ID"])
print("Shared participant IDs:", len(shared))

print("\nRepeated JPS2024 IDs:")
print(jps.loc[jps["ID"].duplicated(False), "ID"]
      .value_counts().sort_index())

print("\nRepeated Minsk2020 IDs:")
print(minsk.loc[minsk["ID"].duplicated(False), "ID"]
      .value_counts().sort_index())

# Check whether each participant has a consistent diagnosis.
label_counts = minsk.groupby("ID")[label_col].nunique()
conflicts = label_counts[label_counts > 1]

print("\nParticipants with conflicting diagnosis labels:",
      len(conflicts))

if len(conflicts):
    print(conflicts)

# Build a label lookup only for IDs with consistent labels.
valid_ids = label_counts[label_counts == 1].index
labels = (
    minsk[minsk["ID"].isin(valid_ids)]
    .drop_duplicates("ID")
    .set_index("ID")[label_col]
)

matched = jps[jps["ID"].isin(labels.index)].copy()
matched["Minsk_label"] = matched["ID"].map(labels)

print("\nJPS rows with a matching Minsk label:", len(matched))
print("Unique matched participants:", matched["ID"].nunique())
print("Matched label counts (encoding not yet verified):")
print(matched.drop_duplicates("ID")["Minsk_label"]
      .value_counts(dropna=False))

print("\nAudit complete.")
