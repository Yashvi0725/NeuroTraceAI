
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

# Show every row belonging to a repeated ID
duplicate_ids = raw.loc[
    raw["ID"].duplicated(keep=False), "ID"
].unique()

print("REPEATED IDS:", sorted(duplicate_ids.tolist()))

for participant_id in sorted(duplicate_ids):
    print("\n" + "=" * 60)
    print("RAW RECORDS FOR ID:", participant_id)

    rows = raw[raw["ID"] == participant_id]
    print(rows.to_string(index=False))

# Compare the three IDs missing from the cleaned dataset
removed_ids = sorted(
    set(raw["ID"].dropna()) - set(clean["ID"].dropna())
)

print("\nIDS REMOVED FROM CLEANED DATA:", removed_ids)

for participant_id in removed_ids:
    print("\n" + "=" * 60)
    print("REMOVED ID:", participant_id)

    rows = raw[raw["ID"] == participant_id]
    print(
        rows[["ID", "Sex", "Age", "Diagnosis (ALS)"]]
        .to_string(index=False)
    )

print("\nInspection complete. No files were modified.")
