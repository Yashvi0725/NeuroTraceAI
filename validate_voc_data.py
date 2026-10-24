
import pandas as pd
from pathlib import Path

PROJECT = Path(r"C:\NeuroTraceAIFinal")

features = pd.read_csv(PROJECT / "audio_acoustic_features.csv")
labels = pd.read_csv(PROJECT / "participant_labels.csv")

# Check that every participant has one diagnosis
label_counts = labels.groupby("ID")["Category"].nunique()
print("Participants with conflicting labels:",
      int((label_counts > 1).sum()))

# Check that feature labels match participant labels
merged = features.merge(
    labels,
    left_on="ParticipantID",
    right_on="ID",
    suffixes=("_feature", "_participant"),
    how="left",
    validate="many_to_one"
)

print("Recordings without a participant label:",
      int(merged["Category_participant"].isna().sum()))

print("Mismatched diagnosis labels:",
      int((merged["Category_feature"] !=
           merged["Category_participant"]).sum()))

# Check the number of tasks per participant
tasks_per_person = features.groupby("ParticipantID")["Task"].nunique()

print("Participants with exactly 8 tasks:",
    int((tasks_per_person == 8).sum()))
print("Participants with missing or extra tasks:",
    int((tasks_per_person != 8).sum()))

# Check whether acoustic features are constant or missing
numeric = features.select_dtypes(include="number")
print("\nMissing numeric values:", int(numeric.isna().sum().sum()))
print("Numeric features with zero variance:")
print(numeric.columns[numeric.nunique(dropna=False) <= 1].tolist())

print("\nValidation complete. No files were modified.")
