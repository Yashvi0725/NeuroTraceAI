
import pandas as pd
from pathlib import Path

PROJECT = Path(r"C:\NeuroTraceAIFinal")

files = [
    PROJECT / "audio_acoustic_features.csv",
    PROJECT / "audio_manifest_labeled.csv",
    PROJECT / "participant_labels.csv",
]

for path in files:
    print("\n" + "=" * 55)
    print(path.name)
    print("=" * 55)

    if not path.exists():
        print("File not found.")
        continue

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    for col in ["Category", "Diagnosis", "Diagnosis (ALS)",
                "Label", "label", "ParticipantID", "Task"]:
        if col in df.columns:
            if col in ["Category", "Diagnosis", "Diagnosis (ALS)",
                    "Label", "label"]:
                print(f"\n{col} counts:")
                print(df[col].value_counts(dropna=False).to_string())

            elif col == "ParticipantID":
                print("\nUnique participants:", df[col].nunique())
                print("Missing participant IDs:", int(df[col].isna().sum()))

            elif col == "Task":
                print("\nTask counts:")
                print(df[col].value_counts(dropna=False).to_string())

print("\nLabel check complete. No files were modified.")
