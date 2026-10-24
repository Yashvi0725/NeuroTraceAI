
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\NeuroTraceAIFinal")

files = {
    "JPS2024": BASE / "JPS2024_ALS_dataset.csv",
    "Minsk2020": (
        BASE / "Minsk2020"
        / "Minsk2020_ALS_database-main"
        / "Minsk2020_ALS_dataset.csv"
    )
}

for name, path in files.items():
    df = pd.read_csv(path)

    print("\n" + "=" * 40)
    print(name)
    print("=" * 40)

    duplicates = df[df["ID"].duplicated(keep=False)]

    if duplicates.empty:
        print("No duplicate IDs found.")
        continue

    for participant_id, rows in duplicates.groupby("ID"):
        print("\nParticipant ID:", participant_id)
        print("Number of rows:", len(rows))

        if "Diagnosis (ALS)" in df.columns:
            print(
                "Diagnosis:",
                rows["Diagnosis (ALS)"].tolist()
            )

        features = [
            c for c in df.columns
            if c not in [
                "ID", "Sex", "Age", "Diagnosis (ALS)"
            ]
        ]

        print(rows[features].to_string(index=False))

print("\nInspection complete!")
