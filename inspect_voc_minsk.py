
import pandas as pd
from pathlib import Path

PROJECT = Path(r"C:\NeuroTraceAIFinal")

files = {
    "VOC-ALS candidate": PROJECT / "audio_acoustic_features.csv",
    "Minsk2020 raw": (
        PROJECT / "Minsk2020"
        / "Minsk2020_ALS_database-main"
        / "Minsk2020_ALS_dataset.csv"
    ),
    "Minsk2020 cleaned": (
        PROJECT / "Minsk2020_ALS_clean_for_training.csv"
    ),
}

for name, path in files.items():
    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    if not path.exists():
        print("FILE NOT FOUND:", path)
        continue

    df = pd.read_csv(path)

    print("File:", path)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Column names:", list(df.columns))
    print("Total missing values:", int(df.isna().sum().sum()))

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    label_cols = [
        c for c in df.columns
        if any(word in c.lower()
            for word in ["diagnosis", "label", "class", "group", "target"])
    ]

    print("\nPossible label columns:", label_cols)

    for col in label_cols:
        print("\nCounts for", col)
        print(df[col].value_counts(dropna=False).to_string())

print("\nInspection complete. No files were modified.")
