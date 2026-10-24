
import pandas as pd

files = [
    "JPS2024_ALS_dataset.csv",
    "JPS2024_ALS_cleaned.csv",
    "Minsk2020_ALS_dataset.csv"
]

for file in files:
    print("\n" + "=" * 60)
    print("FILE:", file)

    try:
        df = pd.read_csv(file)
        print("Shape:", df.shape)
        print("Columns:")
        print(df.columns.tolist())

        print("\nPossible diagnosis columns:")
        for col in df.columns:
            if any(term in col.lower() for term in [
                "diagnosis", "category", "label", "als", "group",
                "disease", "class", "status"
            ]):
                print(col, ":", df[col].value_counts(dropna=False).to_dict())

        print("\nID column candidates:")
        for col in df.columns:
            if col.strip().lower() in ["id", "participant_id", "subject_id"]:
                print(col, "unique IDs:", df[col].nunique())
    except Exception as e:
        print("ERROR:", e)
