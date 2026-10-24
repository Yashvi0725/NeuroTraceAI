
import pandas as pd

df = pd.read_csv(r"C:\NeuroTraceAIFinal\JPS2024_ALS_dataset.csv")

cleaned = df.drop_duplicates()

cleaned.to_csv(
    r"C:\NeuroTraceAIFinal\JPS2024_ALS_cleaned.csv",
    index=False
)

print("Original rows:", len(df))
print("Cleaned rows:", len(cleaned))
print("Unique IDs:", cleaned["ID"].nunique())
print("Saved cleaned copy successfully.")
