
from pathlib import Path
import pandas as pd

base = (
    Path.home()
    / "Desktop"
    / "NeuroTraceAI_ALS"
    / "datasets"
    / "Minsk2020"
    / "Minsk2020_ALS_database-main"
)

csv_path = base / "Minsk2020_ALS_dataset.csv"

data = pd.read_csv(csv_path)

print("CSV rows:", len(data))
print("CSV columns:", len(data.columns))
print("\nDiagnosis counts:")
print(data["Diagnosis (ALS)"].value_counts().sort_index())

print("\nCSV subject details:")
print(
    data[["ID", "Age", "Sex", "Diagnosis (ALS)"]]
    .sort_values(["Diagnosis (ALS)", "ID"])
    .to_string(index=False)
)

print("\nOriginal Excel table counts:")
print("ALS:", len(pd.read_excel(base / "ALS_table.xlsx")))
print("Healthy controls:", len(pd.read_excel(base / "HC_table.xlsx")))
