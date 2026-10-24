
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\NeuroTraceAIFinal")

voc = pd.read_excel(
    BASE / "VOC-ALS.xlsx",
    sheet_name="VOC-ALS_Data",
    header=1
)

jps = pd.read_csv(BASE / "JPS2024_ALS_dataset.csv")

minsk = pd.read_csv(
    BASE / "Minsk2020" / "Minsk2020_ALS_database-main"
    / "Minsk2020_ALS_dataset.csv"
)

datasets = {
    "VOC-ALS": voc,
    "JPS2024": jps,
    "Minsk2020": minsk
}

id_sets = {}

for name, df in datasets.items():
    print("\n", "=" * 40)
    print(name)

    ids = df["ID"].dropna().astype(str)
    id_sets[name] = set(ids)

    print("Total rows:", len(df))
    print("Unique IDs:", ids.nunique())
    print("Duplicate ID rows:", ids.duplicated().sum())
    print("Completely duplicated rows:", df.duplicated().sum())

    if ids.duplicated().any():
        print("Repeated IDs:")
        print(ids[ids.duplicated(keep=False)].tolist())

print("\n", "=" * 40)
print("CROSS-DATASET ID OVERLAP")

names = list(id_sets)

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        common = id_sets[a] & id_sets[b]

        print(f"{a} vs {b}: {len(common)} shared IDs")

        if common:
            print("Examples:", sorted(common)[:10])

print("\nCheck complete.")
