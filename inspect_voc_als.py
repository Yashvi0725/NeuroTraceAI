
import pandas as pd

file_path = r"C:\Users\ypath\Downloads\VOC-ALS.xlsx"

df = pd.read_excel(
    file_path,
    sheet_name="VOC-ALS_Data",
    header=1
)

print("Dataset shape:", df.shape)
print("\nAll column names:")

for column in df.columns:
    print(column)

print("\nPossible participant ID and diagnosis columns:")

for column in df.columns:
    if any(word in str(column).lower()
           for word in ["id", "participant", "subject", "category"]):
        print("\nColumn:", column)
        print(df[column].head(10).to_list())

print("\nDiagnosis counts:")
print(df["Category"].value_counts(dropna=False))
