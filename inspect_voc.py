
import pandas as pd

path = r"C:\Users\ypath\Downloads\VOC-ALS.xlsx"

# Read the sheet using the second row as the header
df = pd.read_excel(
    path,
    sheet_name="VOC-ALS_Data",
    header=1
)

print("Dataset shape:", df.shape)

print("\nDiagnosis counts:")
print(df["Category"].value_counts(dropna=False))

print("\nUnique participants:", df["ID"].nunique())

print("\nDuplicate participant IDs:", df["ID"].duplicated().sum())

print("\nFirst 10 participants:")
print(df[["ID", "Age (years)", "Sex", "Category"]].head(10))

print("\nMissing diagnosis labels:", df["Category"].isna().sum())
