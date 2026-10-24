
import pandas as pd

df = pd.read_excel(
    "VOC-ALS.xlsx",
    sheet_name="VOC-ALS_Data",
    header=1
)

df.columns = df.columns.str.strip()
df = df.dropna(subset=["Category"]).copy()

excluded_names = {
    "id", "category", "age", "age (years)",
    "sex", "gender", "cantagallo_questionnaire"
}

excluded_terms = [
    "alsfrs", "fvc", "progression", "disease duration",
    "king stage", "diagnosis", "survival", "outcome",
    "questionnaire", "medication", "treatment",
    "onset", "severity", "clinical"
]

features = []

for col in df.columns:
    name = col.strip().lower()

    if name in excluded_names:
        continue

    if any(term in name for term in excluded_terms):
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        features.append(col)

print("Participants:", len(df))
print("Selected numeric features:", len(features))
print("\nFEATURE NAMES:")

for feature in features:
    print(feature)
