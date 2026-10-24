
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

BASE = Path(r"C:\NeuroTraceAIFinal")

df = pd.read_excel(
    BASE / "VOC-ALS.xlsx",
    sheet_name="VOC-ALS_Data",
    header=1
)

df.columns = df.columns.str.strip()
df = df.dropna(subset=["Category"]).copy()

# Exclude identifiers, demographics and clinical outcomes
exclude = [
    "ID", "Category", "Age", "Sex",
    "ALSFRS-R", "FVC", "Disease duration",
    "Progression rate", "King stage"
]

# Also exclude clinical fields with similar names
clinical_terms = [
    "alsfrs", "fvc", "duration", "progression",
    "king stage", "severity", "diagnosis"
]

features = []

for col in df.columns:
    name = col.lower().replace("_", " ").strip()

    if col in exclude:
        continue

    if any(term in name for term in clinical_terms):
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        features.append(col)

X = df[features].replace([np.inf, -np.inf], np.nan)
y = df["Category"].astype(str).str.strip()

# Keep folds stratified by diagnosis
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42
        )
    ),

    "Extra Trees": make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42
        )
    ),

    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    ),

    "SVM": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(
            class_weight="balanced",
            kernel="rbf"
        )
    )
}

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": "f1_macro"
}

print("Participants:", len(df))
print("Features used:", len(features))
print("Classes:")
print(y.value_counts())
print("\nCross-validation results (5 folds)")
print("-" * 70)

results = []

for name, model in models.items():
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    row = {
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Balanced Accuracy": scores["test_balanced_accuracy"].mean(),
        "Macro F1": scores["test_macro_f1"].mean()
    }

    results.append(row)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

print(results_df.round(3).to_string(index=False))

results_df.to_csv(
    BASE / "model_comparison.csv",
    index=False
)

print("\nSaved model_comparison.csv")
