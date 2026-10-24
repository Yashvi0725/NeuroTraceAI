
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

BASE = Path(r"C:\NeuroTraceAIFinal")

df = pd.read_excel(
    BASE / "VOC-ALS.xlsx",
    sheet_name="VOC-ALS_Data",
    header=1
)

df.columns = df.columns.str.strip()
df = df.dropna(subset=["Category"])

# Use acoustic features only
exclude = [
    "ID", "Category", "Age", "Sex",
    "ALSFRS-R", "FVC", "Disease duration",
    "Progression rate", "King stage"
]

features = [
    c for c in df.columns
    if c not in exclude
    and pd.api.types.is_numeric_dtype(df[c])
]

X = df[features].replace([np.inf, -np.inf], np.nan)
y = df["Category"].astype(str).str.strip()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    )
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Features used:", len(features))
print("Training participants:", len(X_train))
print("Testing participants:", len(X_test))
print("\nAccuracy:", round(
    accuracy_score(y_test, predictions), 3
))
print("\nClassification report:")
print(classification_report(y_test, predictions))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(
    {"model": model, "features": features},
    BASE / "neurotrace_baseline.pkl"
)

print("\nSaved neurotrace_baseline.pkl")
