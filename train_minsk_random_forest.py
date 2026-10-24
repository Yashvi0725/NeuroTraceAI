
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

# Dataset location
DATA_PATH = Path(
    r"C:\NeuroTraceAIFinal\Minsk2020_ALS_clean_for_training.csv"
)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

# Load the dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Find the diagnosis column
target_candidates = [
    c for c in df.columns
    if "diagnosis" in c.lower()
    and "als" in c.lower()
]

if not target_candidates:
    target_candidates = [
        c for c in df.columns
        if c.lower().strip() == "diagnosis"
    ]

if not target_candidates:
    raise ValueError(
        "Diagnosis column not found. Check the printed column names."
    )

target = target_candidates[0]

# Prepare target labels
y = pd.to_numeric(df[target], errors="coerce")
valid = y.notna()

df = df.loc[valid].copy()
y = y.loc[valid].astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError(
        f"Expected diagnosis labels 0 and 1, found {sorted(y.unique())}"
    )

# Exclude target, identifiers and demographic information
excluded = {
    target, "ID", "Age", "Sex", "Gender",
    "ParticipantID", "SubjectID", "SpeakerID"
}

X = df.drop(
    columns=[
        c for c in df.columns
        if c.lower() in {v.lower() for v in excluded}
    ],
    errors="ignore"
)

# Use numeric acoustic features only
X = X.select_dtypes(include=[np.number])
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=True) > 1]

if X.empty:
    raise ValueError("No usable numeric features were found.")

print("\nRows:", len(df))
print("Healthy control labels (0):", int((y == 0).sum()))
print("ALS labels (1):", int((y == 1).sum()))
print("Numeric features:", X.shape[1])

# Select features inside each CV fold to reduce leakage
k = min(10, X.shape[1])

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("selector", SelectKBest(score_func=f_classif, k=k)),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2
    ))
])

folds = min(5, int(y.value_counts().min()))

cv = StratifiedKFold(
    n_splits=folds,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

print(f"\nTraining and evaluating with {folds}-fold cross-validation...")

results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

# Display results
rows = []

for metric in scoring:
    scores = results["test_" + metric]
    mean = scores.mean()
    std = scores.std(ddof=1) if len(scores) > 1 else 0

    rows.append({
        "Metric": metric,
        "Mean": mean,
        "Standard Deviation": std
    })

    print(f"{metric}: {mean:.3f} +/- {std:.3f}")

# Save evaluation results
output_dir = DATA_PATH.parent

pd.DataFrame(rows).to_csv(
    output_dir / "minsk_training_metrics.csv",
    index=False
)

# Train the final model using all available rows
model.fit(X, y)

# Save model and expected input features
joblib.dump({
    "pipeline": model,
    "input_features": X.columns.tolist(),
    "target_column": target,
    "label_meaning": {
        0: "Healthy control, according to dataset labels",
        1: "ALS, according to dataset labels"
    }
}, output_dir / "minsk_random_forest_model.joblib")

# Save selected feature names
selected = X.columns[
    model.named_steps["selector"].get_support()
]

pd.DataFrame({
    "selected_feature": selected
}).to_csv(
    output_dir / "minsk_selected_features.csv",
    index=False
)

print("\nTraining completed!")
print("Saved: minsk_random_forest_model.joblib")
print("Saved: minsk_training_metrics.csv")
print("Saved: minsk_selected_features.csv")
print("\nThis is a research prototype, not a medical diagnostic tool.")
