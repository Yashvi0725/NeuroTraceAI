
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

PROJECT = Path(r"C:\NeuroTraceAIFinal")

path = PROJECT / "Minsk2020_ALS_clean_for_training.csv"
data = pd.read_csv(path)

target = "Diagnosis (ALS)"

if target not in data.columns:
    raise ValueError("Diagnosis (ALS) column not found.")

y = pd.to_numeric(data[target], errors="raise").astype(int)

if not set(y.unique()).issubset({0, 1}):
    raise ValueError("Diagnosis must contain only 0 and 1.")

# Use numeric acoustic features.
# Exclude diagnosis, identifiers, and demographic variables.
excluded = {
    target, "ID", "Age", "Sex"
}

X = data.drop(
    columns=[c for c in excluded if c in data.columns]
)

X = X.select_dtypes(include="number")

if X.shape[1] == 0:
    raise ValueError("No numeric acoustic features found.")

# Remove columns that contain no usable variation
X = X.loc[:, X.nunique(dropna=True) > 1]

if X.shape[1] < 10:
    k = "all"
else:
    k = 10

print("Rows:", len(data))
print("ALS cases:", int((y == 1).sum()))
print("Healthy controls:", int((y == 0).sum()))
print("Acoustic candidate features:", X.shape[1])
print("Features selected per fold:", k)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SelectKBest(score_func=f_classif, k=k),
        LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=42
        )
    ),

    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        SelectKBest(score_func=f_classif, k=k),
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        )
    ),

    "SVM": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SelectKBest(score_func=f_classif, k=k),
        SVC(class_weight="balanced")
    )
}

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

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

    row = {"Model": name}

    for metric in scoring:
        values = scores["test_" + metric]
        row[metric + "_mean"] = values.mean()
        row[metric + "_std"] = values.std()

    results.append(row)

result_df = pd.DataFrame(results)

result_df.to_csv(
    PROJECT / "minsk_model_comparison.csv",
    index=False
)

print("\nMINSK2020 MODEL COMPARISON")
print(result_df.round(3).to_string(index=False))

print("\nSaved:")
print(PROJECT / "minsk_model_comparison.csv")
