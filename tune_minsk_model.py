
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_validate
)

DATA_PATH = Path(
    r"C:\NeuroTraceAIFinal\Minsk2020_ALS_clean_for_training.csv"
)

df = pd.read_csv(DATA_PATH)

target = "Diagnosis (ALS)"

if target not in df.columns:
    raise ValueError("Diagnosis column not found.")

y = pd.to_numeric(df[target], errors="coerce")
valid = y.notna()

df = df.loc[valid].copy()
y = y.loc[valid].astype(int)

excluded = [
    target, "ID", "Age", "Sex", "Gender",
    "ParticipantID", "SubjectID", "SpeakerID"
]

X = df.drop(
    columns=[
        c for c in df.columns
        if c.lower() in [v.lower() for v in excluded]
    ],
    errors="ignore"
)

X = X.select_dtypes(include=[np.number])
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=True) > 1]

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("selector", SelectKBest(score_func=f_classif)),
    ("classifier", RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])

parameter_grid = {
    "selector__k": [5, 10, 15, 20, 30],
    "classifier__n_estimators": [200, 500],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", 0.5]
}

inner_cv = StratifiedKFold(
    n_splits=3, shuffle=True, random_state=42
)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring="balanced_accuracy",
    cv=inner_cv,
    n_jobs=-1,
    refit=True
)

outer_cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=123
)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

print("Running nested cross-validation...")
print("This may take a few minutes.")

results = cross_validate(
    search,
    X,
    y,
    cv=outer_cv,
    scoring=scoring,
    n_jobs=1
)

print("\nNESTED CROSS-VALIDATION RESULTS")

for metric in scoring:
    scores = results["test_" + metric]
    print(
        f"{metric}: {scores.mean():.3f} "
        f"+/- {scores.std(ddof=1):.3f}"
    )

# Find the best settings using all available records.
# These settings are for the final model, not an
# independent estimate of future performance.
search.fit(X, y)

print("\nBEST SETTINGS")
print(search.best_params_)
print(
    "Best inner-CV balanced accuracy:",
    round(search.best_score_, 3)
)

# Save the tuned model and experiment results.
output_dir = DATA_PATH.parent

import joblib

joblib.dump(
    {
        "pipeline": search.best_estimator_,
        "input_features": X.columns.tolist(),
        "target_column": target
    },
    output_dir / "minsk_tuned_random_forest.joblib"
)

pd.DataFrame({
    "Metric": list(scoring.keys()),
    "Mean": [
        results["test_" + m].mean() for m in scoring
    ],
    "Standard_Deviation": [
        results["test_" + m].std(ddof=1) for m in scoring
    ]
}).to_csv(
    output_dir / "minsk_tuned_cv_metrics.csv",
    index=False
)

print("\nSaved minsk_tuned_random_forest.joblib")
print("Saved minsk_tuned_cv_metrics.csv")
