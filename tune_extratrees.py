
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler

BASE = Path(r"C:\NeuroTraceAIFinal")

df = pd.read_excel(
    BASE / "VOC-ALS.xlsx",
    sheet_name="VOC-ALS_Data",
    header=1
)

df.columns = df.columns.str.strip()
df = df.dropna(subset=["Category"]).copy()

# Exclude identifiers, demographics and clinical outcomes
exclude_exact = {
    "ID", "Category", "Age", "Sex", "Gender",
    "ALSFRS-R", "FVC", "Disease duration",
    "Progression rate", "King stage"
}

exclude_terms = [
    "alsfrs", "fvc", "progression", "disease duration",
    "king stage", "diagnosis", "survival", "outcome",
    "medication", "treatment", "date of diagnosis"
]

features = []

for col in df.columns:
    name = col.lower().strip()

    if col in exclude_exact:
        continue

    if any(term in name for term in exclude_terms):
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        features.append(col)

X = df[features].replace([np.inf, -np.inf], np.nan)
y = df["Category"].astype(str).str.strip()

print("Participants:", len(df))
print("Features available:", len(features))
print("Features being considered:")
print(features)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("select", SelectKBest(score_func=f_classif)),
    ("model", ExtraTreesClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# Search for useful feature counts and model settings
param_distributions = {
    "select__k": [10, 20, 30, "all"],
    "model__n_estimators": [200, 400],
    "model__max_features": ["sqrt", 0.5, 1.0],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_depth": [None, 10, 20]
}

outer_cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

# Outer folds estimate performance on unseen participants
all_predictions = []
all_actual = []

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X, y), start=1
):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    inner_cv = StratifiedKFold(
        n_splits=4, shuffle=True, random_state=42
    )

    search = RandomizedSearchCV(
        estimator=clone(pipeline),
        param_distributions=param_distributions,
        n_iter=25,
        scoring="accuracy",
        cv=inner_cv,
        random_state=fold,
        n_jobs=-1,
        refit=True
    )

    search.fit(X_train, y_train)
    pred = search.predict(X_test)

    all_predictions.extend(pred)
    all_actual.extend(y_test.tolist())

    print(
        f"Fold {fold}: accuracy = "
        f"{accuracy_score(y_test, pred):.3f}"
    )

print("\nNested cross-validation results")
print("Accuracy:",
      round(accuracy_score(all_actual, all_predictions), 3))
print("Balanced accuracy:",
      round(balanced_accuracy_score(all_actual, all_predictions), 3))
print("Macro F1:",
      round(f1_score(all_actual, all_predictions,
                     average="macro"), 3))

# Tune on all participants to create a model for future experiments.
# This model is not an independently validated clinical model.
final_search = RandomizedSearchCV(
    estimator=clone(pipeline),
    param_distributions=param_distributions,
    n_iter=25,
    scoring="accuracy",
    cv=StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    ),
    random_state=42,
    n_jobs=-1,
    refit=True
)

final_search.fit(X, y)

joblib.dump(
    {
        "model": final_search.best_estimator_,
        "features": features,
        "classes": list(final_search.classes_)
    },
    BASE / "neurotrace_extratrees_tuned.pkl"
)

print("\nBest settings:", final_search.best_params_)
print("Best inner CV accuracy:",
    round(final_search.best_score_, 3))
print("Saved neurotrace_extratrees_tuned.pkl")
