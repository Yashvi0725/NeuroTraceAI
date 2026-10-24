
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BASE = Path(r"C:\NeuroTraceAIFinal")

# Load data
df = pd.read_excel(
    BASE / "VOC-ALS.xlsx",
    sheet_name="VOC-ALS_Data",
    header=1
)

df.columns = df.columns.str.strip()
df = df.dropna(subset=["Category"]).copy()

# Keep only eligible acoustic measurements.

# Exclude identifiers, demographics, questionnaires and clinical outcomes.
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

print("\nFEATURES SELECTED FOR REVIEW:")
for feature in features:
    print(feature)

X = df[features].replace([np.inf, -np.inf], np.nan)
y = df["Category"].astype(str).str.strip()


if y.nunique() != 2:
    raise ValueError("Expected two diagnosis classes.")

print("Participants:", len(df))
print("Eligible acoustic features:", len(features))
print("\nFEATURES USED:")
for feature in features:
    print(feature)
print("Classes:")
print(y.value_counts())

# Four candidate models
models = {
    "Extra Trees": (
        ExtraTreesClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        ),
        {
            "model__n_estimators": [200, 400, 600],
            "model__max_features": ["sqrt", 0.5, 1.0],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_depth": [None, 10, 20]
        }
    ),
    "Random Forest": (
        RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        ),
        {
            "model__n_estimators": [200, 400, 600],
            "model__max_features": ["sqrt", 0.5, 1.0],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_depth": [None, 10, 20]
        }
    ),
    "SVM": (
        SVC(class_weight="balanced"),
        {
            "model__C": [0.1, 1, 10, 100],
            "model__kernel": ["rbf", "linear"],
            "model__gamma": ["scale", "auto"]
        }
    ),
    "Logistic Regression": (
        LogisticRegression(
            class_weight="balanced",
            max_iter=3000
        ),
        {
            "model__C": [0.01, 0.1, 1, 10, 100]
        }
    )
}

outer_cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

results = []

for model_name, (estimator, params) in models.items():
    actual = []
    predicted = []

    print("\nTraining:", model_name)

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif)),
            ("model", clone(estimator))
        ])

        # Keep selection and tuning inside the training fold.
        max_k = min(40, X_train.shape[1])
        k_values = sorted(set(
            k for k in [5, 10, 15, 20, 30, max_k]
            if k <= max_k
        ))

        search_params = dict(params)
        search_params["select__k"] = k_values

        inner_cv = StratifiedKFold(
            n_splits=4, shuffle=True, random_state=fold
        )

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=search_params,
            n_iter=20,
            scoring="accuracy",
            cv=inner_cv,
            random_state=42,
            n_jobs=-1,
            refit=True
        )

        search.fit(X_train, y_train)
        pred = search.predict(X_test)

        actual.extend(y_test.tolist())
        predicted.extend(pred.tolist())

        print(
            " Fold", fold,
            "| Accuracy:", round(
                accuracy_score(y_test, pred), 3
            ),
            "| Selected features:",
            search.best_params_["select__k"]
        )

    acc = accuracy_score(actual, predicted)
    bal = balanced_accuracy_score(actual, predicted)
    f1 = f1_score(actual, predicted, average="macro")

    results.append({
        "Model": model_name,
        "Accuracy": acc,
        "Balanced Accuracy": bal,
        "Macro F1": f1
    })

    print(
        " Nested CV:",
        "Accuracy =", round(acc, 3),
        "| Balanced accuracy =", round(bal, 3),
        "| Macro F1 =", round(f1, 3)
    )

results_df = pd.DataFrame(results).sort_values(
    "Accuracy", ascending=False
)

print("\nFINAL MODEL COMPARISON")
print(results_df.round(3).to_string(index=False))

results_df.to_csv(
    BASE / "neurotrace_training_results.csv",
    index=False
)

# Train a deployable model on all available participants.
# Its performance must be estimated from the nested CV above.
best_name = results_df.iloc[0]["Model"]
best_estimator, best_params = models[best_name]

final_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("select", SelectKBest(score_func=f_classif)),
    ("model", clone(best_estimator))
])

max_k = min(40, X.shape[1])
k_values = sorted(set(
    k for k in [5, 10, 15, 20, 30, max_k] if k <= max_k
))

final_params = dict(best_params)
final_params["select__k"] = k_values

final_search = RandomizedSearchCV(
    final_pipeline,
    param_distributions=final_params,
    n_iter=20,
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
        "model_name": best_name,
        "classes": list(final_search.classes_)
    },
    BASE / "neurotrace_best_model.pkl"
)

print("\nBest model:", best_name)
print("Best training settings:", final_search.best_params_)
print("Saved: neurotrace_best_model.pkl")
print("Saved: neurotrace_training_results.csv")
