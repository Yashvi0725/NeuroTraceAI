
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score
)

df = pd.read_csv("Minsk2020_ALS_clean_for_training.csv")

y = df["Diagnosis (ALS)"].astype(int)

X = df.drop(
    columns=["ID", "Sex", "Age", "Diagnosis (ALS)"],
    errors="ignore"
).select_dtypes(include="number")

X = X.dropna(axis=1, how="all")

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("select", SelectKBest(
        score_func=f_classif,
        k=min(30, X.shape[1])
    )),
    ("model", ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

predictions = cross_val_predict(model, X, y, cv=cv)

print("Accuracy:", round(accuracy_score(y, predictions), 3))
print("Balanced accuracy:", round(
    balanced_accuracy_score(y, predictions), 3
))

print("\nConfusion matrix:")
print("Rows = actual [0, 1]; columns = predicted [0, 1]")
print(confusion_matrix(y, predictions, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y,
    predictions,
    labels=[0, 1],
    target_names=["Healthy control", "ALS"],
    zero_division=0
))
