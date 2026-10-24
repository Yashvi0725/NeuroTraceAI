
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    roc_auc_score
)

folder = Path(r"C:\NeuroTraceAIFinal")

data = pd.read_csv(
    folder / "Minsk2020_ALS_clean_for_training.csv"
)

saved = joblib.load(
    folder / "minsk_tuned_random_forest.joblib"
)

target = saved["target_column"]

y = pd.to_numeric(data[target], errors="coerce")
valid = y.notna()
data = data.loc[valid].copy()
y = y.loc[valid].astype(int)

X = data[saved["input_features"]]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=123
)

model = saved["pipeline"]

print("Generating out-of-fold predictions...")

predictions = cross_val_predict(
    clone(model),
    X,
    y,
    cv=cv,
    method="predict",
    n_jobs=-1
)

probabilities = cross_val_predict(
    clone(model),
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

cm = confusion_matrix(y, predictions, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn) if (tp + fn) else 0
specificity = tn / (tn + fp) if (tn + fp) else 0

print("\nCONFUSION MATRIX")
print("Rows = actual labels; columns = predicted labels")
print("                 Predicted healthy   Predicted ALS")
print(f"Actual healthy          {tn:3d}               {fp:3d}")
print(f"Actual ALS              {fn:3d}               {tp:3d}")

print("\nSENSITIVITY:", round(sensitivity, 3))
print("SPECIFICITY:", round(specificity, 3))
print(
    "BALANCED ACCURACY:",
    round(balanced_accuracy_score(y, predictions), 3)
)
print("ROC-AUC:", round(roc_auc_score(y, probabilities), 3))

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["Healthy control", "ALS label"],
        zero_division=0
    )
)

print("\nResearch evaluation complete.")
