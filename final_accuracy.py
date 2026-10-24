
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load the cleaned Minsk2020 dataset
file_path = Path("Minsk2020_ALS_clean_for_training.csv")

if not file_path.exists():
    raise FileNotFoundError(
        "Minsk2020_ALS_clean_for_training.csv was not found."
    )

df = pd.read_csv(file_path)

# Separate labels and features
y = df["Diagnosis (ALS)"].astype(int)

X = df.drop(
    columns=["ID", "Sex", "Age", "Diagnosis (ALS)"],
    errors="ignore"
).select_dtypes(include="number")

# Remove columns containing no usable values
X = X.dropna(axis=1, how="all")

print("Dataset:", file_path.name)
print("Participants:", len(df))
print("Voice features:", X.shape[1])
print("\nClass counts:")
print(y.value_counts().sort_index())

# Model pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("select", SelectKBest(
        score_func=f_classif,
        k=min(30, X.shape[1])
    )),
    ("model", ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Generate one out-of-fold prediction for each participant
predictions = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    n_jobs=1
)

# Calculate metrics
accuracy = accuracy_score(y, predictions)
balanced_accuracy = balanced_accuracy_score(y, predictions)
precision = precision_score(y, predictions, pos_label=1, zero_division=0)
recall = recall_score(y, predictions, pos_label=1, zero_division=0)
f1 = f1_score(y, predictions, pos_label=1, zero_division=0)
macro_f1 = f1_score(y, predictions, average="macro", zero_division=0)

print("\n========== FINAL EVALUATION ==========")
print(f"Accuracy:           {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Balanced Accuracy:  {balanced_accuracy:.4f} ({balanced_accuracy * 100:.2f}%)")
print(f"ALS Precision:      {precision:.4f}")
print(f"ALS Recall:         {recall:.4f}")
print(f"ALS F1-score:       {f1:.4f}")
print(f"Macro F1-score:     {macro_f1:.4f}")

print("\nConfusion Matrix")
print("Rows = actual [Healthy, ALS]")
print("Columns = predicted [Healthy, ALS]")
print(confusion_matrix(y, predictions, labels=[0, 1]))

print("\nClassification Report")
print(classification_report(
    y,
    predictions,
    labels=[0, 1],
    target_names=["Healthy control", "ALS"],
    zero_division=0
))

# Save predictions for later analysis
output = df[["ID"]].copy() if "ID" in df.columns else pd.DataFrame()
output["Actual"] = y.to_numpy()
output["Predicted"] = predictions

output.to_csv("final_accuracy_predictions.csv", index=False)

print("\nPredictions saved to final_accuracy_predictions.csv")
