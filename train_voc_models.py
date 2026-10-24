
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

PROJECT = Path(r"C:\NeuroTraceAIFinal")

features = pd.read_csv(PROJECT / "audio_acoustic_features.csv")
labels = pd.read_csv(PROJECT / "participant_labels.csv")

# Check that each participant-task combination is unique
if features.duplicated(["ParticipantID", "Task"]).any():
    raise ValueError("Duplicate participant-task recordings found.")

# Select acoustic features
feature_cols = [
    "Duration_sec",
    "Pitch_mean",
    "Pitch_std",
    "Pitch_median",
    "RMS_mean",
    "RMS_std",
    "SpectralCentroid_mean",
    "SpectralCentroid_std",
    "ZeroCrossing_mean",
    "HNR_approx"
]

# Convert eight recordings into one row per participant
wide = features.set_index(
    ["ParticipantID", "Task"]
)[feature_cols].unstack("Task")

wide.columns = [
    f"{task}__{feature}"
    for feature, task in wide.columns
]

wide = wide.reset_index()

# Add participant diagnosis
data = wide.merge(
    labels,
    left_on="ParticipantID",
    right_on="ID",
    how="left",
    validate="one_to_one"
)

if data["Category"].isna().any():
    raise ValueError("Some participants have no diagnosis label.")

data = data.drop(columns=["ID"])
data.to_csv(PROJECT / "voc_participant_features.csv", index=False)

print("Participant-level dataset created.")
print("Rows:", len(data))
print("Feature columns:", len(data.columns) - 2)
print("Diagnosis counts:")
print(data["Category"].value_counts())

# ALS = 1, healthy control = 0
X = data.drop(
    columns=["ParticipantID", "Category"]
)
y = data["Category"].map({"HC": 0, "ALS": 1})

if y.isna().any():
    raise ValueError("Unexpected diagnosis category found.")

# Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(
            class_weight="balanced",
            probability=True,
            random_state=42
        )
    )
}

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

results = []
predictions = pd.DataFrame({
    "ParticipantID": data["ParticipantID"],
    "Actual": data["Category"]
})

for name, model in models.items():
    print("\nEvaluating:", name)

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

    # Each participant receives a prediction from a model
    # that did not train on that participant.
    pred = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict",
        n_jobs=-1
    )

    predictions[name] = pd.Series(pred).map({
        0: "HC",
        1: "ALS"
    })

    cm = confusion_matrix(y, pred, labels=[0, 1])

    print("Confusion matrix (rows=actual, columns=predicted)")
    print("Order: HC, ALS")
    print(cm)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(
    PROJECT / "voc_model_comparison.csv",
    index=False
)

predictions.to_csv(
    PROJECT / "voc_oof_predictions.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(
    results_df[
        ["Model", "accuracy_mean", "balanced_accuracy_mean",
        "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean"]
    ].round(3).to_string(index=False)
)

print("\nSaved files:")
print(PROJECT / "voc_participant_features.csv")
print(PROJECT / "voc_model_comparison.csv")
print(PROJECT / "voc_oof_predictions.csv")
print("\nTraining and evaluation complete.")
