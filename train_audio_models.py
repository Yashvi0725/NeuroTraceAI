
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

BASE = Path(r"C:\NeuroTraceAIFinal")

df = pd.read_csv(BASE / "audio_acoustic_features.csv")

# Features extracted from audio
audio_features = [
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

# Confirm each participant has only one diagnosis label
label_counts = df.groupby("ParticipantID")["Category"].nunique()

if (label_counts > 1).any():
    raise ValueError("Some participants have conflicting labels.")

labels = df.groupby("ParticipantID")["Category"].first()

# Create separate feature columns for each recording task
wide = df.pivot(
    index="ParticipantID",
    columns="Task",
    values=audio_features
)

wide.columns = [
    f"{task}_{feature}"
    for feature, task in wide.columns
]

X = wide
y = labels.reindex(X.index)

X = X.replace([np.inf, -np.inf], np.nan)

print("Participants:", len(X))
print("Audio features:", X.shape[1])
print("\nDiagnosis counts:")
print(y.value_counts())

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42
        )
    ),

    "Extra Trees": make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42
        )
    ),

    "SVM": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(class_weight="balanced")
    ),

    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000
        )
    )
}

scoring = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro"
]

results = []

for name, model in models.items():
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    results.append({
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Balanced Accuracy": scores["test_balanced_accuracy"].mean(),
        "Macro F1": scores["test_f1_macro"].mean()
    })

results_df = pd.DataFrame(results).sort_values(
    "Macro F1",
    ascending=False
)

print("\nAudio-only cross-validation results:")
print(results_df.round(3).to_string(index=False))

results_df.to_csv(
    BASE / "audio_model_comparison.csv",
    index=False
)

print("\nSaved audio_model_comparison.csv")
