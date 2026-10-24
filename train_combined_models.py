
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

BASE = Path(r"C:\NeuroTraceAIFinal")

# Load spreadsheet
clinical = pd.read_excel(
    BASE / "VOC-ALS.xlsx",
    sheet_name="VOC-ALS_Data",
    header=1
)

clinical.columns = clinical.columns.str.strip()
clinical = clinical.dropna(subset=["Category"])
clinical["ID"] = clinical["ID"].astype(str).str.strip()

# Exclude identifiers, demographics and clinical outcome fields
exclude = {
    "ID", "Category", "Age", "Sex",
    "ALSFRS-R", "FVC", "Disease duration",
    "Progression rate", "King stage"
}

clinical_features = []

for col in clinical.columns:
    if col in exclude:
        continue

    if not pd.api.types.is_numeric_dtype(clinical[col]):
        continue

    name = col.lower()

    if any(term in name for term in [
        "alsfrs", "fvc", "progression", "disease duration",
        "king stage", "diagnosis", "survival", "outcome"
    ]):
        continue

    clinical_features.append(col)

clinical_X = clinical.set_index("ID")[clinical_features]
clinical_X = clinical_X.replace([np.inf, -np.inf], np.nan)

# Load extracted audio features
audio = pd.read_csv(BASE / "audio_acoustic_features.csv")
audio["ParticipantID"] = audio["ParticipantID"].astype(str).str.strip()

audio_features = [
    "Duration_sec", "Pitch_mean", "Pitch_std", "Pitch_median",
    "RMS_mean", "RMS_std", "SpectralCentroid_mean",
    "SpectralCentroid_std", "ZeroCrossing_mean", "HNR_approx"
]

# Create one row per participant, with a separate set per task
wide = audio.pivot(
    index="ParticipantID",
    columns="Task",
    values=audio_features
)

wide.columns = [
    f"{task}_{feature}" for feature, task in wide.columns
]

wide = wide.replace([np.inf, -np.inf], np.nan)

# Confirm diagnoses agree
clinical_labels = clinical.set_index("ID")["Category"].astype(str).str.strip()
audio_labels = audio.groupby("ParticipantID")["Category"].first().astype(str).str.strip()

if not clinical_labels.reindex(wide.index).equals(
    audio_labels.reindex(wide.index)
):
    raise ValueError("Diagnosis labels do not match.")

# Align every dataset by participant ID
ids = sorted(set(clinical_X.index) & set(wide.index))

clinical_X = clinical_X.loc[ids]
audio_X = wide.loc[ids]
y = clinical_labels.loc[ids]

datasets = {
    "Spreadsheet only": clinical_X,
    "Audio only": audio_X,
    "Combined": pd.concat(
        [clinical_X, audio_X], axis=1
    )
}

print("Participants:", len(ids))
print("Spreadsheet features:", clinical_X.shape[1])
print("Audio features:", audio_X.shape[1])
print("Combined features:", datasets["Combined"].shape[1])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Random Forest": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42
        )
    ),
    "Extra Trees": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42
        )
    ),
    "SVM": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(class_weight="balanced")
    )
}

scoring = ["accuracy", "balanced_accuracy", "f1_macro"]
results = []

for dataset_name, X in datasets.items():
    for model_name, create_model in models.items():
        scores = cross_validate(
            create_model(),
            X,
            y,
            cv=cv,
            scoring=scoring
        )

        results.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Balanced Accuracy": scores["test_balanced_accuracy"].mean(),
            "Macro F1": scores["test_f1_macro"].mean()
        })

results_df = pd.DataFrame(results).sort_values(
    ["Macro F1", "Balanced Accuracy"],
    ascending=False
)

print("\nComparison results:")
print(results_df.round(3).to_string(index=False))

results_df.to_csv(
    BASE / "combined_model_comparison.csv",
    index=False
)

print("\nSaved combined_model_comparison.csv")
