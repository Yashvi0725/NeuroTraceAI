
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

PROJECT = Path(r"C:\NeuroTraceAIFinal")

data = pd.read_csv(
    PROJECT / "voc_participant_features.csv"
)

y = data["Category"].map({"HC": 0, "ALS": 1})

all_features = [
    col for col in data.columns
    if "__" in col
]

tasks = [
    "phonationA",
    "phonationE",
    "phonationI",
    "phonationO",
    "phonationU",
    "rhythmPA",
    "rhythmTA",
    "rhythmKA"
]

feature_sets = {
    task: [
        col for col in all_features
        if col.startswith(task + "__")
    ]
    for task in tasks
}

feature_sets["ALL_TASKS"] = all_features

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    ),
    "Random Forest": __import__(
        "sklearn.ensemble",
        fromlist=["RandomForestClassifier"]
    ).RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(class_weight="balanced", random_state=42)
    )
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

results = []

for set_name, columns in feature_sets.items():
    X = data[columns]

    for model_name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        results.append({
            "Feature_Set": set_name,
            "Model": model_name,
            "Feature_Count": len(columns),
            "Accuracy": scores["test_accuracy"].mean(),
            "Balanced_Accuracy":
                scores["test_balanced_accuracy"].mean(),
            "F1": scores["test_f1"].mean(),
            "ROC_AUC": scores["test_roc_auc"].mean()
        })

result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    ["Balanced_Accuracy", "ROC_AUC"],
    ascending=False
)

result_df.to_csv(
    PROJECT / "voc_task_comparison.csv",
    index=False
)

print(result_df.round(3).to_string(index=False))

print("\nSaved:")
print(PROJECT / "voc_task_comparison.csv")
