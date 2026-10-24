
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier

df = pd.read_csv("Minsk2020_ALS_clean_for_training.csv")

y = df["Diagnosis (ALS)"].astype(int)

X = df.drop(
    columns=["ID", "Sex", "Age", "Diagnosis (ALS)"],
    errors="ignore"
).select_dtypes(include="number")

X = X.dropna(axis=1, how="all")

results_file = Path("overnight_continuous_results.csv")
log_file = Path("overnight_continuous_log.txt")

if results_file.exists():
    results = pd.read_csv(results_file)
    run_number = int(results["Run"].max()) + 1
else:
    results = pd.DataFrame()
    run_number = 1

print("Continuous training started:", datetime.now())
print("Press Ctrl+C to stop safely.")

try:
    while True:
        seed = 1000 + run_number

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("select", SelectKBest(
                score_func=f_classif,
                k=min(30, X.shape[1])
            )),
            ("model", ExtraTreesClassifier(
                n_estimators=300,
                random_state=seed,
                class_weight="balanced",
                n_jobs=-1
            ))
        ])

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed
        )

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=[
                "accuracy",
                "balanced_accuracy",
                "f1_macro"
            ],
            n_jobs=1
        )

        row = {
            "Run": run_number,
            "Seed": seed,
            "Time": datetime.now().isoformat(timespec="seconds"),
            "Accuracy": scores["test_accuracy"].mean(),
            "Balanced_Accuracy":
                scores["test_balanced_accuracy"].mean(),
            "Macro_F1": scores["test_f1_macro"].mean()
        }

        results = pd.concat(
            [results, pd.DataFrame([row])],
            ignore_index=True
        )

        results.to_csv(results_file, index=False)

        message = (
            f"Run {run_number} | "
            f"Accuracy: {row['Accuracy']:.3f} | "
            f"Balanced Accuracy: "
            f"{row['Balanced_Accuracy']:.3f} | "
            f"Macro F1: {row['Macro_F1']:.3f}"
        )

        print(message, flush=True)

        with open(log_file, "a", encoding="utf-8") as log:
            log.write(
                f"{datetime.now().isoformat()} | "
                f"{message}\n"
            )

        run_number += 1

except KeyboardInterrupt:
    print("\nTraining stopped by you.")

finally:
    print("Completed results are saved in:", results_file)
    print("Log saved in:", log_file)
