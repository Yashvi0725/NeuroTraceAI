
import pandas as pd
from datetime import datetime
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

results_file = "overnight_minsk_results.csv"
log_file = "overnight_minsk_log.txt"

all_results = []

with open(log_file, "a", encoding="utf-8") as log:
    log.write("\nStarted: " + str(datetime.now()) + "\n")

    for run in range(1, 51):
        seed = 100 + run

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

        result = {
            "Run": run,
            "Seed": seed,
            "Accuracy": scores["test_accuracy"].mean(),
            "Balanced_Accuracy":
                scores["test_balanced_accuracy"].mean(),
            "Macro_F1": scores["test_f1_macro"].mean()
        }

        all_results.append(result)

        pd.DataFrame(all_results).to_csv(
            results_file, index=False
        )

        message = (
            f"Run {run}/50 | "
            f"Accuracy: {result['Accuracy']:.3f} | "
            f"Balanced Accuracy: "
            f"{result['Balanced_Accuracy']:.3f} | "
            f"Macro F1: {result['Macro_F1']:.3f}"
        )

        print(message)
        log.write(message + "\n")
        log.flush()

print("\nAll 50 experiments completed.")
print("Results saved to", results_file)
print("Log saved to", log_file)
