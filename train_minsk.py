
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Minsk2020_ALS_clean_for_training.csv")

label_col = "Diagnosis (ALS)"

# Separate labels from input features
y = df[label_col].astype(int)

excluded = ["ID", "Sex", "Age", label_col]
X = df.drop(
    columns=[c for c in excluded if c in df.columns],
    errors="ignore"
)

# Keep numeric voice features only
X = X.select_dtypes(include="number")

# Remove columns with no usable measurements
X = X.dropna(axis=1, how="all")

print("Participants:", len(df))
print("Voice features:", X.shape[1])
print("Labels:")
print(y.value_counts())

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "SVM": SVC(kernel="rbf", C=1),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    )
}

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro"
}

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(
            score_func=f_classif,
            k=min(30, X.shape[1])
        )),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
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

result_df = pd.DataFrame(results).sort_values(
    "Balanced Accuracy",
    ascending=False
)

result_df.to_csv("minsk_training_results.csv", index=False)

print("\nMinsk2020 cross-validation results:")
print(result_df.round(3).to_string(index=False))
print("\nSaved results to minsk_training_results.csv")
