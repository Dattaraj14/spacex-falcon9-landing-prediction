
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "falcon9_cleaned.csv"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Cleaned data not found. Run clean_data.py first."
        )

    return pd.read_csv(DATA_PATH)


def build_features(df):
    feature_columns = [
        "FlightNumber",
        "PayloadMass",
        "Orbit",
        "LaunchSite",
        "Flights",
        "GridFins",
        "Reused",
        "Legs",
        "Block",
        "ReusedCount",
    ]

    X = df[feature_columns].copy()
    y = df["Class"].astype(int).copy()

    numeric_features = [
        "FlightNumber",
        "PayloadMass",
        "Flights",
        "Block",
        "ReusedCount",
    ]

    categorical_features = [
        "Orbit",
        "LaunchSite",
        "GridFins",
        "Reused",
        "Legs",
    ]

    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in categorical_features:
        X[col] = X[col].astype("string")

    return X, y, numeric_features, categorical_features


def build_decision_tree(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            DecisionTreeClassifier(
                max_depth=3,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def main():
    df = load_data()
    X, y, numeric_features, categorical_features = build_features(df)

    # There are only 9 unsuccessful records, so use 3 folds.
    # Each validation fold should contain 3 unsuccessful records.
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    models = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Decision Tree": build_decision_tree(
            numeric_features,
            categorical_features,
        ),
    }

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "failure_recall": make_scorer(
            recall_score,
            pos_label=0,
            zero_division=0,
        ),
    }

    print(f"Records: {len(df)}")
    print(f"Class counts:\n{y.value_counts().sort_index()}")
    print("\nThree-fold stratified cross-validation")
    print("Failure recall measures how often class 0 is detected.")

    for name, model in models.items():
        results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            error_score="raise",
        )

        print(f"\n{name}")
        print("-" * len(name))

        for metric in scoring:
            scores = results[f"test_{metric}"]
            print(
                f"{metric}: "
                f"{scores.mean():.3f} "
                f"+/- {scores.std():.3f}"
            )


if __name__ == "__main__":
    main()
