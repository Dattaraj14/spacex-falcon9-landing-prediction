
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "falcon9_cleaned.csv"


# --------------------------------------------------
# 2. Load the cleaned dataset
# --------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load data prepared by clean_data.py."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}\n"
            "Run clean_data.py first."
        )

    return pd.read_csv(DATA_PATH)


# --------------------------------------------------
# 3. Prepare input features and target
# --------------------------------------------------

def prepare_features(df: pd.DataFrame):
    """Separate model inputs (X) from target labels (y)."""

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

    missing_columns = [
        col for col in feature_columns + ["Class"]
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    # X contains information used to make predictions.
    X = df[feature_columns].copy()

    # y contains the answer the model must learn to predict.
    y = df["Class"].copy()

    # Convert date-derived/numeric values to consistent types.
    numeric_features = [
        "FlightNumber",
        "PayloadMass",
        "Flights",
        "Block",
        "ReusedCount",
    ]

    for column in numeric_features:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    categorical_features = [
        "Orbit",
        "LaunchSite",
        "GridFins",
        "Reused",
        "Legs",
    ]

    for column in categorical_features:
        X[column] = X[column].astype("string").fillna(pd.NA)

    return X, y, numeric_features, categorical_features


# --------------------------------------------------
# 4. Build preprocessing and model pipeline
# --------------------------------------------------

def build_pipeline(numeric_features, categorical_features):
    """Handle missing values, encode categories, and train a tree."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    model = DecisionTreeClassifier(
        max_depth=3,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# --------------------------------------------------
# 5. Train and evaluate
# --------------------------------------------------

def main():
    df = load_data()

    X, y, numeric_features, categorical_features = (
        prepare_features(df)
    )

    print("Dataset loaded successfully.")
    print(f"Total labelled records: {len(df)}")
    print(f"Number of input features before encoding: {X.shape[1]}")
    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    # Keep the class proportions similar in both subsets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline(
        numeric_features,
        categorical_features,
    )

    # Learn preprocessing and model parameters from training data.
    pipeline.fit(X_train, y_train)

    # Evaluate on records withheld from training.
    y_pred = pipeline.predict(X_test)

    print("\nEvaluation results")
    print("------------------")
    print(f"Training records: {len(X_train)}")
    print(f"Test records: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y_test, y_pred):.3f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Unsuccessful", "Successful"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


if __name__ == "__main__":
    main()
