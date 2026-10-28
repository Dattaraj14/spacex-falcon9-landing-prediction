
from pathlib import Path

import pandas as pd


# Locate the project root automatically.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset_part_1.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "falcon9_cleaned.csv"


def load_data() -> pd.DataFrame:
    """Load the original Falcon 9 dataset."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    return pd.read_csv(RAW_DATA_PATH)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the dataset and create the landing target."""

    required_columns = [
        "FlightNumber",
        "Date",
        "PayloadMass",
        "Orbit",
        "LaunchSite",
        "Outcome",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    # Work on a copy so the original DataFrame is unchanged.
    cleaned = df.copy()

    # Remove duplicate records, if any.
    cleaned = cleaned.drop_duplicates()

    # Keep rows with known outcomes for the first model.
    # This excludes both confirmed failures and successes
    # with unknown outcomes, so we do not label unknowns as failures.
    known_outcome = cleaned["Outcome"].astype("string").str.startswith(
        ("True", "False"), na=False
    )
    cleaned = cleaned.loc[known_outcome].copy()

    # Create the binary target from confirmed outcomes only.
    cleaned["Class"] = (
        cleaned["Outcome"].astype("string").str.startswith("True")
    ).astype("int8")

    # Convert dates to a consistent datetime representation.
    cleaned["Date"] = pd.to_datetime(
        cleaned["Date"], errors="coerce"
    )

    # Report missing values rather than silently inventing data.
    return cleaned


def main() -> None:
    """Run the cleaning pipeline and save the result."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    original = load_data()
    cleaned = prepare_data(original)

    cleaned.to_csv(OUTPUT_PATH, index=False)

    print("Data preparation completed.")
    print(f"Original rows: {len(original)}")
    print(f"Rows retained: {len(cleaned)}")
    print(f"Rows excluded: {len(original) - len(cleaned)}")
    print("\nTarget class counts:")
    print(cleaned["Class"].value_counts().sort_index())
    print("\nMissing values in retained data:")
    print(cleaned.isna().sum()[cleaned.isna().sum() > 0])
    print(f"\nSaved dataset: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
