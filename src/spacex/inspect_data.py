
from pathlib import Path

import pandas as pd


# Find the project root from this file's location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def inspect_dataset(filename: str) -> None:
    """Print a basic report about a CSV dataset."""

    file_path = RAW_DATA_DIR / filename

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    print("\n" + "=" * 60)
    print(f"Dataset: {filename}")
    print("=" * 60)

    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string())

    print("\nFirst five records:")
    print(df.head().to_string(index=False))


def main() -> None:
    """Inspect all three downloaded datasets."""

    datasets = [
        "dataset_part_1.csv",
        "dataset_part_2.csv",
        "dataset_part_3.csv",
    ]

    for filename in datasets:
        inspect_dataset(filename)


if __name__ == "__main__":
    main()
