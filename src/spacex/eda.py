
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "falcon9_cleaned.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found. Run clean_data.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    sns.set_theme(style="whitegrid")

    # Chart 1: Distribution of landing outcomes
    outcome_counts = (
        df["Class"]
        .map({0: "Unsuccessful", 1: "Successful"})
        .value_counts()
        .reindex(["Successful", "Unsuccessful"], fill_value=0)
    )

    plt.figure(figsize=(7, 5))
    sns.barplot(
        x=outcome_counts.index,
        y=outcome_counts.values,
        hue=outcome_counts.index,
        legend=False,
    )
    plt.title("Falcon 9 Landing Outcome Distribution")
    plt.xlabel("Landing Outcome")
    plt.ylabel("Number of Launches")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "landing_outcomes.png", dpi=150)
    plt.close()

    # Chart 2: Success rate by launch site
    site_rates = (
        df.groupby("LaunchSite", as_index=False)
        .agg(
            total_launches=("Class", "size"),
            success_rate=("Class", "mean"),
        )
        .sort_values("success_rate", ascending=False)
    )
    site_rates["success_rate_percent"] = site_rates["success_rate"] * 100

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=site_rates,
        x="success_rate_percent",
        y="LaunchSite",
        hue="LaunchSite",
        legend=False,
    )
    plt.title("Landing Success Rate by Launch Site")
    plt.xlabel("Success Rate (%)")
    plt.ylabel("Launch Site")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "success_by_site.png", dpi=150)
    plt.close()

    # Chart 3: Payload mass versus landing outcome
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Class",
        y="PayloadMass",
        hue="Class",
    )
    plt.title("Payload Mass by Landing Outcome")
    plt.xlabel("Landing Outcome (0 = Unsuccessful, 1 = Successful)")
    plt.ylabel("Payload Mass (kg)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "payload_vs_outcome.png", dpi=150)
    plt.close()

    print("EDA completed successfully.")
    print(f"Figures saved in: {OUTPUT_DIR}")
    print("\nSuccess rate by launch site:")
    print(
        site_rates[
            ["LaunchSite", "total_launches", "success_rate_percent"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
