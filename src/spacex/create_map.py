
from pathlib import Path

import folium
import pandas as pd
from folium.plugins import MarkerCluster
from xyzservices import providers


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset_part_1.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports"
OUTPUT_PATH = OUTPUT_DIR / "spacex_launch_map.html"


def main():
    # Verify that the dataset exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Check required columns
    required = ["Latitude", "Longitude", "LaunchSite"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean and validate coordinates
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()

    df = df[
        df["Latitude"].between(-90, 90)
        & df["Longitude"].between(-180, 180)
    ]

    if df.empty:
        raise ValueError("No valid launch coordinates found.")

    # Start with a geographic basemap that does not use OSM tiles
    launch_map = folium.Map(
        location=[
            df["Latitude"].mean(),
            df["Longitude"].mean(),
        ],
        zoom_start=4,
        tiles=providers.Esri.WorldGrayCanvas,
        control_scale=True,
    )

    # Group nearby launch markers
    marker_cluster = MarkerCluster(
        name="Falcon 9 Launch Records"
    ).add_to(launch_map)

    for _, row in df.iterrows():
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        site = str(row["LaunchSite"])

        folium.Marker(
            location=[lat, lon],
            tooltip=site,
            popup=folium.Popup(
                f"""
                <b>Launch Site:</b> {site}<br>
                <b>Latitude:</b> {lat:.5f}<br>
                <b>Longitude:</b> {lon:.5f}
                """,
                max_width=300,
            ),
        ).add_to(marker_cluster)

    # Fit the view to include all launch-site coordinates
    bounds = [
        [df["Latitude"].min(), df["Longitude"].min()],
        [df["Latitude"].max(), df["Longitude"].max()],
    ]
    launch_map.fit_bounds(bounds, padding=(30, 30))

    folium.LayerControl().add_to(launch_map)

    # Save the HTML map
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    launch_map.save(str(OUTPUT_PATH))

    print("Map created successfully.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Launch records mapped: {len(df)}")
    print(f"Unique launch sites: {df['LaunchSite'].nunique()}")


if __name__ == "__main__":
    main()
