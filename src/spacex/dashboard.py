
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "falcon9_cleaned.csv"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Cleaned data not found. Run clean_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    df["LandingOutcome"] = df["Class"].map({
        0: "Unsuccessful",
        1: "Successful",
    })

    sites = sorted(df["LaunchSite"].dropna().unique())

    app = Dash(__name__)

    app.layout = html.Div(
        [
            html.H1("SpaceX Falcon 9 Landing Analysis"),
            html.P(
                "Explore historical landing outcomes by launch site."
            ),
            dcc.Dropdown(
                id="site-dropdown",
                options=[
                    {"label": "All Sites", "value": "ALL"}
                ] + [
                    {"label": site, "value": site}
                    for site in sites
                ],
                value="ALL",
                clearable=False,
            ),
            dcc.Graph(id="outcome-chart"),
            dcc.Graph(id="payload-chart"),
        ],
        style={
            "maxWidth": "1100px",
            "margin": "30px auto",
            "fontFamily": "Arial",
        },
    )

    @app.callback(
        Output("outcome-chart", "figure"),
        Output("payload-chart", "figure"),
        Input("site-dropdown", "value"),
    )
    def update_charts(selected_site):
        filtered = df.copy()

        if selected_site != "ALL":
            filtered = filtered[
                filtered["LaunchSite"] == selected_site
            ]

        outcome_counts = (
            filtered["LandingOutcome"]
            .value_counts()
            .reindex(["Successful", "Unsuccessful"], fill_value=0)
            .rename_axis("Landing Outcome")
            .reset_index(name="Number of Launches")
        )

        outcome_fig = px.bar(
            outcome_counts,
            x="Landing Outcome",
            y="Number of Launches",
            color="Landing Outcome",
            title="Landing Outcomes",
        )

        payload_fig = px.box(
            filtered,
            x="LandingOutcome",
            y="PayloadMass",
            color="LandingOutcome",
            title="Payload Mass by Landing Outcome",
            labels={
                "LandingOutcome": "Landing Outcome",
                "PayloadMass": "Payload Mass (kg)",
            },
        )

        return outcome_fig, payload_fig

    app.run(debug=False)


if __name__ == "__main__":
    main()
