
# SpaceX Falcon 9 Landing Prediction

A machine learning project that explores historical SpaceX Falcon 9 launch data, analyzes landing outcomes, and builds a baseline model to predict whether a landing will be successful.

The project includes data inspection, data cleaning, exploratory data analysis, model evaluation, an interactive launch-site map, and a dashboard.

## Project Objectives

- Explore historical Falcon 9 launch records.
- Analyze landing outcomes and success rates by launch site.
- Prepare a dataset for supervised machine learning.
- Train a Decision Tree classification model.
- Evaluate model performance using cross-validation.
- Visualize launch records on an interactive map.
- Present findings through an interactive dashboard.

## Technology Stack

- Python
- pandas
- scikit-learn
- Matplotlib and Seaborn
- Folium
- Dash and Plotly

## Repository Structure

```text
spacex-falcon9-landing-prediction/
├── data/
│   ├── raw/
│   │   ├── dataset_part_1.csv
│   │   ├── dataset_part_2.csv
│   │   └── dataset_part_3.csv
│   └── processed/
│       └── falcon9_cleaned.csv
├── reports/
│   ├── figures/
│   │   ├── landing_outcomes.png
│   │   ├── success_by_site.png
│   │   └── payload_vs_outcome.png
│   └── spacex_launch_map.html
├── src/
│   └── spacex/
│       ├── inspect_data.py
│       ├── clean_data.py
│       ├── eda.py
│       ├── train_model.py
│       ├── evaluate_model.py
│       ├── create_map.py
│       └── dashboard.py
├── requirements.txt
├── .gitignore
├── README.md
└── codes
```

Generated files such as the processed CSV, charts, and HTML map are created by the corresponding scripts. They may not exist until the relevant steps have been run.

## Dataset

The project uses three CSV files containing historical SpaceX launch information.

The dataset includes fields such as:

- Flight number and launch date
- Booster version
- Payload mass and orbit
- Launch site
- Landing outcome
- Reuse-related features
- Launch-site latitude and longitude

The cleaning pipeline starts with 90 records and retains 69 records with confirmed `True` or `False` outcomes. The remaining 21 records have unknown or unconfirmed outcomes and are excluded from the current model dataset.

The resulting target variable is `Class`:

- `1`: Successful landing
- `0`: Unsuccessful landing

The cleaned dataset contains 60 successful outcomes and 9 unsuccessful outcomes.

## Installation

Python 3.11 or a compatible Python version is recommended.

Clone the repository:

```bash
git clone https://github.com/Dattaraj14/spacex-falcon9-landing-prediction.git
cd spacex-falcon9-landing-prediction
```

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Running the Project

Run commands from the repository root.

### 1. Inspect the datasets

```powershell
python src/spacex/inspect_data.py
```

### 2. Clean the data

```powershell
python src/spacex/clean_data.py
```

This creates `data/processed/falcon9_cleaned.csv`.

### 3. Generate exploratory visualizations

```powershell
python src/spacex/eda.py
```

The generated charts are saved under `reports/figures/`.

### 4. Train the prediction model

```powershell
python src/spacex/train_model.py
```

The script trains a Decision Tree classifier using a preprocessing pipeline for numeric and categorical features.

### 5. Evaluate the model

```powershell
python src/spacex/evaluate_model.py
```

This compares the model against a majority-class baseline using stratified cross-validation.

### 6. Generate the interactive map

```powershell
python src/spacex/create_map.py
```

Open `reports/spacex_launch_map.html` in a browser. The map uses an external Esri basemap, so an internet connection is required for the geographic tiles to load.

The markers represent recorded launch-site coordinates, not actual booster landing locations.

### 7. Launch the dashboard

```powershell
python src/spacex/dashboard.py
```

Open the following URL in your browser:

http://127.0.0.1:8050/

The dashboard includes launch-site filtering and interactive visualizations. Stop the local server with `Ctrl+C`.

## Model Evaluation

The current dataset contains 69 labelled records and is imbalanced, with substantially more successful than unsuccessful outcomes.

Previously observed results:

| Metric | Result |
|---|---:|
| Test-set accuracy | 88.9% |
| Cross-validation accuracy | 81.2% |
| Cross-validation balanced accuracy | 60.8% |
| Cross-validation failure recall | 33.3% |

The test-set result is based on only 18 records. Cross-validation estimates also have substantial uncertainty because the dataset contains only nine unsuccessful outcomes.

These results should therefore be treated as preliminary baseline performance, not as evidence of a production-ready landing prediction system.

## Limitations

- The dataset is relatively small.
- Successful and unsuccessful outcomes are imbalanced.
- Performance estimates are uncertain, particularly for unsuccessful landings.
- The current map displays launch-site coordinates rather than landing locations.
- The interactive map and dashboard run locally and are not automatically deployed online.

## Future Improvements

- Evaluate additional classification algorithms.
- Improve performance on the minority class.
- Use more historical launch records and relevant features.
- Tune model hyperparameters using appropriate validation.
- Add automated tests and reproducible experiment tracking.
- Deploy the dashboard for public access.

## Disclaimer

This is an educational machine learning project using historical launch data. Its predictions are experimental and should not be used for operational aerospace decisions.
