# Warehouse Picking Time Prediction

## Overview
This project predicts warehouse picking time with machine learning models using a historical picking dataset.

The main script trains and evaluates four regression models, then saves metrics and plots into the `Outputs/` folder.

## Repository layout
From the repository root, the relevant project folder is:

```bash
src/nhatnguyen_gradproject/Warehouse_Picking_Time_Project
```

Inside that folder, the main files are:
- `main.py` — training, prediction, and visualization logic
- `requirements.txt` — required Python packages
- `README.md` — project instructions
- `data/picking_time_data.csv` — input dataset
- `Outputs/` — generated results and plots

## Input data
The script loads the dataset from:

```bash
src/nhatnguyen_gradproject/Warehouse_Picking_Time_Project/data/picking_time_data.csv
```

If the file is missing, add a CSV file with the same name into the `data/` folder.

## Models
The script trains these models:
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- MLP Regressor (neural network)

## Requirements
1. Python 3.8 or newer
2. A virtual environment is recommended
3. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Setup
From the project folder:

```bash
cd src/nhatnguyen_gradproject/Warehouse_Picking_Time_Project
python -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/Scripts/activate   # Windows
# or
source .venv/bin/activate       # macOS / Linux
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

## Run
Run the main script from the same folder:

```bash
python main.py
```

If you want to use a different input file, update the `source_path` value in `main.py` to point to your CSV.

## Outputs
After running, check the `Outputs/` folder for generated files:

- `Task1_dataset_preview.csv` — dataset preview
- `Task2_training_curve.png` — MLP training loss curve
- `Task3_prediction_vs_actual.png` — best model actual vs predicted plot
- `Task4_model_comparison.csv` — model performance table
- `Task5_model_comparison.png` — RMSE comparison chart
- `Task6_residual_plot.png` — residual plot for the best model
- `Task7_feature_importance.png` — random forest feature importance
- `Task8_all_models_prediction_grid.png` — combined actual vs predicted grid for all four models

## First steps for a new user
1. Clone the repository.
2. Open `src/nhatnguyen_gradproject/Warehouse_Picking_Time_Project`.
3. Verify the file `data/picking_time_data.csv` exists.
4. Create and activate a virtual environment.
5. Install dependencies and run `python main.py`.
6. Inspect the generated outputs in the `Outputs/` folder.

## Notes
- `main.py` scales features for Ridge and MLP models automatically.
- The best model is selected by lowest RMSE and its plots are saved, but the model itself is not serialized.
- To add or change models, modify the `models` list in `main.py` and update any visualization logic if needed.