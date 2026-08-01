import json
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_PATH = "data/raw/dataset.csv"
REQUIRED_COLUMNS = {
    "customerID",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}


def validate_data():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Raw dataset not found at {INPUT_PATH}. Run data_ingestion first.")

    df = pd.read_csv(INPUT_PATH)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    logging.info("Validation passed. Rows: %s, Columns: %s", df.shape[0], df.shape[1])


if __name__ == "__main__":
    validate_data()
