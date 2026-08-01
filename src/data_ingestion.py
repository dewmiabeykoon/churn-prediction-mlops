import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-ibm-cloud/"
    "master/data/Telco-Customer-Churn.csv"
)
OUTPUT_PATH = "data/raw/dataset.csv"


def ingest_data():
    os.makedirs("data/raw", exist_ok=True)

    try:
        logging.info("Downloading Telco Customer Churn dataset...")
        df = pd.read_csv(DATASET_URL)
        df.to_csv(OUTPUT_PATH, index=False)
        logging.info("Data ingestion successful. Saved to %s. Shape: %s", OUTPUT_PATH, df.shape)
    except Exception as exc:
        logging.error("Error during ingestion: %s", exc)
        raise


if __name__ == "__main__":
    ingest_data()
