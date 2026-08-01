import json
import logging
import os

import joblib
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_PATH = "models/model.pkl"
METADATA_PATH = "models/model_metadata.json"


def register_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run training first.")

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, encoding="utf-8") as file:
        metadata = json.load(file)

    mlflow.set_experiment("Churn_Prediction")
    with mlflow.start_run(run_name="Model_Registration"):
        mlflow.log_param("selected_model", metadata["model_name"])
        mlflow.log_metric("f1_score", metadata["f1_score"])
        mlflow.sklearn.log_model(model, "production_model")

    logging.info(
        "Production model logged to MLflow (%s, F1=%s)",
        metadata["model_name"],
        metadata["f1_score"],
    )


if __name__ == "__main__":
    register_model()
