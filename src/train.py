import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier


def train_models():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }

    mlflow.set_experiment("Churn_Prediction")
    os.makedirs("models", exist_ok=True)

    best_model = None
    best_name = None
    best_f1 = -1.0

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            f1 = f1_score(y_test, predictions)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(model, name)

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_name = name

    joblib.dump(best_model, "models/model.pkl")

    metadata = {"model_name": best_name, "f1_score": round(best_f1, 4)}
    with open("models/model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Best model ({best_name}) saved to models/model.pkl with F1={best_f1:.4f}")


if __name__ == "__main__":
    train_models()
