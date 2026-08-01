import json
import os

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Churn Prediction Service",
    description="Predict customer churn for telecom/subscription businesses.",
    version="1.0.0",
)

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODERS_PATH = "models/label_encoders.pkl"
FEATURES_PATH = "models/feature_columns.json"

model = None
scaler = None
label_encoders = {}
feature_columns = []


def load_artifacts():
    global model, scaler, label_encoders, feature_columns

    if not os.path.exists(MODEL_PATH):
        return

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)

    with open(FEATURES_PATH, encoding="utf-8") as file:
        feature_columns = json.load(file)


load_artifacts()


class ChurnInput(BaseModel):
    gender: str = Field(example="Female")
    SeniorCitizen: int = Field(example=0)
    Partner: str = Field(example="Yes")
    Dependents: str = Field(example="No")
    tenure: int = Field(example=12)
    PhoneService: str = Field(example="Yes")
    MultipleLines: str = Field(example="No")
    InternetService: str = Field(example="Fiber optic")
    OnlineSecurity: str = Field(example="No")
    OnlineBackup: str = Field(example="Yes")
    DeviceProtection: str = Field(example="No")
    TechSupport: str = Field(example="No")
    StreamingTV: str = Field(example="No")
    StreamingMovies: str = Field(example="No")
    Contract: str = Field(example="Month-to-month")
    PaperlessBilling: str = Field(example="Yes")
    PaymentMethod: str = Field(example="Electronic check")
    MonthlyCharges: float = Field(example=70.5)
    TotalCharges: float = Field(example=840.0)


def prepare_features(raw_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw_input])

    for column, encoder in label_encoders.items():
        if column in df.columns:
            value = df.at[0, column]
            if value not in encoder.classes_:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid value '{value}' for field '{column}'.",
                )
            df.at[0, column] = encoder.transform([value])[0]

    for column in feature_columns:
        if column not in df.columns:
            df[column] = 0

    ordered = df[feature_columns]
    scaled = scaler.transform(ordered)
    return pd.DataFrame(scaled, columns=feature_columns)


@app.get("/")
def read_root():
    return {
        "status": "API is online",
        "model_loaded": model is not None,
        "message": "Go to /docs for testing",
    }


@app.get("/health")
def health_check():
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run the training pipeline first.",
        )
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: ChurnInput):
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run the training pipeline first.",
        )

    features = prepare_features(data.model_dump())
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return {
        "churn_probability": round(probability, 4),
        "prediction": "Yes" if prediction == 1 else "No",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
