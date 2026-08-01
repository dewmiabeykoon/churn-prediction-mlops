import json
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data():
    input_path = "data/raw/dataset.csv"
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(input_path)

    df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

    encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        if col != "customerID":
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            encoders[col] = encoder

    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(encoders, "models/label_encoders.pkl")
    with open("models/feature_columns.json", "w", encoding="utf-8") as file:
        json.dump(list(X.columns), file)

    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(f"{output_dir}/X_train.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print("Preprocessing complete. Scaler and feature columns saved to models/")


if __name__ == "__main__":
    preprocess_data()
