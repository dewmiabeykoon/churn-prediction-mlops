import pandas as pd
import joblib
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
)

def evaluate_model():
    # 1. Load Data and Model
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    model = joblib.load("models/model.pkl")

    # 2. Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probability for ROC-AUC

    # 3. Calculate Metrics (Assignment requirements)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    mlflow.set_experiment("Churn_Prediction")

    with mlflow.start_run(run_name="Evaluation"):
        # Log Metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # --- ARTIFACT 1: Confusion Matrix ---
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix (Acc: {acc:.4f})")
        cm_path = "models/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # --- ARTIFACT 2: ROC Curve (Assignment requirement) ---
        plt.figure(figsize=(6,4))
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title(f"ROC Curve (AUC: {auc:.4f})")
        roc_path = "models/roc_curve.png"
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(roc_path)
        
        # Save metrics text file
        with open("metrics.txt", "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall: {rec:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"ROC-AUC: {auc:.4f}\n\n")
            f.write(classification_report(y_test, y_pred))

    print(f" Evaluation complete. ROC-AUC: {auc:.4f}, Accuracy: {acc:.4f}")

if __name__ == "__main__":
    evaluate_model()