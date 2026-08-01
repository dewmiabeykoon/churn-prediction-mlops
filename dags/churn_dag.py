import json
import subprocess
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "dewmiabeykoon",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def run_pipeline_task(script_name):
    script_path = f"/usr/local/airflow/src/{script_name}"
    result = subprocess.run(
        [sys.executable, script_path],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")


with DAG(
    dag_id="customer_churn_dag",
    default_args=default_args,
    description="MLOps pipeline for churn prediction",
    schedule=timedelta(days=1),
    catchup=False,
    tags=["mlops", "churn"],
) as dag:
    ingestion = PythonOperator(
        task_id="data_ingestion",
        python_callable=run_pipeline_task,
        op_args=["data_ingestion.py"],
    )

    validation = PythonOperator(
        task_id="data_validation",
        python_callable=run_pipeline_task,
        op_args=["validate_data.py"],
    )

    feature_eng = PythonOperator(
        task_id="feature_engineering",
        python_callable=run_pipeline_task,
        op_args=["preprocessing.py"],
    )

    train = PythonOperator(
        task_id="model_training",
        python_callable=run_pipeline_task,
        op_args=["train.py"],
    )

    evaluate = PythonOperator(
        task_id="model_evaluation",
        python_callable=run_pipeline_task,
        op_args=["evaluate.py"],
    )

    registration = PythonOperator(
        task_id="model_registration",
        python_callable=run_pipeline_task,
        op_args=["register_model.py"],
    )

    ingestion >> validation >> feature_eng >> train >> evaluate >> registration
