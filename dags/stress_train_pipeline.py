from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

sys.path.append("/home/oumaima/MlOps/scripts")
from train import train

default_args = {
    "owner": "oumaima",
    "depends_on_past": False,
    "start_date": datetime(2025, 11, 21),
    "retries": 1,
}

with DAG(
    dag_id="stress_train_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
) as dag:

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train,
        op_kwargs={
            "train_path": "/home/oumaima/MlOps/data/dreaddit-train.csv",
            "test_path": "/home/oumaima/MlOps/data/dreaddit-test.csv",
            "model_path": "/home/oumaima/MlOps/models/text_classification_model.pkl",
        },
    )

train_model_task

