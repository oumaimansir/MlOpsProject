import os
import mlflow
import pandas as pd

def run_evidently():
    # IMPORT EVIDENTLY INSIDE FUNCTION (CRITICAL)
    from evidently.report import Report
    from evidently.metric_preset import ClassificationPreset

    print("✅ Starting Evidently report generation")

    # Example data (replace with real eval data)
    df = pd.read_csv("/home/oumaima/MlOps/data/eval_data.csv")

    report = Report(metrics=[ClassificationPreset()])
    report.run(reference_data=df, current_data=df)

    reports_dir = "/home/oumaima/MlOps/reports"
    os.makedirs(reports_dir, exist_ok=True)

    report_path = os.path.join(reports_dir, "evidently_report.html")
    report.save_html(report_path)

    print(f"✅ Evidently report saved to {report_path}")

    # Log to MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("stress_detection")

    with mlflow.start_run(run_name="evidently_report"):
        mlflow.log_artifact(report_path, artifact_path="evidently")

    print("✅ Evidently report logged to MLflow")

