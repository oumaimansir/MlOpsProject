import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

def evaluate(
    model_path="models/text_classification_model.pkl",
    train_path="data/dreaddit-train.csv",
    test_path="data/dreaddit-test.csv"
):
    mlflow.set_experiment("StressDetection_Pipeline")

    with mlflow.start_run():
        # ---------------------------
        # 1. Charger les datasets
        # ---------------------------
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Séparer features / labels
        X_train = train_df.drop("label", axis=1)
        y_train = train_df["label"]

        X_test = test_df.drop("label", axis=1)
        y_test = test_df["label"]

        # ---------------------------
        # 2. Charger le modèle existant
        # ---------------------------
        model = joblib.load(model_path)

        # ---------------------------
        # 3. Faire les prédictions
        # ---------------------------
        y_pred = model.predict(X_test)

        # ---------------------------
        # 4. Calculer les métriques
        # ---------------------------
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        print("Test Accuracy:", acc)
        print(classification_report(y_test, y_pred))

        # ---------------------------
        # 5. Log dans MLflow
        # ---------------------------
        mlflow.log_metric("test_accuracy", acc)
        # log f1-score pour chaque classe
        mlflow.log_metrics({f"{k}_f1": v["f1-score"] for k, v in report.items() if k not in ["accuracy", "macro avg", "weighted avg"]})

        # ---------------------------
        # 6. Versionner le modèle
        # ---------------------------
        mlflow.sklearn.log_model(model, "model")
        print("Modèle versionné dans MLflow.")

if __name__ == "__main__":
    evaluate()
