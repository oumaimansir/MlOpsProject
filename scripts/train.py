import os
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


def train():
    # =====================================================
    # Environment variables
    # =====================================================
    DATA_DIR = os.getenv("DATA_DIR", "/opt/airflow/MlOps/data")
    MODELS_DIR = os.getenv("MODELS_DIR", "/opt/airflow/MlOps/models")
    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
    EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")

    train_path = os.path.join(DATA_DIR, "dreaddit-train.csv")
    test_path = os.path.join(DATA_DIR, "dreaddit-test.csv")
    model_path = os.path.join(MODELS_DIR, "text_classification_model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # =====================================================
    # MLflow
    # =====================================================
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="train_model"):

        # =====================================================
        # 1. Load data
        # =====================================================
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train_text = train_df["text"]
        y_train = train_df["label"]

        X_test_text = test_df["text"]
        y_test = test_df["label"]

        # =====================================================
        # 2. TF-IDF
        # =====================================================
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

        # =====================================================
        # 3. Model
        # =====================================================
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # =====================================================
        # 4. Evaluation
        # =====================================================
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        print(classification_report(y_test, y_pred))
        mlflow.log_metric("test_accuracy", test_acc)

        # =====================================================
        # 5. Save artifacts
        # =====================================================
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.sklearn.log_model(vectorizer, artifact_path="vectorizer")

        print("✔ Training completed successfully")
        print("✔ Model & vectorizer saved and logged to MLflow")

