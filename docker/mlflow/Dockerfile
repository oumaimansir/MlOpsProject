FROM python:3.12-slim

WORKDIR /mlflow

# Copy requirements from root of build context
COPY requirements.txt /mlflow/

RUN pip install --no-cache-dir -r requirements.txt

# Expose MLflow port
EXPOSE 5000

# MLflow server command
CMD mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root /mlruns \
    --host 0.0.0.0 \
    --port 5000

