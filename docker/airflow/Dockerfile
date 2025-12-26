FROM apache/airflow:2.10.2-python3.12

USER root

# System dependencies
RUN apt-get update && apt-get install -y build-essential && apt-get clean

USER airflow

WORKDIR /airflow

# Copy requirements from root of build context
COPY requirements.txt /airflow/

RUN pip install --no-cache-dir -r requirements.txt

