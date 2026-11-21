FROM python:3.12-slim
# SQLAlchemy pinned: MLflow 2.17 imports a pool class removed in SQLAlchemy 2.1
RUN pip install --no-cache-dir mlflow==2.17.2 psycopg2-binary==2.9.10 sqlalchemy==2.0.36
EXPOSE 5000
