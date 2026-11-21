-- Runs once when the Postgres volume is first created.
-- The "app" database is created by POSTGRES_DB; MLflow gets its own database.
CREATE DATABASE mlflow;
