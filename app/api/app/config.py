import os
from pathlib import Path


class Settings:
    database_url: str = os.getenv("DATABASE_URL", "postgresql+psycopg://app:app@localhost:5433/app")
    mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_public_url: str = os.getenv("MLFLOW_PUBLIC_URL", "http://localhost:5000")
    mlflow_experiment: str = os.getenv("MLFLOW_EXPERIMENT", "well-failure-survival")
    # Raw data: a read-only mounted folder, or files uploaded through the app (kept in a Docker volume)
    data_dir: Path = Path(os.getenv("DATA_DIR", "../../data/Competition Dataset"))
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", "../../data/uploads"))

    # Business assumptions (visible in the app and logged to MLflow)
    price_cad_per_bbl: float = float(os.getenv("PRICE_CAD_PER_BBL", "60"))
    horizon_days: int = 90
    landmark_days: int = 7
    train_cutoff: str = "2025-01-01"


settings = Settings()

# The three competition files the pipeline reads
REQUIRED_FILES = ("daily_production.parquet", "production_failures.parquet", "well_jobs.parquet")


def find_data_file(name: str) -> tuple[Path, str] | None:
    """Locate a data file: the mounted folder wins, then the upload volume."""
    for folder, source in ((settings.data_dir, "mounted"), (settings.upload_dir, "uploaded")):
        p = folder / name
        if p.is_file():
            return p, source
    return None


def data_status() -> dict:
    files = []
    for name in REQUIRED_FILES:
        hit = find_data_file(name)
        files.append({"name": name, "present": hit is not None, "source": hit[1] if hit else None,
                      "size": hit[0].stat().st_size if hit else None})
    return {"ready": all(f["present"] for f in files), "files": files}
