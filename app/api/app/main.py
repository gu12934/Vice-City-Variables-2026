"""Well Failure Risk API.

GET  /api/data                  are the three raw data files available (mounted or uploaded)?
POST /api/data                  upload raw parquet files (multipart) so the pipeline can run
POST /api/pipeline/run          start the full modelling pipeline in the background
GET  /api/pipeline/runs/{id}    progress of a run (steps, status, error)
GET  /api/latest                summary of the latest successful run
GET  /api/charts                all visualization payloads of the latest successful run
GET  /api/wells                 ranked work list
GET  /api/wells/{uwi}           one well: score, SHAP reasons, survival curve, history, production
"""
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import db
from .config import REQUIRED_FILES, data_status, settings

_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_schema()
    db.fail_stale_runs()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Well Failure Risk API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])


def _latest_or_404() -> dict:
    run = db.latest_success()
    if not run:
        raise HTTPException(404, "No successful pipeline run yet. Start one with POST /api/pipeline/run.")
    return run


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/config")
def config():
    return {"mlflow_url": settings.mlflow_public_url, "experiment": settings.mlflow_experiment,
            "price_cad_per_bbl": settings.price_cad_per_bbl, "horizon_days": settings.horizon_days}


@app.get("/api/data")
def data():
    return data_status()


@app.post("/api/data")
def upload_data(files: list[UploadFile] = File(...)):
    """Save uploaded competition files into the upload volume (validated as parquet first)."""
    import pyarrow.parquet as pq

    if db.active_run():
        raise HTTPException(409, "A pipeline run is in progress. Upload after it finishes.")
    for f in files:
        name = os.path.basename(f.filename or "")
        if name not in REQUIRED_FILES:
            raise HTTPException(400, f"Unexpected file '{name}'. Expected: {', '.join(REQUIRED_FILES)}")
        dest = settings.upload_dir / name
        part = dest.with_name(name + ".part")
        with part.open("wb") as out:
            while chunk := f.file.read(1 << 20):
                out.write(chunk)
        try:
            pq.read_schema(part)
        except Exception:
            part.unlink(missing_ok=True)
            raise HTTPException(400, f"'{name}' is not a readable parquet file")
        os.replace(part, dest)
    return data_status()


@app.post("/api/pipeline/run", status_code=202)
def start_pipeline():
    from .pipeline import STEPS, run_pipeline  # heavy imports only when needed

    status = data_status()
    if not status["ready"]:
        missing = [f["name"] for f in status["files"] if not f["present"]]
        raise HTTPException(409, {"message": f"Data files missing: {', '.join(missing)}", "code": "data_missing",
                                  "missing": missing})
    with _lock:
        active = db.active_run()
        if active:
            raise HTTPException(409, {"message": "A pipeline run is already in progress", "run_id": active})
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        steps = [{"key": k, "label": lbl, "status": "pending", "seconds": None} for k, lbl in STEPS]
        db.create_run(run_id, steps)
    threading.Thread(target=run_pipeline, args=(run_id,), daemon=True, name=f"pipeline-{run_id}").start()
    return {"run_id": run_id}


@app.get("/api/pipeline/runs")
def runs(limit: int = 10):
    return db.list_runs(limit)


@app.get("/api/pipeline/runs/{run_id}")
def run_status(run_id: str):
    r = db.get_run(run_id)
    if not r:
        raise HTTPException(404, "Unknown run")
    return r


@app.get("/api/latest")
def latest():
    return {"run": db.latest_success(), "active_run_id": db.active_run()}


@app.get("/api/charts")
def charts():
    run = _latest_or_404()
    return {"run_id": run["run_id"], "charts": db.charts(run["run_id"])}


@app.get("/api/wells")
def wells(limit: int = Query(100, le=2000), area: str | None = None, q: str | None = None):
    run = _latest_or_404()
    return {"run_id": run["run_id"], "areas": db.areas(run["run_id"]),
            "wells": db.wells(run["run_id"], limit, area, q)}


@app.get("/api/wells/{uwi}")
def well(uwi: str):
    run = _latest_or_404()
    w = db.well(run["run_id"], uwi)
    if not w:
        raise HTTPException(404, "Well not found in the latest run")
    return w
