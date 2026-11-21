"""Postgres access: schema setup, run bookkeeping, result writes and reads."""
import json
from pathlib import Path

from sqlalchemy import create_engine, text

from .config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)


def init_schema() -> None:
    raw = (Path(__file__).parent / "schema.sql").read_text()
    sql = "\n".join(line.split("--")[0] for line in raw.splitlines())   # drop comments
    with engine.begin() as c:
        for stmt in (s.strip() for s in sql.split(";")):
            if stmt:
                c.execute(text(stmt))


def fail_stale_runs() -> None:
    """A run still 'running' at startup died with the previous process."""
    with engine.begin() as c:
        c.execute(text("UPDATE pipeline_runs SET status='failed', finished_at=now(), "
                       "error='Interrupted: the API restarted while this run was in progress' "
                       "WHERE status='running'"))


# ---------------------------------------------------------------- run bookkeeping
def create_run(run_id: str, steps: list) -> None:
    with engine.begin() as c:
        c.execute(text("INSERT INTO pipeline_runs (run_id, status, steps) VALUES (:r, 'running', CAST(:s AS JSONB))"),
                  {"r": run_id, "s": json.dumps(steps)})


def update_steps(run_id: str, steps: list, current: str | None) -> None:
    with engine.begin() as c:
        c.execute(text("UPDATE pipeline_runs SET steps=CAST(:s AS JSONB), current_step=:c WHERE run_id=:r"),
                  {"r": run_id, "s": json.dumps(steps), "c": current})


def set_mlflow_run(run_id: str, mlflow_run_id: str) -> None:
    with engine.begin() as c:
        c.execute(text("UPDATE pipeline_runs SET mlflow_run_id=:m WHERE run_id=:r"), {"r": run_id, "m": mlflow_run_id})


def finish_run(run_id: str, status: str, summary: dict | None = None, error: str | None = None) -> None:
    with engine.begin() as c:
        c.execute(text("UPDATE pipeline_runs SET status=:st, finished_at=now(), current_step=NULL, "
                       "summary=CAST(:sm AS JSONB), error=:e WHERE run_id=:r"),
                  {"r": run_id, "st": status, "sm": json.dumps(summary) if summary else None, "e": error})


def active_run() -> str | None:
    with engine.connect() as c:
        return c.execute(text("SELECT run_id FROM pipeline_runs WHERE status='running' LIMIT 1")).scalar()


def get_run(run_id: str) -> dict | None:
    with engine.connect() as c:
        r = c.execute(text("SELECT * FROM pipeline_runs WHERE run_id=:r"), {"r": run_id}).mappings().first()
        return dict(r) if r else None


def list_runs(limit: int = 10) -> list[dict]:
    with engine.connect() as c:
        rows = c.execute(text("SELECT run_id, status, current_step, started_at, finished_at, mlflow_run_id, error "
                              "FROM pipeline_runs ORDER BY started_at DESC LIMIT :n"), {"n": limit}).mappings()
        return [dict(r) for r in rows]


def latest_success() -> dict | None:
    with engine.connect() as c:
        r = c.execute(text("SELECT * FROM pipeline_runs WHERE status='success' "
                           "ORDER BY finished_at DESC LIMIT 1")).mappings().first()
        return dict(r) if r else None


# ---------------------------------------------------------------- result writes
def save_results(run_id: str, charts: dict, wells: list[dict], reasons: list[dict], details: list[dict]) -> None:
    """All-or-nothing write of a run's outputs."""
    with engine.begin() as c:
        for key, data in charts.items():
            c.execute(text("INSERT INTO chart_data (run_id, key, data) VALUES (:r, :k, CAST(:d AS JSONB))"),
                      {"r": run_id, "k": key, "d": json.dumps(data)})
        if wells:
            c.execute(text("""INSERT INTO well_scores (run_id, uwi, rank, area, p_fail_90d, exp_cost_90d, oil_rate_m3d,
                              days_running, repairs_total, repairs_365, rpm, last_repair, top_reason)
                              VALUES (:run_id, :uwi, :rank, :area, :p_fail_90d, :exp_cost_90d, :oil_rate_m3d,
                              :days_running, :repairs_total, :repairs_365, :rpm, :last_repair, :top_reason)"""),
                      [{**w, "run_id": run_id} for w in wells])
        if reasons:
            c.execute(text("""INSERT INTO well_reasons (run_id, uwi, ord, factor, label, shap, multiplier)
                              VALUES (:run_id, :uwi, :ord, :factor, :label, :shap, :multiplier)"""),
                      [{**r, "run_id": run_id} for r in reasons])
        if details:
            c.execute(text("""INSERT INTO well_detail (run_id, uwi, curve, history, production, base_value, model_value)
                              VALUES (:run_id, :uwi, CAST(:curve AS JSONB), CAST(:history AS JSONB),
                              CAST(:production AS JSONB), :base_value, :model_value)"""),
                      [{**d, "run_id": run_id, "curve": json.dumps(d["curve"]), "history": json.dumps(d["history"]),
                        "production": json.dumps(d["production"])} for d in details])


# ---------------------------------------------------------------- result reads
def charts(run_id: str) -> dict:
    with engine.connect() as c:
        rows = c.execute(text("SELECT key, data FROM chart_data WHERE run_id=:r"), {"r": run_id})
        return {k: d for k, d in rows}


def wells(run_id: str, limit: int, area: str | None, q: str | None) -> list[dict]:
    sql = "SELECT * FROM well_scores WHERE run_id=:r"
    p: dict = {"r": run_id, "n": limit}
    if area:
        sql += " AND area=:a"
        p["a"] = area
    if q:
        sql += " AND uwi ILIKE :q"
        p["q"] = f"%{q}%"
    sql += " ORDER BY rank LIMIT :n"
    with engine.connect() as c:
        return [dict(r) for r in c.execute(text(sql), p).mappings()]


def areas(run_id: str) -> list[str]:
    with engine.connect() as c:
        return [r[0] for r in c.execute(text("SELECT area FROM well_scores WHERE run_id=:r GROUP BY area "
                                             "ORDER BY count(*) DESC"), {"r": run_id})]


def well(run_id: str, uwi: str) -> dict | None:
    with engine.connect() as c:
        s = c.execute(text("SELECT * FROM well_scores WHERE run_id=:r AND uwi=:u"), {"r": run_id, "u": uwi}).mappings().first()
        if not s:
            return None
        reasons = c.execute(text("SELECT factor, label, shap, multiplier FROM well_reasons "
                                 "WHERE run_id=:r AND uwi=:u ORDER BY ord"), {"r": run_id, "u": uwi}).mappings()
        d = c.execute(text("SELECT curve, history, production, base_value, model_value FROM well_detail "
                           "WHERE run_id=:r AND uwi=:u"), {"r": run_id, "u": uwi}).mappings().first()
        return {**dict(s), "reasons": [dict(r) for r in reasons], **(dict(d) if d else {})}
