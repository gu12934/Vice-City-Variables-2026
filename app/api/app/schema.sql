-- App tables. Every result row is tagged with the pipeline run that produced it;
-- the API always serves the latest *successful* run, so a failed run never breaks the app.

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id         TEXT PRIMARY KEY,
    status         TEXT NOT NULL,                 -- running | success | failed
    current_step   TEXT,
    steps          JSONB NOT NULL DEFAULT '[]',
    started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at    TIMESTAMPTZ,
    mlflow_run_id  TEXT,
    error          TEXT,
    summary        JSONB
);

-- One JSON document per visualization tab (curves, tables, metrics)
CREATE TABLE IF NOT EXISTS chart_data (
    run_id  TEXT NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    key     TEXT NOT NULL,
    data    JSONB NOT NULL,
    PRIMARY KEY (run_id, key)
);

-- Ranked work list: one row per well currently running
CREATE TABLE IF NOT EXISTS well_scores (
    run_id         TEXT NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    uwi            TEXT NOT NULL,
    rank           INT NOT NULL,
    area           TEXT,
    p_fail_90d     DOUBLE PRECISION,
    exp_cost_90d   DOUBLE PRECISION,
    oil_rate_m3d   DOUBLE PRECISION,
    days_running   INT,
    repairs_total  INT,
    repairs_365    INT,
    rpm            DOUBLE PRECISION,
    last_repair    TEXT,
    top_reason     TEXT,
    PRIMARY KEY (run_id, uwi)
);
CREATE INDEX IF NOT EXISTS ix_well_scores_rank ON well_scores (run_id, rank);

-- SHAP reason codes per well (grouped factors, signed)
CREATE TABLE IF NOT EXISTS well_reasons (
    run_id      TEXT NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    uwi         TEXT NOT NULL,
    ord         INT NOT NULL,
    factor      TEXT NOT NULL,
    label       TEXT NOT NULL,
    shap        DOUBLE PRECISION NOT NULL,
    multiplier  DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, uwi, ord)
);

-- Per-well chart payloads: predicted survival from today, repair history, recent production
CREATE TABLE IF NOT EXISTS well_detail (
    run_id      TEXT NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    uwi         TEXT NOT NULL,
    curve       JSONB,
    history     JSONB,
    production  JSONB,
    base_value  DOUBLE PRECISION,
    model_value DOUBLE PRECISION,
    PRIMARY KEY (run_id, uwi)
);
