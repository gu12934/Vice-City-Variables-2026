# Well Failure Risk: demo application

A containerized demo of how the survival-analysis model would be used in operations.
One button in the web app runs the full pipeline: data → failure events → survival dataset → Kaplan–Meier / hazard
→ log-rank tests → Weibull & friends → Cox model → ML models validated on 2025–26 → risk-scored work list → SHAP
explanations. The pipeline logs every run to **MLflow** and writes the results the app reads to **Postgres**.

```
 React + TS SPA (nginx :8080) ──/api──▶ FastAPI (:8000) ──▶ pipeline (background thread)
   one tab per visualization                │                  ├─▶ MLflow (:5000): params, metrics, chart JSON, model
   + LaTeX (KaTeX) math per tab             │                  └─▶ Postgres (:5433): results, versioned by run
                                            └── reads the latest *successful* run from Postgres
```

## Run it

Requirements: Docker Desktop running.

```bash
cd app
docker compose up -d --build
```

**Results are pre-loaded.** On first start, Postgres loads `db/seed.sql.gz`, a snapshot of a completed pipeline run,
so every tab and the work list work immediately, **without the raw data**.

**To recompute (the Run pipeline button)**, the app needs the three competition files: `daily_production.parquet`,
`production_failures.parquet` and `well_jobs.parquet`. Provide them either way:
- put them in `../data/Competition Dataset/` (relative to this folder), which is mounted read-only, or
- click **Run pipeline**: if the files are missing, the app asks for them, uploads them into the `uploads` Docker
  volume (kept across restarts), and then starts the run.

To refresh the snapshot after a new run: `bash db/export_seed.sh`, then commit `db/seed.sql.gz`.
The seed only loads into an empty database. To reload it locally, run `docker compose down -v && docker compose up -d`
(this deletes your local runs).

| URL | What |
|---|---|
| http://localhost:8080 | the web app (pre-loaded; **Run pipeline** recomputes, about 10 min) |
| http://localhost:5000 | MLflow: experiment `well-failure-survival` |
| http://localhost:8000/docs | FastAPI interactive docs |
| `localhost:5433` | Postgres (`app` / `app`, databases `app` and `mlflow`) |

Stop with `docker compose down`. Add `-v` to also delete the Postgres, MLflow and upload volumes (the seed reloads on the next start).

## Tabs

| # | Tab | Visualization | Math shown |
|---|---|---|---|
| – | Overview | KPIs, pipeline progress | S(t), h(t), H(t), censoring |
| 1 | Failure events | failure log vs repair jobs per quarter | proxy precision / recall |
| 2 | Survival curve | Kaplan–Meier with 95% band | product-limit estimator, Greenwood |
| 3 | Hazard | smoothed hazard, cumulative hazard | Nelson–Aalen, kernel smoothing |
| 4 | Group comparison | KM by repairs / RPM / last repair / area | log-rank χ² |
| 5 | Distributions | exponential, Weibull, log-normal, log-logistic vs KM | Weibull hazard, censored likelihood, AIC |
| 6 | Cox model | hazard-ratio forest plot | Cox PH, partial likelihood, clustered SEs |
| 7 | Validation | test metrics, time-dependent AUC | RSF, boosted Cox, Harrell/Uno C, IBS |
| 8 | Gains | cumulative gains vs random | gains / lift |
| 9 | SHAP | global importance, effect-direction strips | Shapley values, additivity |
| 10 | Cross-check | Cox β vs SHAP slope | slope regression, sign agreement |
| 11 | Work list | ranked wells → reasons, survival outlook, history | conditional failure probability, expected cost |

## Design notes

- **Results are versioned by run.** Every Postgres row carries `run_id`, and the API serves the latest successful run, so a failed or in-progress run never breaks the app. Only one run at a time (HTTP 409 otherwise).
- **The data is mounted read-only** into the API container and never baked into an image. The competition terms forbid redistribution, so don't commit `data/`.
- **Business assumptions** (oil price, horizon) are set through environment variables in `docker-compose.yml` and logged to MLflow with each run.
- Pipeline code: `api/app/pipeline.py` (port of `notebooks/02_survival.ipynb`). Schema: `api/app/schema.sql`.

### Postgres tables (database `app`)

| table | contents |
|---|---|
| `pipeline_runs` | status, per-step progress, MLflow run id, summary KPIs, error |
| `chart_data` | one JSON document per visualization tab |
| `well_scores` | the ranked work list |
| `well_reasons` | SHAP reason codes per well |
| `well_detail` | per-well survival outlook, repair history, recent production |

## Local development (without Docker for the web app)

```bash
docker compose up -d postgres mlflow api     # backend in containers
cd web && npm install && npm run dev          # http://localhost:5173 (proxies /api to :8000)
```

AI assistance: built with Claude Code (Anthropic), as disclosed in the main project README.
