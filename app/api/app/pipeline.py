"""End-to-end survival-analysis pipeline.

load data -> failure events -> survival dataset -> Kaplan-Meier / hazard -> group tests -> parametric fits
-> Cox model -> ML models (time-split validation) -> score active wells -> SHAP -> save (Postgres) + log (MLflow)

Each step updates its status in Postgres so the UI can show progress. All tab data is stored as JSON
documents keyed by run; the ranked work list and reason codes go into relational tables.
"""
import time
import traceback
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import mlflow
import mlflow.sklearn
import shap
from lifelines import (CoxPHFitter, ExponentialFitter, KaplanMeierFitter, LogLogisticFitter, LogNormalFitter,
                       NelsonAalenFitter, WeibullFitter)
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc,
                            integrated_brier_score)
from sksurv.util import Surv

from . import db
from .config import find_data_file, settings

warnings.filterwarnings("ignore")

STEPS = [
    ("load", "Load data"),
    ("events", "Build failure events"),
    ("runs", "Build survival dataset"),
    ("km", "Kaplan–Meier & hazard"),
    ("groups", "Group comparisons"),
    ("parametric", "Parametric fits"),
    ("cox", "Cox model"),
    ("ml", "Train & validate ML models"),
    ("score", "Score active wells"),
    ("shap", "SHAP explanations"),
    ("save", "Save to Postgres & MLflow"),
]

NUMERIC = ["log_oil", "water_cut_10pp", "rpm_100", "disp_100", "efficiency", "torque_1000",
           "age_10y", "prior_fails", "fails_last_365"]

FACTOR_NAMES = {
    "fails_last_365": "Repairs in last 12 months", "prior_fails": "Repairs since 2021", "area": "Field area",
    "last_repair": "Last repair type", "make": "Pump make", "rpm_100": "Pump speed", "log_oil": "Oil rate",
    "water_cut_10pp": "Water cut", "efficiency": "Pump efficiency", "torque_1000": "Rod torque",
    "disp_100": "Pump size", "age_10y": "Well age",
}
FEATURE_LABELS = {
    "log_oil": "Oil rate (log m³/d)", "water_cut_10pp": "Water cut (per 10 pts)", "rpm_100": "Pump speed (per 100 RPM)",
    "disp_100": "Pump size (per 100 m³/d)", "efficiency": "Pump efficiency", "torque_1000": "Rod torque (per 1,000 ft·lb)",
    "age_10y": "Well age (per 10 yrs)", "prior_fails": "Repairs since 2021", "fails_last_365": "Repairs in last 12 months",
    "last_repair_pump": "Last repair: pump (vs rods)", "last_repair_tubing": "Last repair: tubing (vs rods)",
    "last_repair_other": "Last repair: other (vs rods)",
}


# ====================================================================== helpers
def clean(o):
    """Make numpy/pandas values JSON-safe (NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return round(f, 6) if np.isfinite(f) else None
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    return o


def read_parquet(name, columns=None):
    hit = find_data_file(f"{name}.parquet")
    if hit is None:
        raise FileNotFoundError(f"{name}.parquet not found: mount the data folder or upload it in the app")
    t = pq.read_table(hit[0], columns=columns)
    fields = [pa.field(f.name, pa.float32()) if (pa.types.is_decimal(f.type) or pa.types.is_floating(f.type)) else f
              for f in t.schema]
    df = t.cast(pa.schema(fields)).to_pandas(strings_to_categorical=True, date_as_object=False)
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].astype("datetime64[ns]")
    return df


def component(job_type) -> str:
    jt = str(job_type)
    if "Pump" in jt or "Rotor" in jt or "PCP" in jt:
        return "Pump"
    if "Tubing" in jt or "Packer" in jt:
        return "Tubing"
    if "Rod" in jt:
        return "Rods"
    return "Other"


def factor_of(feature: str) -> str:
    """Group one-hot / missing-indicator columns into the factor a person thinks about."""
    if feature.endswith("_missing"):
        feature = feature[: -len("_missing")]
    for prefix, group in (("area_", "area"), ("make_", "make"), ("last_repair_", "last_repair")):
        if feature.startswith(prefix):
            return group
    return feature


def safe_metric_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-. /" else "_" for ch in s).replace(" ", "_").lower()


def sf_at(fn, t):
    """Evaluate an sksurv StepFunction, clipping t into its domain."""
    return float(fn(float(np.clip(t, fn.x[0], fn.x[-1]))))


# ====================================================================== pipeline
class Pipeline:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.steps = [{"key": k, "label": lbl, "status": "pending", "seconds": None} for k, lbl in STEPS]
        self.charts: dict = {}
        self.metrics: dict = {}
        self.summary: dict = {}

    # ------------------------------------------------------------ progress
    @contextmanager
    def step(self, key):
        s = next(s for s in self.steps if s["key"] == key)
        s["status"] = "running"
        db.update_steps(self.run_id, self.steps, key)
        t0 = time.time()
        try:
            yield
        except Exception:
            s["status"] = "failed"
            s["seconds"] = round(time.time() - t0, 1)
            db.update_steps(self.run_id, self.steps, key)
            raise
        s["status"] = "done"
        s["seconds"] = round(time.time() - t0, 1)
        db.update_steps(self.run_id, self.steps, None)
        mlflow.log_metric(f"step_seconds/{key}", s["seconds"])

    def run(self):
        try:
            mlflow.set_tracking_uri(settings.mlflow_uri)
            mlflow.set_experiment(settings.mlflow_experiment)
            with mlflow.start_run(run_name=f"pipeline-{self.run_id}") as mr:
                db.set_mlflow_run(self.run_id, mr.info.run_id)
                mlflow.set_tag("app_run_id", self.run_id)
                mlflow.log_params({"landmark_days": settings.landmark_days, "train_cutoff": settings.train_cutoff,
                                   "horizon_days": settings.horizon_days,
                                   "price_cad_per_bbl": settings.price_cad_per_bbl,
                                   "event_definition": "failure-repair jobs (Rod Truck/Workover repair job types)"})
                with self.step("load"): self.load()
                with self.step("events"): self.events()
                with self.step("runs"): self.build_runs()
                with self.step("km"): self.kaplan_meier()
                with self.step("groups"): self.group_tests()
                with self.step("parametric"): self.parametric()
                with self.step("cox"): self.cox()
                with self.step("ml"): self.ml()
                with self.step("score"): self.score()
                with self.step("shap"): self.explain()
                with self.step("save"): self.save()
            db.finish_run(self.run_id, "success", summary=clean(self.summary))
        except Exception:
            db.finish_run(self.run_id, "failed", error=traceback.format_exc()[-4000:])

    # ------------------------------------------------------------ 1. load
    def load(self):
        cols = ["uwi", "production_date", "well_daily_status", "field_operating_area", "on_production_date",
                "oil_produced_m3", "water_produced_m3", "hours_on_prod", "hours_down",
                "pcp_pump_rpm", "pcp_pump_make", "pcp_pump_displacement_m3d100rpm", "pump_efficiency_pct",
                "pcp_pump_rod_torque_ftlb", "casing_pressure_kpa"]
        self.dp = read_parquet("daily_production", cols)
        self.fails = read_parquet("production_failures")
        self.jobs = read_parquet("well_jobs")
        self.jobs["uwi"] = self.jobs.uwi.astype(str)
        self.fails["uwi"] = self.fails.uwi.astype(str)
        mlflow.log_metrics({"rows_daily": len(self.dp), "rows_failures": len(self.fails), "rows_jobs": len(self.jobs)})

    # ------------------------------------------------------------ 2. events
    def events(self):
        jobs, fails = self.jobs, self.fails
        repair = (jobs.job_category.astype(str).isin(["Rod Truck", "Workover"]) &
                  jobs.job_type.astype(str).str.contains("Repair|Replacement|Break|Rotor", regex=True))
        rep = jobs[repair].copy()

        ov = rep[(rep.start_date >= "2025-04-01") & (rep.start_date <= fails.failure_start_date.max())]
        m = ov[["uwi", "start_date"]].reset_index(names="jid").merge(fails[["uwi", "failure_start_date"]], on="uwi")
        precision = m[(m.start_date - m.failure_start_date).dt.days.between(-30, 3)].jid.nunique() / max(len(ov), 1)
        m2 = fails[["uwi", "failure_start_date"]].reset_index(names="fid").merge(rep[["uwi", "start_date"]], on="uwi")
        recall = m2[(m2.start_date - m2.failure_start_date).dt.days.between(-3, 30)].fid.nunique() / max(len(fails), 1)

        qj = rep.set_index("start_date").resample("QS").size()
        qf = fails.set_index("failure_start_date").resample("QS").size()
        idx = qj.index.union(qf.index)
        qj, qf = qj.reindex(idx, fill_value=0), qf.reindex(idx, fill_value=0)
        self.charts["events"] = {
            "quarters": [{"quarter": f"{d.year} Q{d.quarter}", "repairs": int(a), "logged": int(b)}
                         for d, a, b in zip(idx, qj.values, qf.values)],
            "precision": precision, "recall": recall, "n_repairs": len(rep), "n_logged": len(fails),
        }
        mlflow.log_metrics({"proxy_precision": precision, "proxy_recall": recall, "n_repair_events": len(rep)})

        f = rep.rename(columns={"start_date": "failure_start_date"})[["uwi", "failure_start_date", "job_type",
                                                                      "field_est_total"]]
        f = f.assign(cause=f.job_type.map(component), failure_type=f.job_type.astype(str))
        f = f.sort_values(["uwi", "failure_start_date"])
        f = f[~(f.groupby("uwi").failure_start_date.diff().dt.days <= 3)].copy()
        f["k"] = f.groupby("uwi").cumcount()
        self.f = f

    # ------------------------------------------------------------ 3. survival dataset
    def build_runs(self):
        dp, f, L = self.dp, self.f, settings.landmark_days
        cut = pd.Timestamp(settings.train_cutoff)
        self.data_end = dp.production_date.max()

        prod = dp[dp.hours_on_prod > 0].copy()
        prod["uwi"] = prod.uwi.astype(str)
        prod = prod.drop_duplicates(["uwi", "production_date"]).sort_values("production_date")
        for c in ["casing_pressure_kpa", "pcp_pump_rod_torque_ftlb"]:
            prod.loc[prod[c] >= 99_999, c] = np.nan
        prod = pd.merge_asof(prod, f[["uwi", "failure_start_date", "k"]].sort_values("failure_start_date")
                             .rename(columns={"failure_start_date": "prev_fail"}),
                             left_on="production_date", right_on="prev_fail", by="uwi",
                             direction="backward", allow_exact_matches=False)
        prod["run_id"] = (prod.k + 1).fillna(0).astype(int)
        prod = prod.drop(columns="k").sort_values(["uwi", "production_date"])
        prod["day_in_run"] = prod.groupby(["uwi", "run_id"]).cumcount()
        prod["pre_cut"] = prod.production_date < cut
        self.prod = prod

        runs = (prod.groupby(["uwi", "run_id"])
                .agg(start=("production_date", "min"), end=("production_date", "max"),
                     duration=("production_date", "size"), dur_pre_cut=("pre_cut", "sum")).reset_index())
        runs = runs.merge(f[["uwi", "k", "failure_start_date", "cause"]], left_on=["uwi", "run_id"],
                          right_on=["uwi", "k"], how="left").drop(columns="k")
        runs["event"] = runs.failure_start_date.notna()

        early = (prod[prod.day_in_run < L].groupby(["uwi", "run_id"])
                 .agg(oil=("oil_produced_m3", "median"), water=("water_produced_m3", "median"),
                      rpm=("pcp_pump_rpm", "median"), disp=("pcp_pump_displacement_m3d100rpm", "median"),
                      eff=("pump_efficiency_pct", "median"), torque=("pcp_pump_rod_torque_ftlb", "median"),
                      make=("pcp_pump_make", "first"), area=("field_operating_area", "first"),
                      onprod=("on_production_date", "first")))
        runs = runs.merge(early.reset_index(), on=["uwi", "run_id"], how="left")

        prev = f[["uwi", "k", "cause"]].rename(columns={"cause": "prev_cause"}).assign(run_id=lambda d: d.k + 1)
        runs = runs.merge(prev[["uwi", "run_id", "prev_cause"]], on=["uwi", "run_id"], how="left")
        tmp = runs[["uwi", "run_id", "start"]].merge(
            f[["uwi", "failure_start_date"]].rename(columns={"failure_start_date": "fd"}), on="uwi")
        tmp = tmp[(tmp.fd < tmp.start) & (tmp.fd >= tmp.start - pd.Timedelta(days=365))]
        runs = runs.merge(tmp.groupby(["uwi", "run_id"]).size().rename("fails_365").reset_index(),
                          on=["uwi", "run_id"], how="left")
        runs["fails_365"] = runs.fails_365.fillna(0)

        runs["water_cut"] = runs.water / (runs.water + runs.oil)
        runs["age_yrs"] = (runs.start - runs.onprod).dt.days / 365.25
        runs["prior_fails"] = runs.run_id
        runs["prev_cause_grp"] = runs.prev_cause.fillna("None").astype(str)
        runs["area"] = runs.area.astype(str)
        runs["make"] = runs.make.astype(str).replace("nan", "Unknown")
        self.top_areas = runs.area.value_counts().index[:6].tolist()
        self.top_makes = [m for m in runs.make.value_counts().index if m != "Unknown"][:3]
        runs["area_grp"] = runs.area.where(runs.area.isin(self.top_areas), "Other")
        runs["make_grp"] = runs.make.where(runs.make.isin(self.top_makes), "Other/unknown")

        after = runs[runs.run_id >= 1]
        infant = after[after.duration <= L]
        S = after[after.duration > L].copy().reset_index(drop=True)
        S["T"] = S.duration - L
        S["E"] = S.event.astype(bool)
        S["prior_grp"] = pd.cut(S.prior_fails, [0, 1, 3, 7, 999], labels=["1", "2–3", "4–7", "8+"]).astype(str)
        S["rpm_grp"] = pd.cut(S.rpm, [0, 150, 250, 350, 1e4], labels=["< 150", "150–250", "250–350", "350+"]).astype(str)
        self.S = S
        self.summary.update({"n_runs": len(S), "n_failures": int(S.E.sum()), "n_wells": S.uwi.nunique(),
                             "infant_failures": int(infant.event.sum()),
                             "infant_share": float(infant.event.sum() / max(after.event.sum(), 1)),
                             "data_end": self.data_end.strftime("%Y-%m-%d")})
        mlflow.log_metrics({"n_runs": len(S), "n_failures": int(S.E.sum()), "n_wells": S.uwi.nunique()})

    # ------------------------------------------------------------ 4. Kaplan-Meier & hazard
    def kaplan_meier(self):
        S = self.S
        G = np.arange(0, 731, 5)
        kmf = KaplanMeierFitter().fit(S["T"], S["E"])
        ci = kmf.confidence_interval_
        ci = ci.reindex(ci.index.union(G)).ffill().loc[G]
        s = kmf.survival_function_at_times(G).to_numpy()
        milestones = {str(t): float(kmf.predict(t)) for t in (30, 90, 180, 365)}
        self.kmf = kmf
        self.charts["km"] = {
            "points": [{"t": int(t), "s": a, "lo": b, "hi": c}
                       for t, a, b, c in zip(G, s, ci.iloc[:, 0].to_numpy(), ci.iloc[:, 1].to_numpy())],
            "median": kmf.median_survival_time_, "milestones": milestones,
            "n": len(S), "events": int(S.E.sum()),
        }

        naf = NelsonAalenFitter().fit(S["T"], S["E"])
        H = naf.cumulative_hazard_.iloc[:, 0]
        h = naf.smoothed_hazard_(bandwidth=30).iloc[:, 0]
        G2 = np.arange(0, 701, 5)
        Hg = H.reindex(H.index.union(G2)).ffill().loc[G2]
        hg = h.reindex(h.index.union(G2)).ffill().bfill().loc[G2]
        self.charts["hazard"] = {"points": [{"t": int(t), "h": float(a) * 100, "H": float(b)}
                                            for t, a, b in zip(G2, hg.to_numpy(), Hg.to_numpy())],
                                 "bandwidth": 30}
        self.summary.update({"median_run_days": kmf.median_survival_time_, "s90": milestones["90"],
                             "s365": milestones["365"]})
        mlflow.log_metrics({"km_median_days": float(kmf.median_survival_time_), "km_s90": milestones["90"],
                            "km_s365": milestones["365"]})

    # ------------------------------------------------------------ 5. group comparisons
    def group_tests(self):
        S = self.S
        G = np.arange(0, 731, 5)
        specs = [("repairs", "Repairs since 2021", "prior_grp", ["1", "2–3", "4–7", "8+"], True),
                 ("rpm", "Pump speed (RPM)", "rpm_grp", ["< 150", "150–250", "250–350", "350+"], True),
                 ("last_repair", "Last repair type", "prev_cause_grp", ["Rods", "Pump", "Tubing", "Other"], False),
                 ("area", "Field area", "area_grp", self.top_areas, False)]
        out = []
        for key, label, col, order, ordinal in specs:
            groups = []
            for g in order:
                d = S[S[col] == g]
                if len(d) < 30:
                    continue
                km = KaplanMeierFitter().fit(d["T"], d["E"])
                groups.append({"name": g, "n": len(d), "events": int(d.E.sum()),
                               "median": km.median_survival_time_, "s90": float(km.predict(90)),
                               "s365": float(km.predict(365)), "s": km.survival_function_at_times(G).to_numpy()})
            sub = S[S[col].isin([g["name"] for g in groups])]
            res = multivariate_logrank_test(sub["T"], sub[col], sub["E"])
            out.append({"key": key, "label": label, "ordinal": ordinal, "p": float(res.p_value),
                        "chi2": float(res.test_statistic), "df": len(groups) - 1, "groups": groups})
            mlflow.log_metric(f"logrank_chi2/{key}", float(res.test_statistic))
        self.charts["groups"] = {"t": G, "groupings": out}

    # ------------------------------------------------------------ 6. parametric fits
    def parametric(self):
        S = self.S
        T = np.arange(0, 901, 10)
        t1 = np.maximum(T, 1)
        fitters = {"Exponential": ExponentialFitter(), "Weibull": WeibullFitter(),
                   "Log-normal": LogNormalFitter(), "Log-logistic": LogLogisticFitter()}
        for name, m in fitters.items():
            m.fit(S["T"], S["E"], label=name)
        best_aic = min(m.AIC_ for m in fitters.values())
        wb = fitters["Weibull"]
        rho_ci = wb.summary.loc["rho_", ["coef lower 95%", "coef upper 95%"]].to_numpy()
        th = np.arange(5, 901, 5)
        hz = []
        for lab, mask in [("1–3 repairs so far", S.prior_fails <= 3), ("4+ repairs so far", S.prior_fails >= 4)]:
            w = WeibullFitter().fit(S.loc[mask, "T"], S.loc[mask, "E"])
            hz.append({"name": lab, "rho": float(w.rho_), "h": w.hazard_at_times(th).to_numpy() * 100})
        self.charts["parametric"] = {
            "t": T, "km": self.kmf.survival_function_at_times(T).to_numpy(),
            "models": [{"name": n, "aic": float(m.AIC_), "aic_delta": float(m.AIC_ - best_aic),
                        "loglik": float(m.log_likelihood_), "s": m.survival_function_at_times(t1).to_numpy()}
                       for n, m in fitters.items()],
            "weibull": {"rho": float(wb.rho_), "rho_lo": rho_ci[0], "rho_hi": rho_ci[1], "lambda": float(wb.lambda_)},
            "hazard_t": th, "hazard_groups": hz,
        }
        self.summary["weibull_rho"] = float(wb.rho_)
        mlflow.log_metrics({"weibull_rho": float(wb.rho_), "weibull_lambda": float(wb.lambda_)})
        for n, m in fitters.items():
            mlflow.log_metric(f"aic/{safe_metric_name(n)}", float(m.AIC_))

    # ------------------------------------------------------------ 7. Cox model
    def design(self, d, med=None, cols=None):
        X = pd.DataFrame(index=d.index)
        X["log_oil"] = np.log1p(d.oil.clip(lower=0))
        X["water_cut_10pp"] = d.water_cut * 10
        X["rpm_100"] = d.rpm / 100
        X["disp_100"] = d.disp / 100
        X["efficiency"] = d.eff.clip(0, 2)
        X["torque_1000"] = d.torque.clip(0, 6000) / 1000
        X["age_10y"] = d.age_yrs.clip(0, 75) / 10
        X["prior_fails"] = d.prior_fails.clip(upper=10)
        X["fails_last_365"] = d.fails_365.clip(upper=10)
        if med is None:
            med = X.median()
        for c in NUMERIC:
            X[c + "_missing"] = X[c].isna().astype(float)
            X[c] = X[c].fillna(med[c])
        for v in ["Pump", "Tubing", "Other"]:
            X[f"last_repair_{v.lower()}"] = (d.prev_cause_grp == v).astype(float)
        for a in self.top_areas:
            X["area_" + a.replace(" ", "_")] = (d.area_grp == a).astype(float)
        for m in self.top_makes:
            X["make_" + m.replace(" ", "_")] = (d.make_grp == m).astype(float)
        if cols is None:
            cols = [c for c in X.columns if X[c].std() > 0]
        return X[cols].astype(float), med, cols

    def cox(self):
        S = self.S
        self.X_all, self.med_all, self.cols_all = self.design(S)
        df = self.X_all.assign(T=S["T"], E=S["E"].astype(int), uwi=S["uwi"])
        cph = CoxPHFitter(penalizer=0.01).fit(df, "T", "E", cluster_col="uwi")
        self.cph = cph
        sm = cph.summary
        rows = []
        for feat, r in sm.iterrows():
            if feat.endswith("_missing"):
                continue
            label = FEATURE_LABELS.get(feat) or (f"Area: {feat[5:].replace('_', ' ')}" if feat.startswith("area_")
                                                 else f"Pump make: {feat[5:].replace('_', ' ')}")
            rows.append({"feature": feat, "label": label, "coef": r["coef"], "hr": r["exp(coef)"],
                         "lo": r["exp(coef) lower 95%"], "hi": r["exp(coef) upper 95%"], "p": r["p"]})
        rows.sort(key=lambda x: x["hr"])
        try:
            ph = proportional_hazard_test(cph, df, time_transform="rank").summary["p"]
            ph_rows = [{"feature": k, "p": float(v)} for k, v in ph.sort_values().items() if not k.endswith("_missing")]
        except Exception:
            ph_rows = []
        self.charts["cox"] = {"rows": rows, "concordance": cph.concordance_index_, "n": len(df),
                              "events": int(df.E.sum()), "penalizer": 0.01, "ph": ph_rows[:10]}
        mlflow.log_metric("cox_concordance_in_sample", float(cph.concordance_index_))
        mlflow.log_text(sm.to_csv(), "cox/summary.csv")

    # ------------------------------------------------------------ 8. ML models, time-split validation
    def ml(self):
        S, L = self.S, settings.landmark_days
        cut = pd.Timestamp(settings.train_cutoff)
        tr = S[S.start < cut].copy()
        tr["T"] = tr.dur_pre_cut - L
        tr["E"] = tr.E & (tr.failure_start_date < cut)
        tr = tr[tr["T"] > 0]
        te = S[S.start >= cut].copy()
        X_tr, med, cols = self.design(tr)
        X_te, _, _ = self.design(te, med, cols)
        y_tr = Surv.from_arrays(tr.E.to_numpy(bool), tr["T"].to_numpy(float))
        y_te = Surv.from_arrays(te.E.to_numpy(bool), te["T"].to_numpy(float))
        self.cols = cols

        hist = ["prior_fails", "fails_last_365"]
        models = {
            "Baseline (history only)": (CoxPHSurvivalAnalysis(alpha=0.01), hist),
            "Cox (all features)": (CoxPHSurvivalAnalysis(alpha=0.01), cols),
            "Random survival forest": (RandomSurvivalForest(n_estimators=200, min_samples_leaf=25,
                                                            max_features="sqrt", n_jobs=-1, random_state=0), cols),
            "Gradient-boosted Cox": (GradientBoostingSurvivalAnalysis(n_estimators=300, learning_rate=0.05,
                                                                      max_depth=3, subsample=0.8, random_state=0), cols),
        }
        for name, (m, c) in models.items():
            m.fit(X_tr[c], y_tr)
            mlflow.log_params({f"{safe_metric_name(name)}/{k}": v for k, v in m.get_params().items()
                               if k in ("alpha", "n_estimators", "min_samples_leaf", "max_depth", "learning_rate")})

        times = np.arange(30, 361, 30)
        times = times[times < te["T"].max()]
        res, auc, risk = [], {}, {}
        for name, (m, c) in models.items():
            r = m.predict(X_te[c])
            risk[name] = r
            surv = np.vstack([fn(times) for fn in m.predict_survival_function(X_te[c])])
            a_t, a_mean = cumulative_dynamic_auc(y_tr, y_te, r, times)
            row = {"model": name,
                   "harrell_c": concordance_index_censored(y_te["event"], y_te["time"], r)[0],
                   "uno_c": concordance_index_ipcw(y_tr, y_te, r, tau=365)[0],
                   "mean_auc": a_mean, "ibs": integrated_brier_score(y_tr, y_te, surv, times)}
            res.append(row)
            auc[name] = a_t
            for k in ("harrell_c", "uno_c", "mean_auc", "ibs"):
                mlflow.log_metric(f"test/{safe_metric_name(name)}/{k}", float(row[k]))
        best = max(res, key=lambda r: r["uno_c"])["model"]

        H = settings.horizon_days
        ok = te.E.to_numpy() | (te["T"].to_numpy() >= H)
        fail_h = (te.E.to_numpy() & (te["T"].to_numpy() <= H))[ok]
        gains = {}
        for name in (best, "Baseline (history only)"):
            order = np.argsort(-risk[name][ok])
            cap = np.concatenate([[0], np.cumsum(fail_h[order]) / fail_h.sum()])
            x = np.linspace(0, 1, len(cap))
            grid = np.linspace(0, 1, 101)
            gains[name] = np.interp(grid, x, cap)
        self.models, self.best = models, best
        self.charts["validation"] = {"metrics": res, "times": times, "auc": auc, "best": best,
                                     "n_train": len(tr), "n_test": len(te), "events_train": int(tr.E.sum()),
                                     "events_test": int(te.E.sum()), "cutoff": settings.train_cutoff}
        self.charts["gains"] = {"x": np.linspace(0, 1, 101), "curves": gains, "best": best, "horizon": H,
                                "n": int(ok.sum()), "events": int(fail_h.sum()),
                                "top20": float(gains[best][20]), "top20_baseline": float(gains["Baseline (history only)"][20])}
        b = next(r for r in res if r["model"] == best)
        base = next(r for r in res if r["model"] == "Baseline (history only)")
        coxr = next(r for r in res if r["model"] == "Cox (all features)")
        self.summary.update({"best_model": best, "uno_c": b["uno_c"], "uno_c_baseline": base["uno_c"],
                             "uno_c_cox": coxr["uno_c"], "top20_capture": float(gains[best][20])})
        mlflow.set_tag("best_model", best)

    # ------------------------------------------------------------ 9. score wells running today
    def score(self):
        S, H = self.S, settings.horizon_days
        proto, fcols = self.models[self.best]
        final = type(proto)(**proto.get_params())
        final.fit(self.X_all[fcols], Surv.from_arrays(S.E.to_numpy(bool), S["T"].to_numpy(float)))
        self.final, self.fcols = final, fcols

        active = S[(~S.E) & (S.end >= self.data_end - pd.Timedelta(days=7))].copy()
        fns = final.predict_survival_function(self.X_all.loc[active.index, fcols])
        U = np.arange(0, 181, 5)
        p, curves = [], []
        for fn, t0 in zip(fns, active["T"].to_numpy(float)):
            s0 = max(sf_at(fn, t0), 1e-9)
            p.append(1 - sf_at(fn, t0 + H) / s0)
            curves.append([sf_at(fn, t0 + u) / s0 for u in U])
        active["p_fail"] = p
        self.curve_map = dict(zip(active.index, curves))     # keyed by S index, before re-sorting
        self.U = U

        # cost assumptions
        f, dp = self.f, self.dp
        fd = f[["uwi", "failure_start_date"]].merge(dp[["uwi", "production_date", "hours_down"]]
                                                    .assign(uwi=lambda d: d.uwi.astype(str)), on="uwi")
        fd = fd[(fd.production_date >= fd.failure_start_date) &
                (fd.production_date < fd.failure_start_date + pd.Timedelta(days=21))]
        down_days = float((fd.groupby(["uwi", "failure_start_date"]).hours_down.sum() / 24).median())
        repair_cost = float(f.field_est_total.mean())
        price_m3 = settings.price_cad_per_bbl * 6.29

        recent = self.prod[self.prod.production_date >= self.data_end - pd.Timedelta(days=30)]
        rate = recent.groupby("uwi").oil_produced_m3.median()
        active["oil_rate"] = active.uwi.map(rate).fillna(0)
        active["exp_cost"] = active.p_fail * (active.oil_rate * down_days * price_m3 + repair_cost)
        active = active.sort_values("exp_cost", ascending=False)
        active["rank"] = np.arange(1, len(active) + 1)
        self.active = active

        tot = float(active.exp_cost.sum())
        top10 = float(active.exp_cost.head(max(int(0.1 * len(active)), 1)).sum())
        self.summary.update({"n_active": len(active), "expected_failures_90d": float(active.p_fail.sum()),
                             "expected_cost_90d": tot, "top10_share": top10 / tot if tot else None,
                             "down_days": down_days, "repair_cost": repair_cost,
                             "price_cad_per_bbl": settings.price_cad_per_bbl, "horizon_days": H})
        mlflow.log_metrics({"active_wells": len(active), "expected_failures_90d": float(active.p_fail.sum()),
                            "expected_cost_90d": tot, "median_downtime_days": down_days,
                            "mean_repair_cost": repair_cost})
        mlflow.sklearn.log_model(final, artifact_path="survival_model")

    # ------------------------------------------------------------ 10. SHAP
    def explain(self):
        rng = np.random.default_rng(0)
        Xf = self.X_all[self.fcols]
        background = Xf.iloc[rng.choice(len(Xf), 150, replace=False)]
        explainer = shap.Explainer(self.final.predict, shap.maskers.Independent(background, max_samples=150),
                                   feature_names=self.fcols)
        factors = [factor_of(c) for c in self.fcols]
        uniq = list(dict.fromkeys(factors))
        G = np.zeros((len(self.fcols), len(uniq)))
        for i, fct in enumerate(factors):
            G[i, uniq.index(fct)] = 1.0

        # global view
        gidx = rng.choice(len(Xf), 500, replace=False)
        sv = explainer(Xf.iloc[gidx], silent=True)
        grouped = sv.values @ G
        imp = np.abs(grouped).mean(0)
        order = np.argsort(-imp)
        strips = []
        for j in order[:8]:
            fct = uniq[j]
            if fct not in NUMERIC:
                continue
            vals = Xf.iloc[gidx][fct].to_numpy()
            pct = pd.Series(vals).rank(pct=True).to_numpy()
            pts = [{"shap": float(a), "pct": float(b)} for a, b in zip(grouped[:, j], pct)]
            strips.append({"factor": fct, "name": FACTOR_NAMES.get(fct, fct), "points": pts[:300]})
        self.charts["shap"] = {
            "importance": [{"factor": uniq[j], "name": FACTOR_NAMES.get(uniq[j], uniq[j]), "mean_abs": float(imp[j])}
                           for j in order],
            "strips": strips[:6], "base_value": float(np.mean(sv.base_values)), "n": len(gidx),
        }

        # Cox cross-check (ungrouped columns, same log-hazard scale)
        xg = Xf.iloc[gidx]
        cc = []
        for j, name in enumerate(self.fcols):
            if name.endswith("_missing") or name not in self.cph.params_.index or xg[name].std() == 0:
                continue
            slope = float(np.polyfit(xg[name].to_numpy(), sv.values[:, j], 1)[0])
            beta = float(self.cph.params_[name])
            p = float(self.cph.summary.loc[name, "p"])
            mabs = float(np.abs(sv.values[:, j]).mean())
            cc.append({"feature": name, "label": FEATURE_LABELS.get(name, name.replace("_", " ")),
                       "cox_beta": beta, "shap_slope": slope, "cox_p": p, "mean_abs_shap": mabs,
                       "agree": bool(np.sign(beta) == np.sign(slope))})
        med = np.median([c["mean_abs_shap"] for c in cc])
        for c in cc:
            c["important"] = bool(c["cox_p"] < 0.05 or c["mean_abs_shap"] > med)
        imp_rows = [c for c in cc if c["important"]]
        agree = float(np.mean([c["agree"] for c in imp_rows])) if imp_rows else None
        self.charts["crosscheck"] = {"rows": cc, "agreement": agree, "n_important": len(imp_rows)}
        self.summary["crosscheck_agreement"] = agree
        mlflow.log_metric("crosscheck_sign_agreement", agree or 0)

        # reason codes for every active well
        A = self.active
        sva = explainer(Xf.loc[A.index], silent=True)
        ga = sva.values @ G
        self.base_value = float(np.mean(sva.base_values))
        reasons, top_reason = [], {}
        for i, (idx, row) in enumerate(A.iterrows()):
            contrib = ga[i]
            order = np.argsort(-np.abs(contrib))[:6]
            k = 0
            for j in order:
                fct = uniq[j]
                lbl = self.reason_label(fct, row)
                if lbl is None:
                    continue
                reasons.append({"uwi": row.uwi, "ord": k, "factor": FACTOR_NAMES.get(fct, fct), "label": lbl,
                                "shap": float(contrib[j]), "multiplier": float(np.exp(contrib[j]))})
                if row.uwi not in top_reason and contrib[j] > 0:
                    top_reason[row.uwi] = lbl
                k += 1
        self.reasons, self.top_reason = reasons, top_reason
        self.model_values = dict(zip(A.uwi, sva.values.sum(1) + sva.base_values))

    @staticmethod
    def reason_label(fct, r):
        def has(v):
            return v is not None and not (isinstance(v, float) and np.isnan(v))
        if fct == "fails_last_365":
            return f"{r.fails_365:.0f} repairs in the 12 months before the last restart"
        if fct == "prior_fails":
            return f"{r.prior_fails:.0f} repairs since 2021"
        if fct == "area":
            return f"Field area: {r.area}"
        if fct == "last_repair":
            return f"Last repair was {r.prev_cause_grp.lower()}"
        if fct == "make":
            return f"Pump make: {r.make}"
        if fct == "rpm_100":
            return f"Pump speed {r.rpm:.0f} RPM" if has(r.rpm) else "Pump speed not recorded"
        if fct == "log_oil":
            return f"Oil rate {r.oil:.1f} m³/d after restart" if has(r.oil) else "Oil rate not recorded"
        if fct == "water_cut_10pp":
            return f"Water cut {r.water_cut:.0%}" if has(r.water_cut) else "Water cut not recorded"
        if fct == "efficiency":
            return f"Pump efficiency {r.eff:.0%}" if has(r.eff) else "Pump efficiency not recorded"
        if fct == "torque_1000":
            return f"Rod torque {r.torque:,.0f} ft·lb" if has(r.torque) else "Rod torque not recorded"
        if fct == "disp_100":
            return f"Pump size {r.disp:.0f} m³/d @100 RPM" if has(r.disp) else "Pump size not recorded"
        if fct == "age_10y":
            return f"Well age {r.age_yrs:.0f} years" if has(r.age_yrs) else "Well age unknown"
        return None

    # ------------------------------------------------------------ 11. save
    def save(self):
        A, f = self.active, self.f
        wells = [{"uwi": r.uwi, "rank": int(r["rank"]), "area": r.area, "p_fail_90d": float(r.p_fail),
                  "exp_cost_90d": float(r.exp_cost), "oil_rate_m3d": float(r.oil_rate), "days_running": int(r["T"]),
                  "repairs_total": int(r.prior_fails), "repairs_365": int(r.fails_365),
                  "rpm": None if pd.isna(r.rpm) else float(r.rpm), "last_repair": r.prev_cause_grp,
                  "top_reason": self.top_reason.get(r.uwi)} for _, r in A.iterrows()]

        hist = f[f.uwi.isin(set(A.uwi))]
        hist_by = {u: [{"date": d.strftime("%Y-%m-%d"), "component": c, "job_type": jt,
                        "cost": None if pd.isna(cost) else float(cost)}
                       for d, c, jt, cost in zip(g.failure_start_date, g.cause, g.failure_type, g.field_est_total)]
                   for u, g in hist.groupby("uwi")}
        recent = self.prod[(self.prod.production_date >= self.data_end - pd.Timedelta(days=182)) &
                           (self.prod.uwi.isin(set(A.uwi)))]
        weekly = (recent.groupby(["uwi", pd.Grouper(key="production_date", freq="W")]).oil_produced_m3.mean()
                  .reset_index())
        prod_by = {u: [{"week": d.strftime("%Y-%m-%d"), "oil": None if pd.isna(o) else float(o)}
                       for d, o in zip(g.production_date, g.oil_produced_m3)] for u, g in weekly.groupby("uwi")}
        details = [{"uwi": r.uwi,
                    "curve": [{"day": int(u), "s": float(s)} for u, s in zip(self.U, self.curve_map[idx])],
                    "history": hist_by.get(r.uwi, []), "production": prod_by.get(r.uwi, []),
                    "base_value": self.base_value, "model_value": float(self.model_values[r.uwi])}
                   for idx, r in A.iterrows()]

        charts = {k: clean(v) for k, v in self.charts.items()}
        db.save_results(self.run_id, charts, clean(wells), clean(self.reasons), clean(details))
        for k, v in charts.items():
            mlflow.log_dict(v, f"charts/{k}.json")
        mlflow.log_dict(clean(self.summary), "summary.json")


def run_pipeline(run_id: str) -> None:
    Pipeline(run_id).run()
