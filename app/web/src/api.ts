export type StepStatus = "pending" | "running" | "done" | "failed";
export interface Step { key: string; label: string; status: StepStatus; seconds: number | null }

export interface Summary {
  n_runs: number; n_failures: number; n_wells: number; infant_share: number; data_end: string;
  median_run_days: number; s90: number; s365: number; weibull_rho: number;
  best_model: string; uno_c: number; uno_c_baseline: number; uno_c_cox: number; top20_capture: number;
  n_active: number; expected_failures_90d: number; expected_cost_90d: number; top10_share: number;
  down_days: number; repair_cost: number; price_cad_per_bbl: number; horizon_days: number;
  crosscheck_agreement: number | null;
}

export interface Run {
  run_id: string; status: "running" | "success" | "failed"; current_step: string | null; steps: Step[];
  started_at: string; finished_at: string | null; mlflow_run_id: string | null; error: string | null;
  summary: Summary | null;
}

export interface EventsChart {
  quarters: { quarter: string; repairs: number; logged: number }[];
  precision: number; recall: number; n_repairs: number; n_logged: number;
}
export interface KmChart {
  points: { t: number; s: number; lo: number; hi: number }[];
  median: number | null; milestones: Record<string, number>; n: number; events: number;
}
export interface HazardChart { points: { t: number; h: number; H: number }[]; bandwidth: number }
export interface Grouping {
  key: string; label: string; ordinal: boolean; p: number; chi2: number; df: number;
  groups: { name: string; n: number; events: number; median: number | null; s90: number; s365: number; s: number[] }[];
}
export interface GroupsChart { t: number[]; groupings: Grouping[] }
export interface ParametricChart {
  t: number[]; km: number[];
  models: { name: string; aic: number; aic_delta: number; loglik: number; s: number[] }[];
  weibull: { rho: number; rho_lo: number; rho_hi: number; lambda: number };
  hazard_t: number[]; hazard_groups: { name: string; rho: number; h: number[] }[];
}
export interface CoxRow { feature: string; label: string; coef: number; hr: number; lo: number; hi: number; p: number }
export interface CoxChart {
  rows: CoxRow[]; concordance: number; n: number; events: number; penalizer: number;
  ph: { feature: string; p: number }[];
}
export interface ModelMetrics { model: string; harrell_c: number; uno_c: number; mean_auc: number; ibs: number }
export interface ValidationChart {
  metrics: ModelMetrics[]; times: number[]; auc: Record<string, number[]>; best: string;
  n_train: number; n_test: number; events_train: number; events_test: number; cutoff: string;
}
export interface GainsChart {
  x: number[]; curves: Record<string, number[]>; best: string; horizon: number; n: number; events: number;
  top20: number; top20_baseline: number;
}
export interface ShapChart {
  importance: { factor: string; name: string; mean_abs: number }[];
  strips: { factor: string; name: string; points: { shap: number; pct: number }[] }[];
  base_value: number; n: number;
}
export interface CrossRow {
  feature: string; label: string; cox_beta: number; shap_slope: number; cox_p: number;
  mean_abs_shap: number; agree: boolean; important: boolean;
}
export interface CrossCheckChart { rows: CrossRow[]; agreement: number | null; n_important: number }

export interface Charts {
  events: EventsChart; km: KmChart; hazard: HazardChart; groups: GroupsChart; parametric: ParametricChart;
  cox: CoxChart; validation: ValidationChart; gains: GainsChart; shap: ShapChart; crosscheck: CrossCheckChart;
}

export interface WellRow {
  uwi: string; rank: number; area: string; p_fail_90d: number; exp_cost_90d: number; oil_rate_m3d: number;
  days_running: number; repairs_total: number; repairs_365: number; rpm: number | null; last_repair: string;
  top_reason: string | null;
}
export interface Reason { factor: string; label: string; shap: number; multiplier: number }
export interface WellDetail extends WellRow {
  reasons: Reason[];
  curve: { day: number; s: number }[];
  history: { date: string; component: string; job_type: string; cost: number | null }[];
  production: { week: string; oil: number | null }[];
  base_value: number; model_value: number;
}

export interface DataFile { name: string; present: boolean; source: "mounted" | "uploaded" | null; size: number | null }
export interface DataStatus { ready: boolean; files: DataFile[] }

/** Upload raw parquet files with progress (XHR, since fetch has no upload progress). */
function uploadData(files: File[], onProgress: (fraction: number) => void): Promise<DataStatus> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f, f.name));
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/data");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    xhr.onload = () => {
      let body: unknown = null;
      try { body = JSON.parse(xhr.responseText); } catch { /* not json */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as DataStatus);
      else {
        const d = (body as { detail?: unknown } | null)?.detail;
        reject(new Error(typeof d === "string" ? d : `Upload failed (${xhr.status} ${xhr.statusText})`));
      }
    };
    xhr.onerror = () => reject(new Error("Upload failed: network error"));
    xhr.send(form);
  });
}

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : j.detail?.message ?? msg; } catch { /* keep */ }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

export const api = {
  config: () => req<{ mlflow_url: string; experiment: string; price_cad_per_bbl: number; horizon_days: number }>("/api/config"),
  latest: () => req<{ run: Run | null; active_run_id: string | null }>("/api/latest"),
  charts: () => req<{ run_id: string; charts: Charts }>("/api/charts"),
  dataStatus: () => req<DataStatus>("/api/data"),
  uploadData,
  startRun: () => req<{ run_id: string }>("/api/pipeline/run", { method: "POST" }),
  run: (id: string) => req<Run>(`/api/pipeline/runs/${id}`),
  wells: (p: { limit?: number; area?: string; q?: string }) => {
    const qs = new URLSearchParams();
    qs.set("limit", String(p.limit ?? 200));
    if (p.area) qs.set("area", p.area);
    if (p.q) qs.set("q", p.q);
    return req<{ run_id: string; areas: string[]; wells: WellRow[] }>(`/api/wells?${qs}`);
  },
  well: (uwi: string) => req<WellDetail>(`/api/wells/${encodeURIComponent(uwi)}`),
};
