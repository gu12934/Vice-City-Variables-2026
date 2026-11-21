import type { Run, Summary } from "../api";
import { Tex } from "../components/Tex";
import { Stat } from "../components/ui";
import { money, num, pct } from "../format";

export function OverviewTab({ summary, run, latestRun, onRun, busy, mlflowUrl }: {
  summary: Summary | null; run: Run | null; latestRun: Run | null; onRun: () => void; busy: boolean; mlflowUrl: string;
}) {
  const shown = run ?? latestRun;
  return (
    <section className="tab-page">
      <header className="tab-head">
        <h2>Which wells will fail next, and why?</h2>
        <p className="lead">
          Survival analysis of rod-pump failures on Strathcona's heavy-oil wells. One click runs the whole pipeline:
          it builds the data, fits the statistical and ML models, validates them on 2025–26, explains every alert with
          SHAP, logs the run to MLflow and writes the results to Postgres. Each tab shows one step, with the math behind it.
        </p>
      </header>

      {summary ? (
        <div className="stats">
          <Stat label="Wells monitored now" value={num(summary.n_active)} hint={`data through ${summary.data_end}`} />
          <Stat label={`Expected failures, next ${summary.horizon_days} days`} value={num(summary.expected_failures_90d)} />
          <Stat label={`Expected failure cost, next ${summary.horizon_days} days`} value={money(summary.expected_cost_90d)}
            hint={`top 10% of wells carry ${pct(summary.top10_share)}`} />
          <Stat label="Median run after a repair" value={`${num(summary.median_run_days)} days`}
            hint={`${pct(summary.s365)} of runs last a year`} />
          <Stat label="Ranking quality (Uno's C, 2025–26)" value={summary.uno_c.toFixed(3)}
            hint={`random 0.500 · history only ${summary.uno_c_baseline.toFixed(3)}`} />
        </div>
      ) : (
        <div className="card empty">
          <p>No results yet. Run the pipeline to train the models and fill every tab.</p>
          <button className="btn primary" onClick={onRun} disabled={busy}>Run pipeline</button>
        </div>
      )}

      <div className="two-col">
        <div className="card">
          <h3>Pipeline {shown ? <span className="note">run {shown.run_id} · {shown.status}</span> : null}</h3>
          {shown ? (
            <ul className="steps">
              {shown.steps.map((s) => (
                <li key={s.key} className={s.status}>
                  <span className="ico" aria-label={s.status}>
                    {s.status === "done" ? "✓" : s.status === "failed" ? "✕" : s.status === "running" ? "•" : ""}
                  </span>
                  <span>{s.label}</span>
                  <span className="secs">{s.seconds != null ? `${s.seconds.toFixed(1)} s` : s.status === "running" ? "running…" : ""}</span>
                </li>
              ))}
            </ul>
          ) : <p className="note">No runs yet.</p>}
          {shown?.mlflow_run_id && (
            <p className="note">MLflow run <code>{shown.mlflow_run_id.slice(0, 8)}</code>: params, metrics, charts and the
              trained model are in <a href={mlflowUrl} target="_blank" rel="noreferrer">MLflow</a>.</p>
          )}
        </div>
        <div className="card math">
          <h3>The math: what survival analysis models</h3>
          <p>For each run (repair → next failure) let <Tex>T</Tex> be the time to failure. Many runs are still going when
            the data ends. We only know <Tex>{String.raw`T > c`}</Tex> (<strong>right-censoring</strong>), so ordinary
            regression or classification would be biased.</p>
          <Tex block>{String.raw`S(t)=P(T>t),\qquad h(t)=\lim_{\Delta\to 0}\frac{P(t\le T<t+\Delta\mid T\ge t)}{\Delta},\qquad H(t)=\int_0^t h(u)\,du`}</Tex>
          <Tex block>{String.raw`S(t)=e^{-H(t)}`}</Tex>
          <p><strong>Survival</strong> <Tex>S(t)</Tex> is the chance a run is still going at day <Tex>t</Tex>.
            The <strong>hazard</strong> <Tex>h(t)</Tex> is the failure rate <em>given it has survived so far</em>, the
            quantity maintenance decisions depend on.</p>
        </div>
      </div>

      <div className="card">
        <h3>How the system fits together</h3>
        <dl className="kv">
          <dt>React SPA</dt><dd>this app: run button, progress, one tab per visualization, work list</dd>
          <dt>FastAPI</dt><dd>starts the pipeline in the background, serves results from Postgres</dd>
          <dt>Pipeline</dt><dd>pandas → lifelines (Kaplan–Meier, Cox, Weibull) → scikit-survival (RSF, boosted Cox) → SHAP</dd>
          <dt>MLflow</dt><dd>every run's parameters, validation metrics, chart data and trained model (<a href={mlflowUrl} target="_blank" rel="noreferrer">open UI</a>)</dd>
          <dt>Postgres</dt><dd>results versioned by run; the app always reads the latest successful run</dd>
        </dl>
      </div>
    </section>
  );
}
