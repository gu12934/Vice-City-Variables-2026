import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Charts, type DataStatus, type Run } from "./api";
import { DataUploadDialog } from "./components/DataUploadDialog";
import { CoxTab } from "./tabs/CoxTab";
import { CrossCheckTab } from "./tabs/CrossCheckTab";
import { EventsTab } from "./tabs/EventsTab";
import { GainsTab } from "./tabs/GainsTab";
import { GroupsTab } from "./tabs/GroupsTab";
import { HazardTab } from "./tabs/HazardTab";
import { KmTab } from "./tabs/KmTab";
import { OverviewTab } from "./tabs/OverviewTab";
import { ParametricTab } from "./tabs/ParametricTab";
import { ShapTab } from "./tabs/ShapTab";
import { ValidationTab } from "./tabs/ValidationTab";
import { WorkListTab } from "./tabs/WorkListTab";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "events", label: "Failure events", n: 1 },
  { id: "km", label: "Survival curve", n: 2 },
  { id: "hazard", label: "Hazard", n: 3 },
  { id: "groups", label: "Group comparison", n: 4 },
  { id: "parametric", label: "Distributions", n: 5 },
  { id: "cox", label: "Cox model", n: 6 },
  { id: "validation", label: "Validation", n: 7 },
  { id: "gains", label: "Gains", n: 8 },
  { id: "shap", label: "SHAP", n: 9 },
  { id: "crosscheck", label: "Cross-check", n: 10 },
  { id: "worklist", label: "Work list", n: 11 },
] as const;
type TabId = (typeof TABS)[number]["id"];

export default function App() {
  const [tab, setTab] = useState<TabId>(() => {
    try { return (localStorage.getItem("tab") as TabId) || "overview"; } catch { return "overview"; }
  });
  const [latest, setLatest] = useState<Run | null>(null);
  const [charts, setCharts] = useState<Charts | null>(null);
  const [active, setActive] = useState<Run | null>(null);
  const [mlflowUrl, setMlflowUrl] = useState("http://localhost:5000");
  const [error, setError] = useState<string | null>(null);
  const [needData, setNeedData] = useState<DataStatus | null>(null);
  const poll = useRef<number | null>(null);

  const loadLatest = useCallback(async () => {
    const l = await api.latest();
    setLatest(l.run);
    if (l.run) setCharts((await api.charts()).charts);
    return l.active_run_id;
  }, []);

  const watch = useCallback((runId: string) => {
    if (poll.current) window.clearInterval(poll.current);
    const tick = async () => {
      try {
        const r = await api.run(runId);
        setActive(r);
        if (r.status !== "running") {
          if (poll.current) window.clearInterval(poll.current);
          poll.current = null;
          if (r.status === "success") { await loadLatest(); setActive(null); }
          else setError(`Pipeline run ${r.run_id} failed:\n${r.error ?? "unknown error"}`);
        }
      } catch (e) { setError((e as Error).message); }
    };
    tick();
    poll.current = window.setInterval(tick, 2000);
  }, [loadLatest]);

  useEffect(() => {
    api.config().then((c) => setMlflowUrl(c.mlflow_url)).catch(() => undefined);
    loadLatest().then((activeId) => { if (activeId) watch(activeId); }).catch((e: Error) => setError(e.message));
    return () => { if (poll.current) window.clearInterval(poll.current); };
  }, [loadLatest, watch]);

  useEffect(() => { try { localStorage.setItem("tab", tab); } catch { /* ignore */ } }, [tab]);

  const launch = async () => {
    try { const { run_id } = await api.startRun(); watch(run_id); }
    catch (e) { setError((e as Error).message); }
  };

  // Results are pre-loaded; the raw files are only needed to recompute. Ask for them if they're missing.
  const start = async () => {
    setError(null);
    try {
      const d = await api.dataStatus();
      if (!d.ready) { setNeedData(d); return; }
      await launch();
    } catch (e) { setError((e as Error).message); }
  };

  const busy = active?.status === "running";
  const doneSteps = active?.steps.filter((s) => s.status === "done").length ?? 0;
  const summary = latest?.summary ?? null;
  const ready = charts && summary && latest;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>Well Failure Risk</h1>
          <p>Survival analysis · Strathcona heavy-oil wells · DSMLC Datathon 2026</p>
        </div>
        <div className="actions">
          {latest && <div className="run-meta">latest run <strong>{latest.run_id}</strong><br />best model: {summary?.best_model}</div>}
          <a className="btn link" href={mlflowUrl} target="_blank" rel="noreferrer">MLflow ↗</a>
          <button className="btn primary" onClick={start} disabled={busy}>
            {busy ? "Running pipeline…" : "Run pipeline"}
          </button>
        </div>
      </header>

      {busy && active && (
        <div className="progress-strip" role="status">
          <strong>Step {Math.min(doneSteps + 1, active.steps.length)}/{active.steps.length}:</strong>
          <span>{active.steps.find((s) => s.status === "running")?.label ?? "starting…"}</span>
          <div className="bar"><div style={{ width: `${(doneSteps / active.steps.length) * 100}%` }} /></div>
          <span className="note" style={{ marginTop: 0 }}>results update when it finishes; current results stay available</span>
        </div>
      )}
      {error && <div className="error-box">{error}</div>}
      {needData && (
        <DataUploadDialog status={needData} onClose={() => setNeedData(null)}
          onReady={() => { setNeedData(null); launch(); }} />
      )}

      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? "on" : ""} onClick={() => setTab(t.id)}>
            {"n" in t && <span className="n">{t.n}</span>}{t.label}
          </button>
        ))}
      </nav>

      {tab === "overview" || !ready ? (
        tab !== "overview" && !ready ? (
          <div className="card empty" style={{ marginTop: 20 }}>
            <p>No results yet. Run the pipeline to fill this tab.</p>
            <button className="btn primary" onClick={start} disabled={busy}>{busy ? "Running…" : "Run pipeline"}</button>
          </div>
        ) : (
          <OverviewTab summary={summary} run={active} latestRun={latest} onRun={start} busy={busy} mlflowUrl={mlflowUrl} />
        )
      ) : (
        <>
          {tab === "events" && <EventsTab d={charts.events} />}
          {tab === "km" && <KmTab d={charts.km} summary={summary} />}
          {tab === "hazard" && <HazardTab d={charts.hazard} />}
          {tab === "groups" && <GroupsTab d={charts.groups} />}
          {tab === "parametric" && <ParametricTab d={charts.parametric} />}
          {tab === "cox" && <CoxTab d={charts.cox} />}
          {tab === "validation" && <ValidationTab d={charts.validation} />}
          {tab === "gains" && <GainsTab d={charts.gains} />}
          {tab === "shap" && <ShapTab d={charts.shap} />}
          {tab === "crosscheck" && <CrossCheckTab d={charts.crosscheck} />}
          {tab === "worklist" && <WorkListTab summary={summary} runId={latest.run_id} />}
        </>
      )}
    </div>
  );
}
