import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Summary, type WellDetail, type WellRow } from "../api";
import { Tex } from "../components/Tex";
import { ChartTitle, Stat, axisProps, gridProps, tooltipStyle } from "../components/ui";
import { AXIS, SERIES, money, num, pct } from "../format";

const riskColor = (p: number) => (p >= 0.75 ? "var(--critical)" : p >= 0.5 ? "var(--serious)" : p >= 0.25 ? "var(--warning)" : "var(--good)");
const riskLabel = (p: number) => (p >= 0.75 ? "very high" : p >= 0.5 ? "high" : p >= 0.25 ? "elevated" : "low");

export function WorkListTab({ summary, runId }: { summary: Summary; runId: string }) {
  const [rows, setRows] = useState<WellRow[]>([]);
  const [areas, setAreas] = useState<string[]>([]);
  const [area, setArea] = useState("");
  const [q, setQ] = useState("");
  const [sel, setSel] = useState<string | null>(null);
  const [detail, setDetail] = useState<WellDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const t = setTimeout(() => {
      api.wells({ limit: 300, area: area || undefined, q: q || undefined })
        .then((r) => { setRows(r.wells); setAreas(r.areas); setErr(null); if (!sel && r.wells[0]) setSel(r.wells[0].uwi); })
        .catch((e: Error) => setErr(e.message));
    }, 200);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [area, q, runId]);

  useEffect(() => {
    if (!sel) return;
    api.well(sel).then(setDetail).catch((e: Error) => setErr(e.message));
  }, [sel, runId]);

  return (
    <section className="tab-page">
      <header className="tab-head">
        <h2>11 · Work list: which wells to check first</h2>
        <p className="lead">Every well running today, ranked by expected failure cost over the next {summary.horizon_days} days, with the
          reasons behind each score. Click a well for its details.</p>
      </header>
      {err && <div className="error-box">{err}</div>}
      <div className="worklist">
        <div className="card">
          <div className="filters">
            <select value={area} onChange={(e) => setArea(e.target.value)} aria-label="Filter by area">
              <option value="">All areas</option>
              {areas.map((a) => <option key={a} value={a}>{a}</option>)}
            </select>
            <input placeholder="Search well ID" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search well" />
            <span className="note" style={{ alignSelf: "center", marginTop: 0 }}>{rows.length} wells shown</span>
          </div>
          <div className="table-scroll">
            <table>
              <thead><tr><th className="num">#</th><th>well</th><th>area</th><th>90-day risk</th><th className="num">expected cost</th><th>main reason</th></tr></thead>
              <tbody>{rows.map((r) => (
                <tr key={r.uwi} className={`clickable ${sel === r.uwi ? "selected" : ""}`} onClick={() => setSel(r.uwi)}>
                  <td className="num">{r.rank}</td>
                  <td><code>{r.uwi}</code></td>
                  <td>{r.area}</td>
                  <td><span className="risk-pill"><span className="dot" style={{ background: riskColor(r.p_fail_90d) }} />{pct(r.p_fail_90d)}
                    <span className="reason-small">{riskLabel(r.p_fail_90d)}</span></span></td>
                  <td className="num">{money(r.exp_cost_90d)}</td>
                  <td className="reason-small">{r.top_reason ?? "–"}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        </div>
        {detail ? <WellPanel w={detail} summary={summary} /> : <div className="card empty">Select a well</div>}
      </div>
    </section>
  );
}

function WellPanel({ w, summary }: { w: WellDetail; summary: Summary }) {
  const maxAbs = Math.max(0.2, ...w.reasons.map((r) => Math.abs(r.shap)));
  const prod = w.production.filter((p) => p.oil != null);
  const s90 = w.curve.find((c) => c.day === 90)?.s;
  return (
    <div className="card" style={{ display: "grid", gap: 16 }}>
      <div>
        <h3 style={{ marginBottom: 2 }}><code>{w.uwi}</code></h3>
        <p className="note" style={{ marginTop: 0 }}>{w.area} · rank #{w.rank} · running {num(w.days_running)} days since last repair ({w.last_repair.toLowerCase()})</p>
      </div>
      <div className="stats">
        <Stat label={`Chance of failure, ${summary.horizon_days} days`} value={<span style={{ color: riskColor(w.p_fail_90d) }}>{pct(w.p_fail_90d)}</span>} hint={riskLabel(w.p_fail_90d)} />
        <Stat label="Expected cost" value={money(w.exp_cost_90d)} hint="lost oil + repair, risk-weighted" />
        <Stat label="Repairs last 12 months" value={num(w.repairs_365)} hint={`${num(w.repairs_total)} since 2021`} />
      </div>

      <div>
        <ChartTitle sub="SHAP contributions: how each factor multiplies this well's failure rate vs the fleet average">Why this well is flagged</ChartTitle>
        <div className="reasons">
          {w.reasons.map((r) => {
            const width = (Math.abs(r.shap) / maxAbs) * 50;
            return (
              <div className="reason" key={r.factor}>
                <span>{r.label}</span>
                <span className="track">
                  <span className="fill" style={{
                    left: r.shap >= 0 ? "50%" : `${50 - width}%`, width: `${width}%`,
                    background: r.shap >= 0 ? "var(--s2)" : "var(--s1)",
                  }} />
                </span>
                <span className="mult">×{r.multiplier.toFixed(2)}</span>
              </div>
            );
          })}
        </div>
        <p className="note">Orange raises risk, blue lowers it. These are associations the model learned, not proven causes.</p>
      </div>

      <div>
        <ChartTitle sub="predicted chance it keeps running, from today">Survival outlook</ChartTitle>
        <div className="chart-box short">
          <ResponsiveContainer>
            <LineChart data={w.curve} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="day" type="number" domain={[0, 180]} ticks={[0, 30, 60, 90, 120, 150, 180]} {...axisProps} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => pct(v)} {...axisProps} />
              <Tooltip {...tooltipStyle} labelFormatter={(d) => `in ${d} days`} formatter={(v) => [pct(v as number, 1), "still running"]} />
              <ReferenceLine x={90} stroke={AXIS} strokeDasharray="4 4" label={{ value: `90 d: ${pct(s90)}`, fill: "var(--ink-2)", fontSize: 11, position: "insideTopRight" }} />
              <Line dataKey="s" stroke={SERIES[0]} strokeWidth={2} dot={false} type="monotone" isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {prod.length > 2 && (
        <div>
          <ChartTitle sub="weekly average, last 6 months">Oil production (m³/d)</ChartTitle>
          <div className="chart-box short">
            <ResponsiveContainer>
              <LineChart data={prod} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="week" {...axisProps} tickFormatter={(v: string) => v.slice(5)} minTickGap={24} />
                <YAxis {...axisProps} domain={[0, "auto"]} />
                <Tooltip {...tooltipStyle} formatter={(v) => [num(v as number, 1), "m³/d"]} />
                <Line dataKey="oil" stroke={SERIES[2]} strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div>
        <ChartTitle sub={`${w.history.length} failure-repair jobs since 2021 (most recent first)`}>Repair history</ChartTitle>
        <div className="table-scroll" style={{ maxHeight: 220 }}>
          <table>
            <thead><tr><th>date</th><th>component</th><th>job</th><th className="num">cost</th></tr></thead>
            <tbody>{[...w.history].reverse().map((h, i) => (
              <tr key={i}><td>{h.date}</td><td>{h.component}</td><td>{h.job_type}</td><td className="num">{money(h.cost)}</td></tr>
            ))}</tbody>
          </table>
        </div>
      </div>

      <div className="math">
        <h3>The math behind this card</h3>
        <p>The well has already run <Tex>t_0</Tex> = {num(w.days_running)} days, so its failure chance is <strong>conditional</strong> on surviving so far:</p>
        <Tex block>{String.raw`P\big(T\le t_0+90 \mid T>t_0\big)=1-\frac{\hat S(t_0+90\mid\mathbf x)}{\hat S(t_0\mid\mathbf x)}`}</Tex>
        <p>Expected cost weighs that chance by what a failure costs: oil lost during ~{summary.down_days.toFixed(1)} days down at
          ${summary.price_cad_per_bbl}/bbl (assumed), plus the mean repair cost ({money(summary.repair_cost)}):</p>
        <Tex block>{String.raw`\mathbb E[\text{cost}]=P_{90}\cdot\big(q_{\text{oil}}\cdot d_{\text{down}}\cdot 6.29\cdot p_{\text{bbl}}+c_{\text{repair}}\big)`}</Tex>
        <p>SHAP adds up exactly: fleet average {w.base_value.toFixed(2)} + Σ contributions = this well's log-hazard {w.model_value.toFixed(2)},
          i.e. ×{Math.exp(w.model_value - w.base_value).toFixed(2)} the average failure rate.</p>
      </div>
    </div>
  );
}
