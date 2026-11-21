import { CartesianGrid, LabelList, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import type { CrossCheckChart } from "../api";
import { Tex } from "../components/Tex";
import { ChartTitle, TabPage, axisProps, tooltipStyle } from "../components/ui";
import { AXIS, MUTED, SERIES, pct, pval } from "../format";

export function CrossCheckTab({ d }: { d: CrossCheckChart }) {
  const lim = 1.1 * Math.max(0.1, ...d.rows.flatMap((r) => [Math.abs(r.cox_beta), Math.abs(r.shap_slope)]));
  const agree = d.rows.filter((r) => r.important && r.agree);
  const disagree = d.rows.filter((r) => r.important && !r.agree);
  const minor = d.rows.filter((r) => !r.important);
  const imp = [...d.rows].filter((r) => r.important).sort((a, b) => b.mean_abs_shap - a.mean_abs_shap);
  return (
    <TabPage
      title="10 · Cross-check: does the ML model agree with Cox?"
      lead="Both models are on the log-hazard scale, so each feature's Cox coefficient can be compared directly with the slope the boosted model learned (from SHAP)."
      chart={<div className="chart-row">
        <div>
          <ChartTitle sub="points on the diagonal = same effect in both models">Cox β vs SHAP slope</ChartTitle>
          <div className="legend">
            <span className="legend-item"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill={SERIES[0]} /></svg>same direction</span>
            <span className="legend-item"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill={SERIES[1]} /></svg>disagree</span>
            <span className="legend-item"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill={MUTED} /></svg>minor feature</span>
          </div>
          <div className="chart-box tall">
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 8, right: 24, left: 0, bottom: 16 }}>
                <CartesianGrid stroke="var(--grid)" />
                <XAxis type="number" dataKey="cox_beta" domain={[-lim, lim]} {...axisProps} tickFormatter={(v) => Number(v).toFixed(2)}
                  label={{ value: "Cox β (log-hazard per unit)", position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 11 }} />
                <YAxis type="number" dataKey="shap_slope" domain={[-lim, lim]} {...axisProps} tickFormatter={(v) => Number(v).toFixed(2)} />
                <ZAxis range={[60, 60]} />
                <ReferenceLine segment={[{ x: -lim, y: -lim }, { x: lim, y: lim }]} stroke={AXIS} strokeDasharray="5 4" />
                <ReferenceLine x={0} stroke={AXIS} />
                <ReferenceLine y={0} stroke={AXIS} />
                <Tooltip {...tooltipStyle} cursor={false}
                  content={({ payload }) => {
                    const r = payload?.[0]?.payload as CrossCheckChart["rows"][number] | undefined;
                    if (!r) return null;
                    return <div style={{ ...tooltipStyle.contentStyle, padding: 8 }}>
                      <strong>{r.label}</strong><br />Cox β {r.cox_beta.toFixed(3)} (p = {pval(r.cox_p)})<br />SHAP slope {r.shap_slope.toFixed(3)}
                    </div>;
                  }} />
                <Scatter data={minor} fill={MUTED} isAnimationActive={false} />
                <Scatter data={agree} fill={SERIES[0]} isAnimationActive={false}>
                  <LabelList dataKey="label" position="right" fill="var(--ink-2)" fontSize={10.5} />
                </Scatter>
                <Scatter data={disagree} fill={SERIES[1]} isAnimationActive={false}>
                  <LabelList dataKey="label" position="right" fill="var(--ink-2)" fontSize={10.5} />
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div>
          <ChartTitle sub="important = Cox p < 0.05 or above-median SHAP importance">Important features</ChartTitle>
          <table>
            <thead><tr><th>feature</th><th className="num">Cox β</th><th className="num">SHAP slope</th><th>agree</th></tr></thead>
            <tbody>{imp.map((r) => (
              <tr key={r.feature}><td>{r.label}</td><td className="num">{r.cox_beta >= 0 ? "+" : ""}{r.cox_beta.toFixed(3)}</td>
                <td className="num">{r.shap_slope >= 0 ? "+" : ""}{r.shap_slope.toFixed(3)}</td><td>{r.agree ? "✓ yes" : "✕ no"}</td></tr>
            ))}</tbody>
          </table>
        </div>
      </div>}
      math={<>
        <p>In Cox the log-hazard is linear, so feature <Tex>j</Tex> has one global slope <Tex>{String.raw`\beta_j`}</Tex>. For the boosted model,
          regress each run's SHAP value on its feature value to get the <strong>average slope the model learned</strong>:</p>
        <Tex block>{String.raw`\hat\gamma_j=\arg\min_{a,\gamma}\sum_i\big(\phi_j(\mathbf x_i)-a-\gamma\,x_{ij}\big)^2`}</Tex>
        <p>If <Tex>{String.raw`\operatorname{sign}(\hat\gamma_j)=\operatorname{sign}(\beta_j)`}</Tex> and the sizes are similar, the ML model has learned the
          same relationship as the transparent model. Disagreements point to non-linear effects, interactions, or a data problem worth investigating.</p>
      </>}
      insight={<ul>
        <li><strong>{pct(d.agreement)}</strong> of the {d.n_important} important features agree in direction between Cox and the boosted model.</li>
        <li>Points sit close to the diagonal, so the effect <em>sizes</em> match too. The ML model isn't relying on anything the Cox model contradicts.</li>
        <li>That's why its predictions can be trusted, and why the Cox hazard ratios are a fair summary for a non-technical audience.</li>
      </ul>}
    />
  );
}
