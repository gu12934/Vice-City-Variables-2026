import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ValidationChart } from "../api";
import { Tex } from "../components/Tex";
import { ChartTitle, Legend, TabPage, axisProps, gridProps, tooltipStyle } from "../components/ui";
import { AXIS, MUTED, SERIES, num } from "../format";

const COLORS = [MUTED, SERIES[0], SERIES[2], SERIES[1]];

export function ValidationTab({ d }: { d: ValidationChart }) {
  const names = d.metrics.map((m) => m.model);
  const data = d.times.map((t, i) => {
    const row: Record<string, number> = { t };
    names.forEach((n) => { row[n] = d.auc[n][i]; });
    return row;
  });
  const best = d.metrics.find((m) => m.model === d.best)!;
  const base = d.metrics.find((m) => m.model.startsWith("Baseline"))!;
  const cox = d.metrics.find((m) => m.model.startsWith("Cox"))!;
  const maxOf = (k: "harrell_c" | "uno_c" | "mean_auc") => Math.max(...d.metrics.map((m) => m[k]));
  const minIbs = Math.min(...d.metrics.map((m) => m.ibs));
  return (
    <TabPage
      title="7 · Validation on the future (scikit-survival)"
      lead={`Models are trained only on runs before ${d.cutoff} and scored on runs that start after it, the way they would be used in practice.`}
      chart={<>
        <ChartTitle sub={`train ${num(d.n_train)} runs (${num(d.events_train)} failures) · test ${num(d.n_test)} runs (${num(d.events_test)} failures)`}>
          Test-set performance
        </ChartTitle>
        <table>
          <thead><tr><th>model</th><th className="num">Harrell's C</th><th className="num">Uno's C (τ=365)</th>
            <th className="num">mean AUC(t)</th><th className="num">integrated Brier ↓</th></tr></thead>
          <tbody>{d.metrics.map((m) => (
            <tr key={m.model}>
              <td>{m.model}{m.model === d.best ? "  ★" : ""}</td>
              <td className="num" style={{ fontWeight: m.harrell_c === maxOf("harrell_c") ? 700 : 400 }}>{m.harrell_c.toFixed(3)}</td>
              <td className="num" style={{ fontWeight: m.uno_c === maxOf("uno_c") ? 700 : 400 }}>{m.uno_c.toFixed(3)}</td>
              <td className="num" style={{ fontWeight: m.mean_auc === maxOf("mean_auc") ? 700 : 400 }}>{m.mean_auc.toFixed(3)}</td>
              <td className="num" style={{ fontWeight: m.ibs === minIbs ? 700 : 400 }}>{m.ibs.toFixed(3)}</td>
            </tr>
          ))}</tbody>
        </table>
        <div style={{ height: 16 }} />
        <ChartTitle sub="separating runs that fail by day t from runs that survive past it (0.5 = random)">Time-dependent AUC</ChartTitle>
        <Legend items={names.map((n, i) => ({ label: n, color: COLORS[i] }))} />
        <div className="chart-box">
          <ResponsiveContainer>
            <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="t" type="number" domain={[0, "dataMax"]} {...axisProps} />
              <YAxis domain={[0.45, "auto"]} {...axisProps} tickFormatter={(v) => Number(v).toFixed(2)} />
              <Tooltip {...tooltipStyle} labelFormatter={(t) => `horizon ${t} days`} formatter={(v, n) => [Number(v).toFixed(3), n]} />
              <ReferenceLine y={0.5} stroke={AXIS} strokeDasharray="4 4" />
              {names.map((n, i) => (
                <Line key={n} type="monotone" dataKey={n} stroke={COLORS[i]} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </>}
      math={<>
        <p><strong>Random survival forest</strong>: trees split where the log-rank statistic between children is largest; leaves hold
          Nelson–Aalen curves, averaged over <Tex>B</Tex> trees: <Tex>{String.raw`\hat H(t\mid\mathbf x)=\tfrac1B\sum_b \hat H_b(t\mid\mathbf x)`}</Tex>.
          <strong> Boosted Cox</strong> builds <Tex>{String.raw`f(\mathbf x)=\sum_m \nu\,g_m(\mathbf x)`}</Tex> from small trees by gradient descent on the negative partial log-likelihood.</p>
        <p><strong>Concordance</strong>: of comparable pairs, how often does the riskier score fail first? Uno's version re-weights by the censoring distribution <Tex>{String.raw`\hat G`}</Tex>:</p>
        <Tex block>{String.raw`C=P\big(\hat r_i>\hat r_j \mid T_i<T_j\big),\qquad C_{\text{Uno}}=\frac{\sum_{i,j}\delta_i\,\hat G(T_i)^{-2}\,\mathbb 1\{T_i<T_j,\,T_i<\tau\}\,\mathbb 1\{\hat r_i>\hat r_j\}}{\sum_{i,j}\delta_i\,\hat G(T_i)^{-2}\,\mathbb 1\{T_i<T_j,\,T_i<\tau\}}`}</Tex>
        <p><strong>Integrated Brier score</strong>: the squared error of the predicted survival probabilities, weighted for censoring (lower is better):</p>
        <Tex block>{String.raw`\mathrm{IBS}=\frac{1}{t_{\max}}\int_0^{t_{\max}}\frac1n\sum_i w_i(t)\big(\mathbb 1\{T_i>t\}-\hat S(t\mid\mathbf x_i)\big)^2\,dt`}</Tex>
      </>}
      insight={<ul>
        <li>Best model: <strong>{d.best}</strong>, Uno's C <strong>{best.uno_c.toFixed(3)}</strong>, against 0.500 for random and {base.uno_c.toFixed(3)} for repair history alone.</li>
        <li>The transparent Cox model scores {cox.uno_c.toFixed(3)}. The ML gain over Cox is <strong>small ({(best.uno_c - cox.uno_c).toFixed(3)})</strong>, so most of the signal is in a model you can fully explain.</li>
        <li>Repair history carries most of the predictive power; the other factors add measurable lift on top.</li>
      </ul>}
    />
  );
}
