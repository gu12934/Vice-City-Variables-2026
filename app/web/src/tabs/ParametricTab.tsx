import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ParametricChart } from "../api";
import { Tex } from "../components/Tex";
import { ChartTitle, Legend, TabPage, axisProps, gridProps, tooltipStyle } from "../components/ui";
import { INK2, SERIES, num, pct } from "../format";

export function ParametricTab({ d }: { d: ParametricChart }) {
  const data = d.t.map((t, i) => {
    const row: Record<string, number> = { t, km: d.km[i] };
    d.models.forEach((m) => { row[m.name] = m.s[i]; });
    return row;
  });
  const hdata = d.hazard_t.map((t, i) => {
    const row: Record<string, number> = { t };
    d.hazard_groups.forEach((g) => { row[g.name] = g.h[i]; });
    return row;
  });
  const ranked = [...d.models].sort((a, b) => a.aic_delta - b.aic_delta);
  const w = d.weibull;
  return (
    <TabPage
      title="5 · Which distribution? Exponential vs Weibull"
      lead="Fitting a named distribution gives a smooth curve and one interpretable number: the Weibull shape ρ, which says whether risk falls, stays flat or rises over a run."
      chart={<div className="chart-row">
        <div>
          <ChartTitle sub="dashed = fitted models, solid = Kaplan–Meier data">Fitted survival vs data</ChartTitle>
          <Legend items={[{ label: "Kaplan–Meier", color: INK2 }, ...d.models.map((m, i) => ({ label: m.name, color: SERIES[i], dashed: true }))]} />
          <div className="chart-box">
            <ResponsiveContainer>
              <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="t" type="number" domain={[0, 900]} {...axisProps} />
                <YAxis domain={[0, 1]} tickFormatter={(v) => pct(v)} {...axisProps} />
                <Tooltip {...tooltipStyle} labelFormatter={(t) => `day ${t}`} formatter={(v, n) => [pct(v as number, 1), n === "km" ? "Kaplan–Meier" : n]} />
                <Line type="stepAfter" dataKey="km" stroke={INK2} strokeWidth={3} dot={false} isAnimationActive={false} />
                {d.models.map((m, i) => (
                  <Line key={m.name} type="monotone" dataKey={m.name} stroke={SERIES[i]} strokeWidth={1.8}
                    strokeDasharray="5 4" dot={false} isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div>
          <ChartTitle sub="failures per 100 running wells per day">Weibull hazard by repair history</ChartTitle>
          <Legend items={d.hazard_groups.map((g, i) => ({ label: `${g.name}: ρ = ${g.rho.toFixed(2)}`, color: SERIES[i] }))} />
          <div className="chart-box">
            <ResponsiveContainer>
              <LineChart data={hdata} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="t" type="number" domain={[0, 900]} {...axisProps} />
                <YAxis {...axisProps} tickFormatter={(v) => Number(v).toFixed(2)} />
                <Tooltip {...tooltipStyle} labelFormatter={(t) => `day ${t}`} formatter={(v, n) => [Number(v).toFixed(3), n]} />
                {d.hazard_groups.map((g, i) => (
                  <Line key={g.name} type="monotone" dataKey={g.name} stroke={SERIES[i]} strokeWidth={2} dot={false} isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>}
      math={<>
        <p><strong>Exponential</strong>: constant hazard, <em>memoryless</em>. A pump that has run 400 days is "as good as new":</p>
        <Tex block>{String.raw`h(t)=\lambda,\qquad S(t)=e^{-\lambda t},\qquad P(T>s+t\mid T>s)=P(T>t)`}</Tex>
        <p><strong>Weibull</strong> adds a shape parameter <Tex>\rho</Tex>:</p>
        <Tex block>{String.raw`S(t)=\exp\!\Big[-\Big(\tfrac{t}{\lambda}\Big)^{\rho}\Big],\qquad h(t)=\frac{\rho}{\lambda}\Big(\frac{t}{\lambda}\Big)^{\rho-1}`}</Tex>
        <p><Tex>{String.raw`\rho<1`}</Tex>: falling hazard (early failures) · <Tex>{String.raw`\rho=1`}</Tex>: exponential · <Tex>{String.raw`\rho>1`}</Tex>: wear-out.
          Parameters maximise the censored log-likelihood (failures add density, censored runs add survival):</p>
        <Tex block>{String.raw`\log L(\theta)=\sum_i\Big[\delta_i\log f(t_i;\theta)+(1-\delta_i)\log S(t_i;\theta)\Big],\qquad \mathrm{AIC}=2p-2\log L`}</Tex>
      </>}
      insight={<>
        <ul>
          <li>Weibull shape <strong>ρ = {w.rho.toFixed(2)}</strong> (95% CI {w.rho_lo.toFixed(2)}–{w.rho_hi.toFixed(2)}), clearly <strong>below 1</strong>: the hazard falls over a run.</li>
          <li>So failures cluster soon after a restart. That means repair quality and recurring root causes, <strong>not wear-out</strong>.</li>
          <li>The exponential (memoryless) fits worst. {ranked[0].name} fits best by AIC.</li>
        </ul>
        <table>
          <thead><tr><th>model</th><th className="num">log-likelihood</th><th className="num">ΔAIC</th></tr></thead>
          <tbody>{ranked.map((m) => (
            <tr key={m.name}><td>{m.name}</td><td className="num">{num(m.loglik)}</td><td className="num">{m.aic_delta === 0 ? "best" : `+${num(m.aic_delta)}`}</td></tr>
          ))}</tbody>
        </table>
      </>}
    />
  );
}
