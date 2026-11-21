import { CartesianGrid, Line, LineChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { GainsChart } from "../api";
import { Tex } from "../components/Tex";
import { ChartTitle, Legend, TabPage, axisProps, gridProps, tooltipStyle } from "../components/ui";
import { AXIS, MUTED, SERIES, num, pct } from "../format";

export function GainsTab({ d }: { d: GainsChart }) {
  const baseName = Object.keys(d.curves).find((k) => k !== d.best) ?? "";
  const data = d.x.map((x, i) => ({ x, model: d.curves[d.best][i], base: d.curves[baseName]?.[i], random: x }));
  return (
    <TabPage
      title="8 · Gains: does the ranking catch failures?"
      lead={`If crews could visit only the riskiest x% of wells, what share of the failures in the next ${d.horizon} days would they reach?`}
      chart={<>
        <ChartTitle sub={`2025–26 test runs · ${num(d.n)} runs with a known ${d.horizon}-day outcome · ${num(d.events)} failed within ${d.horizon} days`}>
          Cumulative gains
        </ChartTitle>
        <Legend items={[{ label: d.best, color: SERIES[1] }, { label: baseName, color: MUTED }, { label: "Random", color: AXIS, dashed: true }]} />
        <div className="chart-box tall">
          <ResponsiveContainer>
            <LineChart data={data} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="x" type="number" domain={[0, 1]} tickFormatter={(v) => pct(v)} {...axisProps}
                label={{ value: "share of wells checked (highest risk first)", position: "insideBottom", offset: -4, fill: "var(--muted)", fontSize: 11 }} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => pct(v)} {...axisProps} />
              <Tooltip {...tooltipStyle} labelFormatter={(x) => `check top ${pct(Number(x))}`}
                formatter={(v, n) => [pct(v as number, 1), n === "model" ? d.best : n === "base" ? baseName : "Random"]} />
              <Line dataKey="random" stroke={AXIS} strokeDasharray="5 4" dot={false} strokeWidth={1.5} isAnimationActive={false} />
              <Line dataKey="base" stroke={MUTED} dot={false} strokeWidth={2} isAnimationActive={false} />
              <Line dataKey="model" stroke={SERIES[1]} dot={false} strokeWidth={2.4} isAnimationActive={false} />
              <ReferenceDot x={0.2} y={d.top20} r={5} fill={SERIES[1]} stroke="var(--surface)" strokeWidth={2}
                label={{ value: `top 20% → ${pct(d.top20)} of failures`, position: "right", fill: "var(--ink-2)", fontSize: 12, offset: 10 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </>}
      math={<>
        <p>Sort test runs by predicted risk <Tex>{String.raw`\hat r`}</Tex>, highest first. The gains curve at <Tex>q</Tex> is the share of
          <Tex>{String.raw`\;H`}</Tex>-day failures inside the top <Tex>q</Tex> fraction:</p>
        <Tex block>{String.raw`G(q)=\frac{\sum_{i\,\in\,\text{top }q}\mathbb 1\{T_i\le H,\ \delta_i=1\}}{\sum_{i}\mathbb 1\{T_i\le H,\ \delta_i=1\}}`}</Tex>
        <p>Random selection gives <Tex>G(q)=q</Tex>. The <strong>lift</strong> at <Tex>q</Tex> is <Tex>{String.raw`G(q)/q`}</Tex>.
          Runs censored before day <Tex>H</Tex> without failing have an unknown outcome and are left out.</p>
      </>}
      insight={<ul>
        <li>Checking the riskiest <strong>20%</strong> reaches <strong>{pct(d.top20)}</strong> of upcoming failures, a lift of <strong>{(d.top20 / 0.2).toFixed(2)}×</strong> over random.</li>
        <li>Repair history alone gets {pct(d.top20_baseline)}, so most of the lift comes from history and the full model adds a little on top.</li>
        <li>It's a prioritisation tool, not an oracle: many failures still come from wells lower on the list.</li>
      </ul>}
    />
  );
}
