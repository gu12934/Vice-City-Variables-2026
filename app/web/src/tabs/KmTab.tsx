import { Area, CartesianGrid, ComposedChart, Line, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { KmChart, Summary } from "../api";
import { Tex } from "../components/Tex";
import { ChartTitle, TabPage, axisProps, gridProps, tooltipStyle } from "../components/ui";
import { SERIES, days, num, pct } from "../format";

export function KmTab({ d, summary }: { d: KmChart; summary: Summary }) {
  const data = d.points.map((p) => ({ t: p.t, s: p.s, band: [p.lo, p.hi] }));
  const ms = Object.entries(d.milestones).map(([t, s]) => ({ t: Number(t), s }));
  return (
    <TabPage
      title="2 · Kaplan–Meier survival curve"
      lead="Of all runs that start after a repair, what share are still running after t days? No distribution is assumed."
      chart={<>
        <ChartTitle sub={`${num(d.n)} runs · ${num(d.events)} failures · shaded band = 95% confidence interval`}>
          Share of runs not yet failed, Ŝ(t)
        </ChartTitle>
        <div className="chart-box tall">
          <ResponsiveContainer>
            <ComposedChart data={data} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="t" type="number" domain={[0, 730]} ticks={[0, 90, 180, 270, 365, 455, 545, 635, 730]} {...axisProps}
                label={{ value: "run-days since restart (after 7-day landmark)", position: "insideBottom", offset: -4, fill: "var(--muted)", fontSize: 11 }} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => pct(v)} {...axisProps} />
              <Tooltip {...tooltipStyle} labelFormatter={(t) => `day ${t}`}
                formatter={(v, name) => name === "band" && Array.isArray(v) ? [`${pct(v[0] as number)} – ${pct(v[1] as number)}`, "95% CI"] : [pct(v as number, 1), "Ŝ(t)"]} />
              <Area type="stepAfter" dataKey="band" stroke="none" fill={SERIES[0]} fillOpacity={0.15} isAnimationActive={false} />
              <Line type="stepAfter" dataKey="s" stroke={SERIES[0]} strokeWidth={2.2} dot={false} isAnimationActive={false} />
              {ms.map((m) => (
                <ReferenceDot key={m.t} x={m.t} y={m.s} r={5} fill={SERIES[0]} stroke="var(--surface)" strokeWidth={2}
                  label={{ value: `day ${m.t}: ${pct(m.s)}`, position: "top", fill: "var(--ink-2)", fontSize: 11, offset: 10 }} />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </>}
      math={<>
        <p>Sort the distinct failure times <Tex>{String.raw`t_1<t_2<\dots`}</Tex>. At each, <Tex>n_i</Tex> runs are still
          <strong> at risk</strong> and <Tex>d_i</Tex> fail. The product-limit estimator multiplies the conditional chances of surviving each step:</p>
        <Tex block>{String.raw`\hat S(t)=\prod_{t_i\le t}\left(1-\frac{d_i}{n_i}\right)`}</Tex>
        <p>A censored run leaves the risk set <Tex>n_i</Tex> without ever counting as a failure. That is how censoring is
          handled without bias. The band uses <strong>Greenwood's formula</strong>:</p>
        <Tex block>{String.raw`\widehat{\operatorname{Var}}\big[\hat S(t)\big]=\hat S(t)^2\sum_{t_i\le t}\frac{d_i}{n_i\,(n_i-d_i)}`}</Tex>
        <p>The <strong>median survival</strong> is the first <Tex>t</Tex> where <Tex>{String.raw`\hat S(t)\le 0.5`}</Tex>.</p>
      </>}
      insight={<ul>
        <li>Median time from a repair to the next failure: <strong>{days(d.median)}</strong> of running.</li>
        <li>Only <strong>{pct(d.milestones["90"])}</strong> of runs reach day 90, and <strong>{pct(d.milestones["365"])}</strong> reach a year.</li>
        <li>Another <strong>{pct(summary.infant_share)}</strong> of failures happen in the first week after a repair (excluded by the landmark, see the Hazard tab).</li>
      </ul>}
    />
  );
}
