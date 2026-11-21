import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { HazardChart } from "../api";
import { Tex } from "../components/Tex";
import { ChartTitle, TabPage, axisProps, gridProps, tooltipStyle } from "../components/ui";
import { SERIES } from "../format";

export function HazardTab({ d }: { d: HazardChart }) {
  const early = d.points.find((p) => p.t === 10)?.h ?? 0;
  const late = d.points.find((p) => p.t === 300)?.h ?? 0;
  return (
    <TabPage
      title="3 · The hazard: when are wells most likely to fail?"
      lead="The hazard is the failure rate among wells still running. Its shape tells you whether failures come from wear-out or from repairs that don't hold."
      chart={<div className="chart-row">
        <div>
          <ChartTitle sub={`kernel-smoothed, bandwidth ${d.bandwidth} days`}>Hazard ĥ(t): failures per 100 running wells per day</ChartTitle>
          <div className="chart-box">
            <ResponsiveContainer>
              <LineChart data={d.points} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="t" type="number" domain={[0, 700]} {...axisProps} />
                <YAxis {...axisProps} domain={[0, "auto"]} tickFormatter={(v) => Number(v).toFixed(2)} />
                <Tooltip {...tooltipStyle} labelFormatter={(t) => `day ${t}`} formatter={(v) => [Number(v).toFixed(3), "per 100 wells/day"]} />
                <Line type="monotone" dataKey="h" stroke={SERIES[1]} strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div>
          <ChartTitle sub="Nelson–Aalen estimator">Cumulative hazard Ĥ(t)</ChartTitle>
          <div className="chart-box">
            <ResponsiveContainer>
              <LineChart data={d.points} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="t" type="number" domain={[0, 700]} {...axisProps} />
                <YAxis {...axisProps} tickFormatter={(v) => Number(v).toFixed(1)} />
                <Tooltip {...tooltipStyle} labelFormatter={(t) => `day ${t}`} formatter={(v) => [Number(v).toFixed(2), "Ĥ(t)"]} />
                <Line type="stepAfter" dataKey="H" stroke={SERIES[0]} strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>}
      math={<>
        <p>The <strong>Nelson–Aalen</strong> estimator adds up the observed failure fraction at each failure time:</p>
        <Tex block>{String.raw`\hat H(t)=\sum_{t_i\le t}\frac{d_i}{n_i}`}</Tex>
        <p>Its slope is the hazard. Smoothing the jumps with a kernel <Tex>K</Tex> of bandwidth <Tex>b</Tex> estimates <Tex>h(t)</Tex> directly:</p>
        <Tex block>{String.raw`\hat h(t)=\frac{1}{b}\sum_{i}K\!\left(\frac{t-t_i}{b}\right)\frac{d_i}{n_i}`}</Tex>
        <p>Reading the shape: <strong>falling</strong> means early failures (problems right after restart),
          <strong> flat</strong> means random failures, and <strong>rising</strong> means wear-out.</p>
      </>}
      insight={<ul>
        <li>The hazard is <strong>highest right after a restart</strong> (~{early.toFixed(2)} per 100 wells/day at day 10) and falls to ~{late.toFixed(2)} by day 300.</li>
        <li>A falling hazard means failures are driven by <strong>repairs that don't hold</strong> or unresolved root causes, not by parts wearing out.</li>
        <li>Practical consequence: scheduled preventive replacement would <em>not</em> reduce failures. Better repairs on chronic wells would.</li>
      </ul>}
    />
  );
}
