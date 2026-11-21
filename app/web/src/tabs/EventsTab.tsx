import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { EventsChart } from "../api";
import { Tex } from "../components/Tex";
import { ChartTitle, Legend, TabPage, axisProps, gridProps, tooltipStyle } from "../components/ui";
import { SERIES, num, pct } from "../format";

export function EventsTab({ d }: { d: EventsChart }) {
  return (
    <TabPage
      title="1 · Defining a failure"
      lead="The failure log only begins in 2025, so failure-repair jobs (rod, pump, tubing repairs) stand in as the failure event for 2021–26."
      chart={<>
        <ChartTitle sub="Events per quarter. Logged failures appear only from Q2 2025, while repair jobs cover all five years.">
          Failure log vs failure-repair jobs
        </ChartTitle>
        <Legend items={[{ label: "Failure-repair jobs (well_jobs)", color: SERIES[0] },
          { label: "Logged failures (production_failures)", color: SERIES[1] }]} />
        <div className="chart-box">
          <ResponsiveContainer>
            <LineChart data={d.quarters} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="quarter" {...axisProps} interval={2} />
              <YAxis {...axisProps} />
              <Tooltip {...tooltipStyle} />
              <Line type="monotone" dataKey="repairs" name="Repair jobs" stroke={SERIES[0]} strokeWidth={2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="logged" name="Logged failures" stroke={SERIES[1]} strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </>}
      math={<>
        <p>Where both sources exist (Apr 2025 on), we check how well a repair job identifies a real failure. A job and a
          failure on the same well are <strong>matched</strong> if they fall within a few days of each other.</p>
        <Tex block>{String.raw`\text{precision}=\frac{\#\{\text{repair jobs matched to a logged failure}\}}{\#\{\text{repair jobs}\}}`}</Tex>
        <Tex block>{String.raw`\text{recall}=\frac{\#\{\text{logged failures matched to a repair job}\}}{\#\{\text{logged failures}\}}`}</Tex>
        <p>Repairs on the same well less than 3 days apart are merged into one event, numbered
          <Tex>{String.raw`\;k=0,1,2,\dots`}</Tex> The time between consecutive events is one <strong>run</strong>, the unit of analysis.</p>
      </>}
      insight={<ul>
        <li>The proxy has <strong>{pct(d.precision)} precision</strong> and <strong>{pct(d.recall)} recall</strong>: most repair jobs are real failures, and almost every logged failure gets a repair job.</li>
        <li><strong>{num(d.n_repairs)}</strong> repair events over 5 years, against <strong>{num(d.n_logged)}</strong> logged failures in ~18 months.</li>
        <li>Without this fix, 2021–24 would look failure-free and any model trained on it would be meaningless.</li>
      </ul>}
    />
  );
}
