import { useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { GroupsChart } from "../api";
import { Tex } from "../components/Tex";
import { ChartTitle, Legend, Segmented, TabPage, axisProps, gridProps, tooltipStyle } from "../components/ui";
import { ORDINAL, SERIES, days, num, pct, pval } from "../format";

export function GroupsTab({ d }: { d: GroupsChart }) {
  const [key, setKey] = useState(d.groupings[0]?.key ?? "");
  const g = d.groupings.find((x) => x.key === key) ?? d.groupings[0];
  const colors = g.ordinal ? ORDINAL : SERIES;
  const data = d.t.map((t, i) => {
    const row: Record<string, number> = { t };
    g.groups.forEach((grp) => { row[grp.name] = grp.s[i]; });
    return row;
  });
  const best = [...g.groups].sort((a, b) => (b.median ?? 1e9) - (a.median ?? 1e9))[0];
  const worst = [...g.groups].sort((a, b) => (a.median ?? 1e9) - (b.median ?? 1e9))[0];

  return (
    <TabPage
      title="4 · Comparing groups: log-rank test"
      lead="Do some kinds of wells fail sooner? Kaplan–Meier curves per group, and a test of whether the difference could be chance."
      chart={<>
        <Segmented value={key} onChange={setKey}
          options={d.groupings.map((x) => ({ value: x.key, label: x.label }))} />
        <ChartTitle sub={`log-rank χ² = ${num(g.chi2, 1)} on ${g.df} df · p = ${pval(g.p)}`}>
          Survival by {g.label.toLowerCase()}
        </ChartTitle>
        <Legend items={g.groups.map((grp, i) => ({ label: `${grp.name} (n=${num(grp.n)})`, color: colors[i] }))} />
        <div className="chart-box tall">
          <ResponsiveContainer>
            <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="t" type="number" domain={[0, 730]} ticks={[0, 90, 180, 270, 365, 455, 545, 635, 730]} {...axisProps} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => pct(v)} {...axisProps} />
              <Tooltip {...tooltipStyle} labelFormatter={(t) => `day ${t}`} formatter={(v, n) => [pct(v as number, 1), n]} />
              {g.groups.map((grp, i) => (
                <Line key={grp.name} type="stepAfter" dataKey={grp.name} stroke={colors[i]} strokeWidth={2}
                  dot={false} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <table>
          <thead><tr><th>{g.label}</th><th className="num">runs</th><th className="num">failures</th>
            <th className="num">median run</th><th className="num">S(90)</th><th className="num">S(365)</th></tr></thead>
          <tbody>{g.groups.map((grp) => (
            <tr key={grp.name}><td>{grp.name}</td><td className="num">{num(grp.n)}</td><td className="num">{num(grp.events)}</td>
              <td className="num">{days(grp.median)}</td><td className="num">{pct(grp.s90)}</td><td className="num">{pct(grp.s365)}</td></tr>
          ))}</tbody>
        </table>
      </>}
      math={<>
        <p>Under <Tex>{String.raw`H_0: h_1(t)=\dots=h_G(t)`}</Tex>, the failures at time <Tex>t_i</Tex> should split between
          groups in proportion to how many runs each has at risk. The expected count for group <Tex>g</Tex> is:</p>
        <Tex block>{String.raw`E_{gi}=d_i\,\frac{n_{gi}}{n_i}`}</Tex>
        <p>Summing observed minus expected over all failure times gives the log-rank statistic:</p>
        <Tex block>{String.raw`\chi^2=(\mathbf O-\mathbf E)^{\top}\,\mathbf V^{-1}\,(\mathbf O-\mathbf E)\;\sim\;\chi^2_{G-1}`}</Tex>
        <p>With thousands of runs, almost any gap is "significant". Judge the <strong>size</strong> of the difference
          (medians, S(90)), not just <Tex>p</Tex>.</p>
      </>}
      insight={<ul>
        <li>Longest-lasting group: <strong>{best.name}</strong>, median {days(best.median)}. Shortest: <strong>{worst.name}</strong>, median {days(worst.median)}.</li>
        <li>Try <em>Last repair type</em>: runs after a <strong>rod</strong> repair fail far sooner than after a pump or tubing repair.</li>
        <li>Each comparison is one factor at a time. The Cox model (tab 6) separates their effects.</li>
      </ul>}
    />
  );
}
