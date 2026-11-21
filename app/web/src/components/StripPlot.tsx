import type { ShapChart } from "../api";

/** SHAP beeswarm-style strip plot: x = SHAP value, colour = feature value percentile (low blue → high orange). */
export function StripPlot({ strips }: { strips: ShapChart["strips"] }) {
  const W = 760, rowH = 46, left = 170, right = 20, top = 10, bottom = 34;
  const H = top + bottom + strips.length * rowH;
  const all = strips.flatMap((s) => s.points.map((p) => p.shap));
  const ext = Math.max(0.2, ...all.map(Math.abs)) * 1.05;
  const x = (v: number) => left + ((v + ext) / (2 * ext)) * (W - left - right);
  const ticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1].filter((t) => Math.abs(t) <= ext);
  // deterministic jitter
  const jitter = (i: number) => (((i * 2654435761) % 1000) / 1000 - 0.5) * (rowH * 0.62);
  const color = (p: number) =>
    p < 0.5
      ? `color-mix(in srgb, var(--s1) ${Math.round((1 - p * 2) * 100)}%, var(--axis))`
      : `color-mix(in srgb, var(--s2) ${Math.round((p - 0.5) * 200)}%, var(--axis))`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="SHAP value distribution per factor">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top} y2={H - bottom} stroke={t === 0 ? "var(--ink-2)" : "var(--grid)"} />
          <text x={x(t)} y={H - bottom + 16} textAnchor="middle" fontSize="11" fill="var(--muted)">{t > 0 ? `+${t}` : t}</text>
        </g>
      ))}
      <text x={(left + W - right) / 2} y={H - 4} textAnchor="middle" fontSize="11" fill="var(--muted)">
        SHAP value (log-hazard): + raises failure rate, − lowers it
      </text>
      {strips.map((s, r) => {
        const cy = top + r * rowH + rowH / 2;
        return (
          <g key={s.factor}>
            <text x={left - 12} y={cy + 4} textAnchor="end" fontSize="12.5" fill="var(--ink)">{s.name}</text>
            {s.points.map((p, i) => (
              <circle key={i} cx={x(p.shap)} cy={cy + jitter(i + r * 997)} r={2.6} fill={color(p.pct)} opacity={0.85} />
            ))}
          </g>
        );
      })}
    </svg>
  );
}
