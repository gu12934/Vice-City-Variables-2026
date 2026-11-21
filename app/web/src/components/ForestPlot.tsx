import type { CoxRow } from "../api";
import { pval } from "../format";

/** Hazard ratios with 95% CIs on a log axis. Colour: raises risk / lowers risk / not significant. */
export function ForestPlot({ rows }: { rows: CoxRow[] }) {
  const W = 760, rowH = 26, left = 250, right = 70, top = 10, bottom = 34;
  const H = top + bottom + rows.length * rowH;
  const lo = Math.max(0.25, Math.min(...rows.map((r) => r.lo)) * 0.9);
  const hi = Math.min(4, Math.max(...rows.map((r) => r.hi)) * 1.1);
  const x = (v: number) => left + ((Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * (W - left - right);
  const ticks = [0.25, 0.33, 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 3, 4].filter((t) => t >= lo && t <= hi);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Cox hazard ratios forest plot">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top} y2={H - bottom} stroke={t === 1 ? "var(--ink-2)" : "var(--grid)"} strokeWidth={t === 1 ? 1.2 : 1} />
          <text x={x(t)} y={H - bottom + 16} textAnchor="middle" fontSize="11" fill="var(--muted)">{t}×</text>
        </g>
      ))}
      <text x={(left + W - right) / 2} y={H - 4} textAnchor="middle" fontSize="11" fill="var(--muted)">hazard ratio (log scale)</text>
      {rows.map((r, i) => {
        const y = top + i * rowH + rowH / 2;
        const sig = r.p < 0.05;
        const c = !sig ? "var(--muted)" : r.hr > 1 ? "var(--s2)" : "var(--s1)";
        return (
          <g key={r.feature}>
            <title>{`${r.label}: HR ${r.hr.toFixed(2)} (95% CI ${r.lo.toFixed(2)}–${r.hi.toFixed(2)}), p = ${pval(r.p)}`}</title>
            <rect x={0} y={y - rowH / 2} width={W} height={rowH} fill="transparent" />
            <text x={left - 10} y={y + 4} textAnchor="end" fontSize="12" fill={sig ? "var(--ink)" : "var(--muted)"}>{r.label}</text>
            <line x1={x(Math.max(r.lo, lo))} x2={x(Math.min(r.hi, hi))} y1={y} y2={y} stroke={c} strokeWidth={2.2} strokeLinecap="round" />
            <circle cx={x(Math.min(Math.max(r.hr, lo), hi))} cy={y} r={5} fill={c} stroke="var(--surface)" strokeWidth={2} />
            <text x={W - right + 8} y={y + 4} fontSize="11.5" fill="var(--ink-2)">{r.hr.toFixed(2)}×</text>
          </g>
        );
      })}
    </svg>
  );
}
