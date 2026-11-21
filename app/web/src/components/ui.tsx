import type { ReactNode } from "react";
import { AXIS, GRID, MUTED } from "../format";

export function TabPage({ title, lead, chart, math, insight }: {
  title: string; lead: ReactNode; chart: ReactNode; math: ReactNode; insight?: ReactNode;
}) {
  return (
    <section className="tab-page">
      <header className="tab-head">
        <h2>{title}</h2>
        <p className="lead">{lead}</p>
      </header>
      <div className="card viz">{chart}</div>
      <div className="two-col">
        <div className="card math">
          <h3>The math</h3>
          {math}
        </div>
        {insight && (
          <div className="card insight">
            <h3>What it shows here</h3>
            {insight}
          </div>
        )}
      </div>
    </section>
  );
}

export function ChartTitle({ children, sub }: { children: ReactNode; sub?: ReactNode }) {
  return (
    <div className="chart-title">
      <h3>{children}</h3>
      {sub && <p>{sub}</p>}
    </div>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}

export function Segmented<T extends string>({ value, options, onChange }: {
  value: T; options: { value: T; label: string }[]; onChange: (v: T) => void;
}) {
  return (
    <div className="segmented" role="tablist">
      {options.map((o) => (
        <button key={o.value} role="tab" aria-selected={o.value === value}
          className={o.value === value ? "on" : ""} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="legend">
      {items.map((i) => (
        <span key={i.label} className="legend-item">
          <svg width="18" height="10" aria-hidden>
            <line x1="1" y1="5" x2="17" y2="5" stroke={i.color} strokeWidth="2.5"
              strokeDasharray={i.dashed ? "4 3" : undefined} strokeLinecap="round" />
          </svg>
          {i.label}
        </span>
      ))}
    </div>
  );
}

/** Shared axis/grid styling for Recharts */
export const axisProps = {
  stroke: AXIS,
  tick: { fill: MUTED, fontSize: 11 },
  tickLine: false,
} as const;
export const gridProps = { stroke: GRID, strokeDasharray: "", vertical: false } as const;
export const tooltipStyle = {
  contentStyle: {
    background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8,
    fontSize: 12, color: "var(--ink)", boxShadow: "0 4px 16px rgba(0,0,0,0.08)",
  },
  labelStyle: { color: "var(--ink-2)", marginBottom: 4 },
  itemStyle: { padding: 0 },
} as const;
