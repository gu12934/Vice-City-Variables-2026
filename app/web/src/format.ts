export const pct = (x: number | null | undefined, d = 0) =>
  x == null || !Number.isFinite(x) ? "–" : `${(x * 100).toFixed(d)}%`;
export const num = (x: number | null | undefined, d = 0) =>
  x == null || !Number.isFinite(x) ? "–" : x.toLocaleString("en-CA", { minimumFractionDigits: d, maximumFractionDigits: d });
export const money = (x: number | null | undefined) => {
  if (x == null || !Number.isFinite(x)) return "–";
  if (Math.abs(x) >= 1e6) return `$${(x / 1e6).toFixed(1)}M`;
  if (Math.abs(x) >= 1e3) return `$${(x / 1e3).toFixed(1)}k`;
  return `$${x.toFixed(0)}`;
};
export const pval = (p: number) => (p < 1e-4 ? p.toExponential(0) : p.toFixed(4));
export const days = (x: number | null | undefined) => (x == null || !Number.isFinite(x) ? "not reached" : `${num(x)} days`);

// Chart colors resolve from CSS custom properties, so light/dark themes swap in one place
export const SERIES = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s6)", "var(--s7)", "var(--s8)"];
export const ORDINAL = ["var(--o1)", "var(--o2)", "var(--o3)", "var(--o4)"];
export const INK = "var(--ink)";
export const INK2 = "var(--ink-2)";
export const MUTED = "var(--muted)";
export const GRID = "var(--grid)";
export const AXIS = "var(--axis)";
