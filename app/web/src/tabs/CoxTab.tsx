import type { CoxChart } from "../api";
import { ForestPlot } from "../components/ForestPlot";
import { Tex } from "../components/Tex";
import { ChartTitle, TabPage } from "../components/ui";
import { num, pval } from "../format";

export function CoxTab({ d }: { d: CoxChart }) {
  const sig = d.rows.filter((r) => r.p < 0.05);
  const up = [...sig].filter((r) => r.hr > 1).sort((a, b) => b.hr - a.hr).slice(0, 4);
  const down = [...sig].filter((r) => r.hr < 1).sort((a, b) => a.hr - b.hr).slice(0, 3);
  return (
    <TabPage
      title="6 · Cox proportional hazards: what drives failure"
      lead="All risk factors at once. Each hazard ratio says how much a factor multiplies the failure rate, holding the others fixed."
      chart={<>
        <ChartTitle sub={`${num(d.n)} runs · ${num(d.events)} failures · robust SEs clustered by well · ridge penalty ${d.penalizer} · in-sample concordance ${d.concordance.toFixed(3)}`}>
          Hazard ratios with 95% confidence intervals
        </ChartTitle>
        <div className="legend">
          <span className="legend-item"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--s2)" /></svg>raises risk (p &lt; 0.05)</span>
          <span className="legend-item"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--s1)" /></svg>lowers risk (p &lt; 0.05)</span>
          <span className="legend-item"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--muted)" /></svg>not significant</span>
        </div>
        <ForestPlot rows={d.rows} />
      </>}
      math={<>
        <p>The hazard for a run with features <Tex>{String.raw`\mathbf x`}</Tex> is a baseline hazard of <em>any</em> shape, scaled by a log-linear risk score:</p>
        <Tex block>{String.raw`h(t\mid\mathbf x)=h_0(t)\,\exp\!\big(\beta_1x_1+\dots+\beta_px_p\big)`}</Tex>
        <p><Tex>{String.raw`\mathrm{HR}_j=e^{\beta_j}`}</Tex> multiplies the failure rate per unit of <Tex>x_j</Tex>, at every <Tex>t</Tex>.
          <Tex>{String.raw`h_0`}</Tex> cancels from the <strong>partial likelihood</strong>: at each failure, how likely was it <em>this</em> run, out of all runs at risk?</p>
        <Tex block>{String.raw`L(\boldsymbol\beta)=\prod_{i:\,\delta_i=1}\frac{\exp(\mathbf x_i^{\top}\boldsymbol\beta)}{\sum_{j\in R(t_i)}\exp(\mathbf x_j^{\top}\boldsymbol\beta)},\qquad R(t_i)=\{j:t_j\ge t_i\}`}</Tex>
        <p>Runs on the same well aren't independent, so standard errors use the <strong>robust sandwich</strong> estimator clustered by well,
          and a ridge penalty <Tex>{String.raw`\tfrac{\lambda}{2}\lVert\boldsymbol\beta\rVert^2`}</Tex> stabilises correlated features.</p>
      </>}
      insight={<>
        <ul>
          {up.map((r) => <li key={r.feature}><strong>{r.label}</strong>: {r.hr.toFixed(2)}× the failure rate (p = {pval(r.p)})</li>)}
          {down.map((r) => <li key={r.feature}><strong>{r.label}</strong>: {r.hr.toFixed(2)}× ({((1 - r.hr) * 100).toFixed(0)}% lower, p = {pval(r.p)})</li>)}
        </ul>
        <p className="note">These are associations, not proof of cause. Proportional-hazards test: the smallest p-values are for {d.ph.slice(0, 3).map((x) => x.feature).join(", ")},
          so those HRs are averages over the run's life.</p>
      </>}
    />
  );
}
