import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ShapChart } from "../api";
import { StripPlot } from "../components/StripPlot";
import { Tex } from "../components/Tex";
import { ChartTitle, TabPage, axisProps, tooltipStyle } from "../components/ui";
import { SERIES } from "../format";

export function ShapTab({ d }: { d: ShapChart }) {
  const imp = d.importance.slice(0, 10);
  const top = imp[0];
  return (
    <TabPage
      title="9 · SHAP: why the model scores a well as risky"
      lead="SHAP splits every prediction into additive contributions, one per factor. One-hot columns (area, pump make, last repair type) are summed into a single factor."
      chart={<div className="chart-row">
        <div>
          <ChartTitle sub={`mean |SHAP| over ${d.n} sampled runs, in log-hazard units`}>Global importance</ChartTitle>
          <div className="chart-box tall">
            <ResponsiveContainer>
              <BarChart data={imp} layout="vertical" margin={{ top: 4, right: 48, left: 8, bottom: 0 }}>
                <CartesianGrid stroke="var(--grid)" horizontal={false} />
                <XAxis type="number" {...axisProps} tickFormatter={(v) => Number(v).toFixed(2)} />
                <YAxis type="category" dataKey="name" width={150} {...axisProps} tick={{ fill: "var(--ink-2)", fontSize: 12 }} />
                <Tooltip {...tooltipStyle} formatter={(v) => [Number(v).toFixed(3), "mean |SHAP|"]} cursor={{ fill: "var(--wash)" }} />
                <Bar dataKey="mean_abs" fill={SERIES[0]} radius={[0, 4, 4, 0]} barSize={16} isAnimationActive={false}>
                  <LabelList dataKey="mean_abs" position="right" formatter={(v: number) => v.toFixed(3)} fill="var(--ink-2)" fontSize={11} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div>
          <ChartTitle sub="each dot is one run">Direction of each effect</ChartTitle>
          <div className="strip-legend">feature value: low <span className="grad" /> high</div>
          <StripPlot strips={d.strips} />
        </div>
      </div>}
      math={<>
        <p>The <strong>Shapley value</strong> of factor <Tex>j</Tex> is its average marginal contribution over every order of adding factors,
          where <Tex>v(S)</Tex> is the expected model output when only the factors in <Tex>S</Tex> are known:</p>
        <Tex block>{String.raw`\phi_j(\mathbf x)=\sum_{S\subseteq F\setminus\{j\}}\frac{|S|!\,(|F|-|S|-1)!}{|F|!}\Big[v(S\cup\{j\})-v(S)\Big]`}</Tex>
        <p>The contributions add up <em>exactly</em> to the prediction (local accuracy):</p>
        <Tex block>{String.raw`f(\mathbf x)=\underbrace{\mathbb E\big[f(\mathbf X)\big]}_{\text{fleet average}}+\sum_j\phi_j(\mathbf x)`}</Tex>
        <p>Here <Tex>f</Tex> is the boosted model's <strong>log-hazard</strong>, so <Tex>{String.raw`e^{\phi_j}`}</Tex> is a <strong>multiplier</strong> on the well's
          failure rate relative to the fleet average. Additivity also lets one-hot columns be summed into one factor: <Tex>{String.raw`\phi_{\text{area}}=\sum_{a}\phi_{\text{area}=a}`}</Tex>.</p>
      </>}
      insight={<ul>
        <li>The strongest factor is <strong>{top?.name}</strong> (mean |φ| = {top?.mean_abs.toFixed(2)}, about ×{Math.exp(top?.mean_abs ?? 0).toFixed(2)} on a typical well's rate).</li>
        <li>In the strip plot, orange dots (high values) on the right mean "higher value → higher risk". Repairs in the last year and pump speed behave that way.</li>
        <li>SHAP explains the <em>model</em>, not the physics. Treat "lower RPM → lower risk" as a hypothesis to test in the field.</li>
      </ul>}
    />
  );
}
