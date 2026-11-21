import katex from "katex";
import { useMemo } from "react";

/** Renders LaTeX with KaTeX. Use String.raw`...` for the source so backslashes survive. */
export function Tex({ children, block = false }: { children: string; block?: boolean }) {
  const html = useMemo(
    () => katex.renderToString(children, { displayMode: block, throwOnError: false, strict: "ignore" }),
    [children, block],
  );
  return block
    ? <div className="tex-block" dangerouslySetInnerHTML={{ __html: html }} />
    : <span className="tex-inline" dangerouslySetInnerHTML={{ __html: html }} />;
}
