import { useEffect, useRef, useState } from "react";

/**
 * Renders a Plotly figure from server-produced JSON.
 *
 * Plotly is imported dynamically. It is ~3MB, and nothing on the landing page
 * or during the first minutes of a run needs it — loading it eagerly would put
 * that weight in front of the reviewer before there is a single chart to show.
 */

interface Props {
  figureJson: string;
  title?: string;
  caption?: string;
}

export function PlotlyFigure({ figureJson, title, caption }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    // Only the two calls this component makes, so the local type does not have
    // to mirror the whole module shape.
    let plotly: {
      newPlot: (
        root: HTMLElement,
        data: unknown[],
        layout?: Record<string, unknown>,
        config?: Record<string, unknown>,
      ) => Promise<HTMLElement>;
      purge: (root: HTMLElement) => void;
    } | null = null;

    const element = container.current;
    if (!element) return;

    (async () => {
      try {
        const module = await import("plotly.js-dist-min");
        if (cancelled) return;
        plotly = module.default ?? module;

        const figure = JSON.parse(figureJson) as {
          data: unknown[];
          layout: Record<string, unknown>;
        };

        await plotly.newPlot(element, figure.data, figure.layout, {
          displaylogo: false,
          responsive: true,
          // Trimmed to the controls a reader of a chart actually wants;
          // the full bar is mostly drawing tools that make no sense here.
          modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
        });
      } catch (exc) {
        if (!cancelled) setError(exc instanceof Error ? exc.message : "figure failed to render");
      }
    })();

    return () => {
      cancelled = true;
      if (plotly && element) plotly.purge(element);
    };
  }, [figureJson]);

  if (error) {
    return (
      <div className="rounded-lg border border-rose-900/50 bg-rose-950/20 p-4">
        <p className="text-sm text-rose-400">Could not render figure: {error}</p>
      </div>
    );
  }

  return (
    <figure className="rounded-lg border border-[var(--color-edge)] bg-[var(--color-panel)] p-3">
      {title && <figcaption className="text-sm text-slate-300 mb-2">{title}</figcaption>}
      <div ref={container} className="w-full min-h-[320px]" />
      {caption && <p className="text-xs text-slate-500 mt-2 leading-relaxed">{caption}</p>}
    </figure>
  );
}
