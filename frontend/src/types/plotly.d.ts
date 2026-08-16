/**
 * Minimal declarations for `plotly.js-dist-min`.
 *
 * The prebuilt bundle ships no types, and the full `@types/plotly.js` package
 * pulls in a large dependency tree to describe an API surface we touch in
 * exactly three places. Declaring those three is cheaper and does not go stale
 * with the rest of Plotly.
 */
declare module "plotly.js-dist-min" {
  export interface PlotConfig {
    displaylogo?: boolean;
    responsive?: boolean;
    modeBarButtonsToRemove?: string[];
    [key: string]: unknown;
  }

  export function newPlot(
    root: HTMLElement,
    data: unknown[],
    layout?: Record<string, unknown>,
    config?: PlotConfig,
  ): Promise<HTMLElement>;

  export function purge(root: HTMLElement): void;

  const Plotly: {
    newPlot: typeof newPlot;
    purge: typeof purge;
  };
  export default Plotly;
}
