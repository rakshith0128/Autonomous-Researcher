/** Thin API client. Same-origin — FastAPI serves this bundle, so no base URL. */

import type { AgentName, RunSummary } from "./types";

export interface TopologyNode {
  id: string;
  /** Typed rather than `string` so the graph can index the agent palette
   *  directly; a backend agent with no frontend colour fails at compile time
   *  instead of rendering as an invisible node. */
  agent: AgentName;
  label: string;
}
export interface TopologyEdge {
  from: string;
  to: string;
  kind: "flow" | "reroute";
}
export interface Topology {
  nodes: TopologyNode[];
  edges: TopologyEdge[];
}

export interface Health {
  status: string;
  providers_configured: string[];
  search_configured: boolean;
  active_runs: number;
  max_concurrent: number;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...init,
  });

  if (!response.ok) {
    // The backend returns structured detail for the cases a user can act on —
    // a run already in progress, a cooldown, a spent daily budget — so it is
    // surfaced rather than flattened into a status code.
    let detail: unknown = null;
    try {
      detail = (await response.json())?.detail ?? null;
    } catch {
      detail = await response.text().catch(() => null);
    }
    const message =
      typeof detail === "string"
        ? detail
        : ((detail as { message?: string })?.message ?? `Request failed (${response.status})`);
    throw new ApiError(message, response.status, detail);
  }

  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/api/health"),
  topology: () => request<Topology>("/api/topology"),
  startRun: () => request<{ run_id: string; status: string }>("/api/runs", { method: "POST" }),
  listRuns: (limit = 25) => request<{ runs: RunSummary[] }>(`/api/runs?limit=${limit}`),
  getRun: (id: string) => request<{ run: RunSummary; live: boolean }>(`/api/runs/${id}`),
  cancelRun: (id: string) => request<unknown>(`/api/runs/${id}/cancel`, { method: "POST" }),
  paperUrl: (id: string) => `/api/runs/${id}/paper.md`,
  manifestUrl: (id: string) => `/api/runs/${id}/manifest.json`,
};
