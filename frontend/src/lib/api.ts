/**
 * Thin API client.
 *
 * Defaults to same-origin, which is correct when one container serves both the
 * API and this bundle. Set `VITE_API_BASE` at build time to point at a
 * separately-hosted backend — the frontend then loads instantly from a CDN
 * while the backend, on a free tier that sleeps after 15 minutes of
 * inactivity, wakes up in the background.
 */

import type { AgentName, RunSummary } from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

/** Absolute URL for an API path. Exported because EventSource needs it too. */
export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

/** True when the backend lives on another origin and may need waking. */
export const isSplitDeployment = API_BASE !== "";

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
  const response = await fetch(apiUrl(path), {
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

  /**
   * Poll health until the backend answers.
   *
   * A sleeping free-tier instance takes 30-60s to wake, during which every
   * request fails outright. Without this the landing page would simply report
   * "Backend unreachable" to a reviewer whose only mistake was arriving first.
   */
  async wake(
    onAttempt?: (attempt: number) => void,
    { attempts = 20, intervalMs = 4000 } = {},
  ): Promise<Health | null> {
    for (let attempt = 1; attempt <= attempts; attempt += 1) {
      onAttempt?.(attempt);
      try {
        return await request<Health>("/api/health");
      } catch {
        await new Promise((resolve) => setTimeout(resolve, intervalMs));
      }
    }
    return null;
  },

  topology: () => request<Topology>("/api/topology"),
  startRun: () => request<{ run_id: string; status: string }>("/api/runs", { method: "POST" }),
  listRuns: (limit = 25) => request<{ runs: RunSummary[] }>(`/api/runs?limit=${limit}`),
  getRun: (id: string) => request<{ run: RunSummary; live: boolean }>(`/api/runs/${id}`),
  cancelRun: (id: string) => request<unknown>(`/api/runs/${id}/cancel`, { method: "POST" }),
  paperUrl: (id: string) => apiUrl(`/api/runs/${id}/paper.md`),
  manifestUrl: (id: string) => apiUrl(`/api/runs/${id}/manifest.json`),
  eventsUrl: (id: string, fromSeq = 0) =>
    apiUrl(`/api/runs/${id}/events?from_seq=${fromSeq}`),
};
