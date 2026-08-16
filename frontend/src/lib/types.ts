/**
 * Mirrors backend/schemas/events.py.
 *
 * Kept hand-written rather than generated: the surface is small, and a
 * generation step is one more thing to run correctly under a deadline. The
 * tests in backend/tests/test_events_contract.py assert the two stay in sync,
 * so drift fails CI rather than surfacing as a blank panel in the demo.
 */

export type EventType =
  | "run_started"
  | "run_completed"
  | "run_failed"
  | "run_cancelled"
  | "node_enter"
  | "node_exit"
  | "edge_traversed"
  | "cycle_started"
  | "reroute"
  | "agent_message"
  | "tool_call"
  | "quip"
  | "artifact"
  | "vitals"
  | "abstention"
  | "conflict"
  | "warning";

export type AgentName =
  | "supervisor"
  | "domain_scout"
  | "peer_review_panel"
  | "question_generator"
  | "data_alchemist"
  | "experiment_designer"
  | "executor"
  | "uncertainty_quantifier"
  | "critic"
  | "paper_writer";

export type Level = "debug" | "info" | "success" | "warn" | "error";

export type ArtifactKind =
  | "domain_candidates"
  | "emergence_chart"
  | "domain_selected"
  | "question_set"
  | "question_selected"
  | "source_acquired"
  | "dataset_summary"
  | "experiment_spec"
  | "experiment_result"
  | "figure"
  | "critique"
  | "confidence_report"
  | "paper";

export type RunStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface RunEvent {
  seq: number;
  run_id: string;
  type: EventType;
  agent: AgentName | null;
  level: Level;
  message: string;
  payload: Record<string, unknown>;
  cycle: number;
  at: string;
}

export interface RunSummary {
  run_id: string;
  status: RunStatus;
  started_at: string;
  finished_at: string | null;
  domain: string;
  question: string;
  paper_title: string;
  cycles_used: number;
  overall_confidence: number;
  abstained_count: number;
  accepted_by_critic: boolean;
  error: string;
  event_count: number;
}

/** Display metadata per agent. The order here is the order in the live graph. */
export const AGENTS: Record<AgentName, { label: string; color: string; short: string }> = {
  supervisor: { label: "Supervisor", color: "var(--color-agent-supervisor)", short: "SUP" },
  domain_scout: { label: "Domain Scout", color: "var(--color-agent-scout)", short: "SCT" },
  peer_review_panel: { label: "Peer Review Panel", color: "var(--color-agent-panel)", short: "PNL" },
  question_generator: {
    label: "Question Generator",
    color: "var(--color-agent-question)",
    short: "QGN",
  },
  data_alchemist: { label: "Data Alchemist", color: "var(--color-agent-alchemist)", short: "ALC" },
  experiment_designer: {
    label: "Experiment Designer",
    color: "var(--color-agent-designer)",
    short: "EXP",
  },
  executor: { label: "Executor", color: "var(--color-agent-executor)", short: "RUN" },
  uncertainty_quantifier: {
    label: "Uncertainty Quantifier",
    color: "var(--color-agent-uncertainty)",
    short: "UQ",
  },
  critic: { label: "Critic", color: "var(--color-agent-critic)", short: "CRT" },
  paper_writer: { label: "Paper Writer", color: "var(--color-agent-writer)", short: "WRT" },
};

/**
 * Agent name to graph node id. Mirrors NODE_TO_AGENT in backend/graph/build.py.
 *
 * Reroute events identify their source by agent, while the graph draws edges
 * between node ids, and the two vocabularies differ (`data_alchemist` vs
 * `data`). Without this the reroute edge silently fails to match and never
 * lights up.
 */
export const AGENT_TO_NODE: Record<AgentName, string> = {
  supervisor: "scout",
  domain_scout: "scout",
  peer_review_panel: "panel",
  question_generator: "question",
  data_alchemist: "data",
  experiment_designer: "design",
  executor: "execute",
  uncertainty_quantifier: "uncertainty",
  critic: "critic",
  paper_writer: "writer",
};

export const LEVEL_STYLES: Record<Level, string> = {
  debug: "text-slate-500",
  info: "text-slate-300",
  success: "text-emerald-400",
  warn: "text-amber-400",
  error: "text-rose-400",
};
