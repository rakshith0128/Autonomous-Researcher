import { useMemo } from "react";
import type { AgentName, ArtifactKind, RunEvent } from "@/lib/types";

/**
 * Derives everything the UI renders from the raw event stream.
 *
 * The event log is the single source of truth. Nothing is fetched separately
 * and no state is mutated in place, which is what makes a live run and a
 * replayed one render identically — the console cannot tell the difference,
 * so a reviewer arriving after the fact sees exactly what someone watching
 * live saw.
 */

export interface Artifact {
  kind: ArtifactKind;
  data: Record<string, unknown>;
  message: string;
  cycle: number;
  seq: number;
}

export interface RunState {
  status: "idle" | "running" | "completed" | "failed" | "cancelled";
  activeAgents: Set<AgentName>;
  completedAgents: Set<AgentName>;
  cycle: number;
  maxCycles: number;
  domain: string;
  question: string;
  confidence: number;
  artifacts: Artifact[];
  byKind: Partial<Record<ArtifactKind, Artifact>>;
  figures: Artifact[];
  reroutes: { from: string; to: string; reason: string; cycle: number }[];
  abstentions: number;
  toolFailures: number;
  failureReason: string;
  elapsedSeconds: number;
  tokens: number;
  degradedTools: string[];
}

const TERMINAL: Record<string, RunState["status"]> = {
  run_completed: "completed",
  run_failed: "failed",
  run_cancelled: "cancelled",
};

export function useRunState(events: RunEvent[]): RunState {
  return useMemo(() => {
    const state: RunState = {
      status: events.length ? "running" : "idle",
      activeAgents: new Set(),
      completedAgents: new Set(),
      cycle: 0,
      maxCycles: 5,
      domain: "",
      question: "",
      confidence: 0,
      artifacts: [],
      byKind: {},
      figures: [],
      reroutes: [],
      abstentions: 0,
      toolFailures: 0,
      failureReason: "",
      elapsedSeconds: 0,
      tokens: 0,
      degradedTools: [],
    };

    for (const event of events) {
      if (event.cycle > state.cycle) state.cycle = event.cycle;

      switch (event.type) {
        case "run_started":
          state.maxCycles = Number(event.payload.max_cycles ?? 5);
          break;

        case "node_enter":
          if (event.agent) state.activeAgents.add(event.agent);
          break;

        case "node_exit":
          if (event.agent) {
            state.activeAgents.delete(event.agent);
            state.completedAgents.add(event.agent);
          }
          break;

        case "reroute":
          state.reroutes.push({
            // The Critic is the usual source, but the Alchemist and Designer
            // reroute too — assuming "critic" would draw the wrong red edge.
            from: String(event.payload.source ?? event.agent ?? "critic"),
            to: String(event.payload.target ?? ""),
            reason: String(event.payload.reason ?? event.message),
            cycle: event.cycle,
          });
          break;

        case "tool_call":
          if (event.payload.ok === false) state.toolFailures += 1;
          break;

        case "artifact": {
          const kind = event.payload.kind as ArtifactKind;
          const artifact: Artifact = {
            kind,
            data: (event.payload.data ?? {}) as Record<string, unknown>,
            message: event.message,
            cycle: event.cycle,
            seq: event.seq,
          };
          state.artifacts.push(artifact);

          // Figures accumulate; everything else keeps only the latest, because
          // a later cycle supersedes an earlier one for the same kind.
          if (kind === "figure" || kind === "emergence_chart") {
            state.figures.push(artifact);
          }
          state.byKind[kind] = artifact;

          if (kind === "domain_selected") {
            state.domain = String(artifact.data.chosen ?? "");
          }
          if (kind === "question_set") {
            const questions = (artifact.data.questions ?? []) as {
              text: string;
              selected: boolean;
            }[];
            const selected = questions.find((q) => q.selected);
            if (selected) state.question = selected.text;
          }
          if (kind === "confidence_report") {
            state.confidence = Number(artifact.data.overall ?? 0);
            state.abstentions = Number(artifact.data.abstained ?? 0);
          }
          if (kind === "paper") {
            state.confidence = Number(artifact.data.confidence ?? state.confidence);
          }
          break;
        }

        default: {
          const terminal = TERMINAL[event.type];
          if (terminal) {
            state.status = terminal;
            state.activeAgents.clear();
            state.elapsedSeconds = Number(event.payload.elapsed_seconds ?? 0);
            state.tokens = Number(event.payload.tokens ?? 0);
            state.degradedTools = (event.payload.degraded_tools ?? []) as string[];
            if (terminal !== "completed") {
              state.failureReason = String(
                event.payload.failure_reason ?? event.message ?? "",
              );
            }
          }
          break;
        }
      }
    }

    return state;
  }, [events]);
}

/** Human-readable elapsed time. */
export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds % 60)}s`;
}
