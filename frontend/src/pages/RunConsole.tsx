import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { AgentFeed } from "@/components/AgentFeed";
import { LiveGraph } from "@/components/LiveGraph";
import { ResultsPanel } from "@/components/ResultsPanel";
import { Vitals } from "@/components/Vitals";
import { useRunStream } from "@/hooks/useRunStream";
import { formatDuration, useRunState } from "@/hooks/useRunState";
import { api, type Topology } from "@/lib/api";

/**
 * The run console.
 *
 * Live runs and finished runs render through exactly the same path: the
 * backend replays a completed run's events over the same SSE endpoint, so a
 * reviewer opening yesterday's run sees the identical console. That is why
 * there is no separate "results page" — results are just the state the event
 * stream ends in.
 */

export function RunConsole() {
  const { runId = "" } = useParams();
  const [topology, setTopology] = useState<Topology | null>(null);
  const [elapsed, setElapsed] = useState(0);

  const { events, state: streamState, error } = useRunStream(runId || null);
  const state = useRunState(events);

  useEffect(() => {
    api.topology().then(setTopology).catch(() => setTopology(null));
  }, []);

  // Tick while running; freeze at the reported total once finished, so a
  // replayed run shows how long it actually took rather than counting up again.
  useEffect(() => {
    if (state.status !== "running") {
      if (state.elapsedSeconds) setElapsed(state.elapsedSeconds);
      return;
    }
    const started = events[0]?.at ? new Date(events[0].at).getTime() : Date.now();
    const timer = setInterval(() => setElapsed((Date.now() - started) / 1000), 1000);
    return () => clearInterval(timer);
  }, [state.status, state.elapsedSeconds, events]);

  const finished = state.status === "completed" || state.status === "failed";

  return (
    <div className="min-h-full px-4 sm:px-6 py-5">
      <header className="flex items-center justify-between gap-4 mb-4 max-w-[1500px] mx-auto">
        <div className="flex items-center gap-3 min-w-0">
          <Link to="/" className="text-slate-500 hover:text-slate-300 text-sm shrink-0">
            ←
          </Link>
          <div className="min-w-0">
            <h1 className="text-lg font-medium truncate">
              {state.domain || "Discovering a domain…"}
            </h1>
            <p className="text-xs text-slate-500 mono">
              run {runId} · {formatDuration(elapsed)}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <StatusPill status={state.status} streamState={streamState} />
        </div>
      </header>

      {error && state.status === "running" && (
        <div className="max-w-[1500px] mx-auto mb-4 rounded-lg border border-amber-900/50 bg-amber-950/20 p-3">
          <p className="text-sm text-amber-400">{error} — reconnecting…</p>
        </div>
      )}

      {state.status === "failed" && (
        <div className="max-w-[1500px] mx-auto mb-4 rounded-lg border border-rose-900/50 bg-rose-950/20 p-3">
          <p className="text-sm text-rose-400">
            This run ended without a paper. {state.failureReason}
          </p>
        </div>
      )}

      <div className="max-w-[1500px] mx-auto grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)_300px] gap-4 mb-4">
        <div className="h-[440px] order-2 lg:order-1">
          <AgentFeed events={events} />
        </div>
        <div className="h-[440px] order-1 lg:order-2">
          <LiveGraph state={state} topology={topology} />
        </div>
        <div className="order-3">
          <Vitals state={state} elapsed={elapsed} />
        </div>
      </div>

      {(finished || state.artifacts.length > 0) && (
        <div className="max-w-[1500px] mx-auto">
          <ResultsPanel runId={runId} state={state} />
        </div>
      )}
    </div>
  );
}

function StatusPill({
  status,
  streamState,
}: {
  status: string;
  streamState: string;
}) {
  const map: Record<string, { label: string; className: string }> = {
    running: { label: "running", className: "bg-indigo-500/15 text-indigo-300" },
    completed: { label: "complete", className: "bg-emerald-500/15 text-emerald-400" },
    failed: { label: "failed", className: "bg-rose-500/15 text-rose-400" },
    cancelled: { label: "cancelled", className: "bg-slate-500/15 text-slate-400" },
    idle: { label: streamState, className: "bg-slate-500/15 text-slate-400" },
  };
  const { label, className } = map[status] ?? map.idle;

  return (
    <span className={`px-2.5 py-1 rounded-full text-xs ${className}`}>
      {status === "running" && (
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-current mr-1.5 animate-pulse" />
      )}
      {label}
    </span>
  );
}
