import { useEffect, useState } from "react";
import { AGENTS, LEVEL_STYLES, type RunEvent } from "@/lib/types";

/**
 * Block 0 shell.
 *
 * The button currently drives `/api/demo/stream`, which exercises the exact
 * production SSE path without needing the agent graph. That is deliberate:
 * it lets the container, the reverse proxy, and the browser transport all be
 * validated on the real deployment before there is anything to deploy.
 * Block 4 swaps this for the run console and the real POST /api/runs.
 */

interface Health {
  status: string;
  providers_configured: string[];
  search_configured: boolean;
  frontend_built: boolean;
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const start = () => {
    setEvents([]);
    setRunning(true);
    const source = new EventSource("/api/demo/stream");
    source.onmessage = (raw) => {
      const event = JSON.parse(raw.data) as RunEvent;
      setEvents((prev) => [...prev, event]);
      if (event.type === "run_completed") {
        source.close();
        setRunning(false);
      }
    };
    source.onerror = () => {
      source.close();
      setRunning(false);
    };
  };

  const ready = (health?.providers_configured.length ?? 0) > 0;

  return (
    <div className="min-h-full flex flex-col items-center justify-center px-6 py-16">
      <div className="w-full max-w-3xl">
        <header className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[var(--color-panel-2)] border border-[var(--color-edge)] text-xs tracking-wide text-slate-400 mb-6">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            zero input · zero cost · self-critiquing
          </div>

          <h1 className="text-4xl sm:text-5xl font-semibold tracking-tight mb-4">
            Autonomous Research Agent
          </h1>

          <p className="text-slate-400 leading-relaxed max-w-xl mx-auto">
            Nine agents discover an emerging scientific domain, invent a question that
            cannot simply be looked up, gather and clean their own data, run real
            experiments, attack their own findings, and write the paper. You press one
            button.
          </p>
        </header>

        <div className="flex flex-col items-center gap-4 mb-12">
          <button
            onClick={start}
            disabled={running}
            className="group relative px-8 py-4 rounded-xl bg-gradient-to-b from-indigo-500 to-indigo-600 hover:from-indigo-400 hover:to-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed font-medium text-white shadow-lg shadow-indigo-950/50 transition-all"
          >
            {running ? "Running…" : "Start Research"}
          </button>
          <p className="text-xs text-slate-500">
            No sign-in. No prompt. Typical run: 5–10 minutes.
          </p>
        </div>

        {events.length > 0 && (
          <div className="rounded-xl border border-[var(--color-edge)] bg-[var(--color-panel)] overflow-hidden mb-8">
            <div className="px-4 py-2 border-b border-[var(--color-edge)] text-xs uppercase tracking-wider text-slate-500">
              Agent feed
            </div>
            <div className="p-4 space-y-2 mono text-sm">
              {events.map((event) => {
                const agent = event.agent ? AGENTS[event.agent] : null;
                return (
                  <div key={event.seq} className="flex gap-3">
                    <span className="text-slate-600 shrink-0">
                      {String(event.seq).padStart(2, "0")}
                    </span>
                    {agent && (
                      <span className="shrink-0 w-10" style={{ color: agent.color }}>
                        {agent.short}
                      </span>
                    )}
                    <span className={LEVEL_STYLES[event.level]}>{event.message}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        <footer className="text-center text-xs text-slate-600 space-y-2">
          {health ? (
            <p>
              Backend {health.status} ·{" "}
              {ready ? (
                <span className="text-emerald-500">
                  {health.providers_configured.join(", ")} configured
                </span>
              ) : (
                <span className="text-amber-500">no LLM keys configured yet</span>
              )}
              {health.search_configured ? " · search ready" : " · search key missing"}
            </p>
          ) : (
            <p className="text-rose-500">Backend unreachable</p>
          )}
        </footer>
      </div>
    </div>
  );
}
