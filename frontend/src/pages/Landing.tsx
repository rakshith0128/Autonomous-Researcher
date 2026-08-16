import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, api, type Health } from "@/lib/api";
import type { RunSummary } from "@/lib/types";

/**
 * The landing page.
 *
 * One button, no inputs, no sign-in — the assessment's success criterion is
 * "paste nothing, click Start Research". The gallery link exists because this
 * is a public URL spending a shared free-tier budget: when the quota is gone,
 * a reviewer must still find a finished paper rather than an error.
 */

export function Landing() {
  const navigate = useNavigate();
  const [health, setHealth] = useState<Health | null>(null);
  const [recent, setRecent] = useState<RunSummary[]>([]);
  const [starting, setStarting] = useState(false);
  const [waking, setWaking] = useState(false);
  const [error, setError] = useState("");
  const [activeRunId, setActiveRunId] = useState("");

  useEffect(() => {
    let cancelled = false;

    api
      .health()
      .then((h) => !cancelled && setHealth(h))
      .catch(async () => {
        // On a free tier the backend sleeps after 15 minutes, and the first
        // visitor after a quiet spell pays a 30-60s cold start. Reporting
        // "unreachable" would be accurate for a second and wrong thereafter,
        // so poll until it answers and say what is happening meanwhile.
        if (cancelled) return;
        setWaking(true);
        const woken = await api.wake();
        if (cancelled) return;
        setWaking(false);
        setHealth(woken);
        if (woken) {
          api
            .listRuns(5)
            .then((r) => !cancelled && setRecent(r.runs))
            .catch(() => undefined);
        }
      });

    api
      .listRuns(5)
      .then((r) => !cancelled && setRecent(r.runs))
      .catch(() => !cancelled && setRecent([]));

    return () => {
      cancelled = true;
    };
  }, []);

  const start = async () => {
    setStarting(true);
    setError("");
    setActiveRunId("");
    try {
      const { run_id } = await api.startRun();
      navigate(`/run/${run_id}`);
    } catch (exc) {
      if (exc instanceof ApiError) {
        setError(exc.message);
        const detail = exc.detail as { active_run_id?: string } | null;
        if (detail?.active_run_id) setActiveRunId(detail.active_run_id);
      } else {
        setError("Could not reach the server.");
      }
      setStarting(false);
    }
  };

  const ready = (health?.providers_configured.length ?? 0) > 0;
  const completed = recent.filter((r) => r.status === "completed");

  return (
    <div className="min-h-full flex flex-col items-center justify-center px-6 py-16">
      <div className="w-full max-w-3xl">
        <header className="text-center mb-10">
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

        <div className="flex flex-col items-center gap-3 mb-10">
          <button
            onClick={start}
            disabled={starting || waking || !ready}
            className="px-8 py-4 rounded-xl bg-gradient-to-b from-indigo-500 to-indigo-600 hover:from-indigo-400 hover:to-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed font-medium text-white shadow-lg shadow-indigo-950/50 transition-all"
          >
            {waking ? "Waking the backend…" : starting ? "Starting…" : "Start Research"}
          </button>
          <p className="text-xs text-slate-500">
            {waking
              ? "The free-tier backend sleeps when idle. This takes about a minute."
              : "No sign-in. No prompt. A run takes up to 15 minutes."}
          </p>

          {error && (
            <div className="mt-2 max-w-md text-center rounded-lg border border-amber-900/50 bg-amber-950/20 p-3">
              <p className="text-sm text-amber-400">{error}</p>
              {activeRunId && (
                <button
                  onClick={() => navigate(`/run/${activeRunId}`)}
                  className="text-xs text-sky-400 hover:underline mt-1.5"
                >
                  Watch the run in progress →
                </button>
              )}
            </div>
          )}
        </div>

        {completed.length > 0 && (
          <section className="mb-10">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xs uppercase tracking-wider text-slate-500">
                Completed runs
              </h2>
              <button
                onClick={() => navigate("/runs")}
                className="text-xs text-slate-500 hover:text-slate-300"
              >
                view all →
              </button>
            </div>
            <div className="space-y-2">
              {completed.slice(0, 3).map((run) => (
                <button
                  key={run.run_id}
                  onClick={() => navigate(`/run/${run.run_id}`)}
                  className="w-full text-left p-3 rounded-lg border border-[var(--color-edge)] bg-[var(--color-panel)] hover:border-slate-600 transition-colors"
                >
                  <p className="text-sm text-slate-200 line-clamp-1">
                    {run.paper_title || run.domain || run.run_id}
                  </p>
                  <p className="text-xs text-slate-500 mt-0.5">
                    {run.domain} · {run.cycles_used} cycle
                    {run.cycles_used === 1 ? "" : "s"} ·{" "}
                    {Math.round(run.overall_confidence * 100)}% confidence
                    {run.abstained_count > 0 && ` · ${run.abstained_count} abstained`}
                  </p>
                </button>
              ))}
            </div>
          </section>
        )}

        <footer className="text-center text-xs text-slate-600">
          {health ? (
            <p>
              {ready ? (
                <span className="text-emerald-600">
                  {health.providers_configured.join(", ")} ready
                </span>
              ) : (
                <span className="text-amber-500">no LLM provider configured</span>
              )}
              {health.search_configured ? " · search ready" : " · search unavailable"}
              {health.active_runs > 0 && ` · ${health.active_runs} run in progress`}
            </p>
          ) : waking ? (
            <p className="text-amber-500">Backend waking from sleep…</p>
          ) : (
            <p className="text-rose-500">
              Backend unreachable — browse completed runs instead
            </p>
          )}
        </footer>
      </div>
    </div>
  );
}
