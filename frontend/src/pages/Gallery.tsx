import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "@/lib/api";
import type { RunSummary } from "@/lib/types";

/**
 * Every run, finished or otherwise.
 *
 * This exists because the demo runs on a shared free-tier budget. When the
 * day's quota is spent, a reviewer clicking "Start Research" gets a 429 — and
 * a submission whose live URL shows only an error is a failed submission. The
 * gallery guarantees there is always a completed paper to read.
 *
 * Failed runs are listed too, with their reason. Hiding them would misrepresent
 * how often free-tier infrastructure actually falls over.
 */

export function Gallery() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .listRuns(50)
      .then((r) => setRuns(r.runs))
      .catch(() => setRuns([]))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="min-h-full px-6 py-10">
      <div className="max-w-3xl mx-auto">
        <header className="flex items-center gap-3 mb-6">
          <Link to="/" className="text-slate-500 hover:text-slate-300">
            ←
          </Link>
          <h1 className="text-2xl font-medium">Runs</h1>
          <span className="text-sm text-slate-500">{runs.length}</span>
        </header>

        {loading && <p className="text-sm text-slate-600">Loading…</p>}
        {!loading && runs.length === 0 && (
          <p className="text-sm text-slate-600">
            No runs yet. Start one from the home page.
          </p>
        )}

        <div className="space-y-2">
          {runs.map((run) => (
            <Link
              key={run.run_id}
              to={`/run/${run.run_id}`}
              className="block p-4 rounded-lg border border-[var(--color-edge)] bg-[var(--color-panel)] hover:border-slate-600 transition-colors"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-sm text-slate-200">
                    {run.paper_title || run.domain || "(no paper produced)"}
                  </p>
                  <p className="text-xs text-slate-500 mt-1">
                    {run.domain && <span>{run.domain} · </span>}
                    {new Date(run.started_at).toLocaleString()}
                  </p>
                  {run.error && (
                    <p className="text-xs text-rose-400/80 mt-1 line-clamp-1">{run.error}</p>
                  )}
                </div>

                <div className="text-right shrink-0">
                  <StatusBadge status={run.status} />
                  {run.status === "completed" && (
                    <p className="text-xs text-slate-500 mt-1.5 mono">
                      {Math.round(run.overall_confidence * 100)}% · {run.cycles_used}c
                      {run.abstained_count > 0 && ` · ${run.abstained_count} abstained`}
                    </p>
                  )}
                </div>
              </div>

              {run.status === "completed" && !run.accepted_by_critic && (
                <p className="text-[11px] text-amber-500/80 mt-2">
                  reached the cycle limit with objections outstanding
                </p>
              )}
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    completed: "bg-emerald-500/15 text-emerald-400",
    running: "bg-indigo-500/15 text-indigo-300",
    failed: "bg-rose-500/15 text-rose-400",
    cancelled: "bg-slate-500/15 text-slate-400",
  };
  return (
    <span className={`px-2 py-0.5 rounded-full text-[11px] ${styles[status] ?? styles.cancelled}`}>
      {status}
    </span>
  );
}
