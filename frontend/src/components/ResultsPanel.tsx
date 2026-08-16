import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { PlotlyFigure } from "./PlotlyFigure";
import type { Artifact, RunState } from "@/hooks/useRunState";
import { api } from "@/lib/api";

/**
 * Results, once the run finishes.
 *
 * The tab order is an argument: Paper first because it is the deliverable, but
 * Confidence and Evidence sit ahead of Trace because the credibility of the
 * paper rests on what the system declined to assert and where its data came
 * from. A reviewer who reads only two tabs should read those two.
 */

type Tab = "paper" | "dashboards" | "confidence" | "evidence" | "trace";

interface Props {
  runId: string;
  state: RunState;
}

const TABS: { id: Tab; label: string }[] = [
  { id: "paper", label: "Paper" },
  { id: "dashboards", label: "Dashboards" },
  { id: "confidence", label: "Confidence" },
  { id: "evidence", label: "Evidence" },
  { id: "trace", label: "Trace" },
];

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="text-sm text-slate-600 p-6 text-center">{children}</p>;
}

export function ResultsPanel({ runId, state }: Props) {
  const [tab, setTab] = useState<Tab>("paper");

  const paper = state.byKind.paper;
  const confidence = state.byKind.confidence_report;
  const dataset = state.byKind.dataset_summary;
  const critique = state.byKind.critique;

  const markdown = String(paper?.data.markdown ?? "");
  const verification = paper?.data.verification as
    | {
        citations_found: number;
        citations_verified: number;
        numbers_found: number;
        numbers_verified: number;
        findings: { kind: string; detail: string }[];
      }
    | undefined;

  const counts = useMemo(
    () => ({
      dashboards: state.figures.length,
      evidence: ((dataset?.data.sources as unknown[]) ?? []).length,
    }),
    [state.figures.length, dataset],
  );

  return (
    <div className="rounded-xl border border-[var(--color-edge)] bg-[var(--color-panel)] overflow-hidden">
      <div className="flex items-center gap-1 px-3 py-2 border-b border-[var(--color-edge)] overflow-x-auto">
        {TABS.map(({ id, label }) => {
          const badge =
            id === "dashboards"
              ? counts.dashboards
              : id === "evidence"
                ? counts.evidence
                : id === "confidence"
                  ? state.abstentions || undefined
                  : undefined;
          return (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`px-3 py-1.5 rounded-lg text-sm whitespace-nowrap transition-colors ${
                tab === id
                  ? "bg-[var(--color-panel-2)] text-slate-100"
                  : "text-slate-500 hover:text-slate-300"
              }`}
            >
              {label}
              {badge ? <span className="ml-1.5 text-xs text-slate-500">{badge}</span> : null}
            </button>
          );
        })}
        <div className="ml-auto flex items-center gap-2 pl-3">
          <a
            href={api.paperUrl(runId)}
            className="text-xs text-slate-500 hover:text-slate-300"
            target="_blank"
            rel="noreferrer"
          >
            paper.md
          </a>
          <a
            href={api.manifestUrl(runId)}
            className="text-xs text-slate-500 hover:text-slate-300"
            target="_blank"
            rel="noreferrer"
          >
            manifest
          </a>
        </div>
      </div>

      <div className="p-4 max-h-[70vh] overflow-y-auto">
        {tab === "paper" &&
          (markdown ? (
            <>
              {verification && verification.findings.length > 0 && (
                <div className="mb-4 rounded-lg border border-amber-900/50 bg-amber-950/20 p-3">
                  <p className="text-sm text-amber-400 mb-1.5">
                    Verification caught {verification.findings.length} problem
                    {verification.findings.length > 1 ? "s" : ""} in the generated text
                  </p>
                  <ul className="text-xs text-amber-300/80 space-y-1">
                    {verification.findings.map((finding, i) => (
                      <li key={i} className="mono">
                        [{finding.kind}] {finding.detail}
                      </li>
                    ))}
                  </ul>
                  <p className="text-[11px] text-slate-500 mt-2">
                    {verification.citations_verified}/{verification.citations_found} citations
                    traced to fetched sources ·{" "}
                    {verification.numbers_verified}/{verification.numbers_found} reported
                    numbers matched computed results
                  </p>
                </div>
              )}
              <article className="prose-paper">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdown}</ReactMarkdown>
              </article>
            </>
          ) : (
            <Empty>The paper appears when the run completes.</Empty>
          ))}

        {tab === "dashboards" &&
          (state.figures.length ? (
            <div className="space-y-4">
              {state.figures.map((figure) => (
                <PlotlyFigure
                  key={figure.seq}
                  figureJson={String(figure.data.figure ?? "")}
                  title={String(figure.data.title ?? "")}
                  caption={String(figure.data.caption ?? "")}
                />
              ))}
            </div>
          ) : (
            <Empty>Figures appear as the agents produce them.</Empty>
          ))}

        {tab === "confidence" && <ConfidenceTab artifact={confidence} />}
        {tab === "evidence" && <EvidenceTab dataset={dataset} critique={critique} />}
        {tab === "trace" && <TraceTab state={state} />}
      </div>
    </div>
  );
}

function ConfidenceTab({ artifact }: { artifact?: Artifact }) {
  if (!artifact) return <Empty>Confidence scores appear after the experiment runs.</Empty>;

  const threshold = Number(artifact.data.threshold ?? 0.6);
  const claims = (artifact.data.claims ?? []) as {
    text: string;
    confidence: number;
    abstained: boolean;
    abstain_reason: string;
    components: Record<string, number | null>;
  }[];

  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-400">
        Confidence is measured, not self-reported: self-consistency across resamples,
        agreement with a second model family, the statistics themselves, and evidence
        quality. Claims below {Math.round(threshold * 100)}% are not asserted.
      </p>
      {claims.map((claim, i) => (
        <div
          key={i}
          className={`rounded-lg border p-3 ${
            claim.abstained
              ? "border-amber-900/50 bg-amber-950/20"
              : "border-[var(--color-edge)] bg-[var(--color-panel-2)]"
          }`}
        >
          <div className="flex items-start justify-between gap-3">
            <p className="text-sm text-slate-200 leading-relaxed">{claim.text}</p>
            <span
              className={`text-sm mono shrink-0 ${
                claim.abstained ? "text-amber-400" : "text-emerald-400"
              }`}
            >
              {Math.round(claim.confidence * 100)}%
            </span>
          </div>
          {claim.abstained && (
            <p className="text-xs text-amber-400/80 mt-1.5">ABSTAINED — {claim.abstain_reason}</p>
          )}
          <div className="flex flex-wrap gap-3 mt-2 text-[11px] text-slate-500 mono">
            {Object.entries(claim.components).map(([name, value]) => (
              <span key={name}>
                {name.replace(/_/g, " ")}: {value === null ? "n/a" : value.toFixed(2)}
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function EvidenceTab({ dataset, critique }: { dataset?: Artifact; critique?: Artifact }) {
  if (!dataset) return <Empty>Sources appear once the Data Alchemist finishes.</Empty>;

  const sources = (dataset.data.sources ?? []) as {
    url: string;
    modality: string;
    title: string;
    sha256: string;
    note: string;
  }[];
  const conflicts = (dataset.data.conflicts ?? []) as {
    subject: string;
    value_a: string;
    source_a: string;
    value_b: string;
    source_b: string;
  }[];
  const objections = (critique?.data.objections ?? []) as {
    severity: string;
    claim: string;
    rationale: string;
    url: string;
    verified: boolean;
    verification_note: string;
  }[];

  return (
    <div className="space-y-5">
      <section>
        <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-2">
          Sources ({sources.length}) · every one content-hashed
        </h3>
        <div className="space-y-1.5">
          {sources.map((source, i) => (
            <div
              key={i}
              className="flex items-start gap-2 text-xs p-2 rounded bg-[var(--color-panel-2)]"
            >
              <span className="mono text-slate-500 shrink-0">{source.modality}</span>
              <a
                href={source.url}
                target="_blank"
                rel="noreferrer"
                className="text-sky-400 hover:underline min-w-0 break-all"
              >
                {source.title || source.url}
              </a>
              <span className="mono text-slate-600 shrink-0 ml-auto">
                {source.sha256}
              </span>
            </div>
          ))}
        </div>
      </section>

      {conflicts.length > 0 && (
        <section>
          <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-2">
            Source conflicts ({conflicts.length}) · surfaced, not silently resolved
          </h3>
          {conflicts.map((conflict, i) => (
            <div key={i} className="text-xs p-2 rounded bg-amber-950/20 mb-1.5">
              <p className="text-amber-300">{conflict.subject}</p>
              <p className="text-slate-400 mono mt-0.5">
                {conflict.source_a}: {conflict.value_a} ≠ {conflict.source_b}: {conflict.value_b}
              </p>
            </div>
          ))}
        </section>
      )}

      {objections.length > 0 && (
        <section>
          <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-2">
            Critic objections · citations fetched and checked
          </h3>
          {objections.map((objection, i) => (
            <div key={i} className="text-xs p-2 rounded bg-[var(--color-panel-2)] mb-1.5">
              <div className="flex items-center gap-2">
                <span className="text-rose-400 mono">[{objection.severity}]</span>
                <span className={objection.verified ? "text-emerald-500" : "text-rose-500"}>
                  {objection.verified ? "citation verified" : "DISCARDED — unverifiable"}
                </span>
              </div>
              <p className="text-slate-300 mt-1">{objection.claim}</p>
              <p className="text-slate-500 mt-0.5">{objection.rationale}</p>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}

function TraceTab({ state }: { state: RunState }) {
  return (
    <div className="space-y-4 text-sm">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          ["Cycles", `${state.cycle}/${state.maxCycles}`],
          ["Reroutes", state.reroutes.length],
          ["Tool failures", state.toolFailures],
          ["Tokens", state.tokens.toLocaleString()],
        ].map(([label, value]) => (
          <div key={String(label)} className="rounded-lg bg-[var(--color-panel-2)] p-3">
            <p className="text-[11px] text-slate-500">{label}</p>
            <p className="text-lg mono">{value}</p>
          </div>
        ))}
      </div>

      {state.reroutes.length > 0 && (
        <section>
          <h3 className="text-xs uppercase tracking-wider text-slate-500 mb-2">
            Critic reroutes · the iteration loop, itemised
          </h3>
          {state.reroutes.map((reroute, i) => (
            <div key={i} className="text-xs p-2 rounded bg-rose-950/20 mb-1.5">
              <span className="mono text-rose-400">
                cycle {reroute.cycle} → {reroute.to}
              </span>
              <p className="text-slate-400 mt-0.5">{reroute.reason}</p>
            </div>
          ))}
        </section>
      )}

      <p className="text-xs text-slate-600">
        Total cost: <span className="text-emerald-500">$0.00</span> — free-tier providers only.
      </p>
    </div>
  );
}
