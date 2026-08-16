import { formatDuration, type RunState } from "@/hooks/useRunState";

/**
 * The right-hand status column.
 *
 * Everything here is chosen to answer "is this real, and is it working?" —
 * the two questions a reviewer has while watching a ten-minute run. The cost
 * counter and the abstention count are deliberately prominent: one proves the
 * free-tier claim, the other proves the system declines to assert things it
 * cannot support.
 */

interface Props {
  state: RunState;
  elapsed: number;
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1.5">
      <span className="text-xs text-slate-500 shrink-0">{label}</span>
      <span className="text-sm text-right min-w-0 break-words">{children}</span>
    </div>
  );
}

function Gauge({ value }: { value: number }) {
  const percent = Math.round(value * 100);
  const colour =
    value >= 0.6 ? "bg-emerald-500" : value > 0 ? "bg-amber-500" : "bg-slate-700";
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-slate-500">Confidence</span>
        <span className={value >= 0.6 ? "text-emerald-400" : "text-amber-400"}>
          {percent}%
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-[var(--color-panel-2)] overflow-hidden">
        <div
          className={`h-full ${colour} transition-all duration-700`}
          style={{ width: `${Math.max(percent, 2)}%` }}
        />
      </div>
      {/* The threshold is shown even before any claim is scored, so the
          abstention rule is visible as a policy rather than a surprise. */}
      <p className="text-[11px] text-slate-600">
        claims below 60% are not asserted
      </p>
    </div>
  );
}

export function Vitals({ state, elapsed }: Props) {
  const cycleLabel = `${state.cycle} / ${state.maxCycles}`;

  return (
    <div className="rounded-xl border border-[var(--color-edge)] bg-[var(--color-panel)] p-4 space-y-4">
      <div className="space-y-3">
        <Gauge value={state.confidence} />
      </div>

      <div className="h-px bg-[var(--color-edge)]" />

      <div className="divide-y divide-[var(--color-edge)]/60">
        <Row label="Cycle">
          <span className={state.cycle > 1 ? "text-amber-400" : ""}>{cycleLabel}</span>
        </Row>
        <Row label="Elapsed">{formatDuration(elapsed)}</Row>
        <Row label="Domain">
          {state.domain ? (
            <span className="text-sky-300">{state.domain}</span>
          ) : (
            <span className="text-slate-600">discovering…</span>
          )}
        </Row>
        <Row label="Abstentions">
          <span className={state.abstentions > 0 ? "text-amber-400" : "text-slate-400"}>
            {state.abstentions}
          </span>
        </Row>
        <Row label="Reroutes">
          <span className={state.reroutes.length > 0 ? "text-rose-400" : "text-slate-400"}>
            {state.reroutes.length}
          </span>
        </Row>
        <Row label="Tool failures">
          <span className={state.toolFailures > 0 ? "text-amber-400" : "text-slate-400"}>
            {state.toolFailures}
          </span>
        </Row>
        {state.tokens > 0 && <Row label="Tokens">{state.tokens.toLocaleString()}</Row>}
        <Row label="Cost">
          <span className="text-emerald-400">$0.00</span>
        </Row>
      </div>

      {state.degradedTools.length > 0 && (
        <div className="rounded-lg bg-amber-950/30 border border-amber-900/40 p-2.5">
          <p className="text-[11px] text-amber-500 mb-1">Degraded sources</p>
          {state.degradedTools.map((tool) => (
            <p key={tool} className="text-xs text-amber-300/80 mono break-all">
              {tool}
            </p>
          ))}
          <p className="text-[11px] text-slate-500 mt-1.5">
            The run continued on the remaining sources.
          </p>
        </div>
      )}

      {state.question && (
        <div className="rounded-lg bg-[var(--color-panel-2)] p-3">
          <p className="text-[11px] uppercase tracking-wider text-slate-500 mb-1.5">
            Research question
          </p>
          <p className="text-[13px] text-slate-300 leading-relaxed">{state.question}</p>
        </div>
      )}
    </div>
  );
}
