import { useEffect, useRef, useState } from "react";
import { AGENTS, LEVEL_STYLES, type RunEvent } from "@/lib/types";

/**
 * The narration column.
 *
 * Two behaviours that matter more than they look:
 *
 * - **Autoscroll that yields to the reader.** It follows the tail until the
 *   user scrolls up, then stops until they return to the bottom. Fighting
 *   someone trying to read an earlier objection is the fastest way to make a
 *   live feed useless.
 * - **Failures are never filtered out.** Warnings and errors stay in the feed
 *   at full prominence. The system's credibility rests on showing its retries,
 *   its dropped objections and its abstentions, not on looking smooth.
 */

interface Props {
  events: RunEvent[];
}

const HIDDEN_TYPES = new Set(["vitals", "node_enter", "edge_traversed"]);

export function AgentFeed({ events }: Props) {
  const [following, setFollowing] = useState(true);
  const [showDetail, setShowDetail] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const visible = events.filter((event) => {
    if (HIDDEN_TYPES.has(event.type)) return false;
    if (!showDetail && event.type === "tool_call") return false;
    if (!showDetail && event.type === "node_exit") return false;
    return true;
  });

  useEffect(() => {
    if (following) bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [visible.length, following]);

  const onScroll = () => {
    const element = scrollRef.current;
    if (!element) return;
    const atBottom =
      element.scrollHeight - element.scrollTop - element.clientHeight < 60;
    setFollowing(atBottom);
  };

  return (
    <div className="flex flex-col h-full rounded-xl border border-[var(--color-edge)] bg-[var(--color-panel)] overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--color-edge)] shrink-0">
        <span className="text-xs uppercase tracking-wider text-slate-500">
          Agent feed
        </span>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowDetail((v) => !v)}
            className={`text-xs px-2 py-0.5 rounded transition-colors ${
              showDetail
                ? "bg-slate-700 text-slate-200"
                : "text-slate-500 hover:text-slate-300"
            }`}
          >
            tools
          </button>
          {!following && (
            <button
              onClick={() => {
                setFollowing(true);
                bottomRef.current?.scrollIntoView({ behavior: "smooth" });
              }}
              className="text-xs px-2 py-0.5 rounded bg-indigo-600/80 hover:bg-indigo-500 text-white"
            >
              ↓ live
            </button>
          )}
        </div>
      </div>

      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="flex-1 overflow-y-auto p-3 space-y-1 mono text-[13px] leading-relaxed"
      >
        {visible.length === 0 && (
          <p className="text-slate-600 p-3">Waiting for the first agent…</p>
        )}

        {visible.map((event) => {
          const agent = event.agent ? AGENTS[event.agent] : null;
          const isReroute = event.type === "reroute";

          return (
            <div
              key={event.seq}
              className={`flex gap-2 px-2 py-1 rounded ${
                isReroute ? "bg-rose-950/40 border border-rose-900/50" : ""
              }`}
            >
              <span
                className="shrink-0 w-9 text-right select-none"
                style={{ color: agent?.color ?? "#475569" }}
                title={agent?.label ?? "system"}
              >
                {agent?.short ?? "—"}
              </span>
              {event.cycle > 0 && (
                <span className="shrink-0 text-slate-600 select-none">
                  c{event.cycle}
                </span>
              )}
              <span className={`${LEVEL_STYLES[event.level]} break-words min-w-0`}>
                {event.message}
              </span>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
