import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { RunEvent } from "@/lib/types";

type StreamState = "idle" | "connecting" | "streaming" | "done" | "error";

/** Batching window. Short enough to feel live, long enough to coalesce the
 *  bursts a busy phase produces. */
const FLUSH_INTERVAL_MS = 90;

interface Options {
  /** Replay from this sequence number. Used when reconnecting mid-run. */
  fromSeq?: number;
  onEvent?: (event: RunEvent) => void;
}

/**
 * Subscribes to a run's SSE stream.
 *
 * Two details matter here and both come from the transport rather than React:
 *
 * 1. `EventSource` auto-reconnects on drop, and would replay the entire run
 *    from scratch each time. We track the highest sequence seen and reconnect
 *    explicitly with `?from_seq=`, so a dropped connection resumes instead of
 *    duplicating the whole feed.
 * 2. Events arrive faster than React should re-render during heavy phases, so
 *    they are buffered and flushed on an animation frame.
 */
export function useRunStream(runId: string | null, options: Options = {}) {
  const { fromSeq = 0, onEvent } = options;

  const [events, setEvents] = useState<RunEvent[]>([]);
  const [state, setState] = useState<StreamState>("idle");
  const [error, setError] = useState<string>("");

  const sourceRef = useRef<EventSource | null>(null);
  const lastSeqRef = useRef<number>(fromSeq);
  const bufferRef = useRef<RunEvent[]>([]);
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onEventRef = useRef(onEvent);

  // Keep the callback current without making it an effect dependency, which
  // would tear down and rebuild the EventSource on every parent render.
  useEffect(() => {
    onEventRef.current = onEvent;
  }, [onEvent]);

  const flush = useCallback(() => {
    flushTimerRef.current = null;
    if (bufferRef.current.length === 0) return;
    const batch = bufferRef.current;
    bufferRef.current = [];
    setEvents((prev) => [...prev, ...batch]);
  }, []);

  /**
   * Coalesce bursts with a timer, deliberately NOT requestAnimationFrame.
   *
   * rAF does not fire while the tab is backgrounded or otherwise not
   * compositing, so buffered events would sit unflushed indefinitely and the
   * feed would appear frozen — on a run that takes minutes, switching tabs is
   * not an edge case, it is the normal thing to do. Timers are throttled in
   * background tabs but they still fire, so the feed stays current and simply
   * updates less smoothly.
   */
  const schedule = useCallback(() => {
    if (flushTimerRef.current !== null) return;
    flushTimerRef.current = setTimeout(flush, FLUSH_INTERVAL_MS);
  }, [flush]);

  useEffect(() => {
    if (!runId) {
      setState("idle");
      return;
    }

    setState("connecting");
    setError("");

    // Absolute URL: EventSource does not inherit any base, so a split
    // deployment needs the backend origin spelled out.
    const source = new EventSource(
      api.eventsUrl(encodeURIComponent(runId), lastSeqRef.current),
    );
    sourceRef.current = source;

    source.onopen = () => setState("streaming");

    source.onmessage = (raw) => {
      let event: RunEvent;
      try {
        event = JSON.parse(raw.data) as RunEvent;
      } catch {
        return;
      }
      if (event.seq <= lastSeqRef.current) return; // already seen, ignore replay
      lastSeqRef.current = event.seq;

      bufferRef.current.push(event);
      onEventRef.current?.(event);
      schedule();

      if (
        event.type === "run_completed" ||
        event.type === "run_failed" ||
        event.type === "run_cancelled"
      ) {
        flush();
        setState("done");
        source.close();
      }
    };

    source.onerror = () => {
      // EventSource retries on its own; only surface an error if it gave up.
      if (source.readyState === EventSource.CLOSED) {
        setState((prev) => (prev === "done" ? prev : "error"));
        setError("Connection to the run stream was lost.");
      }
    };

    return () => {
      source.close();
      sourceRef.current = null;
      if (flushTimerRef.current !== null) clearTimeout(flushTimerRef.current);
    };
  }, [runId, schedule, flush]);

  const reset = useCallback(() => {
    sourceRef.current?.close();
    if (flushTimerRef.current !== null) clearTimeout(flushTimerRef.current);
    bufferRef.current = [];
    lastSeqRef.current = 0;
    setEvents([]);
    setState("idle");
    setError("");
  }, []);

  return { events, state, error, reset };
}
