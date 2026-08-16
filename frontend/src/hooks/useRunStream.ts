import { useCallback, useEffect, useRef, useState } from "react";
import type { RunEvent } from "@/lib/types";

type StreamState = "idle" | "connecting" | "streaming" | "done" | "error";

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
  const frameRef = useRef<number | null>(null);
  const onEventRef = useRef(onEvent);

  // Keep the callback current without making it an effect dependency, which
  // would tear down and rebuild the EventSource on every parent render.
  useEffect(() => {
    onEventRef.current = onEvent;
  }, [onEvent]);

  const flush = useCallback(() => {
    frameRef.current = null;
    if (bufferRef.current.length === 0) return;
    const batch = bufferRef.current;
    bufferRef.current = [];
    setEvents((prev) => [...prev, ...batch]);
  }, []);

  const schedule = useCallback(() => {
    if (frameRef.current !== null) return;
    frameRef.current = requestAnimationFrame(flush);
  }, [flush]);

  useEffect(() => {
    if (!runId) {
      setState("idle");
      return;
    }

    setState("connecting");
    setError("");

    const url = `/api/runs/${encodeURIComponent(runId)}/events?from_seq=${lastSeqRef.current}`;
    const source = new EventSource(url);
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
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    };
  }, [runId, schedule, flush]);

  const reset = useCallback(() => {
    sourceRef.current?.close();
    bufferRef.current = [];
    lastSeqRef.current = 0;
    setEvents([]);
    setState("idle");
    setError("");
  }, []);

  return { events, state, error, reset };
}
