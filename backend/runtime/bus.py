"""In-process event bus with replay.

One bus per run. The graph publishes synchronously (agents are not async-aware
about the UI and should not be), subscribers consume asynchronously, and every
event is retained so that a browser arriving late -- or refreshing mid-run --
receives the full history before the live tail.

That retention is what makes the run gallery and the trace tab free: both are
just the event log rendered differently.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from ..schemas import AgentName, ArtifactKind, EventType, Level, RunEvent

log = logging.getLogger(__name__)

# Bounded so a runaway agent cannot exhaust memory on a free-tier container.
MAX_RETAINED_EVENTS = 5_000
SUBSCRIBER_QUEUE_SIZE = 1_000


class EventBus:
    """Fan-out event stream for a single run."""

    def __init__(self, run_id: str, on_persist: Callable[[RunEvent], None] | None = None) -> None:
        self.run_id = run_id
        self._history: list[RunEvent] = []
        self._subscribers: set[asyncio.Queue[RunEvent | None]] = set()
        self._seq = 0
        self._closed = False
        self._on_persist = on_persist

    # ------------------------------------------------------------- publishing

    def publish(self, event: RunEvent) -> RunEvent:
        """Emit an event. Safe to call from synchronous agent code."""
        if self._closed:
            return event

        self._seq += 1
        event.seq = self._seq
        event.run_id = self.run_id

        self._history.append(event)
        if len(self._history) > MAX_RETAINED_EVENTS:
            # Drop from the middle: the opening events explain how the run
            # started and the recent ones are what anyone is actually watching.
            del self._history[MAX_RETAINED_EVENTS // 4 : MAX_RETAINED_EVENTS // 2]

        if self._on_persist is not None:
            try:
                self._on_persist(event)
            except Exception:  # noqa: BLE001 - persistence must never break a run
                log.exception("failed to persist event %s", event.seq)

        dead: list[asyncio.Queue[RunEvent | None]] = []
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A stalled client must not slow the run down. Drop it; it will
                # replay from history when it reconnects.
                log.warning("subscriber queue full on run %s; dropping client", self.run_id)
                dead.append(queue)
        for queue in dead:
            self._subscribers.discard(queue)

        return event

    # --------------------------------------------------------- convenience API

    def message(
        self,
        agent: AgentName,
        text: str,
        *,
        level: Level = Level.INFO,
        cycle: int = 0,
        **payload: Any,
    ) -> RunEvent:
        return self.publish(
            RunEvent(
                type=EventType.AGENT_MESSAGE,
                agent=agent,
                level=level,
                message=text,
                cycle=cycle,
                payload=payload,
            )
        )

    def artifact(
        self,
        agent: AgentName,
        kind: ArtifactKind,
        data: dict[str, Any],
        *,
        message: str = "",
        cycle: int = 0,
    ) -> RunEvent:
        """Publish a substantive result the UI renders as a summary card."""
        return self.publish(
            RunEvent(
                type=EventType.ARTIFACT,
                agent=agent,
                level=Level.SUCCESS,
                message=message,
                cycle=cycle,
                payload={"kind": kind.value, "data": data},
            )
        )

    def node_enter(self, agent: AgentName, cycle: int = 0, note: str = "") -> RunEvent:
        return self.publish(
            RunEvent(
                type=EventType.NODE_ENTER,
                agent=agent,
                message=note or f"{agent.value} started",
                cycle=cycle,
            )
        )

    def node_exit(
        self, agent: AgentName, cycle: int = 0, ok: bool = True, note: str = ""
    ) -> RunEvent:
        return self.publish(
            RunEvent(
                type=EventType.NODE_EXIT,
                agent=agent,
                level=Level.SUCCESS if ok else Level.ERROR,
                message=note or f"{agent.value} finished",
                cycle=cycle,
                payload={"ok": ok},
            )
        )

    def edge(self, source: AgentName, target: AgentName, cycle: int = 0, reason: str = "") -> RunEvent:
        return self.publish(
            RunEvent(
                type=EventType.EDGE_TRAVERSED,
                message=reason,
                cycle=cycle,
                payload={"from": source.value, "to": target.value, "reason": reason},
            )
        )

    def reroute(self, target: str, reason: str, cycle: int) -> RunEvent:
        """The Critic sending work back. Rendered as a red edge in the UI."""
        return self.publish(
            RunEvent(
                type=EventType.REROUTE,
                agent=AgentName.CRITIC,
                level=Level.WARN,
                message=f"Sending work back to {target}: {reason}",
                cycle=cycle,
                payload={"target": target, "reason": reason},
            )
        )

    def tool_call(
        self,
        agent: AgentName,
        tool: str,
        *,
        ok: bool = True,
        detail: str = "",
        duration_ms: int = 0,
        cycle: int = 0,
    ) -> RunEvent:
        return self.publish(
            RunEvent(
                type=EventType.TOOL_CALL,
                agent=agent,
                level=Level.INFO if ok else Level.WARN,
                message=detail or tool,
                cycle=cycle,
                payload={"tool": tool, "ok": ok, "duration_ms": duration_ms},
            )
        )

    def vitals(self, data: dict[str, Any]) -> RunEvent:
        return self.publish(RunEvent(type=EventType.VITALS, payload=data))

    # ------------------------------------------------------------ subscribing

    async def subscribe(self, from_seq: int = 0) -> AsyncIterator[RunEvent]:
        """Yield history from `from_seq`, then live events until the run closes.

        Backfilling before subscribing is deliberate: a client that connects
        during cycle 3 still sees how the system got there, which is most of
        what makes the trace worth reading.
        """
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        backlog = [e for e in self._history if e.seq > from_seq]
        self._subscribers.add(queue)

        try:
            for event in backlog:
                yield event

            if self._closed:
                return

            last_seq = backlog[-1].seq if backlog else from_seq
            while True:
                event = await queue.get()
                if event is None:
                    return
                # Guard against re-delivering anything already sent as backlog.
                if event.seq > last_seq:
                    last_seq = event.seq
                    yield event
        finally:
            self._subscribers.discard(queue)

    def close(self) -> None:
        """Signal end-of-stream to every subscriber."""
        if self._closed:
            return
        self._closed = True
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    # ----------------------------------------------------------------- access

    @property
    def history(self) -> list[RunEvent]:
        return list(self._history)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
