"""Tests for the event bus and the SSE wire format."""

from __future__ import annotations

import asyncio
import json

import pytest

from backend.runtime.bus import EventBus
from backend.schemas import AgentName, ArtifactKind, EventType, Level, RunEvent


class TestSseWireFormat:
    def test_event_field_is_omitted(self):
        """Regression guard.

        Setting SSE's `event:` field makes browsers dispatch a custom event
        type, so `EventSource.onmessage` silently never fires. This cost a
        debugging cycle once; it should not cost a second one.
        """
        frame = RunEvent(type=EventType.AGENT_MESSAGE, message="hi").to_sse()
        assert "event" not in frame
        assert set(frame) == {"id", "data"}

    def test_type_travels_inside_the_payload(self):
        frame = RunEvent(type=EventType.REROUTE, message="back to data").to_sse()
        assert json.loads(frame["data"])["type"] == "reroute"

    def test_id_carries_the_sequence_for_reconnection(self):
        bus = EventBus("r")
        event = bus.message(AgentName.SCOUT, "one")
        assert event.to_sse()["id"] == str(event.seq)


class TestSequencing:
    def test_sequence_numbers_are_monotonic_and_start_at_one(self):
        bus = EventBus("r")
        seqs = [bus.message(AgentName.SCOUT, f"m{i}").seq for i in range(5)]
        assert seqs == [1, 2, 3, 4, 5]

    def test_run_id_is_stamped_on_every_event(self):
        bus = EventBus("run-abc")
        assert bus.message(AgentName.CRITIC, "x").run_id == "run-abc"

    def test_publishing_after_close_is_a_no_op(self):
        bus = EventBus("r")
        bus.message(AgentName.SCOUT, "before")
        bus.close()
        bus.message(AgentName.SCOUT, "after")
        assert len(bus.history) == 1


class TestPersistenceHook:
    def test_every_event_is_offered_to_the_persister(self):
        seen: list[RunEvent] = []
        bus = EventBus("r", on_persist=seen.append)
        bus.message(AgentName.SCOUT, "a")
        bus.artifact(AgentName.SCOUT, ArtifactKind.DOMAIN_CANDIDATES, {"n": 5})
        assert len(seen) == 2

    def test_a_failing_persister_never_breaks_a_run(self):
        """Losing the audit trail is bad; losing the run is worse."""

        def explode(_: RunEvent) -> None:
            raise RuntimeError("disk full")

        bus = EventBus("r", on_persist=explode)
        bus.message(AgentName.SCOUT, "still works")
        assert len(bus.history) == 1


class TestSubscription:
    async def test_late_subscriber_receives_full_backlog(self):
        bus = EventBus("r")
        for i in range(3):
            bus.message(AgentName.SCOUT, f"m{i}")
        bus.close()

        got = [e.message async for e in bus.subscribe()]
        assert got == ["m0", "m1", "m2"]

    async def test_from_seq_resumes_without_duplicating(self):
        bus = EventBus("r")
        for i in range(5):
            bus.message(AgentName.SCOUT, f"m{i}")
        bus.close()

        got = [e.seq async for e in bus.subscribe(from_seq=3)]
        assert got == [4, 5]

    async def test_live_events_follow_the_backlog(self):
        bus = EventBus("r")
        bus.message(AgentName.SCOUT, "historic")

        received: list[str] = []

        async def consume() -> None:
            async for event in bus.subscribe():
                received.append(event.message)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)

        bus.message(AgentName.CRITIC, "live")
        await asyncio.sleep(0.05)
        bus.close()
        await asyncio.wait_for(task, timeout=2.0)

        assert received == ["historic", "live"]

    async def test_subscriber_is_released_on_close(self):
        bus = EventBus("r")

        async def consume() -> None:
            async for _ in bus.subscribe():
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        assert bus.subscriber_count == 1

        bus.close()
        await asyncio.wait_for(task, timeout=2.0)
        assert bus.subscriber_count == 0


class TestConvenienceHelpers:
    def test_reroute_is_attributed_to_the_critic_and_warns(self):
        bus = EventBus("r")
        event = bus.reroute("experiment", "p=0.41 exceeds alpha", cycle=2)
        assert event.type == EventType.REROUTE
        assert event.agent == AgentName.CRITIC
        assert event.level == Level.WARN
        assert event.payload["target"] == "experiment"
        assert event.cycle == 2

    def test_artifact_carries_kind_and_data(self):
        bus = EventBus("r")
        event = bus.artifact(
            AgentName.SCOUT, ArtifactKind.EMERGENCE_CHART, {"figure": "{}"}, message="chart"
        )
        assert event.payload["kind"] == "emergence_chart"
        assert event.payload["data"] == {"figure": "{}"}

    def test_failed_tool_call_is_warned_not_errored(self):
        """A single tool failure is expected and recoverable; the circuit
        breaker escalates only after repeats."""
        bus = EventBus("r")
        event = bus.tool_call(AgentName.ALCHEMIST, "fetch", ok=False, detail="403")
        assert event.level == Level.WARN
        assert event.payload["ok"] is False


@pytest.mark.parametrize("agent", list(AgentName))
def test_every_agent_has_a_frontend_colour(agent: AgentName):
    """Guards against an agent being added in Python but not in the UI, which
    renders as an invisible row in the feed."""
    from pathlib import Path

    types_ts = Path(__file__).resolve().parents[2] / "frontend" / "src" / "lib" / "types.ts"
    assert f"{agent.value}:" in types_ts.read_text(encoding="utf-8"), (
        f"{agent.value} missing from AGENTS in frontend/src/lib/types.ts"
    )
