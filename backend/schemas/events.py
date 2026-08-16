"""The event contract shared by the graph, the SSE stream, and the React UI.

Every event is persisted to SQLite as it is emitted. That single decision buys
three features at once: a mid-run browser refresh replays cleanly, the run
gallery can show finished runs without re-running anything, and the trace tab
is just the event log rendered differently.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .common import AgentName


class EventType(str, Enum):
    # lifecycle
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"

    # graph topology -- drives the live React Flow visualisation
    NODE_ENTER = "node_enter"
    NODE_EXIT = "node_exit"
    EDGE_TRAVERSED = "edge_traversed"
    CYCLE_STARTED = "cycle_started"
    REROUTE = "reroute"

    # narration
    AGENT_MESSAGE = "agent_message"
    TOOL_CALL = "tool_call"
    QUIP = "quip"

    # substantive results, rendered as summary cards the moment they land
    ARTIFACT = "artifact"
    VITALS = "vitals"
    ABSTENTION = "abstention"
    CONFLICT = "conflict"
    WARNING = "warning"


class Level(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    SUCCESS = "success"
    WARN = "warn"
    ERROR = "error"


class RunEvent(BaseModel):
    """One thing that happened, in order.

    `seq` is assigned by the event bus and is what makes replay deterministic;
    clients reconnecting send the last seq they saw and receive only the gap.
    """

    seq: int = 0
    run_id: str = ""
    type: EventType
    agent: AgentName | None = None
    level: Level = Level.INFO
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    cycle: int = 0
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_sse(self) -> dict[str, str]:
        """Shape expected by sse-starlette.

        The SSE `event:` field is deliberately **not** set. Naming it makes the
        browser dispatch a custom event type, which means `EventSource.onmessage`
        never fires and every event type needs its own `addEventListener` --
        a silent failure that looks exactly like a dead backend. The type is
        already inside the JSON payload, so a single handler switching on
        `data.type` is both simpler and harder to get wrong.
        """
        return {"id": str(self.seq), "data": self.model_dump_json()}


class ArtifactKind(str, Enum):
    """Substantive payloads the UI knows how to render as a card."""

    DOMAIN_CANDIDATES = "domain_candidates"
    EMERGENCE_CHART = "emergence_chart"
    DOMAIN_SELECTED = "domain_selected"
    QUESTION_SET = "question_set"
    QUESTION_SELECTED = "question_selected"
    SOURCE_ACQUIRED = "source_acquired"
    DATASET_SUMMARY = "dataset_summary"
    EXPERIMENT_SPEC = "experiment_spec"
    EXPERIMENT_RESULT = "experiment_result"
    FIGURE = "figure"
    CRITIQUE = "critique"
    CONFIDENCE_REPORT = "confidence_report"
    PAPER = "paper"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunSummary(BaseModel):
    """Row in the gallery, and the payload of GET /api/runs/{id}."""

    run_id: str
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None = None
    domain: str = ""
    question: str = ""
    paper_title: str = ""
    cycles_used: int = 0
    overall_confidence: float = 0.0
    abstained_count: int = 0
    accepted_by_critic: bool = False
    error: str = ""
    event_count: int = 0

    @property
    def duration_seconds(self) -> float:
        if not self.finished_at:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()
