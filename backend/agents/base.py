"""Shared scaffolding for every agent in the graph.

Each agent is a LangGraph node: it takes the run state and returns a partial
update. `BaseAgent` wraps that with the concerns every node shares -- timing,
event emission, and failure containment -- so an individual agent's code is
only its actual reasoning.

The failure policy is the important part. An agent that raises does **not**
crash the run. It records the error into state and returns, letting the
supervisor's routing decide whether the run can continue without it. A domain
scout that dies should end the run; a figure OCR that dies should not. That
judgement belongs to the graph, not to whichever node happened to throw.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..llm import LLMRouter
from ..llm.budget import BudgetExhausted
from ..memory.vector import VectorMemory
from ..runtime.bus import EventBus
from ..schemas import AgentName, ArtifactKind, Level
from ..tools.http import Fetcher

log = logging.getLogger(__name__)


class AgentFailure(RuntimeError):
    """Raised by an agent when it cannot complete its job.

    Distinct from an unexpected exception: this one is deliberate, carries a
    message written for the event feed, and marks whether the run can go on
    without this node's output.
    """

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


@dataclass
class AgentContext:
    """Everything an agent needs from the outside world.

    Passed in rather than imported so tests can substitute a fake router and
    a mocked fetcher without patching module globals.
    """

    run_id: str
    settings: Settings
    router: LLMRouter
    fetcher: Fetcher
    bus: EventBus
    #: Run-scoped vector store. Optional: when unavailable the agents fall back
    #: to titles and abstracts, which is worse but not fatal.
    memory: VectorMemory | None = None


class BaseAgent(ABC):
    """Base class for every node in the research graph."""

    name: AgentName
    #: Whether a failure in this agent should end the run. Discovery agents are
    #: fatal (there is nothing to research without them); enrichment is not.
    fatal_on_failure: bool = False

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    # ------------------------------------------------------------ node entry

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph node entrypoint."""
        cycle = state.get("cycle", 0)
        self.ctx.bus.node_enter(self.name, cycle=cycle)
        started = time.perf_counter()

        try:
            update = await self.execute(state)

        except BudgetExhausted as exc:
            # Every provider is out of free-tier budget. Nothing downstream can
            # recover from this, so it always ends the run -- and says why,
            # rather than producing a mysteriously empty paper.
            elapsed = int((time.perf_counter() - started) * 1000)
            self.ctx.bus.node_exit(self.name, cycle=cycle, ok=False, note=str(exc))
            log.error("%s: budget exhausted after %dms", self.name.value, elapsed)
            return {
                "errors": [f"{self.name.value}: {exc}"],
                "finished": True,
                "failure_reason": f"Free-tier budget exhausted during {self.name.value}.",
            }

        except AgentFailure as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            self.ctx.bus.node_exit(self.name, cycle=cycle, ok=False, note=str(exc))
            log.warning("%s failed after %dms: %s", self.name.value, elapsed, exc)
            update = {"errors": [f"{self.name.value}: {exc}"]}
            if exc.fatal or self.fatal_on_failure:
                update |= {"finished": True, "failure_reason": str(exc)}
            return update

        except Exception as exc:  # noqa: BLE001 - containment is the whole point
            elapsed = int((time.perf_counter() - started) * 1000)
            log.exception("%s raised unexpectedly after %dms", self.name.value, elapsed)
            self.ctx.bus.node_exit(
                self.name, cycle=cycle, ok=False, note=f"unexpected error: {exc}"
            )
            update = {"errors": [f"{self.name.value}: unexpected {type(exc).__name__}: {exc}"]}
            if self.fatal_on_failure:
                update |= {
                    "finished": True,
                    "failure_reason": f"{self.name.value} failed: {exc}",
                }
            return update

        elapsed = int((time.perf_counter() - started) * 1000)
        self.ctx.bus.node_exit(self.name, cycle=cycle, note=f"done in {elapsed / 1000:.1f}s")
        log.info("%s completed in %dms", self.name.value, elapsed)
        return update

    @abstractmethod
    async def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Do the work. Return a partial state update."""

    # -------------------------------------------------------------- helpers

    def say(self, text: str, *, level: Level = Level.INFO, cycle: int = 0, **payload: Any) -> None:
        """Narrate to the live feed."""
        self.ctx.bus.message(self.name, text, level=level, cycle=cycle, **payload)

    def publish(
        self,
        kind: ArtifactKind,
        data: dict[str, Any],
        *,
        message: str = "",
        cycle: int = 0,
    ) -> None:
        """Publish a result the UI renders as a card the moment it lands.

        Called as soon as each piece of substance exists rather than at the end
        of the node, so a reviewer watching a ten-minute run sees the five
        discovered domains within the first minute instead of a spinner.
        """
        self.ctx.bus.artifact(self.name, kind, data, message=message, cycle=cycle)

    def tool(
        self, tool_name: str, *, ok: bool = True, detail: str = "", ms: int = 0, cycle: int = 0
    ) -> None:
        self.ctx.bus.tool_call(
            self.name, tool_name, ok=ok, detail=detail, duration_ms=ms, cycle=cycle
        )

    @property
    def settings(self) -> Settings:
        return self.ctx.settings

    @property
    def router(self) -> LLMRouter:
        return self.ctx.router

    @property
    def fetcher(self) -> Fetcher:
        return self.ctx.fetcher

    @property
    def memory(self) -> VectorMemory | None:
        return self.ctx.memory

    def recall(self, query: str, *, k: int = 5, max_chars: int = 4000) -> str:
        """Retrieve evidence relevant to `query` from the acquired documents.

        Returns an empty string when vector memory is unavailable, so callers
        can interpolate it unconditionally and simply get a thinner prompt.
        """
        if self.ctx.memory is None or not self.ctx.memory.available:
            return ""
        return self.ctx.memory.context_for(query, k=k, max_chars=max_chars)
