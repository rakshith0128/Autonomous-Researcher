"""Run orchestration: build the context, execute the graph, own the lifecycle.

Everything above this is stateless machinery; this is where a *run* exists.
It owns the event bus, the shared HTTP client, the LLM router, and the graph
itself, and guarantees they are all torn down whatever happens.

Two guarantees matter to the demo:

* **A run always terminates.** A wall-clock timeout wraps the whole graph, on
  top of the cycle cap. A reviewer watching a live URL must never be left with
  a spinner that never resolves.
* **A run always produces something.** Even a failure ends with a state a
  caller can render and a reason a human can read.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from ..agents.alchemist import DataAlchemist
from ..agents.base import AgentContext
from ..agents.critic import Critic
from ..agents.designer import ExperimentDesigner
from ..agents.executor import Executor
from ..agents.panel import PeerReviewPanel
from ..agents.question import QuestionGenerator
from ..agents.scout import DomainScout
from ..agents.uncertainty import UncertaintyQuantifier
from ..agents.writer import PaperWriter
from ..config import Settings, get_settings
from ..graph import build as graph_build
from ..graph.state import ResearchState, initial_state, summarise
from ..llm import LLMRouter
from ..memory.vector import VectorMemory
from ..schemas import (
    AgentName,
    EventType,
    Level,
    RunEvent,
    RunManifest,
)
from ..tools.http import Fetcher
from .bus import EventBus

log = logging.getLogger(__name__)


class ResearchRunner:
    """One autonomous research run, start to finish."""

    def __init__(
        self,
        run_id: str | None = None,
        settings: Settings | None = None,
        on_persist: Any = None,
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.settings = settings or get_settings()
        self.bus = EventBus(self.run_id, on_persist=on_persist)
        self.state: ResearchState = initial_state(
            self.run_id, max_cycles=self.settings.max_cycles
        )
        self.manifest = RunManifest(run_id=self.run_id, seed=42)
        self.memory: VectorMemory | None = None
        self._started = 0.0

    async def run(self) -> ResearchState:
        """Execute the full graph. Never raises; failures land in state."""
        self._started = time.perf_counter()
        self.bus.publish(
            RunEvent(
                type=EventType.RUN_STARTED,
                agent=AgentName.SUPERVISOR,
                message="Starting autonomous research run.",
                payload={"max_cycles": self.settings.max_cycles},
            )
        )

        router = LLMRouter(settings=self.settings, on_event=self.bus.publish)
        fetcher = Fetcher(settings=self.settings)

        try:
            if not router.configured:
                raise RuntimeError(
                    "No LLM provider is configured. Set at least one API key in .env."
                )

            await router.preflight()

            # Loading the embedding model takes a few seconds, so it happens
            # off the event loop. Failure is non-fatal: agents fall back to
            # titles and abstracts, which is thinner evidence but still a run.
            memory = VectorMemory(self.run_id)
            if await asyncio.to_thread(memory.initialise):
                self.bus.message(
                    AgentName.SUPERVISOR, "Vector memory ready for document retrieval."
                )
            else:
                self.bus.message(
                    AgentName.SUPERVISOR,
                    "Vector memory unavailable; agents will work from abstracts only.",
                    level=Level.WARN,
                )
            self.memory = memory

            ctx = AgentContext(
                run_id=self.run_id,
                settings=self.settings,
                router=router,
                fetcher=fetcher,
                bus=self.bus,
                memory=memory,
            )
            graph = graph_build.build_graph(_nodes(ctx))

            # Recursion limit sized from the worst-case path, with slack.
            #
            # Discovery costs 4 super-steps (scout, panel, question, data). A
            # cycle rerouted all the way back to the question costs 6
            # (question, data, design, execute, uncertainty, critic), and the
            # write-up costs 1. LangGraph counts conditional edges as steps
            # too, so an exact figure runs out mid-cycle -- an earlier
            # `max_cycles * 6 + 12` aborted a run with GraphRecursionError
            # instead of writing the paper it already had material for.
            #
            # The cycle cap is the real limit; this is only a backstop against
            # a routing bug, so it is set generously.
            config = {"recursion_limit": self.settings.max_cycles * 8 + 20}

            final = await asyncio.wait_for(
                graph.ainvoke(self.state, config=config),
                timeout=self.settings.run_timeout_seconds,
            )
            self.state.update(final)
            self._finish(router, fetcher, ok=True)

        except TimeoutError:
            self.state["failure_reason"] = (
                f"Run exceeded its {self.settings.run_timeout_seconds}s time budget "
                "and was stopped."
            )
            self._finish(router, fetcher, ok=False)

        except Exception as exc:  # noqa: BLE001 - a run must never take the server with it
            log.exception("run %s failed", self.run_id)
            self.state["failure_reason"] = f"{type(exc).__name__}: {exc}"
            self._finish(router, fetcher, ok=False)

        finally:
            await fetcher.aclose()
            await router.aclose()
            if self.memory is not None:
                self.memory.close()
            self.bus.close()

        return self.state

    def _finish(self, router: LLMRouter, fetcher: Fetcher, *, ok: bool) -> None:
        """Record the manifest and emit the terminal event."""
        self.manifest.finished_at = datetime.now(UTC)
        self.manifest.token_usage = router.budget.usage()
        self.manifest.provider_failovers = router.failovers
        self.manifest.cycles_used = self.state.get("cycle", 0)
        self.manifest.tool_failures = {
            host: info["total_failures"] for host, info in fetcher.health().items()
        }
        self.manifest.models_used = {
            f"{u.provider}": u.model for u in router.budget.usage()
        }

        bundle = self.state.get("data_bundle")
        if bundle is not None:
            self.manifest.source_hashes = {
                d.url: d.provenance.sha256 for d in bundle.documents
            }

        self.state["manifest"] = self.manifest
        self.state["degraded_tools"] = fetcher.degraded_hosts()
        self.state["finished"] = True

        elapsed = time.perf_counter() - self._started
        paper = self.state.get("paper")
        succeeded = ok and paper is not None

        summary = summarise(self.state)
        summary |= {
            "elapsed_seconds": round(elapsed, 1),
            "tokens": self.manifest.total_tokens,
            "cost_usd": 0.0,
            "degraded_tools": self.state["degraded_tools"],
        }

        self.bus.publish(
            RunEvent(
                type=EventType.RUN_COMPLETED if succeeded else EventType.RUN_FAILED,
                agent=AgentName.SUPERVISOR,
                level=Level.SUCCESS if succeeded else Level.ERROR,
                message=(
                    f"Run complete in {elapsed:.0f}s: {paper.title}"
                    if succeeded
                    else f"Run ended without a paper: {self.state.get('failure_reason', 'unknown')}"
                ),
                payload=summary,
            )
        )


def _nodes(ctx: AgentContext) -> dict[str, Any]:
    """Map graph node names to agent instances."""
    return {
        graph_build.SCOUT: DomainScout(ctx),
        graph_build.PANEL: PeerReviewPanel(ctx),
        graph_build.QUESTION: QuestionGenerator(ctx),
        graph_build.DATA: DataAlchemist(ctx),
        graph_build.DESIGN: ExperimentDesigner(ctx),
        graph_build.EXECUTE: Executor(ctx),
        graph_build.UNCERTAINTY: UncertaintyQuantifier(ctx),
        graph_build.CRITIC: Critic(ctx),
        graph_build.WRITER: PaperWriter(ctx),
    }
