"""The state that flows through the agent graph.

One typed object carries the whole run. Every node reads what it needs and
writes only its own slice, which is what makes the graph auditable: at any
point you can serialise this and see exactly what the system believes and how
it got there.

Two design choices are worth stating outright.

**The cycle counter and reroute target live here, not in an agent.** No single
agent decides how many times the system iterates -- the graph's routing
function reads this state and enforces the cap. An agent cannot loop forever
by choosing to, which is the difference between a controlled iteration limit
and a hopeful instruction in a prompt.

**Rejected work is retained, not overwritten.** `rejected_questions` and the
full `critiques` history stay in state for the whole run so later cycles can
see what has already failed. A system that forgets its own rejected approaches
will propose them again, and does, reliably.
"""

from __future__ import annotations

import operator
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict

from ..schemas import (
    Claim,
    Conflict,
    Critique,
    DataBundle,
    DomainSelection,
    ExperimentResult,
    ExperimentSpec,
    Paper,
    QuestionSet,
    ResearchQuestion,
    RunManifest,
)


class Phase:
    """Named checkpoints, used for UI progress and for resume logic."""

    DISCOVERY = "discovery"
    SELECTION = "selection"
    QUESTION = "question"
    DATA = "data"
    EXPERIMENT = "experiment"
    UNCERTAINTY = "uncertainty"
    CRITIQUE = "critique"
    WRITING = "writing"
    DONE = "done"
    FAILED = "failed"


class ResearchState(TypedDict, total=False):
    """Everything the run knows.

    `total=False` because LangGraph nodes return partial updates -- a node
    returns only the keys it changed and the framework merges them.
    """

    # --- identity ---
    run_id: str
    started_at: datetime
    seed: int
    #: Iteration cap for this run.
    #:
    #: Must be declared here, not merely written onto the dict before invoke.
    #: LangGraph propagates only *declared* channels, so an undeclared key is
    #: silently dropped between nodes -- which made the routing function fall
    #: back to its default of 5 and run past a configured limit of 2.
    max_cycles: int

    # --- phase 1: domain discovery ---
    domain_selection: DomainSelection | None
    domain_name: str

    # --- phase 2: question formulation ---
    question_set: QuestionSet | None
    question: ResearchQuestion | None
    # Accumulated across cycles so the generator can be told what has already
    # been tried and rejected, rather than rediscovering the same dead ends.
    rejected_questions: Annotated[list[str], operator.add]

    # --- phase 3: data ---
    data_bundle: DataBundle | None
    conflicts: Annotated[list[Conflict], operator.add]

    # --- phase 4: experimentation ---
    experiment_spec: ExperimentSpec | None
    experiment_result: ExperimentResult | None
    # Every attempt, not just the surviving one: the paper's methods section
    # should be able to say how many designs were tried before one held up.
    experiment_history: Annotated[list[dict[str, Any]], operator.add]

    # --- phase 5: uncertainty ---
    claims: list[Claim]
    abstained_claims: list[Claim]
    overall_confidence: float

    # --- phase 6: critique and control ---
    critiques: Annotated[list[Critique], operator.add]
    cycle: int
    reroute_to: str
    reroute_reason: str

    # --- output ---
    paper: Paper | None
    manifest: RunManifest | None

    # --- control and diagnostics ---
    phase: str
    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    degraded_tools: list[str]
    finished: bool
    failure_reason: str


def initial_state(run_id: str, seed: int = 0, max_cycles: int = 5) -> ResearchState:
    """A fresh run.

    Every collection is initialised explicitly. LangGraph's reducers append to
    existing values, and an absent key on the first write behaves differently
    from an empty one -- initialising here removes a whole class of
    first-cycle-only bugs.
    """
    return ResearchState(
        run_id=run_id,
        started_at=datetime.now(UTC),
        seed=seed,
        max_cycles=max_cycles,
        domain_selection=None,
        domain_name="",
        question_set=None,
        question=None,
        rejected_questions=[],
        data_bundle=None,
        conflicts=[],
        experiment_spec=None,
        experiment_result=None,
        experiment_history=[],
        claims=[],
        abstained_claims=[],
        overall_confidence=0.0,
        critiques=[],
        cycle=0,
        reroute_to="",
        reroute_reason="",
        paper=None,
        manifest=None,
        phase=Phase.DISCOVERY,
        errors=[],
        warnings=[],
        degraded_tools=[],
        finished=False,
        failure_reason="",
    )


def summarise(state: ResearchState) -> dict[str, Any]:
    """Compact view for logs, the vitals panel, and the run gallery."""
    question = state.get("question")
    result = state.get("experiment_result")
    critiques = state.get("critiques") or []

    return {
        "run_id": state.get("run_id", ""),
        "phase": state.get("phase", ""),
        "cycle": state.get("cycle", 0),
        "domain": state.get("domain_name", ""),
        "question": question.text if question else "",
        "sources": len(state["data_bundle"].documents) if state.get("data_bundle") else 0,
        "modalities": (
            sorted(m.value for m in state["data_bundle"].modalities)
            if state.get("data_bundle")
            else []
        ),
        "experiment_ok": bool(result and result.executed_ok),
        "p_value": (
            result.primary.p_value if result and result.primary else None
        ),
        "critiques": len(critiques),
        "last_verdict": critiques[-1].verdict.value if critiques else "",
        "claims": len(state.get("claims") or []),
        "abstained": len(state.get("abstained_claims") or []),
        "confidence": round(state.get("overall_confidence", 0.0), 3),
        "errors": len(state.get("errors") or []),
    }
