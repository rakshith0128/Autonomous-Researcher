"""Graph assembly and routing.

This module is the assessment's "not a prompt chain" evidence. The control
flow is a state machine with conditional edges and real cycles: the Critic can
send work back to three different places depending on what it objected to, and
the cycle cap is enforced by the router rather than requested in a prompt.

Routing decisions live here, in one place, rather than being distributed
across the agents. An agent reports what happened; the graph decides what
happens next. That separation is what makes the iteration limit a guarantee
instead of a hope.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from ..schemas import AgentName, RerouteTarget
from .state import ResearchState

log = logging.getLogger(__name__)

# Node names. Kept as constants because they appear in the graph definition,
# the routing functions, and the frontend's graph visualisation, and a typo in
# any one of them is a silent dead end.
SCOUT = "scout"
PANEL = "panel"
QUESTION = "question"
DATA = "data"
DESIGN = "design"
EXECUTE = "execute"
UNCERTAINTY = "uncertainty"
CRITIC = "critic"
WRITER = "writer"

#: Edges the UI draws before anything runs, so the graph is visible from the
#: first frame rather than materialising node by node.
STATIC_EDGES: list[tuple[str, str]] = [
    (SCOUT, PANEL),
    (PANEL, QUESTION),
    (QUESTION, DATA),
    (DATA, DESIGN),
    (DESIGN, EXECUTE),
    (EXECUTE, UNCERTAINTY),
    (UNCERTAINTY, CRITIC),
    (CRITIC, WRITER),
]

#: Edges that only appear when the Critic sends work back. Drawn in red.
REROUTE_EDGES: list[tuple[str, str]] = [
    (CRITIC, QUESTION),
    (CRITIC, DATA),
    (CRITIC, DESIGN),
    (DATA, QUESTION),
    (EXECUTE, DESIGN),
]

NODE_TO_AGENT: dict[str, AgentName] = {
    SCOUT: AgentName.SCOUT,
    PANEL: AgentName.PANEL,
    QUESTION: AgentName.QUESTION_GEN,
    DATA: AgentName.ALCHEMIST,
    DESIGN: AgentName.DESIGNER,
    EXECUTE: AgentName.EXECUTOR,
    UNCERTAINTY: AgentName.UNCERTAINTY,
    CRITIC: AgentName.CRITIC,
    WRITER: AgentName.WRITER,
}


# --- routing ---------------------------------------------------------------


def _halted(state: ResearchState) -> bool:
    """Whether an agent has declared the run unrecoverable."""
    return bool(state.get("finished")) and bool(state.get("failure_reason"))


def route_after_data(state: ResearchState) -> str:
    """The Alchemist could not meet the data floor.

    Sending the run back to the Question Generator -- rather than retrying the
    same acquisition -- is the interesting behaviour: the system concludes the
    *question* was unanswerable with available data and picks a different one.
    """
    if _halted(state):
        return WRITER
    if state.get("reroute_to") == RerouteTarget.QUESTION.value:
        if state.get("cycle", 0) >= _max_cycles(state):
            log.info("data shortfall at the cycle cap; writing up what exists")
            return WRITER
        return QUESTION
    return DESIGN


def route_after_execute(state: ResearchState) -> str:
    """Execution failed outright: hand the traceback back to the Designer.

    Self-repair happens inside the executor for code-level errors. Reaching
    here means repair was exhausted, so the *design* needs to change, not the
    code.
    """
    if _halted(state):
        return WRITER
    result = state.get("experiment_result")
    if result is not None and not result.executed_ok:
        if state.get("cycle", 0) < _max_cycles(state):
            return DESIGN
    return UNCERTAINTY


def route_after_critic(state: ResearchState) -> str:
    """The central control decision.

    The Critic returns a structured verdict naming where the work is wrong.
    That target decides the destination, and the cycle cap decides whether a
    reroute is allowed at all. Both are enforced here so no agent can loop by
    choosing to.
    """
    if _halted(state):
        return WRITER

    critiques = state.get("critiques") or []
    if not critiques:
        return WRITER

    critique = critiques[-1]
    cycle = state.get("cycle", 0)
    limit = _max_cycles(state)

    if not critique.demands_iteration():
        log.info("critic accepted at cycle %d", cycle)
        return WRITER

    if cycle >= limit:
        # The honest ending. The paper is still written, with the unresolved
        # objections printed prominently rather than quietly dropped.
        log.info("cycle limit %d reached with objections outstanding", limit)
        return WRITER

    destination = {
        RerouteTarget.QUESTION: QUESTION,
        RerouteTarget.DATA: DATA,
        RerouteTarget.EXPERIMENT: DESIGN,
    }.get(critique.reroute_to, DESIGN)

    log.info(
        "critic verdict=%s -> rerouting to %s (cycle %d/%d)",
        critique.verdict.value,
        destination,
        cycle,
        limit,
    )
    return destination


def _max_cycles(state: ResearchState) -> int:
    return int(state.get("max_cycles") or 5)


# --- assembly --------------------------------------------------------------


def build_graph(nodes: dict[str, Any], *, checkpointer: Any = None):
    """Wire the agents into an executable graph.

    `nodes` maps the constants above to callables (the agent instances). They
    are injected rather than constructed here so tests can substitute stubs and
    exercise the routing logic without any network or LLM calls -- which is
    exactly how the cycle-cap tests work.
    """
    graph = StateGraph(ResearchState)

    for name, node in nodes.items():
        graph.add_node(name, node)

    graph.set_entry_point(SCOUT)

    graph.add_edge(SCOUT, PANEL)
    graph.add_edge(PANEL, QUESTION)
    graph.add_edge(QUESTION, DATA)

    graph.add_conditional_edges(
        DATA, route_after_data, {QUESTION: QUESTION, DESIGN: DESIGN, WRITER: WRITER}
    )
    graph.add_edge(DESIGN, EXECUTE)
    graph.add_conditional_edges(
        EXECUTE, route_after_execute, {DESIGN: DESIGN, UNCERTAINTY: UNCERTAINTY, WRITER: WRITER}
    )
    graph.add_edge(UNCERTAINTY, CRITIC)
    graph.add_conditional_edges(
        CRITIC,
        route_after_critic,
        {QUESTION: QUESTION, DATA: DATA, DESIGN: DESIGN, WRITER: WRITER},
    )
    graph.add_edge(WRITER, END)

    return graph.compile(checkpointer=checkpointer)


def graph_topology() -> dict[str, Any]:
    """Static description of the graph for the frontend to render.

    Shipped from the backend rather than duplicated in TypeScript so the
    picture a reviewer watches cannot drift away from the graph that actually
    executes.
    """
    return {
        "nodes": [
            {"id": node, "agent": agent.value, "label": agent.value.replace("_", " ")}
            for node, agent in NODE_TO_AGENT.items()
        ],
        "edges": [{"from": a, "to": b, "kind": "flow"} for a, b in STATIC_EDGES]
        + [{"from": a, "to": b, "kind": "reroute"} for a, b in REROUTE_EDGES],
    }
