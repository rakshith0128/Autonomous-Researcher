"""Tests for graph routing and the iteration cap.

These run the real compiled graph with stub nodes — no network, no LLM. That
is the point: the cycle cap is a *control* guarantee, and it should be provable
without any of the machinery it governs.
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.graph.build import (
    CRITIC,
    DATA,
    DESIGN,
    EXECUTE,
    PANEL,
    QUESTION,
    SCOUT,
    UNCERTAINTY,
    WRITER,
    build_graph,
    route_after_critic,
    route_after_data,
    route_after_execute,
)
from backend.graph.state import initial_state
from backend.schemas import Critique, RerouteTarget, StatFlags, Verdict


def critique(
    *,
    verdict: Verdict = Verdict.REVISE,
    target: RerouteTarget = RerouteTarget.EXPERIMENT,
    cycle: int = 1,
    flags: StatFlags | None = None,
) -> Critique:
    return Critique(
        cycle=cycle,
        verdict=verdict,
        reroute_to=target,
        stat_flags=flags or StatFlags(),
    )


class TestCycleCap:
    def test_reroutes_while_under_the_cap(self):
        state = initial_state("r", max_cycles=5)
        state["cycle"] = 2
        state["critiques"] = [critique(target=RerouteTarget.DATA)]
        assert route_after_critic(state) == DATA

    def test_writes_up_once_the_cap_is_reached(self):
        """The honest ending: a paper with unresolved objections, not an
        endless loop and not a crash."""
        state = initial_state("r", max_cycles=3)
        state["cycle"] = 3
        state["critiques"] = [critique(target=RerouteTarget.DATA)]
        assert route_after_critic(state) == WRITER

    def test_cap_comes_from_state_not_a_default(self):
        """Regression guard.

        `max_cycles` must be a declared channel on ResearchState. LangGraph
        propagates only declared keys, so writing it onto the dict before
        invoke silently dropped it and routing fell back to a default of 5 --
        a run configured for 2 cycles ran to 3 and then died on the recursion
        limit.
        """
        state = initial_state("r", max_cycles=1)
        assert state["max_cycles"] == 1
        state["cycle"] = 1
        state["critiques"] = [critique()]
        assert route_after_critic(state) == WRITER

    def test_acceptance_ends_the_loop_early(self):
        state = initial_state("r", max_cycles=5)
        state["cycle"] = 1
        state["critiques"] = [critique(verdict=Verdict.ACCEPT, target=RerouteTarget.NONE)]
        assert route_after_critic(state) == WRITER

    def test_blocking_flags_override_acceptance(self):
        """p > alpha forces iteration even when the Critic says accept."""
        state = initial_state("r", max_cycles=5)
        state["cycle"] = 1
        state["critiques"] = [
            critique(
                verdict=Verdict.ACCEPT,
                target=RerouteTarget.NONE,
                flags=StatFlags(p_gt_alpha=True),
            )
        ]
        assert route_after_critic(state) != WRITER


class TestRerouteTargets:
    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            (RerouteTarget.QUESTION, QUESTION),
            (RerouteTarget.DATA, DATA),
            (RerouteTarget.EXPERIMENT, DESIGN),
        ],
    )
    def test_critic_target_selects_the_destination(self, target, expected):
        """Adaptive routing is what separates this from a fixed retry loop."""
        state = initial_state("r", max_cycles=5)
        state["cycle"] = 1
        state["critiques"] = [critique(target=target)]
        assert route_after_critic(state) == expected


class TestDataShortfall:
    def test_insufficient_data_reroutes_to_a_new_question(self):
        state = initial_state("r", max_cycles=5)
        state["reroute_to"] = RerouteTarget.QUESTION.value
        assert route_after_data(state) == QUESTION

    def test_sufficient_data_proceeds_to_design(self):
        assert route_after_data(initial_state("r", max_cycles=5)) == DESIGN

    def test_shortfall_at_the_cap_still_writes_up(self):
        state = initial_state("r", max_cycles=2)
        state["cycle"] = 2
        state["reroute_to"] = RerouteTarget.QUESTION.value
        assert route_after_data(state) == WRITER


class TestHaltPropagation:
    def test_a_halted_run_routes_straight_to_the_writer(self):
        """A fatal failure must still produce a document explaining itself."""
        state = initial_state("r", max_cycles=5)
        state["finished"] = True
        state["failure_reason"] = "scout could not reach any source"
        assert route_after_critic(state) == WRITER
        assert route_after_data(state) == WRITER
        assert route_after_execute(state) == WRITER


class TestCompiledGraph:
    async def test_full_loop_terminates_at_the_cap(self):
        """End-to-end with stub nodes: three forced reroutes, cap of 3, and
        the graph must reach the writer exactly once."""
        visits: list[str] = []
        cap = 3

        def node(name: str, update: dict[str, Any] | None = None):
            async def run(state: dict[str, Any]) -> dict[str, Any]:
                visits.append(name)
                return dict(update or {})

            return run

        async def critic_node(state: dict[str, Any]) -> dict[str, Any]:
            visits.append(CRITIC)
            cycle = state.get("cycle", 0) + 1
            return {
                "critiques": [critique(target=RerouteTarget.EXPERIMENT, cycle=cycle)],
                "cycle": cycle,
            }

        graph = build_graph(
            {
                SCOUT: node(SCOUT),
                PANEL: node(PANEL),
                QUESTION: node(QUESTION),
                DATA: node(DATA),
                DESIGN: node(DESIGN),
                EXECUTE: node(EXECUTE),
                UNCERTAINTY: node(UNCERTAINTY),
                CRITIC: critic_node,
                WRITER: node(WRITER, {"finished": True}),
            }
        )

        final = await graph.ainvoke(
            initial_state("r", max_cycles=cap), config={"recursion_limit": cap * 8 + 20}
        )

        assert visits.count(CRITIC) == cap, "critic should run exactly max_cycles times"
        assert visits.count(WRITER) == 1, "the writer must run exactly once"
        assert visits[-1] == WRITER
        assert final["cycle"] == cap
