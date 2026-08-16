"""Tests for the Emergence Index.

This is the scoring that decides which domain the entire run investigates, so
its failure modes are the expensive kind: a wrong number here is invisible and
sends every downstream agent to work on the wrong field.
"""

from __future__ import annotations

from backend.analysis.emergence import (
    WEIGHTS,
    SignalVector,
    _transform,
    _zscores,
    rank,
    score_candidates,
)
from backend.schemas import DomainCandidate, DomainProposal, EmergenceSignals

ALL_SIGNALS = list(WEIGHTS)


def vec(name: str, *, measured: bool = True, **values: float) -> SignalVector:
    return SignalVector(
        name=name,
        values={s: values.get(s, 0.0) for s in ALL_SIGNALS},
        measured={s: measured for s in ALL_SIGNALS},
    )


class TestWeights:
    def test_weights_sum_to_one(self):
        """Otherwise the index silently changes scale when a signal is added."""
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_publication_signals_dominate(self):
        """A GitHub trend with no literature is not an emerging science field."""
        publication = WEIGHTS["arxiv_growth_ratio"] + WEIGHTS["openalex_growth_ratio"]
        code = WEIGHTS["github_repos"] + WEIGHTS["github_star_velocity"]
        assert publication > code


class TestTransform:
    def test_heavy_tails_are_compressed(self):
        """Raw, 300 would swamp 25; logged, they stay on a comparable scale."""
        assert _transform("arxiv_growth_ratio", 300) / _transform(
            "arxiv_growth_ratio", 25
        ) < 2.0

    def test_negative_ratios_are_floored_not_reflected(self):
        assert _transform("arxiv_growth_ratio", -5) == 0.0

    def test_signed_signals_keep_their_direction(self):
        """A decelerating field must score below a flat one, not above it."""
        assert _transform("arxiv_relative_slope", -2.0) < 0
        assert _transform("arxiv_relative_slope", 2.0) > 0

    def test_transform_is_monotonic(self):
        values = [0, 1, 5, 25, 300]
        transformed = [_transform("arxiv_growth_ratio", v) for v in values]
        assert transformed == sorted(transformed)


class TestZScores:
    def test_standardises_to_zero_mean(self):
        z = _zscores([1.0, 2.0, 3.0])
        assert abs(sum(z)) < 1e-9
        assert z[0] < z[1] < z[2]

    def test_missing_values_impute_to_the_mean(self):
        """A rate-limited source must not be scored as a zero measurement."""
        z = _zscores([1.0, None, 3.0])
        assert z[1] == 0.0

    def test_identical_values_carry_no_information(self):
        assert _zscores([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]

    def test_single_candidate_cannot_be_standardised(self):
        assert _zscores([7.0]) == [0.0]

    def test_all_missing_is_all_zero(self):
        assert _zscores([None, None]) == [0.0, 0.0]


class TestScoring:
    def test_faster_growth_scores_higher(self):
        scores = score_candidates(
            [
                vec("fast", arxiv_growth_ratio=25.0, openalex_growth_ratio=20.0),
                vec("slow", arxiv_growth_ratio=1.2, openalex_growth_ratio=1.1),
            ]
        )
        assert scores["fast"]["index"] > scores["slow"]["index"]

    def test_component_z_scores_are_exposed(self):
        """The UI chart and the paper both show the breakdown, so a bare index
        is not enough."""
        scores = score_candidates([vec("a", arxiv_growth_ratio=10.0), vec("b")])
        assert "arxiv_growth_ratio_z" in scores["a"]
        assert scores["a"]["arxiv_growth_ratio_z"] > 0

    def test_an_outage_does_not_penalise_a_candidate(self):
        """Same measured evidence, but one had GitHub fail. It must not lose
        purely because of an outage on our side."""
        complete = vec("complete", arxiv_growth_ratio=10.0, github_repos=0.0)
        partial = SignalVector(
            name="partial",
            values={s: 0.0 for s in ALL_SIGNALS} | {"arxiv_growth_ratio": 10.0},
            measured={s: s not in ("github_repos", "github_star_velocity") for s in ALL_SIGNALS},
        )
        scores = score_candidates([complete, partial])
        assert scores["partial"]["index"] >= scores["complete"]["index"] - 1e-9

    def test_completeness_is_reported(self):
        partial = SignalVector(
            name="p",
            values={s: 1.0 for s in ALL_SIGNALS},
            measured={s: i % 2 == 0 for i, s in enumerate(ALL_SIGNALS)},
        )
        scores = score_candidates([partial, vec("other")])
        assert 0.0 < scores["p"]["completeness"] < 1.0

    def test_empty_input_is_safe(self):
        assert score_candidates([]) == {}


def make_candidate(name: str, *, arxiv_ratio: float, measured: bool = True) -> DomainCandidate:
    return DomainCandidate(
        proposal=DomainProposal(
            name=name, description="d", why_emerging="w", search_terms=[name]
        ),
        signals=EmergenceSignals(
            arxiv_growth_ratio=arxiv_ratio,
            openalex_growth_ratio=arxiv_ratio,
            measured_ok={"arxiv": measured, "openalex": measured, "github": measured, "forum": measured},
        ),
    )


class TestRanking:
    def test_ranks_best_first(self):
        ranked = rank(
            [
                make_candidate("slow", arxiv_ratio=1.1),
                make_candidate("fast", arxiv_ratio=30.0),
                make_candidate("mid", arxiv_ratio=5.0),
            ]
        )
        assert [c.name for c in ranked] == ["fast", "mid", "slow"]

    def test_unmeasurable_candidates_are_disqualified(self):
        """Scoring 0.0 on every component lands mid-table by construction, so
        a domain with no evidence at all could otherwise win a weak field."""
        ranked = rank(
            [
                make_candidate("measured", arxiv_ratio=2.0),
                make_candidate("ghost", arxiv_ratio=0.0, measured=False),
            ]
        )
        ghost = next(c for c in ranked if c.name == "ghost")
        assert ghost.disqualified
        assert "no growth signal" in ghost.disqualified_reason
        assert ranked[0].name == "measured"

    def test_scores_are_written_back_onto_candidates(self):
        candidates = [make_candidate("a", arxiv_ratio=10.0), make_candidate("b", arxiv_ratio=1.0)]
        rank(candidates)
        assert candidates[0].emergence_index != 0.0
        assert candidates[0].component_z, "component breakdown must be populated for the UI"
