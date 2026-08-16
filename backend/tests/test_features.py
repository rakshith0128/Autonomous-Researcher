"""Tests for question-specific feature derivation.

The failure these guard against was the most consequential in the project's
history and produced no error at all. Across three consecutive runs the
assembled table held only structural metadata -- author counts, title lengths,
category counts -- while the questions asked about properties of the *research*:
whether a paper used reinforcement learning, whether it released code. Nothing
connected the two, so the Designer correctly refused, the run looped to its
cycle limit, and the write-up fell back to whatever trivia had run. One paper
was titled "The Relationship Between Author Count and Title Length" beneath a
research question about citation counts.

The abstracts were in memory the entire time.
"""

from __future__ import annotations

import pytest

from backend.agents.alchemist import (
    _CODE_HOST_RE,
    _min_group_size,
    _safe_column,
    _shadows,
)


class TestColumnNameSafety:
    """Names reach pandas, the experiment registry, and the paper's tables."""

    def test_spaces_become_underscores(self):
        assert _safe_column("uses reinforcement learning") == "uses_reinforcement_learning"

    def test_case_is_normalised(self):
        assert _safe_column("UsesRL") == "usesrl"

    def test_punctuation_is_stripped(self):
        assert _safe_column("open-source? (code)") == "open_source_code"

    def test_leading_and_trailing_separators_removed(self):
        assert _safe_column("  __has code__  ") == "has_code"

    def test_name_starting_with_a_digit_is_rejected(self):
        """A leading digit is not a valid identifier and breaks downstream."""
        assert _safe_column("3d_reconstruction") == ""

    def test_empty_input_is_rejected(self):
        assert _safe_column("") == ""
        assert _safe_column("!!!") == ""

    def test_absurdly_long_names_are_truncated(self):
        assert len(_safe_column("a" * 200)) <= 40


class TestCodeReleaseDetection:
    """The single most commonly asked-about property, so it is always derived."""

    @pytest.mark.parametrize(
        "abstract",
        [
            "Our implementation is available at https://github.com/foo/bar",
            "Code and data are hosted on GitLab.",
            "We release our models on huggingface.co/org/model",
            "The dataset is archived on zenodo.org",
            "Our code is publicly available upon publication.",
            "We open-source the full training pipeline.",
            "Code is available in the supplementary material.",
            "We release all checkpoints.",
        ],
    )
    def test_detects_a_code_release(self, abstract: str):
        assert _CODE_HOST_RE.search(abstract)

    @pytest.mark.parametrize(
        "abstract",
        [
            "We propose a novel attention mechanism for quantum circuits.",
            "This paper presents a theoretical bound on channel capacity.",
            "We evaluate three heuristic partitioning strategies.",
        ],
    )
    def test_does_not_fire_on_papers_without_one(self, abstract: str):
        assert not _CODE_HOST_RE.search(abstract)

    def test_matching_is_case_insensitive(self):
        assert _CODE_HOST_RE.search("SEE GITHUB.COM/X FOR CODE")


def apply_keywords(abstracts: list[str], keywords: list[str]) -> list[int]:
    """The classification step, exactly as the Alchemist performs it.

    Deliberately deterministic: the model proposes vocabulary, Python decides
    membership. That keeps the resulting column a function of the text rather
    than a model's per-paper opinion, and keeps it reproducible from the
    manifest.
    """
    lowered = [k.strip().lower() for k in keywords if k.strip()]
    return [1 if any(k in text.lower() for k in lowered) else 0 for text in abstracts]


class TestKeywordClassification:
    ABSTRACTS = [
        "We apply reinforcement learning to circuit allocation, training a policy gradient agent.",
        "A greedy heuristic partitions the circuit across nodes.",
        "We use Q-learning to schedule gate operations.",
        "This work derives an analytical bound on entanglement cost.",
    ]

    def test_splits_papers_by_method(self):
        flags = apply_keywords(
            self.ABSTRACTS, ["reinforcement learning", "policy gradient", "q-learning"]
        )
        assert flags == [1, 0, 1, 0]

    def test_variants_and_abbreviations_are_caught(self):
        """Authors write the same idea several ways; that is why the model is
        asked for vocabulary rather than a single term."""
        assert apply_keywords(["We train an RL agent."], ["rl agent", "reinforcement learning"]) == [1]

    def test_matching_ignores_case(self):
        assert apply_keywords(["REINFORCEMENT LEARNING is used."], ["reinforcement learning"]) == [1]

    def test_a_feature_matching_nothing_is_all_zero(self):
        """All-zero and all-one columns carry no variance; the Alchemist
        discards them rather than handing the Designer a dead column."""
        assert apply_keywords(self.ABSTRACTS, ["quantum annealing"]) == [0, 0, 0, 0]

    def test_a_feature_matching_everything_is_all_one(self):
        assert sum(apply_keywords(self.ABSTRACTS, ["e"])) == len(self.ABSTRACTS)

    def test_blank_keywords_are_ignored(self):
        assert apply_keywords(["reinforcement learning"], ["  ", "reinforcement learning"]) == [1]

    def test_missing_abstract_yields_zero_not_a_crash(self):
        """Not every row has an abstract; a missing one must classify as
        absent rather than break the whole dataset build."""
        assert apply_keywords(["", "reinforcement learning"], ["reinforcement learning"]) == [0, 1]


class TestMinorityGroupFloor:
    """Variance alone is not enough — both groups must be big enough to compare.

    A live run derived a column splitting 3 papers against 57. It passed a bare
    variance check, produced a group comparison on n=3, and returned a
    confident-looking p-value from a test with essentially no power.
    """

    @pytest.mark.parametrize(
        ("positives", "total", "usable"),
        [
            (3, 60, False),   # observed: mentions_sensor_modality
            (56, 60, False),  # observed: is_robust — 4 in the minority
            (11, 60, True),   # observed: releases_code
            (0, 60, False),
            (60, 60, False),
            (7, 14, True),
            (2, 14, False),
        ],
    )
    def test_floor_scales_with_dataset_size(self, positives: int, total: int, usable: bool):
        minority = min(positives, total - positives)
        assert (minority >= _min_group_size(total)) is usable

    def test_small_datasets_still_require_an_absolute_minimum(self):
        """12% of 14 rows is under 2, which would let a 2-vs-12 split through."""
        assert _min_group_size(14) >= 5


class TestShadowedMetadata:
    """A derived feature must not restate metadata already recorded as fact.

    Observed live: `revised_on_arxiv` was derived by scanning abstracts for the
    word "revised", while a real `revised` column taken from arXiv's own version
    history sat beside it. The derived one measures whether an author happened
    to use a word, and looks equally authoritative.
    """

    @pytest.mark.parametrize(
        ("derived", "existing"),
        [
            ("revised_on_arxiv", "revised"),
            ("has_many_authors", "n_authors"),
            ("mentions_year", "year"),
        ],
    )
    def test_restatements_are_detected(self, derived: str, existing: str):
        assert _shadows(derived, existing)

    @pytest.mark.parametrize(
        ("derived", "existing"),
        [
            ("releases_code", "revised"),
            ("uses_reinforcement_learning", "n_authors"),
            ("mentions_sensor_modality", "title_length"),
            ("is_robust", "revised"),
        ],
    )
    def test_genuinely_new_features_pass(self, derived: str, existing: str):
        assert not _shadows(derived, existing)

    def test_empty_names_never_shadow(self):
        assert not _shadows("", "revised")
        assert not _shadows("revised", "")
