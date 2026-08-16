"""Tests for vector memory: chunking, retrieval, and duplicate detection.

The embedding model is real here, so these are slower than the rest of the
suite. That is deliberate for the retrieval cases — the whole value of this
module rests on whether the embeddings actually separate the things we need
separated, and a mocked embedder would prove nothing about that.
"""

from __future__ import annotations

import pytest

from backend.memory.vector import (
    REJECTION_AMBIGUOUS,
    REJECTION_CERTAIN,
    VectorMemory,
    chunk_text,
)
from backend.schemas import Modality, Provenance, SourceDocument


class TestChunking:
    def test_short_text_is_one_chunk(self):
        assert chunk_text("a short paragraph") == ["a short paragraph"]

    def test_empty_text_yields_nothing(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\n  ") == []

    def test_long_text_is_split(self):
        chunks = chunk_text("word " * 1200)
        assert len(chunks) > 1

    def test_chunks_respect_the_size_budget(self):
        chunks = chunk_text("sentence about quantum circuits. " * 200, size=600)
        # Boundary-seeking can overshoot slightly; a hard cap would split words.
        assert all(len(c) <= 800 for c in chunks)

    def test_chunks_overlap_so_findings_are_not_severed(self):
        """A method and the number it produced routinely straddle a boundary;
        without overlap the retrieved chunk describes an approach with no
        result attached."""
        text = "".join(f"Sentence number {i} about the measured effect. " for i in range(200))
        chunks = chunk_text(text, size=600, overlap=150)
        assert len(chunks) >= 2
        tail = chunks[0][-80:]
        assert any(fragment and fragment in chunks[1] for fragment in [tail[-40:]])

    def test_slivers_are_discarded(self):
        for chunk in chunk_text("x " * 900, size=500, overlap=100):
            assert len(chunk) > 60


def _docs() -> list[SourceDocument]:
    return [
        SourceDocument(
            provenance=Provenance(
                url="http://arxiv.org/abs/1", modality=Modality.PDF, title="Quantum SWITCH"
            ),
            text=(
                "The quantum SWITCH enables indefinite causal order. We measured a "
                "12 percent improvement in channel capacity using a superposition of "
                "orders. " * 10
            ),
        ),
        SourceDocument(
            provenance=Provenance(
                url="http://arxiv.org/abs/2", modality=Modality.PDF, title="Citation study"
            ),
            text=(
                "Papers releasing open source code receive on average 1.8x more "
                "citations within two years of publication. We analysed 4000 "
                "preprints. " * 10
            ),
        ),
    ]


@pytest.fixture(scope="module")
def memory() -> VectorMemory:
    store = VectorMemory("test-run")
    if not store.initialise():
        pytest.skip("vector memory unavailable in this environment")
    store.index_documents(_docs())
    return store


@pytest.mark.slow
class TestRetrieval:
    def test_indexing_reports_chunks(self, memory: VectorMemory):
        assert memory.chunks_indexed > 0

    def test_retrieves_the_topically_correct_document(self, memory: VectorMemory):
        hits = memory.search("does releasing code increase citations?", k=2)
        assert hits and hits[0].title == "Citation study"

    def test_retrieves_a_different_document_for_a_different_query(self, memory: VectorMemory):
        hits = memory.search("quantum switch channel capacity gain", k=2)
        assert hits and hits[0].title == "Quantum SWITCH"

    def test_hits_carry_provenance_for_citation(self, memory: VectorMemory):
        """A retrieved passage must be traceable, or quoting it produces a
        citation the verification pass will flag as fabricated."""
        hit = memory.search("open source code citations", k=1)[0]
        assert hit.source_url.startswith("http")
        assert 0.0 < hit.similarity <= 1.0

    def test_context_is_formatted_with_sources(self, memory: VectorMemory):
        context = memory.context_for("citation impact of open source", k=2, max_chars=1200)
        assert "[source:" in context and "http" in context

    def test_context_respects_the_character_budget(self, memory: VectorMemory):
        assert len(memory.context_for("citations", k=5, max_chars=500)) <= 700

    def test_empty_query_returns_nothing(self, memory: VectorMemory):
        assert memory.search("   ") == []


@pytest.mark.slow
class TestDuplicateDetection:
    """The calibration that shaped this design.

    Measured against "Does the presence of a GitHub repository correlate with
    citation counts?" using bge-small-en-v1.5:

        near-identical rewording ......... 0.984
        paraphrase, same meaning ......... 0.828
        reworded, same meaning ........... 0.810
        DIFFERENT question (author count)  0.812
        unrelated ........................ 0.512

    A genuinely different question outranks a true duplicate, so no threshold
    separates them and similarity alone cannot decide. Hence recall from the
    vector store, precision from a model.
    """

    REFERENCE = "Does the presence of a GitHub repository correlate with citation counts?"

    @pytest.fixture(scope="class")
    def rejections(self) -> VectorMemory:
        store = VectorMemory("dup-test")
        if not store.initialise():
            pytest.skip("vector memory unavailable in this environment")
        store.remember_rejection(self.REFERENCE, "trivially searchable")
        return store

    def test_near_identical_is_certain_without_adjudication(self, rejections: VectorMemory):
        hit, ambiguous = rejections.similar_rejection(
            "Does having a GitHub repository correlate with citation count?"
        )
        assert hit is not None and not ambiguous

    def test_paraphrase_is_flagged_for_adjudication(self, rejections: VectorMemory):
        hit, ambiguous = rejections.similar_rejection(
            "Is there a link between having a public code repo and how often a paper is cited?"
        )
        assert hit is not None and ambiguous

    def test_different_question_also_lands_in_the_ambiguous_band(
        self, rejections: VectorMemory
    ):
        """The finding that forced the design.

        This question is genuinely new, yet scores as high as a true duplicate.
        Blocking on similarity alone would silently discard it — which is why
        the ambiguous band is adjudicated rather than rejected.
        """
        hit, ambiguous = rejections.similar_rejection(
            "Does the number of authors correlate with citation counts?"
        )
        assert hit is not None and ambiguous, (
            "a different question scoring in the duplicate range is exactly why "
            "similarity alone must not be the decider"
        )

    def test_unrelated_question_passes_untouched(self, rejections: VectorMemory):
        hit, ambiguous = rejections.similar_rejection(
            "Do quantum error correction codes reduce gate depth?"
        )
        assert hit is None and not ambiguous


class TestThresholds:
    def test_certain_sits_above_ambiguous(self):
        assert REJECTION_CERTAIN > REJECTION_AMBIGUOUS

    def test_certain_is_high_enough_to_exclude_measured_false_positives(self):
        """A different question measured 0.812; the certain threshold must sit
        well clear of that or it blocks valid work without adjudication."""
        assert REJECTION_CERTAIN > 0.90


class TestGracefulDegradation:
    def test_uninitialised_memory_is_inert_rather_than_fatal(self):
        """Retrieval is an enhancement. Losing it should thin the evidence the
        agents see, never end a run."""
        store = VectorMemory("never-initialised")
        assert not store.available
        assert store.index_documents(_docs()) == 0
        assert store.search("anything") == []
        assert store.context_for("anything") == ""
        assert store.similar_rejection("anything") == (None, False)
        assert store.stats() == {"available": False}
        store.remember_rejection("q", "r")  # must not raise
