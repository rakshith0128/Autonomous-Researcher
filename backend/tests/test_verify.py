"""Tests for the anti-fabrication checks.

The failure this guards against is the expensive one for a research system:
a paper that reads perfectly and cites a paper that does not exist. Every case
below is a form of invention observed from real models.
"""

from __future__ import annotations

from backend.analysis.verify import (
    VerificationReport,
    build_allowed_index,
    format_reference_list,
    verify_citations,
    verify_numbers,
)
from backend.schemas import Modality, Provenance, StatResult

SOURCES = [
    Provenance(
        url="https://arxiv.org/abs/2401.12345",
        modality=Modality.PDF,
        title="A Real Paper That Was Fetched",
    ),
    Provenance(
        url="https://api.openalex.org/works?search=x",
        modality=Modality.STRUCTURED_API,
        title="OpenAlex records",
    ),
    Provenance(
        url="https://example.com/article",
        modality=Modality.HTML,
        title="An article",
    ),
]


class TestAllowedIndex:
    def test_indexes_by_url_and_arxiv_id(self):
        index = build_allowed_index(SOURCES)
        assert "arxiv.org/abs/2401.12345" in index
        assert "2401.12345" in index

    def test_pdf_and_abs_urls_are_the_same_paper(self):
        """arXiv serves one paper at several paths; treating them as different
        would flag a correct citation as invented."""
        index = build_allowed_index(SOURCES)
        text = "See https://arxiv.org/pdf/2401.12345 for details."
        _, report = verify_citations(text, SOURCES)
        assert report.citations_verified == 1
        assert index  # sanity

    def test_version_suffix_is_ignored(self):
        text = "See https://arxiv.org/abs/2401.12345v3"
        _, report = verify_citations(text, SOURCES)
        assert report.citations_verified == 1


class TestCitationVerification:
    def test_real_citation_passes(self):
        text = "As shown in https://arxiv.org/abs/2401.12345, the effect holds."
        cleaned, report = verify_citations(text, SOURCES)
        assert report.citations_verified == 1
        assert report.clean
        assert "arxiv.org/abs/2401.12345" in cleaned

    def test_invented_url_is_caught_and_stripped(self):
        text = "Prior work (https://arxiv.org/abs/2408.99999) established this."
        cleaned, report = verify_citations(text, SOURCES)
        assert not report.clean
        assert report.findings[0].kind == "fabricated_citation"
        assert "2408.99999" not in cleaned
        assert "citation removed" in cleaned

    def test_invented_bare_arxiv_id_is_caught(self):
        """Models cite 'arXiv:2405.01234' with no URL at all."""
        text = "This follows arXiv:2405.01234 closely."
        _, report = verify_citations(text, SOURCES)
        assert any(f.kind == "fabricated_citation" for f in report.findings)

    def test_mixed_real_and_invented(self):
        text = (
            "Real: https://arxiv.org/abs/2401.12345 . "
            "Invented: https://arxiv.org/abs/2499.00001 ."
        )
        cleaned, report = verify_citations(text, SOURCES)
        assert report.citations_found == 2
        assert report.citations_verified == 1
        assert "2401.12345" in cleaned and "2499.00001" not in cleaned

    def test_integrity_ratio_is_reported(self):
        text = "https://arxiv.org/abs/2401.12345 and https://fake.test/nope"
        _, report = verify_citations(text, SOURCES)
        assert report.citation_integrity == 0.5

    def test_paper_with_no_citations_is_clean(self):
        _, report = verify_citations("No citations at all here.", SOURCES)
        assert report.clean
        assert report.citation_integrity == 1.0


STATS = [
    StatResult(test_name="pearson", p_value=0.0312, effect_size=0.41, effect_size_name="r", n=120)
]


class TestNumberVerification:
    def test_correctly_reported_statistics_pass(self):
        report = verify_numbers("We found r = 0.41 (p = 0.0312, n = 120).", STATS, VerificationReport())
        assert report.numbers_found == 3
        assert report.numbers_verified == 3
        assert report.clean

    def test_rounding_within_tolerance_is_accepted(self):
        report = verify_numbers("r = 0.41, n = 120", STATS, VerificationReport())
        assert report.clean

    def test_invented_p_value_is_caught(self):
        """The classic: correct table, wrong number restated in the prose."""
        report = verify_numbers("The result was highly significant (p = 0.001).", STATS, VerificationReport())
        assert not report.clean
        assert report.findings[0].kind == "unsupported_number"

    def test_invented_sample_size_is_caught(self):
        report = verify_numbers("Across n = 5000 papers we observed...", STATS, VerificationReport())
        assert any("n = 5000" in f.detail for f in report.findings)

    def test_prose_numbers_are_not_treated_as_claims(self):
        """'roughly a third' is not a precision claim and must not be flagged."""
        report = verify_numbers(
            "Roughly 40 percent of the 2024 cohort showed the pattern.",
            STATS,
            VerificationReport(),
        )
        assert report.numbers_found == 0

    def test_tiny_p_values_compare_on_magnitude(self):
        stats = [StatResult(test_name="t", p_value=0.0000031, n=50)]
        report = verify_numbers("p = 0.000003", stats, VerificationReport())
        assert report.numbers_verified == 1

    def test_no_computed_stats_means_every_number_is_unsupported(self):
        report = verify_numbers("p = 0.04", [], VerificationReport())
        assert report.numbers_found == 1
        assert report.numbers_verified == 0


class TestReporting:
    def test_markdown_shows_failures_rather_than_hiding_them(self):
        text = "https://fake.test/invented"
        _, report = verify_citations(text, SOURCES)
        rendered = report.as_markdown()
        assert "Verification failures" in rendered
        assert "fake.test" in rendered

    def test_markdown_states_success_explicitly_when_clean(self):
        _, report = verify_citations("nothing to cite", SOURCES)
        assert "No fabricated citations" in report.as_markdown()

    def test_reference_list_is_numbered_for_closed_citation(self):
        """The writer cites [n] against this list; anything else is invention."""
        rendered = format_reference_list(SOURCES)
        assert rendered.startswith("[1] A Real Paper That Was Fetched")
        assert "[3]" in rendered
