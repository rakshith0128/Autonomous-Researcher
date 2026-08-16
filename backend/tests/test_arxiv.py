"""Tests for arXiv growth measurement.

These cover the pure numeric logic behind the Emergence Index. That logic is
worth pinning precisely because its failures are silent: a mis-trimmed
histogram still produces a plausible-looking number, and the Scout would then
rank a shrinking field above a growing one with total confidence.
"""

from __future__ import annotations

from datetime import UTC, datetime

from backend.tools.arxiv import (
    ArxivGrowth,
    _fully_observed_months,
    _linear_slope,
    _quote,
    _window,
)

THIS_MONTH = datetime.now(UTC).strftime("%Y-%m")


class TestQueryConstruction:
    def test_multiword_terms_are_quoted(self):
        """Unquoted, arXiv ORs the words together and matches nearly everything."""
        assert _quote("graph neural network") == '"graph neural network"'

    def test_single_words_are_not_quoted(self):
        assert _quote("transformers") == "transformers"

    def test_embedded_quotes_are_stripped_and_whitespace_collapsed(self):
        """A double space inside the quotes reads as a distinct phrase to
        arXiv and silently returns nothing."""
        assert _quote('say "hi" there') == '"say hi there"'
        assert _quote("  spaced   out  ") == '"spaced out"'

    def test_window_uses_arxiv_timestamp_format(self):
        w = _window(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 6, 30, tzinfo=UTC))
        assert w == "submittedDate:[202401010000 TO 202406302359]"


class TestPartialMonthTrimming:
    def test_current_month_is_always_dropped(self):
        """It is mid-month; today's count is not a month's count."""
        monthly = {"2026-05": 50, "2026-06": 60, THIS_MONTH: 7}
        assert THIS_MONTH not in _fully_observed_months(monthly, truncated=False)

    def test_oldest_month_dropped_only_when_the_sample_hit_the_cap(self):
        monthly = {"2026-03": 20, "2026-04": 40, "2026-05": 45}
        assert "2026-03" in _fully_observed_months(monthly, truncated=False)
        assert "2026-03" not in _fully_observed_months(monthly, truncated=True)

    def test_both_ends_trimmed_together(self):
        monthly = {"2026-03": 10, "2026-04": 40, "2026-05": 45, THIS_MONTH: 3}
        assert set(_fully_observed_months(monthly, truncated=True)) == {"2026-04", "2026-05"}

    def test_empty_input_is_safe(self):
        assert _fully_observed_months({}, truncated=True) == {}

    def test_single_partial_month_trims_to_nothing(self):
        assert _fully_observed_months({THIS_MONTH: 5}, truncated=False) == {}


class TestSlope:
    def test_rising_series_has_positive_slope(self):
        assert _linear_slope({"2026-01": 10, "2026-02": 20, "2026-03": 30}) == 10.0

    def test_falling_series_has_negative_slope(self):
        assert _linear_slope({"2026-01": 30, "2026-02": 20, "2026-03": 10}) == -10.0

    def test_flat_series_has_zero_slope(self):
        assert _linear_slope({"2026-01": 7, "2026-02": 7, "2026-03": 7}) == 0.0

    def test_months_are_ordered_before_fitting(self):
        """dict insertion order must not change the answer."""
        scrambled = {"2026-03": 30, "2026-01": 10, "2026-02": 20}
        assert _linear_slope(scrambled) == 10.0

    def test_too_few_points_is_zero_not_an_error(self):
        assert _linear_slope({}) == 0.0
        assert _linear_slope({"2026-01": 5}) == 0.0


class TestGrowthRatio:
    def test_ratio_against_a_real_baseline(self):
        g = ArxivGrowth(query="q", recent_count=200, baseline_count=50)
        assert g.growth_ratio == 4.0

    def test_zero_baseline_does_not_divide_by_zero(self):
        """A field with no prior history is the case we care most about, so it
        must score high and finite rather than raising."""
        g = ArxivGrowth(query="q", recent_count=300, baseline_count=0)
        assert g.growth_ratio == 300.0

    def test_no_activity_at_all_scores_zero(self):
        assert ArxivGrowth(query="q").growth_ratio == 0.0


class TestFailureIsHonest:
    def test_a_failed_measurement_is_marked_not_ok(self):
        """`ok=False` is what stops a dead source silently scoring as zero
        growth, which would look identical to a measured stagnant field."""
        g = ArxivGrowth(query="q", error="timeout")
        assert not g.ok
        assert g.growth_ratio == 0.0

    def test_slope_is_not_reported_when_unmeasurable(self):
        g = ArxivGrowth(query="q", ok=True, monthly_complete={"2026-01": 5, "2026-02": 9})
        assert not g.slope_measurable
        assert g.slope == 0.0
