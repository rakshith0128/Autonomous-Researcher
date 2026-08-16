"""Tests for rate-limit and token accounting.

Time is injected rather than slept, so the whole suite covering minute- and
day-length windows runs in milliseconds.

Budgets are tracked per **model**, not per provider. That granularity is
load-bearing: free-tier request quotas are issued per model and differ by more
than an order of magnitude within one provider, so a provider-level budget
either starves the plentiful models or overspends the scarce one.
"""

from __future__ import annotations

from backend.llm.budget import (
    MODEL_LIMITS,
    PROVIDER_LIMITS,
    BudgetManager,
    RateLimits,
    estimate_messages,
    estimate_tokens,
)

PROVIDER = {"test": RateLimits(rpm=3, tpm=1_000, rpd=10, tpd=5_000)}
PER_MODEL = {"scarce": RateLimits(rpm=3, tpm=1_000, rpd=2, tpd=5_000)}


def mgr() -> BudgetManager:
    return BudgetManager(provider_limits=PROVIDER, model_limits=PER_MODEL)


class TestLimitResolution:
    def test_model_limits_win_over_provider_limits(self):
        b = mgr()
        assert b.limits_for("test", "scarce").rpd == 2
        assert b.limits_for("test", "ordinary").rpd == 10

    def test_unknown_provider_falls_back_to_a_tight_default(self):
        b = mgr()
        assert b.limits_for("mystery", "").rpd > 0


class TestRequestWindows:
    def test_allows_up_to_the_rpm_ceiling(self):
        b = mgr()
        for i in range(3):
            assert b.can_afford("test", "m", 10, now=100.0), f"call {i} should fit"
            b.record("test", "m", 5, 5, now=100.0)
        assert not b.can_afford("test", "m", 10, now=100.0)

    def test_minute_window_slides(self):
        b = mgr()
        for _ in range(3):
            b.record("test", "m", 5, 5, now=100.0)
        assert not b.can_afford("test", "m", 10, now=150.0)
        assert b.can_afford("test", "m", 10, now=161.0)

    def test_daily_request_ceiling_is_enforced(self):
        b = mgr()
        for i in range(10):
            b.record("test", "m", 1, 1, now=100.0 + i * 120)
        assert not b.can_afford("test", "m", 1, now=100.0 + 10 * 120)

    def test_models_have_independent_budgets(self):
        """Exhausting a scarce model must not block its plentiful sibling --
        this is exactly the degradation path that keeps a run alive."""
        b = mgr()
        for i in range(2):
            b.record("test", "scarce", 1, 1, now=100.0 + i * 120)
        assert not b.can_afford("test", "scarce", 1, now=400.0)
        assert b.can_afford("test", "ordinary", 1, now=400.0)


class TestTokenWindows:
    def test_rejects_a_call_larger_than_remaining_tpm(self):
        b = mgr()
        b.record("test", "m", 400, 400, now=100.0)
        assert b.can_afford("test", "m", 100, now=100.0)
        assert not b.can_afford("test", "m", 300, now=100.0)

    def test_daily_token_ceiling_is_enforced(self):
        b = mgr()
        for i in range(5):
            b.record("test", "m", 500, 400, now=100.0 + i * 90)
        assert not b.can_afford("test", "m", 1_000, now=100.0 + 5 * 90)


class TestPenalties:
    def test_penalised_model_is_refused_until_cooldown_expires(self):
        b = mgr()
        b.penalise("test", "m", seconds=60.0, now=100.0)
        assert b.is_blocked("test", "m", now=120.0)
        assert not b.can_afford("test", "m", 1, now=120.0)
        assert not b.is_blocked("test", "m", now=161.0)

    def test_penalising_one_model_leaves_siblings_usable(self):
        """A 429 is a model's quota. Benching the provider would discard a
        sibling that is very likely still available."""
        b = mgr()
        b.penalise("test", "scarce", seconds=1800.0, now=100.0)
        assert b.is_blocked("test", "scarce", now=200.0)
        assert not b.is_blocked("test", "ordinary", now=200.0)
        assert b.can_afford("test", "ordinary", 10, now=200.0)

    def test_provider_wide_penalty_blocks_every_model(self):
        """Bad credentials are provider-wide, signalled by omitting the model."""
        b = mgr()
        b.penalise("test", seconds=3600.0, now=100.0)
        assert b.is_blocked("test", "scarce", now=200.0)
        assert b.is_blocked("test", "ordinary", now=200.0)

    def test_penalty_extends_but_never_shortens(self):
        b = mgr()
        b.penalise("test", "m", seconds=3600.0, now=100.0)
        b.penalise("test", "m", seconds=60.0, now=100.0)
        assert b.is_blocked("test", "m", now=1000.0)

    def test_cooldown_remaining_is_reported_for_the_wait(self):
        b = mgr()
        b.penalise("test", "m", seconds=60.0, now=100.0)
        assert b.cooldown_remaining("test", "m", now=130.0) == 30.0
        assert b.cooldown_remaining("test", "m", now=200.0) == 0.0


class TestUsageAccounting:
    def test_usage_accumulates_per_provider_and_model(self):
        b = mgr()
        b.record("test", "big", 100, 50, now=1.0)
        b.record("test", "big", 100, 50, now=2.0)
        b.record("test", "small", 10, 5, now=3.0)

        usage = {(u.provider, u.model): u for u in b.usage()}
        assert usage[("test", "big")].calls == 2
        assert usage[("test", "big")].total_tokens == 300
        assert usage[("test", "small")].total_tokens == 15
        assert b.total_tokens() == 315

    def test_snapshot_is_keyed_by_provider_and_model(self):
        """A provider-level view would hide a scarce model running out while
        the provider as a whole still looks healthy."""
        b = mgr()
        b.record("test", "m", 100, 100, now=10.0)
        snap = b.snapshot(now=10.0)["test/m"]
        assert snap["requests_this_minute"] == 1
        assert snap["tokens_this_minute"] == 200
        assert snap["rpd_limit"] == 10


class TestShippedLimits:
    def test_every_registry_provider_has_configured_limits(self):
        from backend.config import PROVIDER_REGISTRY

        for spec in PROVIDER_REGISTRY:
            assert spec.name in PROVIDER_LIMITS, f"no limits configured for {spec.name}"

    def test_scarce_gemini_model_is_configured_from_the_measured_quota(self):
        """The API reported:
        GenerateRequestsPerDayPerProjectPerModel-FreeTier ... quotaValue: 20
        """
        assert MODEL_LIMITS["gemini-3.5-flash"].rpd <= 20

    def test_lite_models_are_configured_as_plentiful(self):
        """500/day each, so they can carry the high-volume roles."""
        assert MODEL_LIMITS["gemini-3.5-flash-lite"].rpd >= 400
        assert MODEL_LIMITS["gemini-3.1-flash-lite"].rpd >= 400

    def test_limits_sit_below_round_published_numbers(self):
        """Headroom is the point: we throttle a beat before the provider does."""
        assert PROVIDER_LIMITS["groq"].rpm < 30
        assert MODEL_LIMITS["gemini-3.5-flash-lite"].rpd < 500


class TestEstimation:
    def test_token_estimate_is_never_zero(self):
        assert estimate_tokens("") >= 1
        assert estimate_tokens("hello world") >= 1

    def test_message_estimate_includes_per_message_overhead(self):
        msgs = [{"role": "user", "content": "hello"}, {"role": "system", "content": "hi"}]
        assert estimate_messages(msgs) > estimate_tokens("hellohi")
