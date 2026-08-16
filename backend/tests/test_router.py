"""Tests for provider selection, reserving, and quota calibration.

The behaviour pinned here was learned from live free-tier failures, not from
documentation. Each test corresponds to a way a run died.
"""

from __future__ import annotations

import pytest

from backend.config import PROVIDER_REGISTRY, Role, Settings
from backend.llm.budget import MODEL_LIMITS, PROVIDER_LIMITS, BudgetManager
from backend.llm.router import (
    SHORT_RETRY_CEILING_SECONDS,
    LLMRouter,
    retry_after_seconds,
)


def settings_with(**keys: str) -> Settings:
    return Settings(_env_file=None, **keys)


class TestQuotaCalibration:
    def test_scarce_gemini_model_reflects_the_measured_quota(self):
        """Gemini's free tier returns:

            GenerateRequestsPerDayPerProjectPerModel-FreeTier ... quotaValue: 20

        An earlier provider-level configuration of 1350/day was wrong by two
        orders of magnitude and let one run exhaust the quota before reaching
        the Question Generator.
        """
        assert MODEL_LIMITS["gemini-3.5-flash"].rpd <= 20

    def test_quotas_differ_sharply_within_one_provider(self):
        """The reason budgets are tracked per model rather than per provider:
        a 25x spread between siblings makes any provider-level number wrong."""
        scarce = MODEL_LIMITS["gemini-3.5-flash"].rpd
        plentiful = MODEL_LIMITS["gemini-3.5-flash-lite"].rpd
        assert plentiful > scarce * 20

    def test_every_registry_provider_has_limits(self):
        for spec in PROVIDER_REGISTRY:
            assert spec.name in PROVIDER_LIMITS


class TestRetryHints:
    """Providers state how long to wait. Ignoring it cost a live run.

    The observed 429 read:

        Rate limit reached for model `llama-3.1-8b-instant` ... on tokens per
        minute (TPM): Limit 6000, Used 4076, Requested 1958. Please try again
        in 340ms.

    A 340-millisecond wait was treated as a failed provider, the failover chain
    was walked and exhausted, and the run ended with "every configured provider
    is rate-limited" — over a third of a second of backpressure.
    """

    def test_parses_the_millisecond_hint_groq_actually_sends(self):
        exc = Exception(
            "Error code: 429 - Rate limit reached for model `llama-3.1-8b-instant` "
            "on tokens per minute (TPM): Limit 6000, Used 4076, Requested 1958. "
            "Please try again in 340ms."
        )
        assert retry_after_seconds(exc) == pytest.approx(0.34)

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("please try again in 2s", 2.0),
            ("Please try again in 1.5s", 1.5),
            ("try again in 500ms", 0.5),
            ("try again in 2m", 120.0),
        ],
    )
    def test_parses_each_unit(self, message: str, expected: float):
        assert retry_after_seconds(Exception(message)) == pytest.approx(expected)

    def test_prefers_the_retry_after_header(self):
        class _Response:
            headers = {"retry-after": "7"}

        exc = Exception("try again in 999s")
        exc.response = _Response()
        assert retry_after_seconds(exc) == 7.0

    def test_returns_none_when_no_hint_is_given(self):
        assert retry_after_seconds(Exception("Internal server error")) is None

    def test_short_hints_stay_under_the_wait_ceiling(self):
        """Below the ceiling the router waits; above it, it fails over."""
        assert 0.34 < SHORT_RETRY_CEILING_SECONDS
        assert 120.0 > SHORT_RETRY_CEILING_SECONDS


class TestMeasuredTokenCeilings:
    def test_groq_tpm_matches_the_limit_the_provider_reports(self):
        """The 429 stated "Limit 6000". A configured 14,000 meant the budget
        manager approved calls the provider then rejected — the precise failure
        this module exists to prevent."""
        for model in (
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
        ):
            assert MODEL_LIMITS[model].tpm < 6000, f"{model} exceeds the measured ceiling"

    def test_ceiling_still_admits_a_realistic_prompt(self):
        """Too conservative is its own failure: a Critic prompt runs to roughly
        2,000 tokens and must fit."""
        assert MODEL_LIMITS["llama-3.1-8b-instant"].tpm >= 4_000


class TestGracefulDegradation:
    def test_scarce_model_falls_back_to_a_sibling_when_exhausted(self):
        """The behaviour that keeps a run alive: once the 20/day reasoning
        model is spent, the role degrades to a 500/day lite model rather than
        failing the run."""
        budget = BudgetManager()
        reasoning = "gemini-3.5-flash"
        fast = "gemini-3.5-flash-lite"

        for i in range(MODEL_LIMITS[reasoning].rpd):
            budget.record("gemini", reasoning, 10, 10, now=100.0 + i * 120)

        assert not budget.can_afford("gemini", reasoning, 100, now=9_000.0)
        assert budget.can_afford("gemini", fast, 100, now=9_000.0)

    def test_reasoning_role_lists_a_cheaper_fallback(self):
        """The degrade ladder must actually offer somewhere to fall to."""
        router = LLMRouter(settings=settings_with(groq_api_key="k1", gemini_api_key="k2"))
        models = [model for _, model, _ in router._candidates(Role.REASONING)]
        assert len(set(models)) > 1

    def test_both_providers_serve_ordinary_traffic(self):
        """With per-model budgeting there is no blanket reserve; every
        configured provider is eligible and scarcity is enforced numerically."""
        router = LLMRouter(settings=settings_with(groq_api_key="k1", gemini_api_key="k2"))
        served_by = {state.spec.name for state, _, _ in router._candidates(Role.REASONING)}
        assert served_by == {"groq", "gemini"}

    def test_single_provider_still_works(self):
        router = LLMRouter(settings=settings_with(gemini_api_key="k2"))
        served_by = {state.spec.name for state, _, _ in router._candidates(Role.REASONING)}
        assert served_by == {"gemini"}


class TestProviderConfiguration:
    def test_no_providers_without_keys(self):
        router = LLMRouter(settings=settings_with())
        assert not router.configured
        assert router.provider_names == []

    def test_providers_appear_in_registry_preference_order(self):
        router = LLMRouter(
            settings=settings_with(gemini_api_key="k2", groq_api_key="k1")
        )
        # Groq precedes Gemini in PROVIDER_REGISTRY regardless of env order.
        assert router.provider_names == ["groq", "gemini"]

    def test_thinking_overhead_is_declared_where_it_was_measured(self):
        """gemini-3.5-flash returned finish_reason=length after 32 visible
        completion tokens against an 800-token ceiling."""
        gemini = next(p for p in PROVIDER_REGISTRY if p.name == "gemini")
        assert gemini.thinking_token_overhead > 0

    def test_groq_reasoning_model_supports_structured_output(self):
        """llama-3.3-70b does not accept response_format on Groq; the reasoning
        slot must be a model that does, since nearly every call is schema-bound."""
        groq = next(p for p in PROVIDER_REGISTRY if p.name == "groq")
        assert groq.models[Role.REASONING] != "llama-3.3-70b-versatile"
