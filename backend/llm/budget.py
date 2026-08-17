"""Rate-limit and token accounting.

Free tiers do not fail politely. They return a 429 in the middle of cycle 4,
twenty minutes into a run a reviewer is watching. So the system tracks its own
consumption and refuses to make a call it knows will be rejected, rather than
discovering the limit by hitting it.

Configured ceilings sit deliberately *below* each provider's published limits.
That headroom means we throttle ourselves a beat before the provider throttles
us, which turns a hard failure into a quiet failover.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from ..schemas import TokenUsage


@dataclass(frozen=True)
class RateLimits:
    """Ceilings we hold ourselves to, per provider."""

    rpm: int  # requests per minute
    tpm: int  # tokens per minute
    rpd: int  # requests per day
    tpd: int  # tokens per day


# Set at roughly 90% of each provider's published free-tier allowance. If a
# provider changes its limits the only cost of being wrong here is an earlier
# failover, never a crashed run.
#: Per-provider defaults, used when a model has no specific entry below.
PROVIDER_LIMITS: dict[str, RateLimits] = {
    # Groq is the workhorse: thousands of requests per day, and the per-minute
    # token ceiling is the only thing that ever binds in practice.
    "groq": RateLimits(rpm=27, tpm=14_000, rpd=900, tpd=90_000),
    "gemini": RateLimits(rpm=13, tpm=200_000, rpd=450, tpd=2_000_000),
    "cerebras": RateLimits(rpm=27, tpm=55_000, rpd=850, tpd=900_000),
    "openrouter": RateLimits(rpm=18, tpm=100_000, rpd=45, tpd=1_000_000),
}

#: Per-**model** overrides.
#:
#: This granularity is not optional. Free-tier request quotas are issued per
#: model, and they differ by more than an order of magnitude within a single
#: provider: Gemini's 3.5-flash allows 20 requests/day while its lite siblings
#: allow 500. Budgeting at provider level therefore either starves the plentiful
#: models or overspends the scarce one -- and overspending is what killed a run,
#: with the API returning:
#:
#:     GenerateRequestsPerDayPerProjectPerModel-FreeTier ... quotaValue: 20
#:
#: With correct per-model ceilings the router simply degrades: once the scarce
#: reasoning model is spent, the role falls to a lite model automatically and
#: the run continues.
MODEL_LIMITS: dict[str, RateLimits] = {
    # --- Gemini: a 25x spread between siblings on the same key ---
    "gemini-3.5-flash": RateLimits(rpm=8, tpm=200_000, rpd=18, tpd=2_000_000),
    "gemini-3.5-flash-lite": RateLimits(rpm=13, tpm=200_000, rpd=450, tpd=2_000_000),
    "gemini-3.1-flash-lite": RateLimits(rpm=13, tpm=200_000, rpd=450, tpd=2_000_000),
    # --- Groq: the per-minute TOKEN ceiling is what actually binds ---
    #
    # Read from the provider's own response headers rather than guessed:
    #
    #   x-ratelimit-limit-tokens:   8000   (per minute)
    #   x-ratelimit-limit-requests: 1000   (per day)
    #
    # An earlier value of 14,000 tpm was more than double the truth, so the
    # budget manager believed a call fitted, admitted it, and collected a 429 --
    # the exact failure this module exists to prevent. 7,000 keeps the same
    # headroom discipline against the real 8,000: one large Critic prompt is
    # ~2,000 tokens, so three fit inside a minute and the fourth waits rather
    # than being rejected.
    "openai/gpt-oss-120b": RateLimits(rpm=27, tpm=7_000, rpd=900, tpd=180_000),
    "openai/gpt-oss-20b": RateLimits(rpm=27, tpm=7_000, rpd=900, tpd=180_000),
    "qwen/qwen3.6-27b": RateLimits(rpm=27, tpm=7_000, rpd=900, tpd=180_000),
}

_FALLBACK_LIMITS = RateLimits(rpm=10, tpm=10_000, rpd=200, tpd=100_000)

_MINUTE = 60.0
_DAY = 86_400.0


@dataclass
class _Window:
    """Sliding window of (timestamp, amount) pairs.

    Reads are non-destructive. That is load-bearing rather than stylistic: the
    same window is queried over both a one-minute and a one-day span, so a read
    that evicted anything older than a minute would quietly erase the daily
    total and let the day's quota overrun without ever tripping. Eviction
    happens on write only, against the longest horizon we ever ask about.
    """

    events: deque[tuple[float, int]] = field(default_factory=deque)

    def add(self, amount: int, now: float, retain: float = _DAY) -> None:
        self.events.append((now, amount))
        cutoff = now - retain
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def total(self, span: float, now: float) -> int:
        cutoff = now - span
        return sum(amount for ts, amount in self.events if ts >= cutoff)

    def count(self, span: float, now: float) -> int:
        cutoff = now - span
        return sum(1 for ts, _ in self.events if ts >= cutoff)


class BudgetExhausted(RuntimeError):
    """Raised when every configured provider is out of budget."""


def _key(provider: str, model: str) -> str:
    return f"{provider}/{model}"


class BudgetManager:
    """Per-model consumption tracking, shared across one process.

    Not thread-safe by design -- the graph runs in a single asyncio task per
    run, and `max_concurrent_runs` defaults to 1, so a lock would only add
    contention for a case that cannot currently arise.
    """

    def __init__(
        self,
        provider_limits: dict[str, RateLimits] | None = None,
        model_limits: dict[str, RateLimits] | None = None,
    ) -> None:
        self._provider_limits = provider_limits or PROVIDER_LIMITS
        self._model_limits = model_limits if model_limits is not None else MODEL_LIMITS
        self._requests: dict[str, _Window] = defaultdict(_Window)
        self._tokens: dict[str, _Window] = defaultdict(_Window)
        self._usage: dict[tuple[str, str], TokenUsage] = {}
        self._blocked_until: dict[str, float] = {}

    def limits_for(self, provider: str, model: str = "") -> RateLimits:
        """Most specific limits available: model, then provider, then default."""
        if model and model in self._model_limits:
            return self._model_limits[model]
        return self._provider_limits.get(provider, _FALLBACK_LIMITS)

    def can_afford(
        self,
        provider: str,
        model: str,
        estimated_tokens: int,
        now: float | None = None,
    ) -> bool:
        """Whether a call of roughly this size fits inside every window."""
        now = now if now is not None else time.monotonic()

        # A provider-wide block (bad credentials) outranks any model budget.
        if self._blocked_until.get(provider, 0.0) > now:
            return False
        key = _key(provider, model)
        if self._blocked_until.get(key, 0.0) > now:
            return False

        limits = self.limits_for(provider, model)
        reqs, toks = self._requests[key], self._tokens[key]

        return (
            reqs.count(_MINUTE, now) + 1 <= limits.rpm
            and reqs.count(_DAY, now) + 1 <= limits.rpd
            and toks.total(_MINUTE, now) + estimated_tokens <= limits.tpm
            and toks.total(_DAY, now) + estimated_tokens <= limits.tpd
        )

    def record(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        now: float | None = None,
    ) -> None:
        now = now if now is not None else time.monotonic()
        key = _key(provider, model)

        self._requests[key].add(1, now)
        self._tokens[key].add(prompt_tokens + completion_tokens, now)

        usage = self._usage.get((provider, model))
        if usage is None:
            usage = TokenUsage(provider=provider, model=model)
            self._usage[(provider, model)] = usage
        usage.prompt_tokens += prompt_tokens
        usage.completion_tokens += completion_tokens
        usage.calls += 1

    def penalise(
        self,
        provider: str,
        model: str = "",
        seconds: float = 60.0,
        now: float | None = None,
    ) -> None:
        """Bench a model (or a whole provider) after a failure.

        A 429 is a *model's* quota, so benching the provider would needlessly
        discard its other models. An auth failure is provider-wide, and callers
        signal that by omitting `model`.
        """
        now = now if now is not None else time.monotonic()
        target = _key(provider, model) if model else provider
        self._blocked_until[target] = max(self._blocked_until.get(target, 0.0), now + seconds)

    def is_blocked(self, provider: str, model: str = "", now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        if self._blocked_until.get(provider, 0.0) > now:
            return True
        return bool(model) and self._blocked_until.get(_key(provider, model), 0.0) > now

    def cooldown_remaining(
        self, provider: str, model: str = "", now: float | None = None
    ) -> float:
        now = now if now is not None else time.monotonic()
        remaining = max(0.0, self._blocked_until.get(provider, 0.0) - now)
        if model:
            remaining = max(
                remaining, max(0.0, self._blocked_until.get(_key(provider, model), 0.0) - now)
            )
        return remaining

    def headroom(self, provider: str, model: str, now: float | None = None) -> float:
        """Fraction of this model's per-minute capacity still free, 0.0-1.0.

        Used to spread load across providers rather than hammering the
        preferred one until it throttles. Both the request and token windows
        are considered, because either can be the binding constraint: a
        provider with plenty of requests left but no tokens is just as
        unavailable as one with neither.
        """
        now = now if now is not None else time.monotonic()
        limits = self.limits_for(provider, model)
        key = _key(provider, model)

        request_free = 1.0 - min(self._requests[key].count(_MINUTE, now) / max(limits.rpm, 1), 1.0)
        token_free = 1.0 - min(self._tokens[key].total(_MINUTE, now) / max(limits.tpm, 1), 1.0)
        daily_free = 1.0 - min(self._requests[key].count(_DAY, now) / max(limits.rpd, 1), 1.0)

        return min(request_free, token_free, daily_free)

    def usage(self) -> list[TokenUsage]:
        return list(self._usage.values())

    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self._usage.values())

    def snapshot(self, now: float | None = None) -> dict[str, dict[str, int | float]]:
        """Live vitals for the UI panel, keyed by provider/model.

        Reported per model because that is the level quotas are enforced at --
        a provider-level view would hide the scarce model running out while the
        provider as a whole looks healthy.
        """
        now = now if now is not None else time.monotonic()
        out: dict[str, dict[str, int | float]] = {}
        for key in self._requests:
            provider, _, model = key.partition("/")
            limits = self.limits_for(provider, model)
            out[key] = {
                "requests_this_minute": self._requests[key].count(_MINUTE, now),
                "rpm_limit": limits.rpm,
                "requests_today": self._requests[key].count(_DAY, now),
                "rpd_limit": limits.rpd,
                "tokens_this_minute": self._tokens[key].total(_MINUTE, now),
                "tpm_limit": limits.tpm,
                "tokens_today": self._tokens[key].total(_DAY, now),
                "cooldown_seconds": round(self.cooldown_remaining(provider, model, now), 1),
            }
        return out


def estimate_tokens(text: str) -> int:
    """Rough token count without paying for a tokeniser.

    ~4 characters per token is close enough for budgeting: the cost of being
    10% wrong is a slightly early failover, and importing a real tokeniser for
    every provider would add more weight to the container than the accuracy is
    worth.
    """
    return max(1, len(text) // 4)


def estimate_messages(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(m.get("content", "")) for m in messages) + 8 * len(messages)
