"""Provider-agnostic LLM access with failover, budgeting, and degradation.

Agents ask for a **role** (`REASONING`, `FAST`, `SYNTHESIS`), never a model.
The router decides who serves it, and if that provider is rate-limited, out of
daily budget, returning 429s, or has quietly dropped the model from its
catalogue, the next one takes over without the agent knowing.

The degradation ladder, in order:

    1. preferred provider, preferred model for the role
    2. next configured provider, same role
    3. any provider, one role tier cheaper (REASONING -> FAST)
    4. wait out the shortest cooldown and retry once
    5. BudgetExhausted -- the run ends honestly rather than silently degrading

Everything here is deliberately boring. This layer is load-bearing for a demo
that must survive a live audience on free-tier infrastructure, so it favours
predictability over cleverness.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from openai import APIError, APIStatusError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel

from ..config import PROVIDER_REGISTRY, ProviderSpec, Role, Settings, get_settings
from ..schemas import AgentName, EventType, Level, RunEvent
from .budget import BudgetExhausted, BudgetManager, estimate_messages, estimate_tokens
from .structured import (
    StructuredOutputError,
    json_schema_response_format,
    parse_into,
    repair_messages,
    schema_instructions,
)

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

Messages = list[dict[str, str]]
EventSink = Callable[[RunEvent], None]

#: A 429 asking us to wait less than this is treated as backpressure to sit out
#: rather than a failure to route around. Per-minute token ceilings clear in
#: seconds; benching the model instead throws away a healthy provider.
SHORT_RETRY_CEILING_SECONDS = 8.0

# Cheaper tier to fall back to when a role's providers are all exhausted.
_DEGRADE_TO: dict[Role, Role | None] = {
    Role.REASONING: Role.FAST,
    Role.SYNTHESIS: Role.REASONING,
    Role.FAST: None,
}


@dataclass
class Completion:
    """One successful completion, plus who actually served it."""

    text: str
    provider: str
    model: str
    role: Role
    prompt_tokens: int = 0
    completion_tokens: int = 0
    degraded: bool = False
    attempts: int = 1


@dataclass
class _ProviderState:
    """Live health of one provider."""

    spec: ProviderSpec
    client: AsyncOpenAI
    available_models: set[str] = field(default_factory=set)
    unavailable_models: set[str] = field(default_factory=set)
    # Models that rejected `response_format: json_schema` at runtime. Learned
    # from the provider's own 400 rather than maintained as a static table,
    # because which models support structured output changes per release and a
    # stale table fails closed on a capability that actually exists.
    no_json_schema: set[str] = field(default_factory=set)
    preflight_ok: bool = False
    consecutive_failures: int = 0


class LLMRouter:
    """The only thing in this codebase that talks to an LLM."""

    def __init__(
        self,
        settings: Settings | None = None,
        budget: BudgetManager | None = None,
        on_event: EventSink | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.budget = budget or BudgetManager()
        self._on_event = on_event
        self._states: list[_ProviderState] = []
        self._failovers: list[str] = []
        self._build_clients()

    # ------------------------------------------------------------------ setup

    def _build_clients(self) -> None:
        for spec in PROVIDER_REGISTRY:
            key = self.settings.key_for(spec)
            if not key:
                continue
            client = AsyncOpenAI(
                api_key=key,
                base_url=spec.base_url,
                timeout=self.settings.http_timeout_seconds,
                max_retries=0,  # we own retry policy; the SDK retrying too would double-spend budget
            )
            self._states.append(_ProviderState(spec=spec, client=client))

    @property
    def configured(self) -> bool:
        return bool(self._states)

    @property
    def provider_names(self) -> list[str]:
        return [s.spec.name for s in self._states]

    @property
    def failovers(self) -> list[str]:
        return list(self._failovers)

    async def preflight(self) -> dict[str, bool]:
        """Verify each provider answers and each configured model exists.

        Provider catalogues change without notice; a model id that worked last
        week can 404 today. Discovering that during startup costs one second,
        discovering it in cycle 3 costs the run.
        """
        results: dict[str, bool] = {}
        await asyncio.gather(*(self._preflight_one(state) for state in self._states))
        for state in self._states:
            results[state.spec.name] = state.preflight_ok
        healthy = [n for n, ok in results.items() if ok]
        self._emit(
            EventType.AGENT_MESSAGE,
            Level.SUCCESS if healthy else Level.ERROR,
            f"LLM preflight: {len(healthy)}/{len(results)} providers healthy "
            f"({', '.join(healthy) or 'none'})",
            payload={"providers": results},
        )
        return results

    async def _preflight_one(self, state: _ProviderState) -> None:
        """Verify a provider by *generating*, not by reading its catalogue.

        Listing endpoints are advisory at best. Google's OpenAI-compatible
        endpoint advertises gemini-2.5-flash in /models and then returns 404
        for it on free-tier keys -- a catalogue-only check passes that model
        happily and the failure surfaces three nodes into a run instead.

        So each configured model gets a one-token completion. Three cheap calls
        at startup buy certainty about the thing that actually matters: whether
        this key can generate with this model right now.
        """
        try:
            listing = await state.client.models.list()
            state.available_models = {m.id for m in listing.data}
        except Exception as exc:  # noqa: BLE001 - not every endpoint implements /models
            log.info("preflight listing unavailable for %s: %s", state.spec.name, exc)
            state.available_models = set()

        distinct = list(dict.fromkeys(state.spec.models.values()))
        results = await asyncio.gather(
            *(self._probe_model(state, model) for model in distinct), return_exceptions=True
        )

        for model, usable in zip(distinct, results, strict=True):
            if usable is not True:
                state.unavailable_models.add(model)
                log.warning(
                    "%s cannot generate with %s: %s", state.spec.name, model, usable
                )

        # Healthy if at least one configured model actually answered.
        state.preflight_ok = len(state.unavailable_models) < len(distinct)

    async def _probe_model(self, state: _ProviderState, model: str) -> bool | str:
        """One-token generation probe. Returns True or a reason string."""
        try:
            await state.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ok"}],
                # Generous enough that a model spending tokens on internal
                # reasoning still returns rather than truncating into an error.
                max_tokens=16,
            )
            return True
        except RateLimitError:
            # Being throttled proves the model exists and the key is valid,
            # which is what the probe is asking. The budget manager handles
            # the rest.
            return True
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {str(exc)[:120]}"

    # -------------------------------------------------------------- emission

    def _emit(
        self,
        type_: EventType,
        level: Level,
        message: str,
        payload: dict | None = None,
    ) -> None:
        if self._on_event is None:
            return
        self._on_event(
            RunEvent(
                type=type_,
                agent=AgentName.SUPERVISOR,
                level=level,
                message=message,
                payload=payload or {},
            )
        )

    # ------------------------------------------------------------ core call

    def _candidates(self, role: Role) -> list[tuple[_ProviderState, str, Role]]:
        """Providers to try, in order, including the degraded tier.

        Scarcity is not filtered here -- the budget manager handles it per
        model. When a model with a tight daily quota is spent, `can_afford`
        stops admitting it and the next candidate in this list takes over,
        which is how a 20-request/day reasoning model degrades to a
        500-request/day lite model without any special-casing.
        """
        out: list[tuple[_ProviderState, str, Role]] = []

        for state in self._states:
            model = state.spec.models.get(role)
            if model and model not in state.unavailable_models:
                out.append((state, model, role))

        fallback_role = _DEGRADE_TO.get(role)
        if fallback_role is not None:
            for state in self._states:
                model = state.spec.models.get(fallback_role)
                if model and model not in state.unavailable_models:
                    out.append((state, model, fallback_role))

        # Last resort: any live model anywhere, including a provider's *other*
        # models. Without this, a role whose own model is spent has nowhere left
        # to go even when a sibling on the same key has quota to spare. Observed
        # live: gemini-3.5-flash-lite absorbed 497 of its 500 daily requests --
        # serving both the FAST role and every degraded REASONING call -- while
        # gemini-3.1-flash-lite sat at 79 and was never a candidate for either.
        for state in self._states:
            for model in dict.fromkeys(state.spec.models.values()):
                if model in state.unavailable_models:
                    continue
                if any(model == m and state is s for s, m, _ in out):
                    continue
                # Report the role the model is configured for, so a substitute
                # is correctly flagged as degraded and sorts behind the real one.
                served = next(
                    (r for r, m in state.spec.models.items() if m == model), role
                )
                out.append((state, model, served))
        return out

    def _ordered_candidates(
        self, role: Role, base_tokens: int
    ) -> list[tuple[_ProviderState, str, Role]]:
        """Affordable candidates, most free capacity first.

        Strict preference ordering hammers the top provider until it throttles
        while every other provider sits idle — observed live as a run stalling
        in 45-second backoffs against Groq's per-minute token ceiling while
        Gemini's lite models, with 500 requests a day each, went untouched.

        Ordering by remaining headroom spreads load instead, which roughly
        doubles sustained throughput with two providers configured. Preference
        order still breaks ties, so a single-provider deployment behaves
        exactly as before and the best model is used whenever it is free.
        """
        candidates = self._candidates(role)
        affordable = [
            (state, model, served)
            for state, model, served in candidates
            if not self.budget.is_blocked(state.spec.name, model)
            # Thinking headroom is a property of the provider being called, so
            # it is added per candidate. Budgeting every provider against the
            # *largest* overhead any configured provider needs charged Gemini's
            # 2,500-token reserve to Groq as well, pushing a routine 2,200-token
            # call past Groq's 5,200 per-minute ceiling. Groq was then dropped
            # from the writer, critic, question, scout and panel call sites on a
            # completely fresh budget, and the whole run landed on Gemini.
            and self.budget.can_afford(
                state.spec.name,
                model,
                base_tokens + state.spec.thinking_token_overhead,
            )
        ]

        preference = {id(state): i for i, (state, _, _) in enumerate(candidates)}
        return sorted(
            affordable,
            key=lambda item: (
                -round(self.budget.headroom(item[0].spec.name, item[1]), 2),
                # Prefer the role's own model over a degraded substitute.
                item[2] is not role,
                preference[id(item[0])],
            ),
        )

    async def complete(
        self,
        role: Role,
        messages: Messages,
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> Completion:
        """Get one completion, trying providers until someone answers."""
        if not self._states:
            raise BudgetExhausted(
                "No LLM provider is configured. Set at least one API key in .env "
                "(see .env.example)."
            )

        # Provider-agnostic size of this call. Each candidate adds its own
        # thinking headroom on top; see _ordered_candidates.
        base_tokens = estimate_messages(messages) + max_tokens
        attempts = 0
        last_error: Exception | None = None
        waited = False

        while True:
            for state, model, served_role in self._ordered_candidates(role, base_tokens):
                attempts += 1
                degraded = served_role is not role
                try:
                    return await self._call(
                        state,
                        model,
                        role,
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                        degraded=degraded,
                        attempts=attempts,
                    )
                except (TimeoutError, RateLimitError, APIStatusError, APIError) as exc:
                    last_error = exc

                    # Brief, self-resolving backpressure: the provider told us
                    # precisely how long to wait and it is short. Sleeping and
                    # retrying the same model is strictly better than benching
                    # it and walking the failover chain -- the model is not
                    # unhealthy, we simply arrived a fraction of a second early.
                    hint = retry_after_seconds(exc)
                    if hint is not None and hint <= SHORT_RETRY_CEILING_SECONDS:
                        log.info(
                            "%s/%s asked for %.2fs; waiting rather than failing over",
                            state.spec.name,
                            model,
                            hint,
                        )
                        await asyncio.sleep(hint + 0.25)
                        try:
                            return await self._call(
                                state,
                                model,
                                role,
                                messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                response_format=response_format,
                                degraded=degraded,
                                attempts=attempts,
                            )
                        except (TimeoutError, RateLimitError, APIStatusError, APIError) as retry_exc:
                            last_error = retry_exc
                            exc = retry_exc

                    self._handle_failure(state, model, exc)
                    continue

            # Everyone is blocked or broke. Wait out the shortest cooldown once.
            if not waited:
                waited = True
                cooldowns = [
                    self.budget.cooldown_remaining(state.spec.name, model)
                    for state, model, _ in self._candidates(role)
                ]
                delay = min((c for c in cooldowns if c > 0), default=20.0)
                delay = max(1.0, min(delay, 45.0))
                self._emit(
                    EventType.WARNING,
                    Level.WARN,
                    f"All providers throttled; waiting {delay:.0f}s before retrying.",
                    payload={"budget": self.budget.snapshot()},
                )
                await asyncio.sleep(delay)
                continue

            raise BudgetExhausted(
                "Every configured provider is rate-limited or out of free-tier budget. "
                f"Last error: {last_error}"
            )

    async def _call(
        self,
        state: _ProviderState,
        model: str,
        role: Role,
        messages: Messages,
        *,
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
        degraded: bool,
        attempts: int,
    ) -> Completion:
        started = time.perf_counter()
        # Pad the ceiling for providers whose models spend unreported reasoning
        # tokens against it; see ProviderSpec.thinking_token_overhead.
        effective_max_tokens = max_tokens + state.spec.thinking_token_overhead
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
        }
        use_schema = (
            response_format is not None
            and state.spec.supports_json_schema
            and model not in state.no_json_schema
        )
        if use_schema:
            kwargs["response_format"] = response_format

        try:
            response = await state.client.chat.completions.create(**kwargs)
        except APIStatusError as exc:
            # Self-repair: a provider that advertises structured output may
            # still refuse it on a given model. Rather than failing over to a
            # different provider -- which loses the preferred model for the
            # rest of the run -- drop the constraint, remember the limitation,
            # and retry here. The schema is still enforced downstream by the
            # Pydantic parse and repair loop, so correctness is unchanged.
            if use_schema and _is_response_format_rejection(exc):
                state.no_json_schema.add(model)
                log.info(
                    "%s/%s rejects response_format; falling back to prompt-enforced JSON",
                    state.spec.name,
                    model,
                )
                self._emit(
                    EventType.WARNING,
                    Level.WARN,
                    f"{state.spec.name}/{model} lacks native JSON schema support; "
                    "enforcing the schema in-prompt instead.",
                    payload={"provider": state.spec.name, "model": model},
                )
                kwargs.pop("response_format", None)
                response = await state.client.chat.completions.create(**kwargs)
            else:
                raise

        text = (response.choices[0].message.content or "").strip()

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or estimate_messages(messages)
        completion_tokens = getattr(usage, "completion_tokens", 0) or estimate_tokens(text)

        self.budget.record(state.spec.name, model, prompt_tokens, completion_tokens)
        state.consecutive_failures = 0

        log.debug(
            "%s/%s served %s in %.0fms (%d+%d tokens)",
            state.spec.name,
            model,
            role.value,
            (time.perf_counter() - started) * 1000,
            prompt_tokens,
            completion_tokens,
        )

        if degraded:
            self._emit(
                EventType.WARNING,
                Level.WARN,
                f"Degraded: {role.value} served by {state.spec.name}/{model}.",
                payload={"provider": state.spec.name, "model": model, "role": role.value},
            )

        return Completion(
            text=text,
            provider=state.spec.name,
            model=model,
            role=role,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            degraded=degraded,
            attempts=attempts,
        )

    def _handle_failure(self, state: _ProviderState, model: str, exc: Exception) -> None:
        name = state.spec.name
        state.consecutive_failures += 1
        status = getattr(exc, "status_code", None)

        if isinstance(exc, RateLimitError) or status == 429:
            # A 429 is this *model's* quota, not the provider's. Benching the
            # whole provider would discard its other models, which on a
            # provider whose quotas differ 25x between models is exactly the
            # wrong response -- the lite sibling is very likely still available.
            message = str(exc).lower()
            daily = "perday" in message.replace("_", "").replace(" ", "")

            if daily:
                # "Resets at midnight UTC" -- retrying every minute is pure
                # waste, so bench it properly.
                cooldown = 1800.0
                reason = "daily quota exhausted"
            else:
                # Prefer the provider's own figure over a guess. A per-minute
                # token ceiling clears in well under a minute, and a flat 60s
                # bench discards a model that is about to be usable again.
                hint = retry_after_seconds(exc)
                cooldown = min(hint + 0.5, 60.0) if hint is not None else 20.0
                reason = (
                    f"rate limited, retry in {hint:.2f}s"
                    if hint is not None
                    else "rate limited"
                )
            self.budget.penalise(name, model, seconds=cooldown)
        elif status == 404:
            # The model is gone. Remember that so we stop asking for it.
            state.unavailable_models.add(model)
            reason = f"model {model} not found"
        elif status in (401, 403):
            # Credentials are provider-wide, so bench the whole provider.
            self.budget.penalise(name, seconds=3600.0)
            reason = "authentication rejected"
        elif status == 400:
            # A 400 says our request was malformed, not that the provider is
            # unwell. Benching a perfectly healthy provider for a bad request
            # of our own making is how one bug cascades into losing the whole
            # failover chain, so this one is recorded without a penalty.
            reason = f"rejected request: {str(exc)[:120]}"
        else:
            self.budget.penalise(name, model, seconds=30.0)
            reason = f"{type(exc).__name__}"

        record = f"{name}/{model}: {reason}"
        self._failovers.append(record)
        log.warning("failover -- %s", record)
        self._emit(
            EventType.WARNING,
            Level.WARN,
            f"{name} unavailable ({reason}); failing over.",
            payload={"provider": name, "model": model, "reason": reason},
        )

    # -------------------------------------------------------- structured API

    async def structured(
        self,
        role: Role,
        messages: Messages,
        schema: type[T],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        max_repairs: int = 2,
    ) -> T:
        """Get a validated instance of `schema`, repairing malformed output.

        The schema is injected into the system turn *and* requested via
        `response_format` where supported -- belt and braces, because the
        providers that advertise schema support do not all honour it.
        """
        work = _with_schema_instructions(messages, schema)
        response_format = json_schema_response_format(schema)

        last_error = ""
        last_raw = ""
        for attempt in range(max_repairs + 1):
            completion = await self.complete(
                role,
                work,
                temperature=temperature if attempt == 0 else min(temperature + 0.1, 0.8),
                max_tokens=max_tokens,
                response_format=response_format,
            )
            last_raw = completion.text
            try:
                return parse_into(completion.text, schema)
            except StructuredOutputError as exc:
                last_error = str(exc)
                log.info(
                    "structured parse failed (attempt %d/%d) for %s: %s",
                    attempt + 1,
                    max_repairs + 1,
                    schema.__name__,
                    last_error,
                )
                if attempt < max_repairs:
                    self._emit(
                        EventType.WARNING,
                        Level.WARN,
                        f"Malformed {schema.__name__} from {completion.provider}; "
                        "showing the model its own error and retrying.",
                        payload={"schema": schema.__name__, "error": last_error[:400]},
                    )
                    work = repair_messages(work, completion.text, last_error, schema)

        raise StructuredOutputError(
            f"{schema.__name__} could not be produced after {max_repairs + 1} attempts: "
            f"{last_error}",
            raw=last_raw,
        )

    async def structured_samples(
        self,
        role: Role,
        messages: Messages,
        schema: type[T],
        *,
        k: int,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        concurrency: int = 2,
    ) -> list[T]:
        """Draw k independent structured samples for self-consistency scoring.

        Concurrency is capped deliberately. Budget is checked *before* a call
        and recorded *after* it, so firing k requests simultaneously means all
        k see the same pre-call state, every one passes the affordability
        check, and the burst blows a per-minute rate limit that the budget
        manager believed it was respecting. Observed live: five parallel
        samples per claim across three claims produced fifteen near-instant
        requests and a wall of 429s.

        A semaphore is the honest fix for the actual cause. Reserving budget
        speculatively at dispatch would be more precise but has to reconcile
        against real usage afterwards, and this is the only place in the system
        that fans out.

        Failures are dropped rather than retried: the Uncertainty Quantifier
        treats a smaller sample as weaker evidence, which is the right response
        to a model that could not answer consistently.
        """
        limiter = asyncio.Semaphore(max(1, concurrency))

        async def sample() -> T:
            async with limiter:
                return await self.structured(
                    role,
                    messages,
                    schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_repairs=1,
                )

        settled = await asyncio.gather(*(sample() for _ in range(k)), return_exceptions=True)
        return [r for r in settled if isinstance(r, schema)]

    async def structured_cross_model(
        self,
        role: Role,
        messages: Messages,
        schema: type[T],
        *,
        exclude_provider: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> tuple[T | None, str]:
        """Ask a *different* provider family the same question.

        Used by the Uncertainty Quantifier: two models from different families
        agreeing is meaningfully stronger evidence than one model agreeing with
        itself, which is all self-consistency can ever show. Returns
        (result, provider_name); result is None when no second family exists.
        """
        others = [s for s in self._states if s.spec.name != exclude_provider]
        if not others:
            return None, ""

        saved = self._states
        try:
            self._states = others
            result = await self.structured(
                role, messages, schema, temperature=temperature, max_tokens=max_tokens, max_repairs=1
            )
            return result, others[0].spec.name
        except Exception as exc:  # noqa: BLE001 - absence of a second opinion is not fatal
            log.info("cross-model check unavailable: %s", exc)
            return None, ""
        finally:
            self._states = saved

    async def aclose(self) -> None:
        await asyncio.gather(
            *(s.client.close() for s in self._states), return_exceptions=True
        )


_RETRY_HINT_RE = re.compile(r"try again in\s*([\d.]+)\s*(ms|s|m)\b", re.IGNORECASE)


def retry_after_seconds(exc: Exception) -> float | None:
    """How long the provider actually asked us to wait, if it said.

    Rate-limit responses usually carry the answer, either in a `retry-after`
    header or in the message itself -- Groq writes "Please try again in 340ms".
    Ignoring that and applying a flat cooldown is what turned a *third of a
    second* of backpressure into a benched model, an exhausted failover chain,
    and a run that ended with "every configured provider is rate-limited".

    Returns None when no hint is present, in which case the caller falls back
    to its own policy.
    """
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}) or {}
    raw = header.get("retry-after") or header.get("Retry-After")
    if raw:
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            pass

    match = _RETRY_HINT_RE.search(str(exc))
    if not match:
        return None
    value, unit = float(match.group(1)), match.group(2).lower()
    return value / 1000.0 if unit == "ms" else value * 60.0 if unit == "m" else value


def _is_response_format_rejection(exc: APIStatusError) -> bool:
    """Whether a 400 is about structured output rather than the request itself.

    Two distinct provider behaviours land here:

    * The model does not support ``response_format`` at all.
    * The provider enforced the schema server-side, the model's output missed
      it, and the whole call was rejected ("Failed to validate JSON").

    Both are better handled by dropping the constraint and letting our own
    Pydantic parse-and-repair loop take over -- it can unwrap envelopes, fix
    trailing commas, and re-prompt with the validation error, none of which a
    server-side validator will do. Without this, the strongest model on a
    provider gets abandoned over a schema keyword it nearly satisfied.

    Matched on message text because providers signal this with a plain 400 and
    no machine-readable code. Deliberately narrow: a broad match would swallow
    genuine request errors and retry them pointlessly.
    """
    if getattr(exc, "status_code", None) != 400:
        return False
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "response_format",
            "response format",
            "failed to validate json",
            "failed_generation",
            "json_schema",
        )
    )


def _with_schema_instructions(messages: Messages, schema: type[BaseModel]) -> Messages:
    """Append schema guidance to the system turn, creating one if absent."""
    instructions = schema_instructions(schema)
    out = [dict(m) for m in messages]
    for message in out:
        if message.get("role") == "system":
            message["content"] = f"{message['content']}\n\n{instructions}"
            return out
    return [{"role": "system", "content": instructions}, *out]
