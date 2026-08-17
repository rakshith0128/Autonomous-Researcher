"""Central configuration.

Everything tunable lives here so that no agent, tool, or prompt hardcodes a
value. Note in particular that there is nothing domain-specific anywhere in
this file -- the assessment forbids hardcoded domains, questions, or data
sources, and the seed queries the Domain Scout starts from are themselves
generated at runtime (see agents/scout.py).
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Role(str, Enum):
    """Logical model roles.

    Agents ask for a *role*, never a specific model. That indirection is what
    lets the router fail over between providers without any agent knowing.
    """

    REASONING = "reasoning"  # Critic, Experiment Designer, Question Generator
    FAST = "fast"  # extraction, classification, summarisation, status quips
    SYNTHESIS = "synthesis"  # final paper assembly (needs long context)


class ProviderSpec:
    """A single LLM provider.

    Every provider we use exposes an OpenAI-compatible chat-completions
    endpoint, so one SDK (`openai`) plus a different `base_url` covers all of
    them. This is why there is no per-provider adapter code in this repo.
    """

    def __init__(
        self,
        name: str,
        env_key: str,
        base_url: str,
        models: dict[Role, str],
        supports_json_schema: bool = False,
        thinking_token_overhead: int = 0,
    ) -> None:
        self.name = name
        self.env_key = env_key
        self.base_url = base_url
        self.models = models
        self.supports_json_schema = supports_json_schema
        #: Extra completion budget to add on top of what the caller asks for.
        #: Some models spend hidden reasoning tokens that count against
        #: `max_tokens` but are not reported in `completion_tokens`. Asking for
        #: 800 then gets `finish_reason=length` after 32 visible tokens, and the
        #: JSON arrives truncated -- which looks exactly like a model that
        #: cannot follow a schema. Padding the ceiling fixes it.
        self.thinking_token_overhead = thinking_token_overhead

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<ProviderSpec {self.name}>"


# Ordered by preference. The router walks this list top-down and uses the first
# provider that (a) has a key configured and (b) passed the startup preflight.
# Model ids are defaults only -- preflight verifies each one actually exists and
# silently drops the ones that do not, because provider catalogues change often
# and a 404 on model-not-found should never kill a run.
PROVIDER_REGISTRY: list[ProviderSpec] = [
    ProviderSpec(
        name="groq",
        env_key="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        models={
            # Three distinct models, one per role, chosen so that no two roles
            # share a rate-limit window. Quotas are per model, so spreading the
            # roles roughly triples the sustained throughput available to a run
            # -- which matters because the per-minute token ceiling, not the
            # daily request count, is what actually stalls a run here.
            #
            # gpt-oss-120b takes reasoning because Groq supports
            # `response_format: json_schema` on only part of its catalogue and
            # this model is in it. Nearly every call the system makes is
            # schema-constrained, so native enforcement is worth more than raw
            # parameter count.
            Role.REASONING: "openai/gpt-oss-120b",
            Role.FAST: "openai/gpt-oss-20b",
            # qwen3.6-27b for synthesis: a third distinct rate-limit window, and
            # mid-size prose beats the 20b's. If it declines response_format the
            # router drops the constraint and the Pydantic repair loop covers it,
            # which for the paper writer costs nothing.
            #
            # These replaced llama-3.1-8b-instant and llama-3.3-70b-versatile,
            # which Groq removed from its catalogue on 2026-08-17. Both then
            # 404'd at preflight, leaving gpt-oss-120b as the only Groq model
            # alive -- so the FAST and SYNTHESIS roles had no Groq option at all
            # and every call in them fell to Gemini, burning its daily quota.
            Role.SYNTHESIS: "qwen/qwen3.6-27b",
        },
        supports_json_schema=True,
    ),
    ProviderSpec(
        name="gemini",
        env_key="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        models={
            # Pinned versions rather than the `-latest` aliases: the run
            # manifest records model ids for reproducibility, and an alias that
            # silently repoints makes a recorded run unrepeatable.
            #
            # Daily request quotas differ sharply *within* this provider:
            # 3.5-flash allows 20/day, while the lite models allow 500. So the
            # scarce model takes the reasoning slot where its quality earns its
            # cost, and the plentiful lite models carry the high-volume roles.
            # When 3.5-flash is spent, the budget manager stops offering it and
            # the role degrades to FAST automatically.
            #
            # Note these are 3.x, not 2.5: the 2.5 models appear in this
            # endpoint's /models listing but return 404 on generation for this
            # tier, which is exactly why preflight generates a token instead of
            # trusting the catalogue.
            Role.REASONING: "gemini-3.5-flash",
            Role.FAST: "gemini-3.5-flash-lite",
            Role.SYNTHESIS: "gemini-3.1-flash-lite",
        },
        supports_json_schema=True,
        # Measured: gemini-3.5-flash returned finish_reason=length after 32
        # visible completion tokens against a 800-token ceiling. The hidden
        # reasoning tokens are billed against max_tokens but excluded from
        # completion_tokens, so the ceiling has to be padded.
        thinking_token_overhead=2500,
    ),
    ProviderSpec(
        name="cerebras",
        env_key="CEREBRAS_API_KEY",
        base_url="https://api.cerebras.ai/v1",
        models={
            Role.REASONING: "llama-3.3-70b",
            Role.FAST: "llama3.1-8b",
            Role.SYNTHESIS: "llama-3.3-70b",
        },
    ),
    ProviderSpec(
        name="openrouter",
        env_key="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        models={
            Role.REASONING: "meta-llama/llama-3.3-70b-instruct:free",
            Role.FAST: "meta-llama/llama-3.2-3b-instruct:free",
            Role.SYNTHESIS: "meta-llama/llama-3.3-70b-instruct:free",
        },
    ),
]


class Settings(BaseSettings):
    """Environment-backed settings. See .env.example for how to populate."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- credentials (all optional; the system degrades rather than dies) ---
    groq_api_key: str = ""
    gemini_api_key: str = ""
    cerebras_api_key: str = ""
    openrouter_api_key: str = ""
    tavily_api_key: str = ""
    github_token: str = ""
    contact_email: str = ""

    # --- agentic loop ---
    max_cycles: int = 5
    confidence_abstain_threshold: float = 0.60
    #: Samples drawn per claim for self-consistency scoring. Three is the
    #: smallest number that distinguishes unanimity (3/3) from a split (2/3),
    #: and each extra sample costs a request against a per-minute ceiling that
    #: the rest of the run also needs.
    self_consistency_samples: int = 3

    # --- operational limits ---
    max_concurrent_runs: int = 1
    run_timeout_seconds: int = 900
    per_ip_cooldown_seconds: int = 120
    max_runs_per_day: int = 50

    # --- optional components ---
    #: Both of these are genuinely optional and degrade gracefully, which makes
    #: them the right things to shed on constrained hardware. A free-tier
    #: instance is typically 512MB and a tenth of a CPU; embedding costs ~70ms
    #: per chunk and OCR ~10s per figure on a full core, so on a tenth of one
    #: they alone would exceed the run's entire time budget.
    #:
    #: Disabled, the agents fall back to abstracts rather than retrieved
    #: passages, and the Alchemist skips the image modality while still meeting
    #: its three-modality floor. The run is thinner, not broken -- and the paper
    #: says so.
    enable_vector_memory: bool = True
    enable_ocr: bool = True

    # --- tool behaviour ---
    http_timeout_seconds: float = 30.0
    max_fetch_bytes: int = 12_000_000
    circuit_breaker_threshold: int = 3
    sandbox_timeout_seconds: int = 45
    max_code_repair_attempts: int = 3

    # --- data acquisition floors (enforced, not suggested) ---
    min_distinct_modalities: int = 3
    min_sources_per_question: int = 5

    # --- storage ---
    data_dir: Path = Field(default=Path("./data"))
    log_level: str = "INFO"

    # --- split deployment ---
    #: Origins permitted to call this API from a browser.
    #:
    #: Empty means same-origin only, which is correct when one container serves
    #: both the API and the built frontend. Set this when the frontend is hosted
    #: separately -- a CDN-hosted page loads instantly even while the backend is
    #: still waking from sleep, which on a free tier that sleeps after 15
    #: minutes is the difference between a dead URL and a working one.
    #:
    #: Comma-separated, e.g. "https://my-app.vercel.app,http://localhost:5173".
    cors_origins: str = ""

    @property
    def allowed_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "runs.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    def key_for(self, provider: ProviderSpec) -> str:
        """Look up the configured API key for a provider, or '' if unset."""
        return getattr(self, provider.env_key.lower(), "")

    def configured_providers(self) -> list[ProviderSpec]:
        """Providers that actually have a key, in preference order."""
        return [p for p in PROVIDER_REGISTRY if self.key_for(p)]

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.chroma_path, self.artifacts_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
