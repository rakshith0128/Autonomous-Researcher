"""Types shared across every agent contract.

Design note that runs through this whole package: models are split into
**LLM-authored** and **machine-computed**. Anything an LLM writes is a
proposal; anything that carries a number used in a decision is measured in
Python. `DomainProposal` vs `EmergenceSignals` is the clearest example. This
split is the main defence against the assessment's warning that "the AI lies
to you constantly".
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Modality(str, Enum):
    """Kind of a data source.

    The Data Alchemist must assemble at least `min_distinct_modalities`
    *different* values from this enum before it is allowed to proceed. Three
    PDFs do not count as three sources.
    """

    PDF = "pdf"
    TABULAR = "tabular"  # CSV / TSV / XLSX
    STRUCTURED_API = "structured_api"  # JSON from a REST API
    HTML = "html"  # scraped article text
    IMAGE = "image"  # figure extracted from a PDF, read via OCR


class AgentName(str, Enum):
    """Every node in the graph. Used for event routing and UI colour-coding."""

    SUPERVISOR = "supervisor"
    SCOUT = "domain_scout"
    PANEL = "peer_review_panel"
    QUESTION_GEN = "question_generator"
    ALCHEMIST = "data_alchemist"
    DESIGNER = "experiment_designer"
    EXECUTOR = "executor"
    UNCERTAINTY = "uncertainty_quantifier"
    CRITIC = "critic"
    WRITER = "paper_writer"


class Provenance(BaseModel):
    """Where a piece of data came from, and proof it has not changed since.

    Every row of every dataset the system analyses carries one of these. The
    content hash is what makes the final paper's "Reproducibility" appendix
    meaningful rather than decorative.
    """

    url: str
    modality: Modality
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sha256: str = ""
    byte_size: int = 0
    title: str = ""
    note: str = ""

    @staticmethod
    def hash_content(content: bytes | str) -> str:
        if isinstance(content, str):
            content = content.encode("utf-8", errors="replace")
        return hashlib.sha256(content).hexdigest()


class ConfidenceComponents(BaseModel):
    """The four measured signals behind every confidence score.

    Deliberately *not* a single number an LLM emitted. See
    agents/uncertainty.py for how each component is measured and combined.
    """

    self_consistency: float = Field(
        0.0, ge=0.0, le=1.0, description="Agreement rate across k resamples at T>0"
    )
    cross_model_agreement: float | None = Field(
        None, ge=0.0, le=1.0, description="Agreement with a different model family"
    )
    statistical_evidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Derived from p-value, effect size, CI width, n"
    )
    evidence_quality: float = Field(
        0.0, ge=0.0, le=1.0, description="Independent source count, conflicts, recency"
    )

    def explain(self) -> str:
        parts = [
            f"self-consistency={self.self_consistency:.2f}",
            f"statistical={self.statistical_evidence:.2f}",
            f"evidence={self.evidence_quality:.2f}",
        ]
        if self.cross_model_agreement is not None:
            parts.insert(1, f"cross-model={self.cross_model_agreement:.2f}")
        return ", ".join(parts)


class Claim(BaseModel):
    """A single assertion the system might make in its paper.

    Confidence is attached per claim, not per run, because a run can be
    simultaneously confident about "the dataset has N rows" and unsure about
    "X causes Y". Claims below the abstention threshold are moved to an
    "Abstained" section rather than asserted.
    """

    text: str
    supporting_source_urls: list[str] = Field(default_factory=list)
    components: ConfidenceComponents = Field(default_factory=ConfidenceComponents)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    abstained: bool = False
    abstain_reason: str = ""


class ToolCall(BaseModel):
    """One tool invocation, recorded for the trace view and for debugging."""

    tool: str
    args_summary: str = ""
    ok: bool = True
    error: str = ""
    duration_ms: int = 0
    attempt: int = 1


class TokenUsage(BaseModel):
    """Per-provider token accounting, surfaced live in the UI vitals panel."""

    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Conflict(BaseModel):
    """Two sources disagreeing about the same quantity.

    Conflicts are first-class rather than silently resolved: they depress the
    `evidence_quality` confidence component and the Critic is required to
    address any that remain open. This is the assessment's "handling of
    ambiguous/conflicting information" requirement.
    """

    subject: str
    value_a: str
    source_a: str
    value_b: str
    source_b: str
    discrepancy: str = ""
    resolved: bool = False
    resolution: str = ""

    @field_validator("subject")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("conflict subject cannot be empty")
        return v
