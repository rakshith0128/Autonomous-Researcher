"""Experiment design and execution contracts.

Two execution paths share these types:

1. A **typed registry** of real statistical procedures. The LLM emits only
   validated JSON parameters; Python does the statistics. Reliable enough to
   demo live in front of a reviewer.
2. **Sandboxed code generation** for anything the registry cannot express, with
   the traceback fed back for self-repair.

The registry exists because an agent that writes its own scipy calls fails in
front of an audience roughly half the time, and a demo that reliably runs a
Mann-Whitney U is worth more than one that occasionally invents a new test.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExperimentKind(str, Enum):
    """Procedures the deterministic runner knows how to execute."""

    CORRELATION = "correlation"  # Pearson + Spearman, two numeric columns
    GROUP_COMPARISON = "group_comparison"  # t-test / Mann-Whitney across a categorical split
    REGRESSION = "regression"  # OLS with heteroskedasticity-robust errors
    TREND = "trend"  # Mann-Kendall + Theil-Sen on an ordered series
    CLUSTERING = "clustering"  # KMeans + silhouette, with a k sweep
    CLASSIFICATION = "classification"  # cross-validated, against a stratified baseline
    CUSTOM_CODE = "custom_code"  # escape hatch -> sandboxed codegen


class Hypothesis(BaseModel):
    """A falsifiable statement, pre-registered before the data is touched.

    `prediction` is recorded *before* execution specifically so the Critic can
    catch the system rationalising whatever it happened to find.
    """

    null: str = Field(description="H0, stated so that it could be rejected")
    alternative: str = Field(description="H1")
    prediction: str = Field(description="What we expect to see if H1 holds, stated in advance")
    reasoning: str = ""


class ExperimentSpec(BaseModel):
    """What to run. Parameters are validated against the dataset before use."""

    kind: ExperimentKind
    hypothesis: Hypothesis
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Kind-specific: column names, group keys, k, alpha, etc.",
    )
    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    rationale: str = ""
    code: str = Field("", description="Only populated for CUSTOM_CODE")

    answers_question: bool = Field(
        True,
        description=(
            "False if no available column measures what the research question asks "
            "about. Set this honestly -- answering a different question is worse "
            "than reporting that this one cannot be answered."
        ),
    )
    addresses_question: str = Field(
        "",
        description=(
            "Which specific column corresponds to which part of the research "
            "question. Naming them makes substitution visible."
        ),
    )

    def required_columns(self) -> list[str]:
        """Columns this spec needs, so we can validate before executing."""
        keys = ("x", "y", "group", "value", "time", "outcome", "target")
        out: list[str] = []
        for key in keys:
            val = self.params.get(key)
            if isinstance(val, str):
                out.append(val)
        for key in ("features", "predictors", "columns"):
            val = self.params.get(key)
            if isinstance(val, list):
                out.extend(str(v) for v in val)
        return out


class FigureSpec(BaseModel):
    """A Plotly figure, serialised as JSON for the browser to render.

    Figures are built server-side but never rendered server-side -- shipping
    figure JSON instead of images keeps the container small and the dashboards
    genuinely interactive, which the assessment asks for explicitly.
    """

    title: str
    figure_json: str
    caption: str = ""
    kind: str = ""


class StatResult(BaseModel):
    """The numbers. Everything the Critic needs to attack the methodology."""

    test_name: str
    statistic: float | None = None
    p_value: float | None = None
    effect_size: float | None = None
    effect_size_name: str = ""
    effect_interpretation: str = ""  # negligible / small / medium / large
    ci_low: float | None = None
    ci_high: float | None = None
    n: int = 0
    dof: float | None = None
    comparisons_made: int = Field(
        1, description="Feeds the Critic's multiple-comparisons check"
    )
    p_value_corrected: float | None = None
    correction_method: str = ""
    assumptions_checked: dict[str, bool] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @property
    def significant(self) -> bool:
        p = self.p_value_corrected if self.p_value_corrected is not None else self.p_value
        return p is not None and p < 0.05

    @property
    def effect_is_trivial(self) -> bool:
        """The assessment forces iteration on a trivial effect size, so the
        judgement is encoded here rather than left to the LLM's taste."""
        return self.effect_interpretation.lower() in {"negligible", "trivial"}


class ExperimentResult(BaseModel):
    """Everything one execution produced, successful or not."""

    spec: ExperimentSpec
    stats: list[StatResult] = Field(default_factory=list)
    figures: list[FigureSpec] = Field(default_factory=list)
    summary: str = ""
    tables: dict[str, Any] = Field(default_factory=dict)
    executed_ok: bool = False
    error: str = ""
    traceback: str = ""
    repair_attempts: int = 0
    duration_ms: int = 0
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    @property
    def primary(self) -> StatResult | None:
        return self.stats[0] if self.stats else None

    def forces_iteration(self) -> tuple[bool, str]:
        """The assessment's hard rule: p > 0.05 or a trivial effect sends the
        system back around the loop. Returns (should_iterate, reason)."""
        if not self.executed_ok:
            return True, f"execution failed: {self.error or 'unknown error'}"
        primary = self.primary
        if primary is None:
            return True, "no statistical result was produced"
        if not primary.significant:
            p = primary.p_value_corrected or primary.p_value
            return True, f"p = {p:.4f} exceeds alpha" if p is not None else "no p-value"
        if primary.effect_is_trivial:
            return True, (
                f"effect size {primary.effect_size_name}="
                f"{primary.effect_size:.3f} is {primary.effect_interpretation}"
                if primary.effect_size is not None
                else "effect size is negligible"
            )
        return False, ""
