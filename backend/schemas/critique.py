"""Critic contracts.

The assessment requires the Critic to "cite counterevidence". That is enforced
rather than trusted: every objection must carry a URL, the URL is actually
fetched, and an objection whose citation cannot be verified is **dropped**.
An LLM that invents a plausible-looking citation therefore loses the argument
instead of winning it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"


class RerouteTarget(str, Enum):
    """Where the Critic sends the work back to.

    An adaptive target is what separates this from a fixed retry loop: a
    flawed statistical test goes back to the Designer, but an unanswerable
    question goes all the way back to the Question Generator.
    """

    QUESTION = "question"
    DATA = "data"
    EXPERIMENT = "experiment"
    NONE = "none"


class Severity(str, Enum):
    BLOCKING = "blocking"
    MAJOR = "major"
    MINOR = "minor"


class Objection(BaseModel):
    """One attack on the methodology, with a citation that must survive checks."""

    severity: Severity
    claim_attacked: str
    rationale: str
    counterevidence_url: str = ""
    counterevidence_quote: str = ""
    verified: bool = Field(
        False, description="True only if the URL was fetched and found relevant"
    )
    verification_note: str = ""
    suggested_fix: str = ""


class StatFlags(BaseModel):
    """Mechanical statistical checks, computed in Python from StatResult.

    These are deliberately not left to the LLM: whether p > 0.05 is arithmetic,
    not a matter of opinion, and the Critic should be arguing about
    confounding, not recomputing inequalities.
    """

    p_gt_alpha: bool = False
    trivial_effect: bool = False
    n_too_small: bool = False
    multiple_comparisons_uncorrected: bool = False
    assumptions_violated: bool = False
    wide_confidence_interval: bool = False
    unresolved_conflicts: bool = False

    @property
    def any_blocking(self) -> bool:
        return any(
            (
                self.p_gt_alpha,
                self.trivial_effect,
                self.n_too_small,
                self.multiple_comparisons_uncorrected,
            )
        )

    def as_reasons(self) -> list[str]:
        labels = {
            "p_gt_alpha": "p-value exceeds alpha",
            "trivial_effect": "effect size is negligible",
            "n_too_small": "sample size is too small to support the claim",
            "multiple_comparisons_uncorrected": "multiple comparisons without correction",
            "assumptions_violated": "test assumptions are violated",
            "wide_confidence_interval": "confidence interval is too wide to be informative",
            "unresolved_conflicts": "sources disagree and the conflict is unresolved",
        }
        return [label for field, label in labels.items() if getattr(self, field)]


class Critique(BaseModel):
    """The Critic's full structured verdict for one cycle."""

    cycle: int
    verdict: Verdict
    reroute_to: RerouteTarget = RerouteTarget.NONE
    objections: list[Objection] = Field(default_factory=list)
    stat_flags: StatFlags = Field(default_factory=StatFlags)
    summary: str = ""
    limitations: str = Field(
        "", description="Becomes the paper's Limitations & Future Work section"
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    @property
    def verified_objections(self) -> list[Objection]:
        """Only objections whose citation survived verification carry weight."""
        return [o for o in self.objections if o.verified]

    @property
    def blocking_objections(self) -> list[Objection]:
        return [o for o in self.verified_objections if o.severity == Severity.BLOCKING]

    def demands_iteration(self) -> bool:
        return self.verdict != Verdict.ACCEPT or self.stat_flags.any_blocking
