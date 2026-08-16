"""Final output contracts: the mini-research paper and the run manifest."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from .common import Claim, Conflict, Provenance, TokenUsage
from .experiment import FigureSpec


class PaperSection(BaseModel):
    heading: str
    body_markdown: str
    order: int = 0


class Paper(BaseModel):
    """The deliverable a reviewer actually reads.

    `abstained_claims` is a required section, not an optional one. A run that
    asserts everything it considered is a run that never checked itself, and
    the abstention list is the most credible page in the document.
    """

    title: str
    abstract: str
    domain: str
    research_question: str
    sections: list[PaperSection] = Field(default_factory=list)
    figures: list[FigureSpec] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    abstained_claims: list[Claim] = Field(default_factory=list)
    limitations: str = ""
    references: list[Provenance] = Field(default_factory=list)
    unresolved_conflicts: list[Conflict] = Field(default_factory=list)
    unresolved_objections: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(0.0, ge=0.0, le=1.0)
    cycles_used: int = 0
    accepted_by_critic: bool = False

    def to_markdown(self) -> str:
        """Render the paper. Kept here rather than in a template so the CLI,
        the API, and the tests all produce byte-identical output."""
        lines: list[str] = [f"# {self.title}", ""]

        badge = "accepted by Critic" if self.accepted_by_critic else "cycle limit reached"
        lines += [
            f"> **Domain:** {self.domain}  ",
            f"> **Research question:** {self.research_question}  ",
            f"> **Overall confidence:** {self.overall_confidence:.0%} "
            f"({self.cycles_used} cycle(s), {badge})",
            "",
            "## Abstract",
            "",
            self.abstract,
            "",
        ]

        for section in sorted(self.sections, key=lambda s: s.order):
            lines += [f"## {section.heading}", "", section.body_markdown, ""]

        if self.claims:
            lines += ["## Claims and Confidence", "", "| Claim | Confidence | Basis |", "|---|---|---|"]
            for claim in self.claims:
                basis = claim.components.explain()
                text = claim.text.replace("|", "\\|")
                lines.append(f"| {text} | {claim.confidence:.0%} | {basis} |")
            lines.append("")

        # The honest bit. Always rendered, even when empty, so its absence in a
        # given run is a deliberate statement rather than an oversight.
        lines += ["## Abstained — Insufficient Evidence", ""]
        if self.abstained_claims:
            lines.append(
                "The following statements were considered and are **not** asserted, "
                "because their confidence fell below the abstention threshold:"
            )
            lines.append("")
            for claim in self.abstained_claims:
                lines.append(
                    f"- {claim.text} _(confidence {claim.confidence:.0%} — "
                    f"{claim.abstain_reason})_"
                )
        else:
            lines.append("_No claims fell below the abstention threshold in this run._")
        lines.append("")

        if self.unresolved_conflicts:
            lines += ["## Unresolved Source Conflicts", ""]
            for conflict in self.unresolved_conflicts:
                lines.append(
                    f"- **{conflict.subject}**: {conflict.source_a} reports "
                    f"`{conflict.value_a}`, {conflict.source_b} reports "
                    f"`{conflict.value_b}`. {conflict.discrepancy}"
                )
            lines.append("")

        lines += ["## Limitations and Future Work", "", self.limitations or "_Not produced._", ""]

        if self.unresolved_objections:
            lines += [
                "### Objections left unaddressed at the cycle limit",
                "",
                *(f"- {o}" for o in self.unresolved_objections),
                "",
            ]

        if self.references:
            lines += ["## References", ""]
            for i, ref in enumerate(self.references, 1):
                title = ref.title or ref.url
                stamp = ref.retrieved_at.strftime("%Y-%m-%d")
                lines.append(
                    f"{i}. {title} — <{ref.url}> "
                    f"({ref.modality.value}, retrieved {stamp}, sha256 `{ref.sha256[:12]}`)"
                )
            lines.append("")

        return "\n".join(lines)


class RunManifest(BaseModel):
    """Everything needed to argue the run really happened as described."""

    run_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    seed: int = 0
    models_used: dict[str, str] = Field(default_factory=dict)
    token_usage: list[TokenUsage] = Field(default_factory=list)
    source_hashes: dict[str, str] = Field(default_factory=dict)
    cycles_used: int = 0
    tool_failures: dict[str, int] = Field(default_factory=dict)
    provider_failovers: list[str] = Field(default_factory=list)
    total_cost_usd: float = 0.0  # free tiers only; asserted, and worth asserting

    @property
    def duration_seconds(self) -> float:
        if not self.finished_at:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self.token_usage)
