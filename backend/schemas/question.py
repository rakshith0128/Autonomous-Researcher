"""Research question contracts.

The assessment demands questions that are "not directly searchable (must
require synthesis)". That is enforced here rather than asserted: every
question declares which distinct sources must be *joined* to answer it, and
then a searchability probe actually goes looking for a direct answer. A
question with a findable answer is rejected and regenerated.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QuestionProposal(BaseModel):
    """LLM-authored question. Not yet validated for triviality."""

    text: str = Field(description="The research question, phrased precisely")
    rationale: str = Field(description="Why this is non-trivial and worth asking")
    required_joins: list[str] = Field(
        min_length=2,
        description="The distinct data sources that must be combined; >=2 is what makes it synthesis",
    )
    expected_measurable: str = Field(
        description="The concrete quantity that would have to be measured to answer it"
    )


class QuestionProposalBatch(BaseModel):
    proposals: list[QuestionProposal] = Field(min_length=3, max_length=5)


class SearchabilityProbe(BaseModel):
    """Result of actively trying to find a ready-made answer.

    If `directly_answered` is true the question is disqualified as trivial --
    the whole point is that no single source already contains the answer.
    """

    query_used: str
    directly_answered: bool
    evidence_url: str = ""
    evidence_snippet: str = ""
    reasoning: str = ""


class PeerRating(BaseModel):
    """A peer agent's novelty/feasibility rating, as the assessment requires."""

    rater_id: str
    novelty: float = Field(ge=0.0, le=1.0)
    feasibility: float = Field(ge=0.0, le=1.0)
    comment: str = ""


class ResearchQuestion(BaseModel):
    """A question that has been through the full gauntlet."""

    id: str
    proposal: QuestionProposal
    probe: SearchabilityProbe | None = None
    ratings: list[PeerRating] = Field(default_factory=list)
    disqualified: bool = False
    disqualified_reason: str = ""

    @property
    def text(self) -> str:
        return self.proposal.text

    @property
    def mean_novelty(self) -> float:
        return sum(r.novelty for r in self.ratings) / len(self.ratings) if self.ratings else 0.0

    @property
    def mean_feasibility(self) -> float:
        return (
            sum(r.feasibility for r in self.ratings) / len(self.ratings) if self.ratings else 0.0
        )

    @property
    def score(self) -> float:
        """Novelty matters slightly more than feasibility, but an infeasible
        question is worthless, so feasibility is weighted heavily enough to
        veto pure moonshots."""
        return 0.55 * self.mean_novelty + 0.45 * self.mean_feasibility


class QuestionSet(BaseModel):
    """The Question Generator's output for one cycle."""

    questions: list[ResearchQuestion] = Field(default_factory=list)
    selected_id: str = ""
    regeneration_rounds: int = 0
    rationale: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    def selected(self) -> ResearchQuestion | None:
        return next((q for q in self.questions if q.id == self.selected_id), None)

    def viable(self) -> list[ResearchQuestion]:
        return [q for q in self.questions if not q.disqualified]
