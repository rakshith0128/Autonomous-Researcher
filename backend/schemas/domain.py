"""Domain discovery contracts.

The split here is the heart of the system's anti-hallucination stance:

    DomainProposal   <- written by an LLM. Cheap talk. No numbers trusted.
    EmergenceSignals <- measured by Python against arXiv / GitHub / OpenAlex.
    DomainCandidate  <- proposal + signals + an index computed from signals.

An LLM never assigns the score that decides which domain wins.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DomainProposal(BaseModel):
    """An LLM's guess at an emerging field. Unverified by construction."""

    name: str = Field(description="Short name of the candidate scientific domain")
    description: str = Field(description="One or two sentences on what the field studies")
    why_emerging: str = Field(description="Why this is believed to be post-2024 and rising")
    search_terms: list[str] = Field(
        default_factory=list,
        description="2-4 literal query strings to measure this field's growth with",
    )


class DomainProposalBatch(BaseModel):
    """What the Scout's first LLM call returns."""

    proposals: list[DomainProposal] = Field(min_length=3)


class EmergenceSignals(BaseModel):
    """Measured growth evidence. Every field here is computed, never generated.

    `measured_ok` records whether each upstream source actually answered; a
    domain scored while GitHub was rate-limited is not comparable to one scored
    with full data, and the Scout says so rather than pretending.
    """

    # arXiv: publication volume before vs after the emergence cutoff
    arxiv_recent_count: int = 0
    arxiv_baseline_count: int = 0
    arxiv_growth_ratio: float = 0.0
    arxiv_monthly_slope: float = 0.0

    # GitHub: repos *created* after the cutoff, and how fast they gather stars
    github_repos_created_post_cutoff: int = 0
    github_total_stars: int = 0
    github_star_velocity: float = 0.0  # stars per day, summed across repos

    # OpenAlex: scholarly output and how quickly it is being cited
    openalex_recent_works: int = 0
    openalex_baseline_works: int = 0
    openalex_growth_ratio: float = 0.0
    openalex_citation_velocity: float = 0.0

    # Public attention as a recency tiebreaker
    forum_mentions: int = 0

    sources_consulted: list[str] = Field(default_factory=list)
    measured_ok: dict[str, bool] = Field(default_factory=dict)
    term_used: str = Field(
        "", description="The search phrase these numbers were actually measured with"
    )
    terms_probed: list[str] = Field(
        default_factory=list, description="Phrases tried before one produced evidence"
    )

    @property
    def completeness(self) -> float:
        """Fraction of upstream sources that answered. Feeds confidence."""
        if not self.measured_ok:
            return 0.0
        return sum(1 for ok in self.measured_ok.values() if ok) / len(self.measured_ok)


class DomainCandidate(BaseModel):
    """A proposal that has survived measurement."""

    proposal: DomainProposal
    signals: EmergenceSignals = Field(default_factory=EmergenceSignals)
    emergence_index: float = Field(
        0.0, description="Z-normalised weighted sum of signals; comparable within a run only"
    )
    component_z: dict[str, float] = Field(
        default_factory=dict, description="Per-signal z-scores, shown in the UI chart"
    )
    evidence_urls: list[str] = Field(default_factory=list)
    disqualified: bool = False
    disqualified_reason: str = ""

    @property
    def name(self) -> str:
        return self.proposal.name


class PanelVote(BaseModel):
    """One reviewer agent's scoring of one candidate.

    Two reviewers score independently with different prompts and, where more
    than one provider is configured, different model families.
    """

    scorer_id: str
    domain_name: str
    novelty: float = Field(ge=0.0, le=1.0)
    data_availability: float = Field(ge=0.0, le=1.0)
    tractability: float = Field(
        ge=0.0, le=1.0, description="Can a real experiment run against this in ~10 minutes?"
    )
    non_obviousness: float = Field(ge=0.0, le=1.0)
    rationale: str = ""

    @property
    def total(self) -> float:
        return (
            self.novelty + self.data_availability + self.tractability + self.non_obviousness
        ) / 4.0


class PanelBallot(BaseModel):
    """One reviewer's full ballot across all candidates."""

    votes: list[PanelVote] = Field(min_length=1)


class DomainSelection(BaseModel):
    """The Scout + Panel's joint output: one domain, with its audit trail."""

    candidates: list[DomainCandidate]
    ballots: list[PanelBallot] = Field(default_factory=list)
    chosen_name: str = ""
    combined_scores: dict[str, float] = Field(default_factory=dict)
    reviewer_disagreement: float = Field(
        0.0, description="Max per-candidate spread between reviewers; triggers a tiebreak round"
    )
    tiebreak_used: bool = False
    rationale: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    def chosen(self) -> DomainCandidate | None:
        return next((c for c in self.candidates if c.name == self.chosen_name), None)
