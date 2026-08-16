"""Peer Review Panel: two independent reviewers choose the domain.

The Emergence Index measures whether a field is *growing*. It cannot judge
whether a field is *researchable in ten minutes with public data*, and those
are different questions -- a domain can be exploding and still be impossible to
study without lab access.

So the quantitative ranking and the judgement are separated. Two reviewers
score every candidate independently, with different prompts and, wherever a
second provider is configured, different model families. Genuine independence
matters: two samples from one model at the same temperature agree with
themselves far more than they agree with the truth, and calling that a panel
would be theatre.

When the reviewers disagree sharply, a third tiebreak round runs *with the
disagreement shown to it*. That is the collaboration the assessment asks for --
the reviewers' conflict changes what the deciding call sees.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import Role
from ..schemas import (
    AgentName,
    ArtifactKind,
    DomainCandidate,
    DomainSelection,
    Level,
    PanelBallot,
)
from .base import AgentFailure, BaseAgent

log = logging.getLogger(__name__)

# Above this spread on a 0-1 scale, the reviewers are not disagreeing at the
# margin -- they are reading the candidate differently, and that is worth a
# third opinion.
DISAGREEMENT_THRESHOLD = 0.25

REVIEWER_A = """You are a pragmatic research reviewer. You have seen many promising research \
directions fail because the data was not there.

Score each candidate domain on:
- novelty: how genuinely new is this? Established fields score low.
- data_availability: can public data be gathered RIGHT NOW from papers, APIs, or repositories?
- tractability: could a meaningful statistical experiment run on this in ten minutes of compute?
- non_obviousness: would a non-trivial research question here require real synthesis?

Be harsh on data_availability and tractability. A fascinating domain with no accessible data \
is worthless for this purpose."""

REVIEWER_B = """You are an ambitious research reviewer. You care most about whether a domain \
can produce an interesting, original finding.

Score each candidate domain on:
- novelty: is this a genuine research frontier rather than an incremental relabelling?
- data_availability: is there enough public evidence to support a real analysis?
- tractability: can a concrete hypothesis be tested here?
- non_obviousness: would the answer surprise a domain expert, or is it already known?

Be harsh on novelty and non_obviousness. A well-trodden field dressed in new terminology \
should score near zero."""

BALLOT_PROMPT = """Score every candidate below. Return one vote per candidate, using the \
exact domain_name given.

{candidates}

Measured growth evidence is supplied for context. It tells you the field is growing; it does \
NOT tell you the field is researchable. That judgement is yours."""

TIEBREAK = """Two reviewers disagreed substantially about these domains.

{disagreements}

Full evidence:
{candidates}

Score every candidate yourself. You have seen where the reviewers diverged -- weigh those \
points explicitly. Where a reviewer's concern about data availability conflicts with another's \
enthusiasm about novelty, remember that an unstudiable domain produces no paper at all."""


class PeerReviewPanel(BaseAgent):
    name = AgentName.PANEL
    fatal_on_failure = True

    async def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        selection: DomainSelection | None = state.get("domain_selection")
        if selection is None:
            raise AgentFailure("no domain candidates to review", fatal=True)

        viable = [c for c in selection.candidates if not c.disqualified]
        if not viable:
            raise AgentFailure("every candidate domain was disqualified", fatal=True)

        if len(viable) == 1:
            # Nothing to deliberate about. Say so rather than staging a vote.
            chosen = viable[0]
            self.say(
                f"Only one candidate survived measurement; selecting {chosen.name} without a vote.",
                level=Level.WARN,
            )
            return self._finish(selection, chosen, rationale="sole viable candidate", agreement=0.5)

        rendered = self._render(viable)

        self.say("Two reviewers scoring candidates independently…")
        ballot_a, ballot_b, single_family = await self._collect_ballots(rendered)

        selection.ballots = [b for b in (ballot_a, ballot_b) if b is not None]
        if not selection.ballots:
            raise AgentFailure("no reviewer returned a usable ballot", fatal=True)

        combined, spread = self._combine(viable, selection.ballots)
        selection.reviewer_disagreement = max(spread.values(), default=0.0)

        if selection.reviewer_disagreement > DISAGREEMENT_THRESHOLD and len(selection.ballots) > 1:
            self.say(
                f"Reviewers disagree by {selection.reviewer_disagreement:.2f} on at least one "
                "candidate; running a tiebreak round with the disagreement surfaced.",
                level=Level.WARN,
            )
            tiebreak = await self._tiebreak(rendered, viable, spread)
            if tiebreak is not None:
                selection.ballots.append(tiebreak)
                selection.tiebreak_used = True
                combined, spread = self._combine(viable, selection.ballots)

        # The panel judges researchability; the index measures growth. Both
        # matter, so the final ordering is their product rather than either
        # alone -- a domain must be both rising and studiable to win.
        ranked = sorted(
            viable,
            key=lambda c: combined.get(c.name, 0.0) * _index_weight(c, viable),
            reverse=True,
        )
        chosen = ranked[0]

        agreement = 1.0 - min(selection.reviewer_disagreement / 0.5, 1.0)
        rationale = (
            f"Highest combined score across {len(selection.ballots)} reviewer(s) "
            f"({combined.get(chosen.name, 0):.2f}) with an Emergence Index of "
            f"{chosen.emergence_index:+.2f}."
        )
        if single_family:
            rationale += " Note: only one model family was available, so reviewer independence is limited."

        self._report(ranked, combined, selection)
        return self._finish(selection, chosen, rationale=rationale, agreement=agreement)

    # -------------------------------------------------------------- ballots

    async def _collect_ballots(
        self, rendered: str
    ) -> tuple[PanelBallot | None, PanelBallot | None, bool]:
        """Two reviewers, ideally from different model families."""
        prompt = BALLOT_PROMPT.format(candidates=rendered)

        ballot_a: PanelBallot | None = None
        try:
            ballot_a = await self.router.structured(
                Role.REASONING,
                [{"role": "system", "content": REVIEWER_A}, {"role": "user", "content": prompt}],
                PanelBallot,
                temperature=0.2,
                max_tokens=1800,
            )
        except Exception as exc:  # noqa: BLE001 - one reviewer failing is survivable
            log.warning("reviewer A failed: %s", exc)
            self.say(f"Reviewer A could not vote ({exc}).", level=Level.WARN)

        # Reviewer B from a different provider family where one exists.
        primary = self.router.provider_names[0] if self.router.provider_names else ""
        ballot_b, provider = await self.router.structured_cross_model(
            Role.REASONING,
            [{"role": "system", "content": REVIEWER_B}, {"role": "user", "content": prompt}],
            PanelBallot,
            exclude_provider=primary,
            temperature=0.2,
            max_tokens=1800,
        )

        single_family = ballot_b is None
        if single_family:
            # No second family configured. Fall back to a different model tier
            # with a higher temperature, and be explicit that this is weaker.
            self.say(
                "Only one model family is configured; reviewer B falls back to a different "
                "model tier, which makes the two votes less independent.",
                level=Level.WARN,
            )
            try:
                ballot_b = await self.router.structured(
                    Role.FAST,
                    [
                        {"role": "system", "content": REVIEWER_B},
                        {"role": "user", "content": prompt},
                    ],
                    PanelBallot,
                    temperature=0.6,
                    max_tokens=1600,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("reviewer B fallback failed: %s", exc)
        else:
            self.say(f"Reviewer B scored independently via {provider}.", level=Level.INFO)

        return ballot_a, ballot_b, single_family

    async def _tiebreak(
        self, rendered: str, viable: list[DomainCandidate], spread: dict[str, float]
    ) -> PanelBallot | None:
        contested = sorted(spread.items(), key=lambda kv: kv[1], reverse=True)[:3]
        described = "\n".join(
            f"- {name}: reviewers differed by {value:.2f}" for name, value in contested if value > 0
        )
        try:
            return await self.router.structured(
                Role.REASONING,
                [
                    {
                        "role": "system",
                        "content": "You are the deciding reviewer on a split panel.",
                    },
                    {
                        "role": "user",
                        "content": TIEBREAK.format(
                            disagreements=described or "- (no specific contest)",
                            candidates=rendered,
                        ),
                    },
                ],
                PanelBallot,
                temperature=0.1,
                max_tokens=1800,
            )
        except Exception as exc:  # noqa: BLE001 - fall back to the two votes we have
            log.warning("tiebreak failed: %s", exc)
            return None

    # ------------------------------------------------------------- scoring

    @staticmethod
    def _combine(
        viable: list[DomainCandidate], ballots: list[PanelBallot]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Mean score and inter-reviewer spread per candidate.

        Names are matched case-insensitively and by prefix, because reviewers
        paraphrase the domain name roughly a third of the time and an exact
        match would silently discard those votes.
        """
        by_name: dict[str, list[float]] = {c.name: [] for c in viable}
        lookup = {c.name.lower(): c.name for c in viable}

        for ballot in ballots:
            for vote in ballot.votes:
                key = vote.domain_name.strip().lower()
                canonical = lookup.get(key)
                if canonical is None:
                    canonical = next(
                        (
                            original
                            for lowered, original in lookup.items()
                            if lowered.startswith(key[:18]) or key.startswith(lowered[:18])
                        ),
                        None,
                    )
                if canonical is not None:
                    by_name[canonical].append(vote.total)

        combined = {
            name: (sum(scores) / len(scores) if scores else 0.0)
            for name, scores in by_name.items()
        }
        spread = {
            name: (max(scores) - min(scores) if len(scores) > 1 else 0.0)
            for name, scores in by_name.items()
        }
        return combined, spread

    @staticmethod
    def _render(candidates: list[DomainCandidate]) -> str:
        lines = []
        for candidate in candidates:
            signals = candidate.signals
            lines.append(
                f"### {candidate.name}\n"
                f"{candidate.proposal.description}\n"
                f"Why believed emerging: {candidate.proposal.why_emerging}\n"
                f"Measured (term: {signals.term_used!r}): "
                f"arXiv {signals.arxiv_recent_count} papers since 2024 "
                f"({signals.arxiv_growth_ratio:.1f}x baseline); "
                f"OpenAlex {signals.openalex_recent_works} works "
                f"({signals.openalex_growth_ratio:.1f}x baseline); "
                f"{signals.github_repos_created_post_cutoff} new repositories; "
                f"Emergence Index {candidate.emergence_index:+.2f}"
            )
        return "\n\n".join(lines)

    def _report(
        self,
        ranked: list[DomainCandidate],
        combined: dict[str, float],
        selection: DomainSelection,
    ) -> None:
        self.publish(
            ArtifactKind.DOMAIN_SELECTED,
            {
                "chosen": ranked[0].name,
                "scores": [
                    {
                        "name": c.name,
                        "panel_score": round(combined.get(c.name, 0.0), 3),
                        "emergence_index": round(c.emergence_index, 3),
                    }
                    for c in ranked
                ],
                "reviewers": len(selection.ballots),
                "disagreement": round(selection.reviewer_disagreement, 3),
                "tiebreak_used": selection.tiebreak_used,
            },
            message=f"Panel selected {ranked[0].name}",
        )
        for candidate in ranked:
            self.say(
                f"{candidate.name}: panel {combined.get(candidate.name, 0):.2f}, "
                f"index {candidate.emergence_index:+.2f}",
                level=Level.SUCCESS if candidate is ranked[0] else Level.INFO,
            )

    def _finish(
        self,
        selection: DomainSelection,
        chosen: DomainCandidate,
        *,
        rationale: str,
        agreement: float,
    ) -> dict[str, Any]:
        selection.chosen_name = chosen.name
        selection.rationale = rationale
        selection.confidence = round(0.5 * selection.confidence + 0.5 * agreement, 3)

        self.say(
            f"Selected: {chosen.name}. {rationale}",
            level=Level.SUCCESS,
        )
        return {
            "domain_selection": selection,
            "domain_name": chosen.name,
            "phase": "question",
        }


def _index_weight(candidate: DomainCandidate, viable: list[DomainCandidate]) -> float:
    """Map the Emergence Index onto a positive multiplier.

    The index is a z-score and can be negative, which would flip the sign of a
    panel score and rank a well-reviewed domain last. Shifting to a positive
    range keeps the ordering meaningful while still letting growth evidence
    move the result.
    """
    indices = [c.emergence_index for c in viable]
    low = min(indices)
    span = max(indices) - low
    if span <= 0:
        return 1.0
    return 0.6 + 0.8 * ((candidate.emergence_index - low) / span)
