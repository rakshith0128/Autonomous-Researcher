"""The Emergence Index: ranking candidate domains by measurement, not opinion.

The assessment asks the Domain Scout to find fields that are genuinely rising
post-2024. The obvious implementation -- ask an LLM which fields feel new --
produces confident, unfalsifiable, and frequently wrong answers. This module
exists so that no LLM ever assigns the score that picks the winning domain.

Each candidate carries signals measured independently from arXiv, OpenAlex,
GitHub, and public discussion. Three statistical decisions shape the result:

**Log-scaling before standardising.** Growth ratios are heavy-tailed: a run
might see 1.1, 2.4, 25.7, and 300. A raw z-score over that is dominated by the
single largest value, and every other candidate lands at roughly -0.5,
destroying the ranking among them. `log1p` first compresses the tail so the
comparison reflects order of magnitude, which is the scale the question is
actually asked on.

**Missing measurements impute to the mean, not to zero.** When GitHub is
rate-limited, that candidate's code signal is *unknown*, not *zero*. Scoring it
as zero silently punishes a field for an outage on our side. Imputing z=0
leaves it neutral and lets the surviving signals decide.

**Completeness is reported, not hidden.** A domain scored on two of four
sources gets a lower confidence, so the Peer Review Panel downstream can weigh
it accordingly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..schemas import DomainCandidate

# Weights sum to 1.0. Publication evidence dominates because the question is
# whether a *scientific field* is emerging: code and chatter are corroborating
# signals, not primary ones. A field with real papers and no GitHub presence is
# still a field; a GitHub trend with no literature behind it is not.
WEIGHTS: dict[str, float] = {
    "arxiv_growth_ratio": 0.22,
    "arxiv_relative_slope": 0.18,
    "openalex_growth_ratio": 0.22,
    "openalex_citation_velocity": 0.12,
    "github_repos": 0.10,
    "github_star_velocity": 0.10,
    "forum_mentions": 0.06,
}

# Signals that are already rates or ratios and may legitimately be negative
# (a decelerating field), so log1p would be undefined.
_SIGNED_SIGNALS = {"arxiv_relative_slope"}


@dataclass
class SignalVector:
    """One candidate's raw measurements, before any normalisation."""

    name: str
    values: dict[str, float] = field(default_factory=dict)
    measured: dict[str, bool] = field(default_factory=dict)

    @property
    def completeness(self) -> float:
        if not self.measured:
            return 0.0
        return sum(1 for ok in self.measured.values() if ok) / len(self.measured)


def _transform(signal: str, value: float) -> float:
    """Compress heavy tails before standardising."""
    if signal in _SIGNED_SIGNALS:
        # Symmetric log: preserves the sign of a declining trend while still
        # compressing magnitude on both sides.
        return math.copysign(math.log1p(abs(value)), value)
    return math.log1p(max(value, 0.0))


def _zscores(values: list[float | None]) -> list[float]:
    """Standardise, imputing None at the mean of what was measured.

    Returns all-zero when fewer than two candidates supplied the signal, or
    when every candidate reported the same value -- in both cases the signal
    carries no information for ranking and should not sway the result.
    """
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return [0.0] * len(values)

    mean = sum(present) / len(present)
    variance = sum((v - mean) ** 2 for v in present) / len(present)
    std = math.sqrt(variance)
    if std == 0:
        return [0.0] * len(values)

    return [0.0 if v is None else (v - mean) / std for v in values]


def score_candidates(vectors: list[SignalVector]) -> dict[str, dict[str, float]]:
    """Compute the Emergence Index for every candidate.

    Returns {name: {"index": float, "<signal>_z": float, "completeness": float}}.

    Scores are comparable **within a run only**: standardisation is relative to
    the candidates measured together, so an index of 1.4 means "well above the
    others considered today", never "objectively emerging". The Scout says so
    in the artefact it publishes, because a number that looks absolute and is
    not is worse than no number.
    """
    if not vectors:
        return {}

    columns: dict[str, list[float]] = {}
    for signal in WEIGHTS:
        raw = [
            _transform(signal, vec.values[signal])
            if vec.measured.get(signal, False) and signal in vec.values
            else None
            for vec in vectors
        ]
        columns[signal] = _zscores(raw)

    results: dict[str, dict[str, float]] = {}
    for i, vec in enumerate(vectors):
        components = {signal: columns[signal][i] for signal in WEIGHTS}
        index = sum(components[signal] * weight for signal, weight in WEIGHTS.items())
        results[vec.name] = {
            "index": round(index, 4),
            "completeness": round(vec.completeness, 3),
            **{f"{signal}_z": round(z, 4) for signal, z in components.items()},
        }
    return results


def vector_from_candidate(candidate: DomainCandidate) -> SignalVector:
    """Flatten a candidate's measured signals into a scoring vector."""
    signals = candidate.signals
    ok = signals.measured_ok

    arxiv_ok = ok.get("arxiv", False)
    openalex_ok = ok.get("openalex", False)
    github_ok = ok.get("github", False)
    forum_ok = ok.get("forum", False)

    return SignalVector(
        name=candidate.name,
        values={
            "arxiv_growth_ratio": signals.arxiv_growth_ratio,
            "arxiv_relative_slope": signals.arxiv_monthly_slope,
            "openalex_growth_ratio": signals.openalex_growth_ratio,
            "openalex_citation_velocity": signals.openalex_citation_velocity,
            "github_repos": float(signals.github_repos_created_post_cutoff),
            "github_star_velocity": signals.github_star_velocity,
            "forum_mentions": float(signals.forum_mentions),
        },
        measured={
            "arxiv_growth_ratio": arxiv_ok,
            "arxiv_relative_slope": arxiv_ok,
            "openalex_growth_ratio": openalex_ok,
            "openalex_citation_velocity": openalex_ok,
            "github_repos": github_ok,
            "github_star_velocity": github_ok,
            "forum_mentions": forum_ok,
        },
    )


def rank(candidates: list[DomainCandidate]) -> list[DomainCandidate]:
    """Score candidates in place and return them best-first.

    Candidates with no successful measurements at all are disqualified rather
    than ranked: an unmeasured domain would score 0.0 on every component, land
    mid-table by construction, and could win a weak field on the strength of
    having no evidence against it.
    """
    vectors = [vector_from_candidate(c) for c in candidates]
    scores = score_candidates(vectors)

    for candidate in candidates:
        result = scores.get(candidate.name, {})
        candidate.emergence_index = result.get("index", 0.0)
        candidate.component_z = {
            key[:-2]: value for key, value in result.items() if key.endswith("_z")
        }
        if candidate.signals.completeness == 0.0:
            candidate.disqualified = True
            candidate.disqualified_reason = (
                "no growth signal could be measured from any source"
            )

    return sorted(
        candidates,
        key=lambda c: (not c.disqualified, c.emergence_index),
        reverse=True,
    )
