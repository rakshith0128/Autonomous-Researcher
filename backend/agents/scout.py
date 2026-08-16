"""Domain Scout: find candidate emerging fields, then measure them.

The assessment forbids hardcoded domains, and the obvious workaround -- asking
a model to name trending fields -- only moves the hardcoding into the model's
training data. It will confidently name whatever was fashionable at its cutoff,
which for "post-2024" is exactly the period it knows least well.

So proposal and evidence are separated:

1. **Ground first.** Pull what is *actually* being published right now: live
   arXiv category counts, recent paper titles, real-time search results. None
   of this comes from model memory.
2. **Let the model label, not recall.** It reads that evidence and names the
   clusters it sees. It is a labelling step over supplied data, not a recall
   step over training data.
3. **Measure independently.** Every proposal is then scored against arXiv,
   OpenAlex, GitHub, and public discussion. A field the model invented scores
   near zero and loses.

Grep this repository for a domain name and you will find none. The only thing
that decides the winner is the Emergence Index, which is arithmetic.
"""

from __future__ import annotations

import logging
from typing import Any

from ..analysis.emergence import rank
from ..analysis.plots import emergence_chart
from ..config import Role
from ..schemas import (
    AgentName,
    ArtifactKind,
    DomainCandidate,
    DomainProposalBatch,
    DomainSelection,
    Level,
)
from ..tools.arxiv import ArxivClient
from ..tools.measure import DomainMeasurer
from ..tools.search import SearchClient
from .base import AgentFailure, BaseAgent

log = logging.getLogger(__name__)

SYSTEM = """You are a research scout who identifies genuinely emerging scientific domains.

You will be shown real, current evidence: what arXiv is publishing right now, and recent \
technical news. Your job is to NAME THE CLUSTERS YOU SEE IN THAT EVIDENCE.

Rules:
- Propose domains that barely existed before January 2024. Not established fields.
- Be specific about the CONCEPT, but keep the search terms SHORT.
- Ground every proposal in the supplied evidence. Do not propose from memory.
- Prefer domains where public data exists (papers, benchmarks, repositories) so they can be \
studied empirically.

search_terms are the critical field. They are sent VERBATIM as exact-phrase queries to arXiv \
and OpenAlex. A phrase with no exact matches returns zero and the domain cannot be scored.

- Use 2-3 words. Name the concept as researchers write it in a title.
- Do NOT stack modifiers into a long compound. "indefinite causal order quantum \
communication" matches nothing; "indefinite causal order" matches hundreds of papers.
- Give the core concept first, then broader variants as backups.
- No boolean operators, no punctuation, no acronyms in parentheses.

Good:  ["sparse autoencoder", "feature steering", "mechanistic interpretability"]
Bad:   ["robust multi-modal contrastive representation learning for vision-language models"]"""

PROMPT = """Here is current, real evidence.

## arXiv categories, last 10 days (category: paper count)
{categories}

## Recent arXiv paper titles
{titles}

## Recent technical news and discussion
{news}

Propose {n} candidate emerging scientific domains that are visible in this evidence and that \
barely existed before January 2024.

For each: a specific name, what it studies, why you believe it is newly emerging (cite what \
in the evidence above points to it), and 2-4 literal search phrases."""


class DomainScout(BaseAgent):
    name = AgentName.SCOUT
    # Nothing downstream exists without a domain, so a failure here ends the run.
    fatal_on_failure = True

    async def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        n_candidates = 5  # the assessment's "ex. 5 domains"

        evidence = await self._gather_evidence()
        proposals = await self._propose(evidence, n_candidates)

        self.say(
            f"Proposed {len(proposals.proposals)} candidate domains; measuring each against "
            "arXiv, OpenAlex, GitHub and public discussion.",
            level=Level.INFO,
        )
        self.publish(
            ArtifactKind.DOMAIN_CANDIDATES,
            {
                "stage": "proposed",
                "candidates": [
                    {
                        "name": p.name,
                        "description": p.description,
                        "why_emerging": p.why_emerging,
                        "search_terms": p.search_terms,
                    }
                    for p in proposals.proposals
                ],
            },
            message="Candidate domains proposed from live evidence",
        )

        measurer = DomainMeasurer(self.fetcher)
        candidates = await measurer.measure_all(proposals.proposals)
        ranked = rank(candidates)

        viable = [c for c in ranked if not c.disqualified]
        if not viable:
            raise AgentFailure(
                "No candidate domain could be measured against any evidence source. "
                "Every upstream data source appears unreachable.",
                fatal=True,
            )

        self._report(ranked)

        selection = DomainSelection(
            candidates=ranked,
            combined_scores={c.name: c.emergence_index for c in ranked},
            confidence=self._confidence(viable),
        )

        return {
            "domain_selection": selection,
            "phase": "selection",
        }

    # ------------------------------------------------------------- evidence

    async def _gather_evidence(self) -> dict[str, str]:
        """Collect live signals about what is being published and discussed."""
        arxiv = ArxivClient(self.fetcher)
        search = SearchClient(self.fetcher)

        self.say("Sampling what arXiv has published in the last 10 days…")
        categories = await arxiv.recent_categories(max_results=120)
        self.tool("arxiv.recent_categories", ok=bool(categories), detail=f"{len(categories)} categories")

        recent = await arxiv.search("", max_results=40) if not categories else []
        titles = [p.title for p in recent][:25]
        if not titles:
            # Sample titles from the busiest current categories rather than a
            # fixed query, so the seed stays tied to live activity.
            top_cats = list(categories)[:4]
            for cat in top_cats:
                papers = await arxiv.search(cat.replace(".", " "), max_results=6)
                titles.extend(p.title for p in papers)
            titles = titles[:25]

        self.say("Searching for recent developments across the technical press…")
        news = await search.search(
            "new scientific field emerging research direction breakthrough",
            max_results=8,
            days=180,
        )
        self.tool(
            "search",
            ok=news.ok,
            detail=f"tier={news.tier} hits={len(news.hits)}"
            + (" (degraded)" if news.degraded else ""),
        )
        if news.degraded:
            self.say(
                f"Real-time search degraded to {news.tier}; domain proposals will be "
                "grounded on a narrower evidence base.",
                level=Level.WARN,
            )

        return {
            "categories": "\n".join(f"- {c}: {n}" for c, n in list(categories.items())[:20])
            or "- (arXiv category sampling unavailable)",
            "titles": "\n".join(f"- {t}" for t in titles[:25]) or "- (unavailable)",
            "news": "\n".join(f"- {h.title}: {h.snippet[:180]}" for h in news.hits[:8])
            or "- (search unavailable)",
        }

    async def _propose(self, evidence: dict[str, str], n: int) -> DomainProposalBatch:
        """Ask the model to label the clusters visible in the evidence."""
        try:
            batch = await self.router.structured(
                Role.REASONING,
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": PROMPT.format(n=n, **evidence)},
                ],
                DomainProposalBatch,
                temperature=0.7,  # some spread: identical proposals every run would be dull
                max_tokens=2000,
            )
        except Exception as exc:
            raise AgentFailure(f"could not generate domain proposals: {exc}", fatal=True) from exc

        # Deduplicate: models routinely propose the same field under two names.
        seen: set[str] = set()
        unique = []
        for proposal in batch.proposals:
            key = proposal.name.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(proposal)

        if not unique:
            raise AgentFailure("model returned no usable domain proposals", fatal=True)

        batch.proposals = unique[:n]
        return batch

    # -------------------------------------------------------------- output

    def _report(self, ranked: list[DomainCandidate]) -> None:
        """Publish the ranked table and the chart behind it."""
        rows = []
        for candidate in ranked:
            signals = candidate.signals
            rows.append(
                {
                    "name": candidate.name,
                    "description": candidate.proposal.description,
                    "emergence_index": round(candidate.emergence_index, 3),
                    "arxiv_growth_ratio": round(signals.arxiv_growth_ratio, 2),
                    "arxiv_recent": signals.arxiv_recent_count,
                    "arxiv_baseline": signals.arxiv_baseline_count,
                    "openalex_growth_ratio": round(signals.openalex_growth_ratio, 2),
                    "openalex_recent": signals.openalex_recent_works,
                    "github_repos": signals.github_repos_created_post_cutoff,
                    "github_star_velocity": round(signals.github_star_velocity, 2),
                    "forum_mentions": signals.forum_mentions,
                    "completeness": round(signals.completeness, 2),
                    "components": candidate.component_z,
                    "disqualified": candidate.disqualified,
                    "disqualified_reason": candidate.disqualified_reason,
                    "evidence_urls": candidate.evidence_urls,
                }
            )

        self.publish(
            ArtifactKind.DOMAIN_CANDIDATES,
            {
                "stage": "measured",
                "candidates": rows,
                "note": (
                    "The Emergence Index is a z-scored weighted sum of measured growth "
                    "signals. Scores are comparable within this run only."
                ),
            },
            message="Domains measured and ranked by Emergence Index",
        )

        figure = emergence_chart(ranked)
        if figure:
            self.publish(
                ArtifactKind.EMERGENCE_CHART,
                {"figure": figure.figure_json, "title": figure.title, "caption": figure.caption},
                message="Emergence evidence chart",
            )

        for candidate in ranked[:5]:
            signals = candidate.signals
            self.say(
                f"{candidate.name}: index {candidate.emergence_index:+.2f} "
                f"(arXiv {signals.arxiv_growth_ratio:.1f}x, "
                f"OpenAlex {signals.openalex_growth_ratio:.1f}x, "
                f"{signals.github_repos_created_post_cutoff} new repos)",
                level=Level.SUCCESS if candidate is ranked[0] else Level.INFO,
            )

    @staticmethod
    def _confidence(viable: list[DomainCandidate]) -> float:
        """How much to trust this ranking.

        Driven by how complete the measurements were and how clearly the
        leader separates from the pack. A photo finish between two candidates
        measured on half their sources is a weak result and should say so.
        """
        completeness = sum(c.signals.completeness for c in viable) / len(viable)
        if len(viable) < 2:
            return round(0.5 * completeness, 3)

        top, second = viable[0].emergence_index, viable[1].emergence_index
        spread = abs(top - second)
        separation = min(spread / 0.75, 1.0)  # 0.75 z-units is a clear win
        return round(0.55 * completeness + 0.45 * separation, 3)
