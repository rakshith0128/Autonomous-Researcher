"""Measure one candidate domain across every evidence source.

This is the Domain Scout's instrument panel, kept separate from the agent so it
can be tested and run without an LLM in the loop.

Sources are queried concurrently because they are different hosts: arXiv's
mandatory 3-second gap delays only arXiv, while OpenAlex, GitHub, and Hacker
News proceed in parallel. Measuring five candidates serially against four
sources would cost minutes; this brings it to roughly the cost of the slowest
single source.

Every source failure is caught and recorded rather than raised. A domain
measured on three of four sources is still rankable -- `measured_ok` carries
which ones answered, and the Emergence Index imputes the rest at the mean
rather than scoring them as zero.
"""

from __future__ import annotations

import asyncio
import logging

from ..schemas import DomainCandidate, DomainProposal, EmergenceSignals
from .arxiv import ArxivClient
from .github import GithubClient
from .http import Fetcher
from .openalex import OpenAlexClient
from .search import SearchClient

log = logging.getLogger(__name__)


class DomainMeasurer:
    def __init__(self, fetcher: Fetcher) -> None:
        self.arxiv = ArxivClient(fetcher)
        self.openalex = OpenAlexClient(fetcher)
        self.github = GithubClient(fetcher)
        self.search = SearchClient(fetcher)
        self._probe_cache: dict[str, int] = {}

    # Recent works below this and the phrase is too narrow to compute a
    # meaningful growth ratio from -- a baseline of 0 against a recent count of
    # 2 is noise, not evidence.
    MIN_EVIDENCE = 10

    async def _probe(self, term: str) -> int:
        """Cheap check of whether a phrase has any literature behind it.

        OpenAlex rather than arXiv because arXiv's mandatory 3-second gap would
        cost ~15 seconds per candidate just to choose a search term.

        Results are cached for the process lifetime. OpenAlex meters anonymous
        clients on a *daily budget* rather than a request rate -- exhausting it
        returns "Insufficient budget, resets at midnight UTC" -- so the number
        of distinct requests matters far more than how fast they are sent.
        Candidates frequently share phrases, and paying for the same probe
        twice is pure waste.
        """
        cached = self._probe_cache.get(term.lower())
        if cached is not None:
            return cached
        try:
            count, _ = await self.openalex._count(term, "from_publication_date:2024-01-01")
        except Exception:  # noqa: BLE001 - a failed probe just scores zero
            count = 0
        self._probe_cache[term.lower()] = count
        return count

    async def _select_term(self, proposal: DomainProposal) -> tuple[str, list[str]]:
        """Pick the phrase that actually has literature behind it.

        Models asked for specificity reliably over-correct into phrases so
        narrow that every source returns zero -- "indefinite causal order
        quantum communication" has no hits, while its core concept
        "indefinite causal order" has hundreds. Scoring the first suggestion
        blindly would rank five real domains at exactly zero and make the
        Emergence Index meaningless.

        So every supplied phrase is probed, plus shortened variants of the
        long ones, and the first phrase carrying real evidence wins. Returns
        (chosen_term, all_probed).
        """
        supplied = [t.strip() for t in proposal.search_terms if t.strip()] or [proposal.name]

        variants: list[str] = []
        for term in supplied:
            variants.append(term)
        # Head and tail trigrams of any long phrase: the core concept is
        # usually one end or the other, and probing both is cheap.
        for term in supplied:
            words = term.split()
            if len(words) > 3:
                variants.append(" ".join(words[:3]))
                variants.append(" ".join(words[-3:]))
        if proposal.name not in variants:
            variants.append(proposal.name)

        # Capped hard. Every probe spends OpenAlex daily budget, and the Scout
        # runs this for five candidates at once -- eight variants each would
        # burn forty requests before any real measurement began.
        seen: set[str] = set()
        ordered = [v for v in variants if not (v.lower() in seen or seen.add(v.lower()))][:3]

        counts = await asyncio.gather(*(self._probe(v) for v in ordered))
        scored = list(zip(ordered, counts, strict=True))

        # Prefer the earliest phrase clearing the bar: the model's own first
        # choice is the most faithful label for the domain it meant.
        for term, count in scored:
            if count >= self.MIN_EVIDENCE:
                return term, ordered

        best_term, best_count = max(scored, key=lambda pair: pair[1])
        if best_count > 0:
            return best_term, ordered
        return supplied[0], ordered

    async def measure(self, proposal: DomainProposal) -> DomainCandidate:
        """Gather every growth signal for one proposed domain."""
        term, probed = await self._select_term(proposal)

        arxiv_task = self.arxiv.growth(term)
        openalex_task = self.openalex.measure(term)
        github_task = self.github.measure(term)
        forum_task = self.search.mention_count(term)

        arxiv, openalex, github, forum = await asyncio.gather(
            arxiv_task, openalex_task, github_task, forum_task, return_exceptions=True
        )

        signals = EmergenceSignals(
            sources_consulted=["arxiv", "openalex", "github", "forum"],
            term_used=term,
            terms_probed=probed,
        )
        evidence: list[str] = []

        # --- arXiv -----------------------------------------------------------
        if _ok(arxiv):
            signals.arxiv_recent_count = arxiv.recent_count
            signals.arxiv_baseline_count = arxiv.baseline_count
            signals.arxiv_growth_ratio = arxiv.growth_ratio
            # The *relative* slope is what the index consumes: absolute
            # papers-per-month would rank any large field above any small one
            # regardless of whether either is accelerating.
            signals.arxiv_monthly_slope = arxiv.relative_slope
            signals.measured_ok["arxiv"] = arxiv.ok
            evidence.extend(p.abs_url for p in arxiv.papers[:3] if p.abs_url)
        else:
            signals.measured_ok["arxiv"] = False
            _log_failure("arxiv", term, arxiv)

        # --- OpenAlex --------------------------------------------------------
        if _ok(openalex):
            signals.openalex_recent_works = openalex.recent_works
            signals.openalex_baseline_works = openalex.baseline_works
            signals.openalex_growth_ratio = openalex.growth_ratio
            signals.openalex_citation_velocity = openalex.citation_velocity
            signals.measured_ok["openalex"] = openalex.ok
            evidence.extend(w.url for w in openalex.top_works[:2] if w.url)
        else:
            signals.measured_ok["openalex"] = False
            _log_failure("openalex", term, openalex)

        # --- GitHub ----------------------------------------------------------
        if _ok(github):
            signals.github_repos_created_post_cutoff = github.repos_created_post_cutoff
            signals.github_total_stars = github.total_stars
            signals.github_star_velocity = github.star_velocity
            signals.measured_ok["github"] = github.ok
            evidence.extend(r.url for r in github.top_repos[:2] if r.url)
        else:
            signals.measured_ok["github"] = False
            _log_failure("github", term, github)

        # --- public attention -------------------------------------------------
        if isinstance(forum, tuple):
            count, ok = forum
            signals.forum_mentions = count
            signals.measured_ok["forum"] = ok
        else:
            signals.measured_ok["forum"] = False
            _log_failure("forum", term, forum)

        return DomainCandidate(
            proposal=proposal,
            signals=signals,
            evidence_urls=evidence[:8],
        )

    async def measure_all(self, proposals: list[DomainProposal]) -> list[DomainCandidate]:
        """Measure every candidate.

        Candidates run concurrently too. arXiv's per-host lock serialises the
        arXiv leg automatically, so this stays polite without any coordination
        here.
        """
        results = await asyncio.gather(
            *(self.measure(p) for p in proposals), return_exceptions=True
        )

        candidates: list[DomainCandidate] = []
        for proposal, result in zip(proposals, results, strict=True):
            if isinstance(result, DomainCandidate):
                candidates.append(result)
            else:
                log.warning("measurement crashed for %r: %s", proposal.name, result)
                candidates.append(
                    DomainCandidate(
                        proposal=proposal,
                        signals=EmergenceSignals(
                            measured_ok=dict.fromkeys(
                                ("arxiv", "openalex", "github", "forum"), False
                            )
                        ),
                        disqualified=True,
                        disqualified_reason=f"measurement failed: {result}",
                    )
                )
        return candidates


def _ok(result: object) -> bool:
    return not isinstance(result, BaseException) and getattr(result, "ok", False)


def _log_failure(source: str, term: str, result: object) -> None:
    detail = result if isinstance(result, BaseException) else getattr(result, "error", "unknown")
    log.warning("%s measurement unavailable for %r: %s", source, term, detail)
