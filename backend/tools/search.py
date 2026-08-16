"""Real-time web search, with a substrate that survives losing it.

The assessment requires the Domain Scout to use real-time search, naming
Tavily among the acceptable free options. Tavily is therefore the primary
path. But a demo whose central claim is "fully autonomous" cannot fall over
because one API key hit its monthly ceiling on the morning a reviewer opens
the link, so search degrades in tiers:

    1. Tavily          -- real-time web search, 1000 free credits/month
    2. Hacker News     -- Algolia's free unauthenticated index, no key at all
    3. arXiv listings  -- the literature itself, already required elsewhere

Tiers 2 and 3 are narrower than tier 1, not equivalent to it, and the Scout is
told which tier answered so that confidence drops honestly when running
degraded rather than pretending the evidence is as strong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from ..config import Settings, get_settings
from .http import Fetcher, FetchError

log = logging.getLogger(__name__)

TAVILY_SEARCH = "https://api.tavily.com/search"
HN_SEARCH = "https://hn.algolia.com/api/v1/search"


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""
    score: float = 0.0
    published: str = ""
    source: str = ""


@dataclass
class SearchResult:
    query: str
    hits: list[SearchHit] = field(default_factory=list)
    tier: str = "none"
    degraded: bool = False
    ok: bool = False
    error: str = ""

    @property
    def urls(self) -> list[str]:
        return [h.url for h in self.hits if h.url]


class SearchClient:
    def __init__(self, fetcher: Fetcher, settings: Settings | None = None) -> None:
        self.fetcher = fetcher
        self.settings = settings or get_settings()

    @property
    def has_tavily(self) -> bool:
        return bool(self.settings.tavily_api_key)

    async def search(
        self,
        query: str,
        *,
        max_results: int = 8,
        days: int | None = None,
        depth: str = "basic",
    ) -> SearchResult:
        """Search, falling through tiers until something answers."""
        if self.has_tavily:
            result = await self._tavily(query, max_results=max_results, days=days, depth=depth)
            if result.ok:
                return result
            log.warning("Tavily unavailable (%s); degrading to free tiers", result.error)

        result = await self._hacker_news(query, max_results=max_results, days=days)
        if result.ok and result.hits:
            result.degraded = True
            return result

        return SearchResult(
            query=query,
            tier="none",
            degraded=True,
            ok=False,
            error="no search backend answered",
        )

    # ------------------------------------------------------------ tier 1

    async def _tavily(
        self, query: str, *, max_results: int, days: int | None, depth: str
    ) -> SearchResult:
        result = SearchResult(query=query, tier="tavily")
        payload: dict = {
            "query": query,
            "max_results": max_results,
            # "advanced" costs 2 credits against a 1000/month budget; a run
            # makes dozens of searches, so basic is the default and depth is
            # spent only where the Scout asks for it.
            "search_depth": depth,
            "include_answer": False,
            "include_raw_content": False,
        }
        if days:
            payload["days"] = days
            payload["topic"] = "news"

        try:
            data = await self.fetcher.post_json(
                TAVILY_SEARCH,
                payload,
                headers={"Authorization": f"Bearer {self.settings.tavily_api_key}"},
                max_retries=1,
            )
        except FetchError as exc:
            result.error = str(exc)
            return result

        for item in data.get("results", []):
            result.hits.append(
                SearchHit(
                    title=item.get("title", "")[:300],
                    url=item.get("url", ""),
                    snippet=(item.get("content") or "")[:1000],
                    score=float(item.get("score") or 0.0),
                    published=item.get("published_date") or "",
                    source="tavily",
                )
            )
        result.ok = True
        return result

    # ------------------------------------------------------------ tier 2

    async def _hacker_news(
        self, query: str, *, max_results: int, days: int | None
    ) -> SearchResult:
        """Free, keyless, and a genuinely useful signal for emerging technical
        topics -- which is why it is also a standalone emergence signal."""
        result = SearchResult(query=query, tier="hackernews")
        params: dict = {"query": query, "tags": "story", "hitsPerPage": max_results}
        if days:
            since = int((datetime.now(UTC) - timedelta(days=days)).timestamp())
            params["numericFilters"] = f"created_at_i>{since}"

        try:
            data = await self.fetcher.get_json(HN_SEARCH, params=params, max_retries=1)
        except FetchError as exc:
            result.error = str(exc)
            return result

        for item in data.get("hits", []):
            url = item.get("url") or f"https://news.ycombinator.com/item?id={item.get('objectID')}"
            result.hits.append(
                SearchHit(
                    title=(item.get("title") or item.get("story_title") or "")[:300],
                    url=url,
                    snippet=(item.get("story_text") or "")[:1000],
                    score=float(item.get("points") or 0),
                    published=item.get("created_at") or "",
                    source="hackernews",
                )
            )
        result.ok = True
        return result

    # -------------------------------------------------- emergence signal

    async def mention_count(self, term: str, *, days: int = 540) -> tuple[int, bool]:
        """How often a term surfaced in technical discussion recently.

        Returns (count, ok). Used as the Emergence Index's attention signal --
        a recency tiebreaker between fields with similar publication growth.
        """
        since = int((datetime.now(UTC) - timedelta(days=days)).timestamp())
        try:
            data = await self.fetcher.get_json(
                HN_SEARCH,
                params={
                    "query": term,
                    "tags": "story",
                    "numericFilters": f"created_at_i>{since}",
                    "hitsPerPage": 1,
                },
                max_retries=1,
            )
        except FetchError as exc:
            log.warning("mention count failed for %r: %s", term, exc)
            return 0, False
        return int(data.get("nbHits", 0)), True

    async def is_directly_answerable(self, question: str) -> tuple[bool, str, str]:
        """Probe whether a question already has a findable answer.

        This is the enforcement behind the assessment's "not directly
        searchable" requirement. Returns (answerable, url, snippet). The
        judgement of whether a hit *actually* answers the question is left to
        the calling agent -- this only supplies the evidence.
        """
        result = await self.search(question, max_results=5, depth="basic")
        if not result.ok or not result.hits:
            return False, "", ""
        best = max(result.hits, key=lambda h: h.score)
        return True, best.url, best.snippet
