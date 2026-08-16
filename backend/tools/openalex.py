"""OpenAlex: scholarly output volume and citation velocity.

OpenAlex indexes far more than arXiv -- journals, conferences, and every
discipline rather than just the preprint-heavy ones -- so it acts as the
cross-check on arXiv's view. A domain that looks explosive on arXiv but flat on
OpenAlex is probably a preprint fashion rather than a field, and the Emergence
Index should see both numbers.

Free, no key, no registration. Supplying a contact address in the User-Agent
moves requests into the faster "polite pool", which the Fetcher does globally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .arxiv import EMERGENCE_CUTOFF
from .http import Fetcher, FetchError

log = logging.getLogger(__name__)

OPENALEX_WORKS = "https://api.openalex.org/works"


@dataclass
class Work:
    title: str
    doi: str
    year: int
    cited_by_count: int
    publication_date: str
    url: str = ""

    @property
    def age_years(self) -> float:
        return max(datetime.now(UTC).year - self.year + 0.5, 0.5)

    @property
    def citations_per_year(self) -> float:
        return self.cited_by_count / self.age_years


@dataclass
class OpenAlexSignal:
    term: str
    recent_works: int = 0
    baseline_works: int = 0
    citation_velocity: float = 0.0
    top_works: list[Work] = field(default_factory=list)
    ok: bool = False
    error: str = ""

    @property
    def growth_ratio(self) -> float:
        return self.recent_works / max(self.baseline_works, 1)


class OpenAlexClient:
    def __init__(self, fetcher: Fetcher) -> None:
        self.fetcher = fetcher

    def _polite(self, params: dict) -> dict:
        """Add the polite-pool identifier.

        OpenAlex accepts `mailto` either in the User-Agent (which the Fetcher
        sets globally) or as a query parameter. The query parameter is the
        form its documentation leads with, so both are sent.

        Worth knowing: the free allowance is a *daily budget scoped to your IP*,
        not a request rate. Once spent it returns "Insufficient budget ...
        resets at midnight UTC" and no amount of identification restores it
        until then. That is why the Data Alchemist carries an arXiv fallback
        rather than treating OpenAlex as guaranteed.
        """
        email = self.fetcher.settings.contact_email
        return {**params, "mailto": email} if email else params

    @staticmethod
    def _phrase_filter(term: str) -> str:
        """Build a precise phrase filter.

        Query precision dominates every number this module produces. Measured
        against "mechanistic interpretability" post-2024:

            search=<term>                          166,808 works
            search="<term>"                          7,927
            title_and_abstract.search:<term>        11,881
            title_and_abstract.search:"<term>"       2,745   <-- used

        The default `search` parameter scores loosely across full text, so it
        matches nearly anything containing either word and reports a field 60x
        larger than it is. Restricting to title and abstract *and* quoting the
        phrase is what makes the recent-vs-baseline ratio mean something.

        Commas are stripped because OpenAlex uses them to separate filters.
        """
        cleaned = " ".join(term.replace(",", " ").replace('"', " ").split())
        return f'title_and_abstract.search:"{cleaned}"'

    async def _count(self, term: str, date_filter: str) -> tuple[int, list[dict]]:
        data = await self.fetcher.get_json(
            OPENALEX_WORKS,
            params=self._polite(
                {
                    "filter": f"{self._phrase_filter(term)},{date_filter}",
                    "per-page": 25,
                    "sort": "cited_by_count:desc",
                }
            ),
        )
        return int(data.get("meta", {}).get("count", 0)), data.get("results", [])

    async def measure(
        self, term: str, *, cutoff: datetime = EMERGENCE_CUTOFF
    ) -> OpenAlexSignal:
        """Compare post-cutoff output against an equal-length prior window."""
        signal = OpenAlexSignal(term=term)
        now = datetime.now(UTC)
        span_days = (now - cutoff).days

        try:
            signal.recent_works, results = await self._count(
                term, f"from_publication_date:{cutoff:%Y-%m-%d}"
            )

            baseline_start = cutoff - (now - cutoff)
            signal.baseline_works, _ = await self._count(
                term,
                f"from_publication_date:{baseline_start:%Y-%m-%d},"
                f"to_publication_date:{cutoff:%Y-%m-%d}",
            )

            signal.top_works = [w for w in (_parse_work(r) for r in results) if w]
            if signal.top_works:
                # Mean citations per year across the most-cited recent works.
                # Papers in a genuinely active field accrue citations quickly
                # despite being new, which volume alone would not reveal.
                signal.citation_velocity = sum(
                    w.citations_per_year for w in signal.top_works
                ) / len(signal.top_works)

            signal.ok = True
            log.debug(
                "OpenAlex %r: %d recent / %d baseline over %d days",
                term,
                signal.recent_works,
                signal.baseline_works,
                span_days,
            )
        except FetchError as exc:
            signal.error = str(exc)
            log.warning("OpenAlex measurement failed for %r: %s", term, exc)

        return signal

    async def works_for_question(self, term: str, *, limit: int = 25) -> list[Work]:
        """Structured records for the Data Alchemist's JSON/API modality."""
        try:
            _, results = await self._count(term, f"from_publication_date:{EMERGENCE_CUTOFF:%Y-%m-%d}")
        except FetchError as exc:
            log.warning("OpenAlex work fetch failed for %r: %s", term, exc)
            return []
        return [w for w in (_parse_work(r) for r in results) if w][:limit]


def _parse_work(item: dict) -> Work | None:
    title = item.get("title") or item.get("display_name") or ""
    if not title:
        return None
    try:
        year = int(item.get("publication_year") or 0)
    except (TypeError, ValueError):
        year = 0
    if not year:
        return None

    return Work(
        title=title[:300],
        doi=(item.get("doi") or "").replace("https://doi.org/", ""),
        year=year,
        cited_by_count=int(item.get("cited_by_count") or 0),
        publication_date=item.get("publication_date") or "",
        url=item.get("id", ""),
    )
