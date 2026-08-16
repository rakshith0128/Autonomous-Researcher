"""arXiv API access: publication counts, growth curves, and full texts.

Two jobs, with different callers:

* The **Domain Scout** needs *counts over time* to measure whether a field is
  actually growing. It never reads the papers.
* The **Data Alchemist** needs *full texts and PDFs* for the chosen question.

Request budget shapes the design. arXiv asks for a 3-second gap between calls,
so naively fetching twelve monthly counts per candidate domain would cost 36
seconds per candidate and blow the run's time budget on bookkeeping. Instead a
single windowed search returns up to 200 recent entries, and the monthly
histogram is built locally from their submission dates -- one request instead
of twelve, for the same curve.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from .http import Fetcher, FetchError

log = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"

# The assessment defines "emerging" as post-January-2024, so that date is the
# boundary between the growth window and the historical baseline everywhere.
EMERGENCE_CUTOFF = datetime(2024, 1, 1, tzinfo=UTC)


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    summary: str
    authors: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    published: datetime | None = None
    updated: datetime | None = None
    pdf_url: str = ""
    abs_url: str = ""

    @property
    def year_month(self) -> str:
        return self.published.strftime("%Y-%m") if self.published else ""


@dataclass
class ArxivGrowth:
    """Publication-volume evidence for one search term."""

    query: str
    recent_count: int = 0
    baseline_count: int = 0
    monthly: dict[str, int] = field(default_factory=dict)
    monthly_complete: dict[str, int] = field(default_factory=dict)
    slope: float = 0.0
    relative_slope: float = 0.0
    slope_measurable: bool = False
    papers: list[ArxivPaper] = field(default_factory=list)
    sample_truncated: bool = False
    ok: bool = False
    error: str = ""

    @property
    def growth_ratio(self) -> float:
        """Recent volume relative to the pre-cutoff baseline.

        A brand-new field has a baseline of zero, which would divide by zero
        and, worse, is exactly the case we care most about. Treating an absent
        baseline as 1 makes a field with no prior history score as its raw
        recent volume -- high, which is correct, and finite, which is required.
        """
        return self.recent_count / max(self.baseline_count, 1)


def _quote(term: str) -> str:
    """Prepare a search term for arXiv's query grammar.

    Unquoted, arXiv ORs the words together, so 'graph neural network' matches
    nearly every ML paper ever posted. Whitespace is collapsed as well as
    stripped: a stray double space inside the quotes is treated as a distinct
    phrase and silently returns zero results.
    """
    cleaned = " ".join(term.replace('"', " ").split())
    return f'"{cleaned}"' if " " in cleaned else cleaned


def _window(start: datetime, end: datetime) -> str:
    return f"submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"


def _parse_entry(entry) -> ArxivPaper:  # noqa: ANN001 - feedparser returns untyped objects
    def _dt(value) -> datetime | None:  # noqa: ANN001
        if not value:
            return None
        try:
            return datetime(*value[:6], tzinfo=UTC)
        except (TypeError, ValueError):
            return None

    abs_url = entry.get("id", "")
    arxiv_id = abs_url.rsplit("/", 1)[-1] if abs_url else ""
    pdf_url = next(
        (link.href for link in entry.get("links", []) if link.get("type") == "application/pdf"),
        abs_url.replace("/abs/", "/pdf/") if abs_url else "",
    )

    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=" ".join(entry.get("title", "").split()),
        summary=" ".join(entry.get("summary", "").split()),
        authors=[a.get("name", "") for a in entry.get("authors", [])],
        categories=[t.get("term", "") for t in entry.get("tags", [])],
        published=_dt(entry.get("published_parsed")),
        updated=_dt(entry.get("updated_parsed")),
        pdf_url=pdf_url,
        abs_url=abs_url,
    )


class ArxivClient:
    def __init__(self, fetcher: Fetcher) -> None:
        self.fetcher = fetcher

    async def _query(
        self,
        search_query: str,
        *,
        max_results: int = 100,
        sort_by: str = "submittedDate",
    ) -> tuple[int, list[ArxivPaper]]:
        """Run one API call, returning (total_matching, parsed_entries).

        `total_matching` comes from the OpenSearch header and reflects the
        whole result set, not the page -- which is what makes counting cheap.
        """
        import feedparser

        result = await self.fetcher.fetch(
            ARXIV_API,
            params={
                "search_query": search_query,
                "start": 0,
                "max_results": max_results,
                "sortBy": sort_by,
                "sortOrder": "descending",
            },
        )
        parsed = feedparser.parse(result.content)
        try:
            total = int(parsed.feed.get("opensearch_totalresults", 0))
        except (TypeError, ValueError):
            total = len(parsed.entries)
        return total, [_parse_entry(e) for e in parsed.entries]

    # Minimum fully-observed months before a slope means anything. Two points
    # define a line through any two months of noise; three is the floor at
    # which "trend" is a defensible word.
    MIN_MONTHS_FOR_SLOPE = 3

    async def growth(
        self,
        term: str,
        *,
        cutoff: datetime = EMERGENCE_CUTOFF,
        max_results: int = 1000,
    ) -> ArxivGrowth:
        """Measure how fast a term's publication volume is rising.

        Two requests: one for the post-cutoff window (which also supplies the
        monthly histogram) and one for the equivalent-length window before it.
        """
        now = datetime.now(UTC)
        quoted = _quote(term)
        growth = ArxivGrowth(query=term)

        try:
            recent_q = f"all:{quoted} AND {_window(cutoff, now)}"
            growth.recent_count, growth.papers = await self._query(
                recent_q, max_results=max_results
            )

            # Baseline: the same span immediately before the cutoff, so the two
            # numbers are comparable rather than "recent months vs all history".
            span = now - cutoff
            baseline_q = f"all:{quoted} AND {_window(cutoff - span, cutoff)}"
            growth.baseline_count, _ = await self._query(baseline_q, max_results=1)

            growth.monthly = self._histogram(growth.papers)
            growth.sample_truncated = len(growth.papers) >= max_results
            growth.monthly_complete = _fully_observed_months(
                growth.monthly, truncated=growth.sample_truncated
            )

            observed = growth.monthly_complete
            growth.slope_measurable = len(observed) >= self.MIN_MONTHS_FOR_SLOPE
            if growth.slope_measurable:
                growth.slope = _linear_slope(observed)
                # Normalise by the term's own mean volume. Without this, a field
                # averaging 80 papers/month outranks one averaging 5 purely by
                # being larger, when the Scout is looking for *acceleration*.
                mean = sum(observed.values()) / len(observed)
                growth.relative_slope = growth.slope / mean if mean else 0.0
            growth.ok = True
        except FetchError as exc:
            # A dead source degrades the score honestly rather than failing the
            # run; EmergenceSignals.measured_ok records that this was missing.
            growth.error = str(exc)
            log.warning("arXiv growth measurement failed for %r: %s", term, exc)

        return growth

    @staticmethod
    def _histogram(papers: list[ArxivPaper]) -> dict[str, int]:
        counts = Counter(p.year_month for p in papers if p.year_month)
        return dict(sorted(counts.items()))

    async def search(
        self,
        term: str,
        *,
        max_results: int = 25,
        since: datetime | None = None,
        categories: list[str] | None = None,
    ) -> list[ArxivPaper]:
        """Fetch papers for the Data Alchemist to actually read."""
        clauses = [f"all:{_quote(term)}"]
        if since:
            clauses.append(_window(since, datetime.now(UTC)))
        if categories:
            cats = " OR ".join(f"cat:{c}" for c in categories)
            clauses.append(f"({cats})")

        try:
            _, papers = await self._query(" AND ".join(clauses), max_results=max_results)
            return papers
        except FetchError as exc:
            log.warning("arXiv search failed for %r: %s", term, exc)
            return []

    async def recent_categories(self, max_results: int = 100) -> dict[str, int]:
        """Category frequencies across the newest submissions.

        Feeds the Scout's seed generation: rather than asking a model what is
        trending and trusting the answer, we show it what arXiv has actually
        been publishing this week and let it name the clusters. The literature
        proposes the topics; the model only labels them.
        """
        window = datetime.now(UTC) - timedelta(days=10)
        try:
            _, papers = await self._query(
                f"{_window(window, datetime.now(UTC))}",
                max_results=max_results,
            )
        except FetchError as exc:
            log.warning("arXiv category sampling failed: %s", exc)
            return {}
        return dict(Counter(c for p in papers for c in p.categories).most_common(30))

    async def fetch_pdf(self, paper: ArxivPaper) -> bytes:
        """Download a paper's PDF for text, table, and figure extraction."""
        result = await self.fetcher.fetch(paper.pdf_url or f"https://arxiv.org/pdf/{paper.arxiv_id}")
        return result.content


def _fully_observed_months(monthly: dict[str, int], *, truncated: bool) -> dict[str, int]:
    """Keep only months the sample actually observed in full.

    Two ends of the histogram lie, and both lie downward, which would make a
    fast-growing field look like a shrinking one:

    * The **current month** is still in progress. On the 15th it holds roughly
      half its eventual papers.
    * The **oldest month in the sample** is clipped whenever the result cap was
      hit -- we asked for the newest 200 entries, so the far end of the window
      is a partial slice of that month rather than all of it.

    Dropping both is why the reported slope tracks the growth ratio instead of
    contradicting it.
    """
    if not monthly:
        return {}

    current = datetime.now(UTC).strftime("%Y-%m")
    keys = [k for k in sorted(monthly) if k != current]
    if truncated and keys:
        keys = keys[1:]
    return {k: monthly[k] for k in keys}


def _linear_slope(monthly: dict[str, int]) -> float:
    """Least-squares slope of counts per month, in papers per month.

    Deliberately plain OLS on the raw counts: the Scout compares slopes across
    candidates measured the same way, so a more sophisticated fit would change
    every score identically and explain nothing extra.
    """
    if len(monthly) < 2:
        return 0.0
    ys = [float(v) for _, v in sorted(monthly.items())]
    xs = [float(i) for i in range(len(ys))]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denom
