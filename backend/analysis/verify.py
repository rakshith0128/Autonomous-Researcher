"""Anti-fabrication checks.

An LLM writing a research paper will invent citations. It will produce a
plausible arXiv id that does not exist, attribute a real finding to the wrong
paper, and restate a p-value slightly wrong in the prose while the correct one
sits in the results table. None of that is malice; it is what next-token
prediction does when a citation-shaped gap appears.

So nothing the model writes is trusted about matters of fact. Three rules are
enforced mechanically after generation:

1. **Every citation must exist in the provenance ledger.** The writer is given
   numbered references drawn from documents that were actually fetched, and
   any URL or arXiv id in the output that is not in that ledger is a
   fabrication. It gets flagged and stripped.

2. **Every number must trace to a computed result.** Statistics are calculated
   in Python and injected. Any figure in the prose that does not match a
   computed value within tolerance is flagged.

3. **Cited URLs must resolve.** A reference that 404s is not evidence.

The output is a `VerificationReport`, which is published to the UI and printed
in the paper itself. Showing what was checked -- including anything that failed
-- is more convincing than a clean paper with no audit.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..schemas import Provenance, StatResult

log = logging.getLogger(__name__)

# Bare URLs and markdown links.
_URL_RE = re.compile(r"https?://[^\s\)\]\}<>\"',]+")
# arXiv identifiers come in two shapes and both must be recognised: a prose
# citation ("arXiv:2401.12345") and a URL ("arxiv.org/abs/2401.12345"). One
# pattern cannot cover both -- the URL has ".org/abs/" between the word and the
# id -- and missing the URL form means real fetched papers never get indexed by
# id, so a correct bare-id citation gets reported as fabricated.
_ARXIV_CITE_RE = re.compile(r"arxiv[:\s]\s*(\d{4}\.\d{4,5})", re.IGNORECASE)
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.IGNORECASE)


def _arxiv_ids(text: str) -> set[str]:
    """Every arXiv id in `text`, in either notation."""
    return set(_ARXIV_CITE_RE.findall(text)) | set(_ARXIV_URL_RE.findall(text))
# DOIs.
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)
# Numbers that look like reported statistics: p = 0.03, r = -0.41, n = 120.
_STAT_RE = re.compile(
    r"\b(p|r|rho|tau|n|d|R2|R\^2|beta)\s*[=:]\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


@dataclass
class Finding:
    """One thing that failed verification."""

    kind: str  # fabricated_citation | unresolvable_url | unsupported_number
    detail: str
    excerpt: str = ""

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"[{self.kind}] {self.detail}"


@dataclass
class VerificationReport:
    citations_found: int = 0
    citations_verified: int = 0
    numbers_found: int = 0
    numbers_verified: int = 0
    urls_checked: int = 0
    urls_live: int = 0
    findings: list[Finding] = field(default_factory=list)
    checked_urls: dict[str, bool] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def citation_integrity(self) -> float:
        """Fraction of citations that trace to a fetched source."""
        if self.citations_found == 0:
            return 1.0
        return self.citations_verified / self.citations_found

    @property
    def numeric_integrity(self) -> float:
        if self.numbers_found == 0:
            return 1.0
        return self.numbers_verified / self.numbers_found

    def summary(self) -> str:
        return (
            f"{self.citations_verified}/{self.citations_found} citations traced to "
            f"fetched sources, {self.numbers_verified}/{self.numbers_found} reported "
            f"numbers matched computed results, {self.urls_live}/{self.urls_checked} "
            f"URLs resolved"
        )

    def as_markdown(self) -> str:
        """Rendered into the paper. Failures are shown, not hidden."""
        lines = [
            "This paper was checked mechanically against the run's evidence ledger.",
            "",
            f"- Citations traced to actually-fetched sources: "
            f"**{self.citations_verified}/{self.citations_found}**",
            f"- Reported numbers matching computed results: "
            f"**{self.numbers_verified}/{self.numbers_found}**",
            f"- Cited URLs that resolved: **{self.urls_live}/{self.urls_checked}**",
        ]
        if self.findings:
            lines += ["", "**Verification failures (content was removed or flagged):**", ""]
            lines += [f"- {finding}" for finding in self.findings]
        else:
            lines += ["", "No fabricated citations or unsupported numbers were detected."]
        return "\n".join(lines)

    def model_dump(self) -> dict[str, Any]:
        return {
            "citations_found": self.citations_found,
            "citations_verified": self.citations_verified,
            "numbers_found": self.numbers_found,
            "numbers_verified": self.numbers_verified,
            "urls_checked": self.urls_checked,
            "urls_live": self.urls_live,
            "citation_integrity": round(self.citation_integrity, 3),
            "numeric_integrity": round(self.numeric_integrity, 3),
            "findings": [{"kind": f.kind, "detail": f.detail, "excerpt": f.excerpt} for f in self.findings],
        }


def _normalise_url(url: str) -> str:
    """Compare URLs by their identifying part.

    arXiv alone serves the same paper at /abs/, /pdf/, and with a version
    suffix. Comparing raw strings would call a correct citation fabricated.
    """
    cleaned = url.strip().rstrip(".,);:'\"").lower()
    cleaned = re.sub(r"^https?://", "", cleaned)
    cleaned = re.sub(r"^www\.", "", cleaned)
    cleaned = cleaned.replace("/pdf/", "/abs/")
    cleaned = re.sub(r"v\d+$", "", cleaned)
    return cleaned.rstrip("/")


def build_allowed_index(sources: list[Provenance]) -> dict[str, Provenance]:
    """Index of everything the system actually retrieved.

    Keyed by normalised URL, arXiv id, and DOI, so a citation written in any
    of those forms can be matched back to a real fetch.
    """
    index: dict[str, Provenance] = {}
    for source in sources:
        index[_normalise_url(source.url)] = source

        for arxiv_id in _arxiv_ids(source.url):
            index[arxiv_id] = source

        doi_match = _DOI_RE.search(source.url)
        if doi_match:
            index[doi_match.group(0).lower()] = source
    return index


def verify_citations(text: str, sources: list[Provenance]) -> tuple[str, VerificationReport]:
    """Check every citation against the provenance ledger.

    Returns (cleaned_text, report). Fabricated citations are replaced inline
    with a visible marker rather than deleted silently -- a reader should be
    able to see that the system caught its own invention.
    """
    report = VerificationReport()
    allowed = build_allowed_index(sources)
    cleaned = text

    for raw_url in set(_URL_RE.findall(text)):
        report.citations_found += 1
        key = _normalise_url(raw_url)

        matched = (
            key in allowed
            or any(arxiv_id in allowed for arxiv_id in _arxiv_ids(raw_url))
            or any(key.startswith(known) or known.startswith(key) for known in allowed)
        )

        if matched:
            report.citations_verified += 1
        else:
            report.findings.append(
                Finding(
                    kind="fabricated_citation",
                    detail=f"{raw_url} was never fetched by this run",
                    excerpt=_context(text, raw_url),
                )
            )
            cleaned = cleaned.replace(raw_url, "[citation removed: not in evidence ledger]")

    # Bare arXiv ids cited without a URL.
    for arxiv_id in set(_ARXIV_CITE_RE.findall(text)):
        if arxiv_id in allowed:
            continue
        if any(arxiv_id in key for key in allowed):
            continue
        report.citations_found += 1
        report.findings.append(
            Finding(
                kind="fabricated_citation",
                detail=f"arXiv:{arxiv_id} was never fetched by this run",
                excerpt=_context(text, arxiv_id),
            )
        )

    return cleaned, report


def verify_numbers(
    text: str, stats: list[StatResult], report: VerificationReport, *, tolerance: float = 0.02
) -> VerificationReport:
    """Check reported statistics against what was actually computed.

    Only figures written in the recognisable `symbol = value` form are checked;
    prose numbers ("roughly a third") are not claims of precision and are left
    alone. Tolerance is relative, to allow honest rounding in the write-up.
    """
    computed: dict[str, list[float]] = {}
    for stat in stats:
        for symbol, value in (
            ("p", stat.p_value),
            ("p", stat.p_value_corrected),
            ("n", float(stat.n) if stat.n else None),
            ("r", stat.effect_size),
            ("rho", stat.effect_size),
            ("d", stat.effect_size),
            ("beta", stat.effect_size),
            ("r2", stat.effect_size),
        ):
            if value is not None:
                computed.setdefault(symbol, []).append(float(value))

    for symbol, raw in _STAT_RE.findall(text):
        key = symbol.lower().replace("^", "").replace("R2", "r2")
        try:
            stated = float(raw)
        except ValueError:
            continue

        report.numbers_found += 1
        candidates = computed.get(key, [])
        if any(_close(stated, actual, tolerance) for actual in candidates):
            report.numbers_verified += 1
        else:
            report.findings.append(
                Finding(
                    kind="unsupported_number",
                    detail=(
                        f"{symbol} = {raw} appears in the text but no computed result "
                        f"matches it (computed {key}: "
                        f"{[round(c, 4) for c in candidates] or 'none'})"
                    ),
                    excerpt=_context(text, f"{symbol}"),
                )
            )
    return report


def _close(stated: float, actual: float, tolerance: float) -> bool:
    if actual == 0:
        return abs(stated) < 1e-9
    # Small p-values are written in wildly varying precision; compare on the
    # order of magnitude rather than the digits.
    if abs(actual) < 1e-3:
        return abs(stated) < 1e-2
    return abs(stated - actual) / abs(actual) <= tolerance


async def verify_urls_live(
    urls: list[str], fetcher: Any, report: VerificationReport, *, limit: int = 12
) -> VerificationReport:
    """Confirm cited URLs actually resolve.

    Capped, because this runs at the end of a run against the wall clock and a
    reference list can be long. A URL that fails here is reported, not removed:
    a source that was fetched successfully earlier and is briefly down now is
    still real evidence, and the distinction matters.
    """
    import asyncio

    async def check(url: str) -> tuple[str, bool]:
        try:
            result = await fetcher.fetch(url, max_retries=0)
            return url, 200 <= result.status < 400
        except Exception:  # noqa: BLE001 - unreachable is the answer, not an error
            return url, False

    checked = await asyncio.gather(*(check(u) for u in urls[:limit]), return_exceptions=True)
    for outcome in checked:
        if isinstance(outcome, BaseException):
            continue
        url, live = outcome
        report.urls_checked += 1
        report.checked_urls[url] = live
        if live:
            report.urls_live += 1
        else:
            report.findings.append(
                Finding(kind="unresolvable_url", detail=f"{url} did not resolve when re-checked")
            )
    return report


def _context(text: str, needle: str, width: int = 70) -> str:
    index = text.find(needle)
    if index < 0:
        return ""
    start = max(0, index - width // 2)
    return text[start : index + len(needle) + width // 2].replace("\n", " ").strip()


def format_reference_list(sources: list[Provenance]) -> str:
    """Numbered references for the writer to cite by index.

    Handing the model a closed list and requiring `[n]` citations removes the
    opportunity to invent one. Anything outside this list is caught by
    `verify_citations` afterwards.
    """
    lines = []
    for i, source in enumerate(sources, 1):
        title = source.title or source.url
        lines.append(f"[{i}] {title} — {source.url} ({source.modality.value})")
    return "\n".join(lines)
