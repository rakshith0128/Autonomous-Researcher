"""Network foundation: retries, circuit breakers, and tiered text extraction.

Every outbound request in this system goes through here. Public scientific
APIs are free, which also means they are rate-limited, occasionally down, and
under no obligation to tell you why. The agents above this layer are written
as though the network works; this module is what makes that assumption safe.

Three behaviours matter:

* **Retry only what retrying can fix.** 5xx and timeouts are transient; a 404
  is a fact. Retrying a 403 just burns the budget and the wall clock.
* **Circuit breakers per host.** After repeated failures a host is taken out of
  rotation, so a dead source degrades one branch of the plan instead of
  stalling the run. The Data Alchemist reads breaker state and re-plans around
  what is still alive.
* **Tiered extraction.** Article text is attempted with trafilatura, then a
  BeautifulSoup fallback, then Jina's free reader proxy for JS-heavy pages.
  Each tier is cheaper to reason about than a headless browser, which is why
  Playwright is not in this image.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import httpx

from ..config import Settings, get_settings

log = logging.getLogger(__name__)

# Free reader proxy that renders JS and returns clean markdown. Used only as a
# last resort, and its failure is never fatal.
JINA_READER = "https://r.jina.ai/"

# Minimum seconds between requests to a given host, where the operator asks for
# one. arXiv's terms request a 3-second delay and enforce it with temporary
# blocks; being throttled out of arXiv mid-run would remove the single richest
# source in the system, so this is cheap insurance rather than mere courtesy.
HOST_MIN_INTERVAL: dict[str, float] = {
    "export.arxiv.org": 3.0,
    "arxiv.org": 3.0,
    "api.crossref.org": 1.0,
    "eutils.ncbi.nlm.nih.gov": 0.4,  # NCBI allows 3 req/s unauthenticated
    "r.jina.ai": 1.0,
    # OpenAlex publishes a 10 req/s ceiling, but the Scout's term probing
    # bursts dozens of requests within a second or two and gets throttled well
    # below that in practice. Spacing them tripped the circuit breaker far less
    # often in testing, at a cost of well under a second per candidate.
    "api.openalex.org": 0.15,
}


class BreakerState(str, Enum):
    CLOSED = "closed"  # healthy
    OPEN = "open"  # failing; refuse immediately
    HALF_OPEN = "half_open"  # cooldown elapsed; allow one probe


class FetchError(RuntimeError):
    """Raised when a resource could not be retrieved after all tiers."""

    def __init__(self, message: str, url: str = "", status: int | None = None) -> None:
        super().__init__(message)
        self.url = url
        self.status = status


@dataclass
class _Breaker:
    """Per-host failure tracking."""

    failures: int = 0
    opened_at: float = 0.0
    successes_since_probe: int = 0
    total_failures: int = 0

    def state(self, threshold: int, cooldown: float, now: float) -> BreakerState:
        if self.failures < threshold:
            return BreakerState.CLOSED
        if now - self.opened_at >= cooldown:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN


@dataclass
class FetchResult:
    """One successful retrieval."""

    url: str
    status: int
    content: bytes
    headers: dict[str, str] = field(default_factory=dict)
    elapsed_ms: int = 0
    tier: str = "direct"

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").split(";")[0].strip().lower()


class Fetcher:
    """Shared async HTTP client with resilience built in.

    One instance per run. Reusing connections across the many small API calls
    the Scout makes is worth more than it looks: a cold TLS handshake per
    request roughly doubles domain-discovery wall time.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._breakers: dict[str, _Breaker] = {}
        self._last_request: dict[str, float] = {}
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.settings.http_timeout_seconds, connect=10.0),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            headers={"User-Agent": self._user_agent()},
        )

    def _user_agent(self) -> str:
        """Identify ourselves.

        OpenAlex and Crossref both operate a faster "polite pool" for clients
        that supply a contact address, and unidentified scrapers are the first
        to be throttled. Being a good citizen here is also just cheaper.
        """
        contact = self.settings.contact_email
        base = "AutonomousResearchAgent/0.1 (+https://github.com/; research assistant)"
        return f"{base} mailto:{contact}" if contact else base

    # ------------------------------------------------------------- breakers

    @staticmethod
    def _host(url: str) -> str:
        return urlparse(url).netloc.lower() or url

    def breaker_state(self, url: str, now: float | None = None) -> BreakerState:
        now = now if now is not None else time.monotonic()
        breaker = self._breakers.get(self._host(url))
        if breaker is None:
            return BreakerState.CLOSED
        return breaker.state(self.settings.circuit_breaker_threshold, 120.0, now)

    def _record_success(self, url: str) -> None:
        breaker = self._breakers.get(self._host(url))
        if breaker is not None:
            breaker.failures = 0
            breaker.successes_since_probe += 1

    def _record_failure(self, url: str, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        host = self._host(url)
        breaker = self._breakers.setdefault(host, _Breaker())
        breaker.failures += 1
        breaker.total_failures += 1
        if breaker.failures == self.settings.circuit_breaker_threshold:
            breaker.opened_at = now
            log.warning("circuit breaker opened for %s after %d failures", host, breaker.failures)

    def health(self) -> dict[str, dict[str, Any]]:
        """Per-host health, surfaced in the UI's tool-health indicators."""
        now = time.monotonic()
        return {
            host: {
                "state": breaker.state(
                    self.settings.circuit_breaker_threshold, 120.0, now
                ).value,
                "consecutive_failures": breaker.failures,
                "total_failures": breaker.total_failures,
            }
            for host, breaker in self._breakers.items()
        }

    def degraded_hosts(self) -> list[str]:
        now = time.monotonic()
        return [
            host
            for host, breaker in self._breakers.items()
            if breaker.state(self.settings.circuit_breaker_threshold, 120.0, now)
            is BreakerState.OPEN
        ]

    # -------------------------------------------------------------- politeness

    async def _throttle(self, url: str) -> None:
        """Hold off until this host's minimum interval has elapsed.

        The lock is per host, so waiting on arXiv never delays a concurrent
        GitHub call -- which matters, because the Scout measures every
        candidate domain against several sources in parallel.
        """
        host = self._host(url)
        interval = HOST_MIN_INTERVAL.get(host)
        if not interval:
            return

        lock = self._host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            last = self._last_request.get(host)
            if last is not None:
                wait = interval - (time.monotonic() - last)
                if wait > 0:
                    await asyncio.sleep(wait)
            self._last_request[host] = time.monotonic()

    # ---------------------------------------------------------------- fetch

    async def fetch(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        max_retries: int = 2,
        accept_status: tuple[int, ...] = (200,),
    ) -> FetchResult:
        """GET with retry, size guards, and breaker enforcement."""
        if self.breaker_state(url) is BreakerState.OPEN:
            raise FetchError(f"circuit breaker open for {self._host(url)}", url=url)

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            await self._throttle(url)
            started = time.perf_counter()
            try:
                response = await self._client.get(url, params=params, headers=headers)
                elapsed = int((time.perf_counter() - started) * 1000)

                if response.status_code in accept_status:
                    content = await self._read_guarded(response, url)
                    self._record_success(url)
                    return FetchResult(
                        url=str(response.url),
                        status=response.status_code,
                        content=content,
                        headers=dict(response.headers),
                        elapsed_ms=elapsed,
                    )

                # 4xx is a fact about the request, not a transient condition --
                # retrying cannot change the answer. 429 is the exception.
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    self._record_failure(url)
                    raise FetchError(
                        f"HTTP {response.status_code} for {url}",
                        url=url,
                        status=response.status_code,
                    )

                last_error = FetchError(
                    f"HTTP {response.status_code}", url=url, status=response.status_code
                )

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc

            except FetchError:
                raise

            if attempt < max_retries:
                # Exponential backoff. 429 gets a longer floor because the
                # server has explicitly asked us to slow down.
                status = getattr(last_error, "status", None)
                delay = (4.0 if status == 429 else 1.0) * (2**attempt)
                await asyncio.sleep(min(delay, 12.0))

        self._record_failure(url)
        raise FetchError(f"failed after {max_retries + 1} attempts: {last_error}", url=url)

    async def _read_guarded(self, response: httpx.Response, url: str) -> bytes:
        """Refuse oversized bodies.

        A 400MB dataset dropped into a 16GB free-tier container alongside an
        ONNX model and a Chroma index is how a run dies with an OOM kill and no
        traceback. Cheaper to decline it here.
        """
        limit = self.settings.max_fetch_bytes
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise FetchError(
                f"resource is {int(declared) / 1e6:.1f}MB, over the {limit / 1e6:.0f}MB limit",
                url=url,
            )

        content = response.content
        if len(content) > limit:
            raise FetchError(
                f"resource exceeded the {limit / 1e6:.0f}MB limit while reading", url=url
            )
        return content

    async def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        max_retries: int = 2,
    ) -> Any:
        """GET and parse JSON, treating malformed JSON as a host failure."""
        result = await self.fetch(url, params=params, headers=headers, max_retries=max_retries)
        try:
            import json

            return json.loads(result.content)
        except ValueError as exc:
            self._record_failure(url)
            raise FetchError(f"invalid JSON from {url}: {exc}", url=url) from exc

    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
        max_retries: int = 2,
    ) -> Any:
        """POST JSON and parse the JSON response.

        Kept separate from `fetch` rather than folded in behind a `method`
        argument: POST is used only for search APIs here, and the retry
        semantics differ -- replaying a GET is always safe, replaying a POST
        needs the endpoint to be idempotent, which search endpoints are and
        most other POSTs are not.
        """
        if self.breaker_state(url) is BreakerState.OPEN:
            raise FetchError(f"circuit breaker open for {self._host(url)}", url=url)

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            await self._throttle(url)
            try:
                response = await self._client.post(url, json=payload, headers=headers)

                if response.status_code == 200:
                    self._record_success(url)
                    return response.json()

                if 400 <= response.status_code < 500 and response.status_code != 429:
                    self._record_failure(url)
                    raise FetchError(
                        f"HTTP {response.status_code} for {url}: {response.text[:200]}",
                        url=url,
                        status=response.status_code,
                    )

                last_error = FetchError(
                    f"HTTP {response.status_code}", url=url, status=response.status_code
                )

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
            except FetchError:
                raise
            except ValueError as exc:  # malformed JSON body
                self._record_failure(url)
                raise FetchError(f"invalid JSON from {url}: {exc}", url=url) from exc

            if attempt < max_retries:
                status = getattr(last_error, "status", None)
                await asyncio.sleep(min((4.0 if status == 429 else 1.0) * (2**attempt), 12.0))

        self._record_failure(url)
        raise FetchError(f"failed after {max_retries + 1} attempts: {last_error}", url=url)

    async def get_article_text(self, url: str, *, min_chars: int = 400) -> tuple[str, str]:
        """Extract readable text from a page. Returns (text, tier_used).

        Tiers escalate only on failure, so the common case costs one request.
        """
        # Tier 1: fetch and extract with trafilatura (handles boilerplate removal)
        try:
            result = await self.fetch(url)
            text = _extract_with_trafilatura(result.text, url)
            if len(text) >= min_chars:
                return text, "trafilatura"

            # Tier 2: crude tag-stripping. Wins on pages whose markup confuses
            # trafilatura's density heuristics, notably bare API docs and
            # single-column preprint landing pages.
            text = _extract_with_soup(result.text)
            if len(text) >= min_chars:
                return text, "beautifulsoup"
        except FetchError as exc:
            log.info("direct fetch failed for %s (%s); trying reader proxy", url, exc)

        # Tier 3: reader proxy, which executes JS. Free, rate-limited, and not
        # guaranteed -- so its failure returns empty rather than raising.
        try:
            result = await self.fetch(f"{JINA_READER}{url}", max_retries=1)
            text = result.text.strip()
            if len(text) >= min_chars:
                return text, "jina-reader"
        except FetchError as exc:
            log.info("reader proxy failed for %s: %s", url, exc)

        return "", "none"

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Fetcher:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


def _extract_with_trafilatura(html: str, url: str) -> str:
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )
        return (extracted or "").strip()
    except Exception as exc:  # noqa: BLE001 - extraction is best-effort by nature
        log.debug("trafilatura failed on %s: %s", url, exc)
        return ""


def _extract_with_soup(html: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
            tag.decompose()
        lines = (line.strip() for line in soup.get_text("\n").splitlines())
        return "\n".join(line for line in lines if line)
    except Exception as exc:  # noqa: BLE001
        log.debug("soup extraction failed: %s", exc)
        return ""
