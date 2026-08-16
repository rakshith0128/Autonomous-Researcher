"""Tests for the network layer, with all traffic mocked.

No test here touches the internet: CI must be able to run the entire suite
offline, and resilience behaviour is far easier to assert against a mock that
fails on demand than against a real host that mostly works.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from backend.config import Settings
from backend.tools.http import BreakerState, Fetcher, FetchError

BASE = "https://example.test"


def settings(**overrides) -> Settings:
    return Settings(
        _env_file=None,
        circuit_breaker_threshold=3,
        http_timeout_seconds=5.0,
        max_fetch_bytes=1_000_000,
        **overrides,
    )


@pytest.fixture
async def fetcher():
    f = Fetcher(settings=settings())
    yield f
    await f.aclose()


class TestRetryPolicy:
    @respx.mock
    async def test_success_on_first_attempt(self, fetcher: Fetcher):
        route = respx.get(f"{BASE}/ok").mock(return_value=httpx.Response(200, json={"a": 1}))
        assert await fetcher.get_json(f"{BASE}/ok") == {"a": 1}
        assert route.call_count == 1

    @respx.mock
    async def test_transient_5xx_is_retried_then_succeeds(self, fetcher: Fetcher):
        route = respx.get(f"{BASE}/flaky").mock(
            side_effect=[httpx.Response(503), httpx.Response(200, json={"ok": True})]
        )
        assert await fetcher.get_json(f"{BASE}/flaky") == {"ok": True}
        assert route.call_count == 2

    @respx.mock
    async def test_404_fails_immediately_without_retrying(self, fetcher: Fetcher):
        """A 404 is a fact about the request. Retrying only burns wall clock."""
        route = respx.get(f"{BASE}/missing").mock(return_value=httpx.Response(404))
        with pytest.raises(FetchError) as exc:
            await fetcher.fetch(f"{BASE}/missing")
        assert route.call_count == 1
        assert exc.value.status == 404

    @respx.mock
    async def test_429_is_retried_despite_being_4xx(self, fetcher: Fetcher):
        route = respx.get(f"{BASE}/limited").mock(
            side_effect=[httpx.Response(429), httpx.Response(200, json={})]
        )
        await fetcher.get_json(f"{BASE}/limited")
        assert route.call_count == 2

    @respx.mock
    async def test_timeouts_are_retried_then_reported(self, fetcher: Fetcher):
        respx.get(f"{BASE}/slow").mock(side_effect=httpx.ReadTimeout("too slow"))
        with pytest.raises(FetchError):
            await fetcher.fetch(f"{BASE}/slow", max_retries=1)


class TestSizeGuards:
    @respx.mock
    async def test_oversized_content_length_is_refused_before_download(self, fetcher: Fetcher):
        """An OOM kill leaves no traceback; declining is far cheaper."""
        respx.get(f"{BASE}/huge").mock(
            return_value=httpx.Response(200, headers={"content-length": "99000000"})
        )
        with pytest.raises(FetchError, match="over the"):
            await fetcher.fetch(f"{BASE}/huge")

    @respx.mock
    async def test_body_larger_than_its_declared_length_is_caught(self, fetcher: Fetcher):
        """The header guard is an optimisation, not the guarantee.

        Servers using chunked encoding omit content-length entirely, and
        misconfigured ones under-report it. The post-read check is what
        actually enforces the ceiling.
        """
        respx.get(f"{BASE}/sneaky").mock(
            return_value=httpx.Response(
                200, content=b"x" * 1_200_000, headers={"content-length": "10"}
            )
        )
        with pytest.raises(FetchError, match="exceeded"):
            await fetcher.fetch(f"{BASE}/sneaky")


class TestCircuitBreaker:
    @respx.mock
    async def test_opens_after_threshold_consecutive_failures(self, fetcher: Fetcher):
        respx.get(f"{BASE}/dead").mock(return_value=httpx.Response(404))

        for _ in range(3):
            with pytest.raises(FetchError):
                await fetcher.fetch(f"{BASE}/dead")

        assert fetcher.breaker_state(f"{BASE}/dead") is BreakerState.OPEN
        assert "example.test" in fetcher.degraded_hosts()

    @respx.mock
    async def test_open_breaker_refuses_without_a_request(self, fetcher: Fetcher):
        route = respx.get(f"{BASE}/dead").mock(return_value=httpx.Response(404))
        for _ in range(3):
            with pytest.raises(FetchError):
                await fetcher.fetch(f"{BASE}/dead")
        calls_before = route.call_count

        with pytest.raises(FetchError, match="circuit breaker open"):
            await fetcher.fetch(f"{BASE}/dead")
        assert route.call_count == calls_before, "no request should have been sent"

    @respx.mock
    async def test_success_resets_the_failure_count(self, fetcher: Fetcher):
        respx.get(f"{BASE}/x").mock(
            side_effect=[
                httpx.Response(404),
                httpx.Response(404),
                httpx.Response(200, json={}),
            ]
        )
        for _ in range(2):
            with pytest.raises(FetchError):
                await fetcher.fetch(f"{BASE}/x")
        await fetcher.get_json(f"{BASE}/x")
        assert fetcher.breaker_state(f"{BASE}/x") is BreakerState.CLOSED

    @respx.mock
    async def test_breakers_are_isolated_per_host(self, fetcher: Fetcher):
        """One dead source must degrade one branch of the plan, not the run."""
        respx.get("https://dead.test/a").mock(return_value=httpx.Response(404))
        respx.get("https://alive.test/b").mock(return_value=httpx.Response(200, json={"ok": 1}))

        for _ in range(3):
            with pytest.raises(FetchError):
                await fetcher.fetch("https://dead.test/a")

        assert fetcher.breaker_state("https://dead.test/a") is BreakerState.OPEN
        assert await fetcher.get_json("https://alive.test/b") == {"ok": 1}


class TestJson:
    @respx.mock
    async def test_malformed_json_counts_as_a_host_failure(self, fetcher: Fetcher):
        respx.get(f"{BASE}/bad").mock(return_value=httpx.Response(200, content=b"not json"))
        with pytest.raises(FetchError, match="invalid JSON"):
            await fetcher.get_json(f"{BASE}/bad")

    @respx.mock
    async def test_post_json_round_trip(self, fetcher: Fetcher):
        route = respx.post(f"{BASE}/search").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        assert await fetcher.post_json(f"{BASE}/search", {"q": "x"}) == {"results": []}
        assert route.call_count == 1

    @respx.mock
    async def test_post_4xx_does_not_retry(self, fetcher: Fetcher):
        route = respx.post(f"{BASE}/search").mock(return_value=httpx.Response(401))
        with pytest.raises(FetchError):
            await fetcher.post_json(f"{BASE}/search", {})
        assert route.call_count == 1


class TestPoliteness:
    async def test_user_agent_includes_contact_when_configured(self):
        """OpenAlex and Crossref run a faster pool for identified clients."""
        f = Fetcher(settings=settings(contact_email="a@b.test"))
        assert "mailto:a@b.test" in f._user_agent()
        await f.aclose()

    async def test_user_agent_is_valid_without_contact(self):
        f = Fetcher(settings=settings())
        agent = f._user_agent()
        assert "AutonomousResearchAgent" in agent and "mailto:" not in agent
        await f.aclose()


class TestArticleExtraction:
    @respx.mock
    async def test_falls_back_to_the_reader_proxy_when_direct_fetch_fails(
        self, fetcher: Fetcher
    ):
        respx.get("https://blocked.test/article").mock(return_value=httpx.Response(403))
        respx.get(url__startswith="https://r.jina.ai/").mock(
            return_value=httpx.Response(200, content=b"Recovered article body. " * 40)
        )
        text, tier = await fetcher.get_article_text("https://blocked.test/article")
        assert tier == "jina-reader"
        assert "Recovered article body" in text

    @respx.mock
    async def test_returns_empty_rather_than_raising_when_every_tier_fails(
        self, fetcher: Fetcher
    ):
        """A page we cannot read is a missing source, not a crashed run."""
        respx.get("https://blocked.test/x").mock(return_value=httpx.Response(403))
        respx.get(url__startswith="https://r.jina.ai/").mock(return_value=httpx.Response(500))
        text, tier = await fetcher.get_article_text("https://blocked.test/x")
        assert text == "" and tier == "none"
