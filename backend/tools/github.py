"""GitHub search: repository creation and star velocity as an emergence signal.

The assessment asks the Domain Scout to notice "rising GitHub repos". Raw star
counts cannot show that -- a decade-old library with 40k stars is popular, not
emerging. What distinguishes an emerging field is repositories that did not
exist before the cutoff and are gathering stars *quickly*, so this module
filters on creation date and measures stars per day.

Unauthenticated search is capped at 10 requests/minute, which one run can
exhaust. A no-scope personal access token raises that to 30/minute and is the
single highest-value optional credential in the system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..config import Settings, get_settings
from .arxiv import EMERGENCE_CUTOFF
from .http import Fetcher, FetchError

log = logging.getLogger(__name__)

GITHUB_SEARCH = "https://api.github.com/search/repositories"


@dataclass
class Repo:
    full_name: str
    description: str
    stars: int
    created_at: datetime
    pushed_at: datetime | None
    url: str
    language: str = ""

    @property
    def age_days(self) -> float:
        return max((datetime.now(UTC) - self.created_at).total_seconds() / 86_400.0, 1.0)

    @property
    def star_velocity(self) -> float:
        """Stars per day since creation."""
        return self.stars / self.age_days


@dataclass
class GithubSignal:
    term: str
    repos_created_post_cutoff: int = 0
    total_stars: int = 0
    star_velocity: float = 0.0
    median_star_velocity: float = 0.0
    top_repos: list[Repo] = field(default_factory=list)
    authenticated: bool = False
    ok: bool = False
    error: str = ""


class GithubClient:
    def __init__(self, fetcher: Fetcher, settings: Settings | None = None) -> None:
        self.fetcher = fetcher
        self.settings = settings or get_settings()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.settings.github_token:
            headers["Authorization"] = f"Bearer {self.settings.github_token}"
        return headers

    async def measure(
        self,
        term: str,
        *,
        cutoff: datetime = EMERGENCE_CUTOFF,
        per_page: int = 50,
    ) -> GithubSignal:
        """Count and rate post-cutoff repositories matching a term."""
        signal = GithubSignal(term=term, authenticated=bool(self.settings.github_token))
        phrase = f'"{term}"' if " " in term else term
        query = f"{phrase} created:>{cutoff:%Y-%m-%d}"

        try:
            data = await self.fetcher.get_json(
                GITHUB_SEARCH,
                params={
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": per_page,
                },
                headers=self._headers(),
            )
        except FetchError as exc:
            signal.error = str(exc)
            log.warning("GitHub measurement failed for %r: %s", term, exc)
            return signal

        signal.repos_created_post_cutoff = int(data.get("total_count", 0))
        repos = [r for r in (_parse_repo(item) for item in data.get("items", [])) if r]

        signal.top_repos = repos[:10]
        signal.total_stars = sum(r.stars for r in repos)

        velocities = sorted(r.star_velocity for r in repos)
        signal.star_velocity = sum(velocities)
        if velocities:
            mid = len(velocities) // 2
            # The median resists the single viral repo that would otherwise
            # make an entire field look like it is accelerating.
            signal.median_star_velocity = (
                velocities[mid]
                if len(velocities) % 2
                else (velocities[mid - 1] + velocities[mid]) / 2
            )

        signal.ok = True
        return signal


def _parse_repo(item: dict) -> Repo | None:
    def _dt(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    created = _dt(item.get("created_at"))
    if created is None:
        return None

    return Repo(
        full_name=item.get("full_name", ""),
        description=(item.get("description") or "")[:300],
        stars=int(item.get("stargazers_count", 0)),
        created_at=created,
        pushed_at=_dt(item.get("pushed_at")),
        url=item.get("html_url", ""),
        language=item.get("language") or "",
    )
