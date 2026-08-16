"""SQLite run ledger: persistence, replay, and the gallery.

Every event is written as it is emitted. That single decision buys three
features that would otherwise each need their own machinery:

* **Refresh-safe streaming.** A browser that reloads mid-run replays from the
  ledger and then joins the live tail, instead of losing the run.
* **The gallery.** Completed runs stay browsable, so a reviewer who arrives
  after the free-tier quota is spent still has a finished paper to read.
* **The trace view.** It is the event log rendered differently -- no separate
  logging path to keep in sync.

Writes are deliberately fire-and-forget from the caller's perspective: a
failed insert must never take down a run that is otherwise working.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from ..schemas import RunEvent, RunStatus, RunSummary

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    status            TEXT NOT NULL,
    started_at        TEXT NOT NULL,
    finished_at       TEXT,
    domain            TEXT DEFAULT '',
    question          TEXT DEFAULT '',
    paper_title       TEXT DEFAULT '',
    cycles_used       INTEGER DEFAULT 0,
    overall_confidence REAL DEFAULT 0.0,
    abstained_count   INTEGER DEFAULT 0,
    accepted_by_critic INTEGER DEFAULT 0,
    error             TEXT DEFAULT '',
    paper_markdown    TEXT DEFAULT '',
    manifest_json     TEXT DEFAULT '',
    client_key        TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS events (
    run_id   TEXT NOT NULL,
    seq      INTEGER NOT NULL,
    payload  TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);

-- The gallery lists newest first; without this it is a full scan per request.
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);
"""


class Ledger:
    """Async SQLite store for runs and their event streams."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()
        self._ready = False

    async def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as db:
            # WAL lets the gallery read while a run is still writing events.
            await db.execute("PRAGMA journal_mode=WAL")
            await db.executescript(SCHEMA)
            await db.commit()
        self._ready = True
        log.info("ledger ready at %s", self.path)

    # ------------------------------------------------------------- writing

    async def create_run(self, run_id: str, client_key: str = "") -> None:
        async with self._lock, aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO runs (run_id, status, started_at, client_key) "
                "VALUES (?, ?, ?, ?)",
                (run_id, RunStatus.RUNNING.value, datetime.now(UTC).isoformat(), client_key),
            )
            await db.commit()

    async def append_events(self, run_id: str, events: list[RunEvent]) -> None:
        """Batch-insert events.

        Batched because a busy phase emits dozens of events per second and one
        transaction per event dominates the run's wall clock on the slow disks
        free-tier containers tend to have.
        """
        if not events:
            return
        rows = [(run_id, e.seq, e.model_dump_json()) for e in events]
        try:
            async with self._lock, aiosqlite.connect(self.path) as db:
                await db.executemany(
                    "INSERT OR IGNORE INTO events (run_id, seq, payload) VALUES (?, ?, ?)", rows
                )
                await db.commit()
        except Exception:  # noqa: BLE001 - losing the audit trail beats losing the run
            log.exception("failed to persist %d events for run %s", len(rows), run_id)

    async def finish_run(
        self,
        run_id: str,
        *,
        status: RunStatus,
        state: dict[str, Any] | None = None,
        error: str = "",
    ) -> None:
        state = state or {}
        paper = state.get("paper")
        manifest = state.get("manifest")
        question = state.get("question")

        async with self._lock, aiosqlite.connect(self.path) as db:
            await db.execute(
                """UPDATE runs SET
                     status = ?, finished_at = ?, domain = ?, question = ?,
                     paper_title = ?, cycles_used = ?, overall_confidence = ?,
                     abstained_count = ?, accepted_by_critic = ?, error = ?,
                     paper_markdown = ?, manifest_json = ?
                   WHERE run_id = ?""",
                (
                    status.value,
                    datetime.now(UTC).isoformat(),
                    state.get("domain_name", ""),
                    question.text if question else "",
                    paper.title if paper else "",
                    state.get("cycle", 0),
                    float(state.get("overall_confidence", 0.0)),
                    len(state.get("abstained_claims") or []),
                    1 if (paper and paper.accepted_by_critic) else 0,
                    error or state.get("failure_reason", ""),
                    paper.to_markdown() if paper else "",
                    manifest.model_dump_json() if manifest else "",
                    run_id,
                ),
            )
            await db.commit()

    # ------------------------------------------------------------- reading

    async def get_run(self, run_id: str) -> RunSummary | None:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT r.*, (SELECT COUNT(*) FROM events e WHERE e.run_id = r.run_id) AS event_count "
                "FROM runs r WHERE r.run_id = ?",
                (run_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return _to_summary(row) if row else None

    async def list_runs(self, limit: int = 25) -> list[RunSummary]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT r.*, (SELECT COUNT(*) FROM events e WHERE e.run_id = r.run_id) AS event_count "
                "FROM runs r ORDER BY r.started_at DESC LIMIT ?",
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [_to_summary(row) for row in rows]

    async def get_events(self, run_id: str, from_seq: int = 0) -> list[RunEvent]:
        async with aiosqlite.connect(self.path) as db:
            async with db.execute(
                "SELECT payload FROM events WHERE run_id = ? AND seq > ? ORDER BY seq",
                (run_id, from_seq),
            ) as cursor:
                rows = await cursor.fetchall()

        events: list[RunEvent] = []
        for (payload,) in rows:
            try:
                events.append(RunEvent.model_validate_json(payload))
            except Exception:  # noqa: BLE001 - one corrupt row must not break replay
                log.warning("skipping unparseable event in run %s", run_id)
        return events

    async def get_paper(self, run_id: str) -> str:
        async with aiosqlite.connect(self.path) as db:
            async with db.execute(
                "SELECT paper_markdown FROM runs WHERE run_id = ?", (run_id,)
            ) as cursor:
                row = await cursor.fetchone()
        return row[0] if row and row[0] else ""

    async def get_manifest(self, run_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.path) as db:
            async with db.execute(
                "SELECT manifest_json FROM runs WHERE run_id = ?", (run_id,)
            ) as cursor:
                row = await cursor.fetchone()
        if not row or not row[0]:
            return None
        try:
            return json.loads(row[0])
        except ValueError:
            return None

    # ------------------------------------------------------------ throttling

    async def recent_run_count(self, client_key: str, within_seconds: int) -> int:
        """Runs started by one client recently. Backs the per-IP cooldown."""
        cutoff = datetime.now(UTC).timestamp() - within_seconds
        async with aiosqlite.connect(self.path) as db:
            async with db.execute(
                "SELECT started_at FROM runs WHERE client_key = ?", (client_key,)
            ) as cursor:
                rows = await cursor.fetchall()

        count = 0
        for (started,) in rows:
            try:
                if datetime.fromisoformat(started).timestamp() >= cutoff:
                    count += 1
            except ValueError:
                continue
        return count

    async def runs_today(self) -> int:
        start_of_day = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        async with aiosqlite.connect(self.path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM runs WHERE started_at >= ?", (start_of_day.isoformat(),)
            ) as cursor:
                row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def mark_orphans_failed(self) -> int:
        """Close out runs left RUNNING by a container restart.

        Free-tier hosts sleep and restart containers without warning. Without
        this, the gallery accumulates runs that appear to be in progress
        forever.
        """
        async with self._lock, aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                "UPDATE runs SET status = ?, error = ? WHERE status = ?",
                (
                    RunStatus.FAILED.value,
                    "interrupted by a server restart",
                    RunStatus.RUNNING.value,
                ),
            )
            await db.commit()
            return cursor.rowcount or 0


def _to_summary(row: Any) -> RunSummary:
    def _dt(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    return RunSummary(
        run_id=row["run_id"],
        status=RunStatus(row["status"]),
        started_at=_dt(row["started_at"]) or datetime.now(UTC),
        finished_at=_dt(row["finished_at"]),
        domain=row["domain"] or "",
        question=row["question"] or "",
        paper_title=row["paper_title"] or "",
        cycles_used=row["cycles_used"] or 0,
        overall_confidence=row["overall_confidence"] or 0.0,
        abstained_count=row["abstained_count"] or 0,
        accepted_by_critic=bool(row["accepted_by_critic"]),
        error=row["error"] or "",
        event_count=row["event_count"] if "event_count" in row.keys() else 0,
    )
