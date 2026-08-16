"""HTTP API: start runs, stream them, browse finished ones.

The public surface is small on purpose. The assessment's success criterion is
"paste nothing, click Start Research", so there is exactly one way to begin a
run and no parameters to get wrong.

Concurrency is capped and runs are throttled per client. This is a public URL
spending a shared free-tier budget: without limits, one visitor holding the
button exhausts the day's quota for everyone who follows, and the gallery
becomes the only thing a reviewer ever sees.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from sse_starlette.sse import EventSourceResponse

from ..config import get_settings
from ..graph.build import graph_topology
from ..memory.ledger import Ledger
from ..runtime.runner import ResearchRunner
from ..schemas import RunEvent, RunStatus

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

#: Live runs, keyed by run_id. Bounded by `max_concurrent_runs`.
_active: dict[str, ResearchRunner] = {}
_tasks: dict[str, asyncio.Task] = {}
_ledger: Ledger | None = None


def attach_ledger(ledger: Ledger) -> None:
    global _ledger
    _ledger = ledger


def get_ledger() -> Ledger:
    if _ledger is None:  # pragma: no cover - wiring error, not a runtime path
        raise RuntimeError("ledger not initialised")
    return _ledger


def _client_key(request: Request) -> str:
    """Identify a caller for throttling.

    Behind Hugging Face's proxy the socket address is the proxy, so the
    forwarded header is what distinguishes visitors.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.get("/health")
async def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "providers_configured": [p.name for p in settings.configured_providers()],
        "search_configured": bool(settings.tavily_api_key),
        "active_runs": len(_active),
        "max_concurrent": settings.max_concurrent_runs,
    }


@router.get("/topology")
async def topology() -> dict[str, Any]:
    """The agent graph, for the frontend to draw before anything runs."""
    return graph_topology()


@router.post("/runs")
async def start_run(request: Request) -> dict[str, Any]:
    """Begin a run. No body, no parameters -- that is the point."""
    settings = get_settings()
    ledger = get_ledger()

    if not settings.configured_providers():
        raise HTTPException(
            status_code=503,
            detail="No LLM provider is configured on the server.",
        )

    if len(_active) >= settings.max_concurrent_runs:
        running = next(iter(_active))
        raise HTTPException(
            status_code=429,
            detail={
                "message": "A research run is already in progress.",
                "active_run_id": running,
                "hint": "Watch the running one, or browse completed runs in the gallery.",
            },
        )

    client = _client_key(request)
    if await ledger.recent_run_count(client, settings.per_ip_cooldown_seconds) > 0:
        raise HTTPException(
            status_code=429,
            detail=f"Please wait {settings.per_ip_cooldown_seconds}s between runs.",
        )

    if await ledger.runs_today() >= settings.max_runs_per_day:
        raise HTTPException(
            status_code=429,
            detail="The daily run limit has been reached; the free-tier budget is spent. "
            "Completed runs remain available in the gallery.",
        )

    run_id = uuid.uuid4().hex[:12]
    await ledger.create_run(run_id, client_key=client)

    runner = ResearchRunner(run_id=run_id, settings=settings)
    _active[run_id] = runner

    # Events are persisted in batches from a background drain rather than one
    # insert per event: a busy phase emits dozens per second and per-event
    # transactions would measurably slow the run on free-tier disk.
    pending: list[RunEvent] = []
    runner.bus._on_persist = pending.append

    async def drain() -> None:
        while True:
            await asyncio.sleep(1.0)
            if pending:
                batch, pending[:] = list(pending), []
                await ledger.append_events(run_id, batch)
            if runner.bus.closed and not pending:
                return

    async def execute() -> None:
        drainer = asyncio.create_task(drain())
        try:
            state = await runner.run()
            status = (
                RunStatus.COMPLETED if state.get("paper") is not None else RunStatus.FAILED
            )
            await ledger.finish_run(run_id, status=status, state=state)
        except Exception as exc:  # noqa: BLE001 - never let one run kill the server
            log.exception("run %s crashed", run_id)
            await ledger.finish_run(run_id, status=RunStatus.FAILED, error=str(exc))
        finally:
            await asyncio.wait_for(drainer, timeout=10.0)
            if pending:
                await ledger.append_events(run_id, list(pending))
            _active.pop(run_id, None)
            _tasks.pop(run_id, None)

    _tasks[run_id] = asyncio.create_task(execute())
    log.info("started run %s for %s", run_id, client)
    return {"run_id": run_id, "status": RunStatus.RUNNING.value}


@router.get("/runs")
async def list_runs(limit: int = 25) -> dict[str, Any]:
    """The gallery. Always has something to show once one run has finished."""
    runs = await get_ledger().list_runs(limit=min(limit, 100))
    return {"runs": [r.model_dump(mode="json") for r in runs]}


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    summary = await get_ledger().get_run(run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"No run {run_id}")
    return {"run": summary.model_dump(mode="json"), "live": run_id in _active}


@router.get("/runs/{run_id}/events")
async def stream_events(run_id: str, request: Request, from_seq: int = 0) -> EventSourceResponse:
    """Stream a run.

    Live runs attach to the in-memory bus, which backfills history before the
    live tail. Finished runs replay from the ledger. The client cannot tell the
    difference, so a reviewer opening a completed run sees the same console as
    someone watching one happen.
    """
    runner = _active.get(run_id)

    if runner is not None:

        async def live():
            async for event in runner.bus.subscribe(from_seq=from_seq):
                if await request.is_disconnected():
                    break
                yield event.to_sse()

        return EventSourceResponse(live())

    events = await get_ledger().get_events(run_id, from_seq=from_seq)
    if not events:
        summary = await get_ledger().get_run(run_id)
        if summary is None:
            raise HTTPException(status_code=404, detail=f"No run {run_id}")

    async def replay():
        for event in events:
            yield event.to_sse()

    return EventSourceResponse(replay())


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict[str, Any]:
    task = _tasks.get(run_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} is not active")
    task.cancel()
    runner = _active.get(run_id)
    if runner is not None:
        runner.bus.close()
    await get_ledger().finish_run(
        run_id, status=RunStatus.CANCELLED, error="cancelled by request"
    )
    _active.pop(run_id, None)
    _tasks.pop(run_id, None)
    return {"run_id": run_id, "status": RunStatus.CANCELLED.value}


@router.get("/runs/{run_id}/paper.md", response_class=PlainTextResponse)
async def get_paper(run_id: str) -> str:
    markdown = await get_ledger().get_paper(run_id)
    if not markdown:
        raise HTTPException(status_code=404, detail="No paper for this run")
    return markdown


@router.get("/runs/{run_id}/manifest.json")
async def get_manifest(run_id: str) -> JSONResponse:
    manifest = await get_ledger().get_manifest(run_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="No manifest for this run")
    return JSONResponse(manifest)
