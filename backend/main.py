"""FastAPI application: JSON API, SSE stream, and the built React SPA.

One process serves both the API and the frontend on a single port. That is a
deployment decision, not a code-organisation one: it means one URL, no CORS
configuration, and one container to keep warm on a free tier.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import attach_ledger, router
from .config import get_settings
from .memory.ledger import Ledger

log = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    ledger = Ledger(settings.db_path)
    await ledger.init()
    # Free-tier containers are restarted without warning; runs left mid-flight
    # would otherwise sit in the gallery appearing to be in progress forever.
    orphaned = await ledger.mark_orphans_failed()
    if orphaned:
        log.info("closed out %d run(s) orphaned by a restart", orphaned)
    attach_ledger(ledger)

    configured = [p.name for p in settings.configured_providers()]
    log.info("starting; LLM providers configured: %s", configured or "NONE")
    if not configured:
        log.warning(
            "No LLM API keys are set. The API will start, but runs will fail. "
            "Copy .env.example to .env and fill in at least one key."
        )

    yield
    log.info("shutting down")


app = FastAPI(
    title="Autonomous Research Agent",
    description="A multi-agent system that discovers emerging science and writes its own papers.",
    version="0.1.0",
    lifespan=lifespan,
)

_settings = get_settings()
if _settings.allowed_origins:
    # Split deployment: the frontend is hosted elsewhere (Vercel/Netlify) and
    # calls this API cross-origin. Same-origin deployments leave CORS_ORIGINS
    # empty and no middleware is installed at all.
    #
    # `expose_headers` matters for SSE reconnection: without it the browser
    # cannot read the Last-Event-ID it needs to resume a dropped stream, and a
    # reconnect silently replays the run from the beginning.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Last-Event-ID", "Content-Type"],
    )
    log.info("CORS enabled for: %s", _settings.allowed_origins)

app.include_router(router)


# ------------------------------------------------------------- static frontend
#
# Mounted last so it never shadows an /api route.

if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str) -> FileResponse:
        """Serve the SPA, letting client-side routing own every non-API path."""
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")

else:

    @app.get("/", include_in_schema=False)
    async def no_frontend() -> JSONResponse:
        return JSONResponse(
            {
                "status": "backend running",
                "detail": "Frontend bundle not built. Run `npm run build` in frontend/.",
                "api": ["/api/health", "/api/topology", "/api/runs"],
            }
        )
