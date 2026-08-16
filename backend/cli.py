"""Headless entrypoint.

    python -m backend.cli run

The whole system works without a browser. That is deliberate and load-bearing:
the UI is a view onto a working pipeline, not a prerequisite for it, so the
science can be developed and debugged while the frontend is still a shell --
and a reviewer can reproduce a run without a display.

Output goes to `data/artifacts/<run_id>/`: paper.md, manifest.json, the raw
event log, and every Plotly figure as JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .config import get_settings
from .runtime.runner import ResearchRunner
from .schemas import EventType, Level

# Terminal colours per severity. Disabled automatically when piped.
_COLOURS = {
    Level.DEBUG: "\033[90m",
    Level.INFO: "\033[0m",
    Level.SUCCESS: "\033[92m",
    Level.WARN: "\033[93m",
    Level.ERROR: "\033[91m",
}
_RESET = "\033[0m"
_DIM = "\033[90m"


def _configure_stdout() -> None:
    """Force UTF-8 on the console.

    Windows defaults stdout to cp1252, which cannot encode the characters
    models routinely emit -- non-breaking hyphens, typographic quotes, en
    dashes. Every such event raised UnicodeEncodeError inside the print,
    silently dropping that line from the console.

    The event bus caught those exceptions and kept the run alive (persistence
    failures must never kill a run), which is correct but meant the breakage
    showed up only as missing feed lines. `errors="replace"` guarantees a
    character we cannot encode degrades to a placeholder instead of losing the
    whole message.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic terminals
                pass


def _supports_colour() -> bool:
    return sys.stdout.isatty()


def _print_event(event, *, verbose: bool) -> None:  # noqa: ANN001
    interesting = {
        EventType.AGENT_MESSAGE,
        EventType.WARNING,
        EventType.REROUTE,
        EventType.RUN_STARTED,
        EventType.RUN_COMPLETED,
        EventType.RUN_FAILED,
    }
    if not verbose and event.type not in interesting:
        return
    if verbose and event.type == EventType.ARTIFACT:
        label = event.payload.get("kind", "artifact")
        print(f"  {_DIM}[artifact]{_RESET} {label}: {event.message[:100]}")
        return

    agent = (event.agent.value if event.agent else "system")[:20]
    colour = _COLOURS.get(event.level, "") if _supports_colour() else ""
    reset = _RESET if _supports_colour() else ""
    cycle = f" c{event.cycle}" if event.cycle else ""
    print(f"  {_DIM if _supports_colour() else ''}[{agent:<20}{cycle}]{reset} {colour}{event.message}{reset}")


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.cycles:
        settings.max_cycles = args.cycles
    settings.ensure_dirs()

    if not settings.configured_providers():
        print(
            "No LLM API keys configured. Copy .env.example to .env and set at least one.",
            file=sys.stderr,
        )
        return 2

    runner = ResearchRunner(settings=settings)
    runner.bus._on_persist = lambda e: _print_event(e, verbose=args.verbose)

    print(f"\nRun {runner.run_id} — max {settings.max_cycles} cycles, "
          f"providers: {[p.name for p in settings.configured_providers()]}\n")

    state = await runner.run()

    out_dir = settings.artifacts_dir / runner.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    paper = state.get("paper")
    if paper is not None:
        (out_dir / "paper.md").write_text(paper.to_markdown(), encoding="utf-8")
        for i, figure in enumerate(paper.figures):
            (out_dir / f"figure_{i + 1}.json").write_text(figure.figure_json, encoding="utf-8")

    manifest = state.get("manifest")
    if manifest is not None:
        (out_dir / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )

    (out_dir / "events.jsonl").write_text(
        "\n".join(e.model_dump_json() for e in runner.bus.history), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    if paper is not None:
        print(f"PAPER: {paper.title}")
        print("=" * 78)
        print(f"  domain          : {paper.domain}")
        print(f"  question        : {paper.research_question[:120]}")
        print(f"  cycles used     : {paper.cycles_used}")
        print(f"  critic accepted : {paper.accepted_by_critic}")
        print(f"  confidence      : {paper.overall_confidence:.0%}")
        print(f"  claims asserted : {len(paper.claims)}")
        print(f"  claims abstained: {len(paper.abstained_claims)}")
        print(f"  references      : {len(paper.references)}")
        print(f"  figures         : {len(paper.figures)}")
    else:
        print("NO PAPER PRODUCED")
        print("=" * 78)
        print(f"  reason: {state.get('failure_reason', 'unknown')}")

    if manifest is not None:
        print(f"  tokens          : {manifest.total_tokens:,}")
        print(f"  cost            : ${manifest.total_cost_usd:.2f}")
        print(f"  duration        : {manifest.duration_seconds:.0f}s")
        if manifest.provider_failovers:
            print(f"  failovers       : {manifest.provider_failovers}")
    if state.get("degraded_tools"):
        print(f"  degraded tools  : {state['degraded_tools']}")

    print(f"\n  artifacts -> {out_dir}\n")
    return 0 if paper is not None else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="research-agent", description="Autonomous multi-agent research assistant."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute one full research run.")
    run_parser.add_argument(
        "--cycles", type=int, default=None, help="Override the maximum critique cycles."
    )
    run_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Include tool calls and artifacts."
    )

    subparsers.add_parser("topology", help="Print the agent graph as JSON.")

    _configure_stdout()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )

    if args.command == "topology":
        from .graph.build import graph_topology

        print(json.dumps(graph_topology(), indent=2))
        return 0

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
