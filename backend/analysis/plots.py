"""Plotly figure construction.

Figures are built server-side and shipped as JSON for the browser to render.
That keeps the container free of a rendering stack, keeps the payload small,
and -- the reason that matters here -- keeps the dashboards genuinely
interactive rather than static images, which the assessment asks for
explicitly.

A shared dark template is applied to every figure so charts produced by
different agents in different phases still look like one system.
"""

from __future__ import annotations

import logging
from typing import Any

from ..schemas import DomainCandidate, FigureSpec

log = logging.getLogger(__name__)

# Matches the frontend's agent palette so a chart never clashes with the UI
# it is embedded in.
PALETTE = [
    "#38bdf8",
    "#4ade80",
    "#fbbf24",
    "#f472b6",
    "#c084fc",
    "#34d399",
    "#fb923c",
    "#f87171",
]

_LAYOUT: dict[str, Any] = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#c8d2e8", "family": "ui-sans-serif, system-ui, sans-serif", "size": 12},
    "margin": {"l": 60, "r": 24, "t": 56, "b": 56},
    "xaxis": {"gridcolor": "#232b45", "zerolinecolor": "#2f3a5c"},
    "yaxis": {"gridcolor": "#232b45", "zerolinecolor": "#2f3a5c"},
    "legend": {"orientation": "h", "y": -0.22},
    "hoverlabel": {"bgcolor": "#151b2e", "bordercolor": "#2f3a5c"},
}


def _to_spec(figure, title: str, caption: str, kind: str) -> FigureSpec | None:  # noqa: ANN001
    """Serialise a Plotly figure, never letting a chart failure break a run."""
    try:
        figure.update_layout(**_LAYOUT, title={"text": title, "x": 0.02, "font": {"size": 15}})
        return FigureSpec(
            title=title,
            figure_json=figure.to_json(),
            caption=caption,
            kind=kind,
        )
    except Exception as exc:  # noqa: BLE001 - a missing chart is cosmetic
        log.warning("figure serialisation failed for %r: %s", title, exc)
        return None


def emergence_chart(candidates: list[DomainCandidate]) -> FigureSpec | None:
    """Grouped bars comparing growth evidence across candidate domains.

    Growth *ratios* are plotted rather than raw counts deliberately: raw
    volume would rank every mature field above every emerging one, which is
    the precise confusion the Emergence Index exists to avoid.
    """
    viable = [c for c in candidates if not c.disqualified][:6]
    if not viable:
        return None

    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover
        return None

    names = [c.name if len(c.name) <= 34 else c.name[:31] + "…" for c in viable]

    figure = go.Figure(
        data=[
            go.Bar(
                name="arXiv growth (x baseline)",
                x=names,
                y=[round(c.signals.arxiv_growth_ratio, 2) for c in viable],
                marker_color=PALETTE[0],
                hovertemplate="%{x}<br>arXiv: %{y}x baseline<extra></extra>",
            ),
            go.Bar(
                name="OpenAlex growth (x baseline)",
                x=names,
                y=[round(c.signals.openalex_growth_ratio, 2) for c in viable],
                marker_color=PALETTE[1],
                hovertemplate="%{x}<br>OpenAlex: %{y}x baseline<extra></extra>",
            ),
        ]
    )
    figure.add_trace(
        go.Scatter(
            name="Emergence Index (right axis)",
            x=names,
            y=[round(c.emergence_index, 3) for c in viable],
            mode="markers+lines",
            marker={"size": 11, "color": PALETTE[2], "symbol": "diamond"},
            line={"color": PALETTE[2], "dash": "dot", "width": 2},
            yaxis="y2",
            hovertemplate="%{x}<br>index: %{y}<extra></extra>",
        )
    )
    figure.update_layout(
        barmode="group",
        yaxis={"title": "growth vs. pre-2024 baseline"},
        yaxis2={
            "title": "Emergence Index (z)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
    )

    return _to_spec(
        figure,
        "Candidate domains: measured growth evidence",
        "Publication growth relative to an equal-length pre-2024 baseline, with the "
        "combined Emergence Index. Scores are comparable within this run only.",
        "emergence",
    )


def timeseries_chart(
    monthly: dict[str, int], title: str, caption: str = ""
) -> FigureSpec | None:
    """Monthly publication counts for the chosen domain."""
    if len(monthly) < 2:
        return None
    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover
        return None

    months = sorted(monthly)
    figure = go.Figure(
        go.Scatter(
            x=months,
            y=[monthly[m] for m in months],
            mode="lines+markers",
            line={"color": PALETTE[0], "width": 2},
            marker={"size": 6},
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.12)",
        )
    )
    figure.update_layout(yaxis={"title": "papers per month"}, xaxis={"title": "month"})
    return _to_spec(figure, title, caption, "timeseries")


def scatter_with_fit(
    x: list[float],
    y: list[float],
    *,
    x_label: str,
    y_label: str,
    title: str,
    caption: str = "",
) -> FigureSpec | None:
    """Scatter plus least-squares line, for correlation experiments."""
    if len(x) < 3 or len(x) != len(y):
        return None
    try:
        import numpy as np
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover
        return None

    figure = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="observations",
            marker={"size": 8, "color": PALETTE[0], "opacity": 0.75},
        )
    )
    try:
        slope, intercept = np.polyfit(x, y, 1)
        xs = [min(x), max(x)]
        figure.add_trace(
            go.Scatter(
                x=xs,
                y=[slope * v + intercept for v in xs],
                mode="lines",
                name=f"fit: y = {slope:.3g}x + {intercept:.3g}",
                line={"color": PALETTE[2], "dash": "dash", "width": 2},
            )
        )
    except Exception as exc:  # noqa: BLE001 - the scatter alone is still useful
        log.debug("trend line unavailable: %s", exc)

    figure.update_layout(xaxis={"title": x_label}, yaxis={"title": y_label})
    return _to_spec(figure, title, caption, "scatter")


def group_comparison_chart(
    groups: dict[str, list[float]],
    *,
    value_label: str,
    title: str,
    caption: str = "",
) -> FigureSpec | None:
    """Box plots per group.

    Box rather than bar-of-means on purpose: a bar chart of two means hides
    the spread, and the Critic's first move against any group comparison is to
    ask about the distribution.
    """
    usable = {k: v for k, v in groups.items() if len(v) >= 2}
    if len(usable) < 2:
        return None
    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover
        return None

    figure = go.Figure()
    for i, (label, values) in enumerate(usable.items()):
        figure.add_trace(
            go.Box(
                y=values,
                name=str(label),
                boxpoints="all",
                jitter=0.4,
                pointpos=-1.6,
                marker={"size": 5, "color": PALETTE[i % len(PALETTE)]},
                line={"color": PALETTE[i % len(PALETTE)]},
            )
        )
    figure.update_layout(yaxis={"title": value_label}, showlegend=False)
    return _to_spec(figure, title, caption, "groups")


def confidence_chart(claims: list[tuple[str, float]], threshold: float) -> FigureSpec | None:
    """Per-claim confidence against the abstention threshold.

    The threshold line is the point: it makes the abstained claims visibly
    below a bar the system set for itself, rather than a footnote.
    """
    if not claims:
        return None
    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover
        return None

    labels = [c[:70] + ("…" if len(c) > 70 else "") for c, _ in claims]
    values = [round(v, 3) for _, v in claims]
    colors = ["#f87171" if v < threshold else "#34d399" for v in values]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>confidence: %{x:.0%}<extra></extra>",
        )
    )
    figure.add_vline(
        x=threshold,
        line={"color": "#fbbf24", "dash": "dash", "width": 2},
        annotation={"text": f"abstention threshold ({threshold:.0%})", "font": {"size": 11}},
    )
    figure.update_layout(
        xaxis={"title": "confidence", "range": [0, 1]},
        yaxis={"automargin": True},
        height=max(260, 46 * len(claims)),
    )
    return _to_spec(
        figure,
        "Claim confidence and abstention",
        "Claims below the threshold are not asserted; they appear in the paper's "
        "Abstained section instead.",
        "confidence",
    )
