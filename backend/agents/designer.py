"""Experiment Designer: state a falsifiable hypothesis and choose a test.

Two things make this more than "ask the model for a statistical test".

**The hypothesis is pre-registered.** H0, H1, and the expected direction are
recorded *before* execution. Without that, a system will happily rationalise
whatever it finds -- and with it, the Critic can catch exactly that.

**The spec is validated against the real dataset before it runs.** Column names
are checked, types are checked, group columns are checked for actually having
groups. A model that proposes a correlation on a column that does not exist
gets told so immediately and corrects, rather than producing a traceback three
nodes downstream.

Prior failures are shown to the Designer on every retry, which is what stops
it proposing the same impossible test twice.
"""

from __future__ import annotations

import logging
from typing import Any

from ..analysis.registry import describe_columns
from ..config import Role
from ..schemas import (
    AgentName,
    ArtifactKind,
    DataBundle,
    Dataset,
    ExperimentKind,
    ExperimentSpec,
    Level,
    ResearchQuestion,
)
from .base import AgentFailure, BaseAgent

log = logging.getLogger(__name__)

SYSTEM = """You design statistical experiments that run on a real dataset in seconds.

You must choose ONE procedure from this registry:

- correlation        params: {"x": <numeric col>, "y": <numeric col>}
- group_comparison   params: {"value": <numeric col>, "group": <categorical col>}
- regression         params: {"outcome": <numeric col>, "predictors": [<numeric cols>]}
- trend              params: {"time": <ordered numeric col>, "value": <numeric col>}
- clustering         params: {"columns": [<numeric cols>, at least 2]}
- classification     params: {"target": <categorical col>, "features": [<numeric cols>]}

Rules:
- Use ONLY column names that appear in the supplied list. Inventing a column fails immediately.
- Choose the procedure that genuinely addresses the research question. Do not default to \
correlation.
- State H0 and H1 so that H0 could actually be rejected.
- State your prediction BEFORE seeing results. You will be judged on whether the outcome \
matches it, so an honest prediction is worth more than a safe one.
- A column that is constant, or nearly all one value, is useless as a predictor."""

PROMPT = """Research question:
{question}

What this question requires measuring: {measurable}

Available dataset ({rows} rows). These are the ONLY columns that exist:
{columns}

{history}

Design one experiment that addresses the research question using this data."""

REPAIR = """Your experiment specification could not be run.

Specification: {kind} with params {params}
Error: {error}

The dataset columns, with summary statistics, are:
{columns}

Produce a corrected specification. Use only columns that exist and satisfy the procedure's \
parameter requirements."""


class ExperimentDesigner(BaseAgent):
    name = AgentName.DESIGNER
    fatal_on_failure = False

    async def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        question: ResearchQuestion | None = state.get("question")
        bundle: DataBundle | None = state.get("data_bundle")
        if question is None or bundle is None or bundle.dataset is None:
            raise AgentFailure("no dataset available to design an experiment against")

        dataset = bundle.dataset
        cycle = state.get("cycle", 0)

        spec = await self._design(question, dataset, state, cycle=cycle)

        self.say(
            f"Hypothesis: {spec.hypothesis.alternative}",
            level=Level.SUCCESS,
            cycle=cycle,
        )
        self.say(f"Prediction (recorded before execution): {spec.hypothesis.prediction}", cycle=cycle)

        self.publish(
            ArtifactKind.EXPERIMENT_SPEC,
            {
                "kind": spec.kind.value,
                "params": spec.params,
                "null": spec.hypothesis.null,
                "alternative": spec.hypothesis.alternative,
                "prediction": spec.hypothesis.prediction,
                "reasoning": spec.hypothesis.reasoning,
                "alpha": spec.alpha,
                "rationale": spec.rationale,
            },
            message=f"Experiment designed: {spec.kind.value}",
            cycle=cycle,
        )

        return {"experiment_spec": spec, "phase": "experiment"}

    async def _design(
        self,
        question: ResearchQuestion,
        dataset: Dataset,
        state: dict[str, Any],
        *,
        cycle: int,
    ) -> ExperimentSpec:
        columns = describe_columns(dataset)
        history = self._history_text(state)

        spec = await self._propose(
            PROMPT.format(
                question=question.text,
                measurable=question.proposal.expected_measurable,
                rows=dataset.n_rows,
                columns=columns,
                history=history,
            )
        )

        # Validate before running. Cheap here, expensive downstream.
        for attempt in range(3):
            error = self._validate(spec, dataset)
            if error is None:
                return spec
            self.say(
                f"Specification rejected before execution: {error}",
                level=Level.WARN,
                cycle=cycle,
            )
            if attempt == 2:
                raise AgentFailure(f"could not produce a runnable experiment: {error}")
            spec = await self._propose(
                REPAIR.format(
                    kind=spec.kind.value, params=spec.params, error=error, columns=columns
                )
            )
        return spec

    async def _propose(self, prompt: str) -> ExperimentSpec:
        try:
            return await self.router.structured(
                Role.REASONING,
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                ExperimentSpec,
                temperature=0.3,
                max_tokens=1400,
            )
        except Exception as exc:
            raise AgentFailure(f"experiment design failed: {exc}") from exc

    @staticmethod
    def _validate(spec: ExperimentSpec, dataset: Dataset) -> str | None:
        """Check the spec can actually run. Returns an error message or None.

        Mirrors the registry's own preconditions so failures surface here, with
        the Designer still in context to fix them, rather than as a traceback
        after the data has been loaded.
        """
        if spec.kind == ExperimentKind.CUSTOM_CODE:
            return None if spec.code.strip() else "custom_code selected but no code supplied"

        missing = [c for c in spec.required_columns() if c not in dataset.data]
        if missing:
            return (
                f"columns {missing} do not exist. Available: {', '.join(dataset.columns)}"
            )

        numeric = {c for c, t in dataset.dtypes.items() if t in {"int", "float"}}

        def needs_numeric(key: str) -> str | None:
            value = spec.params.get(key)
            if isinstance(value, str) and value not in numeric:
                return f"{key}={value!r} must be numeric but is {dataset.dtypes.get(value, 'unknown')}"
            if isinstance(value, list):
                bad = [v for v in value if v not in numeric]
                if bad:
                    return f"{key} entries {bad} must be numeric"
            return None

        checks = {
            ExperimentKind.CORRELATION: ("x", "y"),
            ExperimentKind.GROUP_COMPARISON: ("value",),
            ExperimentKind.REGRESSION: ("outcome", "predictors"),
            ExperimentKind.TREND: ("time", "value"),
            ExperimentKind.CLUSTERING: ("columns",),
            ExperimentKind.CLASSIFICATION: ("features",),
        }
        for key in checks.get(spec.kind, ()):
            if spec.params.get(key) in (None, "", []):
                return f"{spec.kind.value} requires a non-empty '{key}' parameter"
            problem = needs_numeric(key)
            if problem:
                return problem

        # Reject relationships that hold by construction rather than by
        # evidence. A derived column correlated against its own source yields a
        # near-perfect coefficient and a vanishing p-value while establishing
        # nothing -- and because the statistics look excellent, neither the
        # confidence scoring nor the Critic reliably catches it downstream.
        dependent_pairs = {
            ExperimentKind.CORRELATION: [("x", "y")],
            ExperimentKind.TREND: [("time", "value")],
        }
        for first, second in dependent_pairs.get(spec.kind, []):
            a, b = spec.params.get(first), spec.params.get(second)
            if isinstance(a, str) and isinstance(b, str) and dataset.are_dependent(a, b):
                return (
                    f"{a!r} and {b!r} are related by construction, not by evidence "
                    f"({a!r} is computed from {b!r} or vice versa). Any correlation "
                    "between them is arithmetic. Choose two independently measured columns."
                )

        if spec.kind == ExperimentKind.REGRESSION:
            outcome = spec.params.get("outcome") or spec.params.get("y")
            predictors = spec.params.get("predictors") or spec.params.get("features") or []
            if isinstance(predictors, str):
                predictors = [predictors]
            tainted = [
                p for p in predictors if isinstance(outcome, str) and dataset.are_dependent(outcome, p)
            ]
            if tainted:
                return (
                    f"predictors {tainted} are components of the outcome {outcome!r}. "
                    "The model would be predicting a quantity from its own ingredients. "
                    "Choose independently measured predictors."
                )

        # A grouping column with one distinct value cannot be compared.
        if spec.kind == ExperimentKind.GROUP_COMPARISON:
            group = spec.params.get("group")
            if not group:
                return "group_comparison requires a 'group' parameter"
            if group not in dataset.data:
                return f"group column {group!r} does not exist"
            distinct = {str(v) for v in dataset.data[group] if v is not None}
            if len(distinct) < 2:
                return (
                    f"group column {group!r} has only {len(distinct)} distinct value(s), "
                    "so there is nothing to compare"
                )

        if dataset.n_rows < 8:
            return f"dataset has only {dataset.n_rows} rows; no test would be defensible"

        return None

    @staticmethod
    def _history_text(state: dict[str, Any]) -> str:
        """Show prior attempts and critic objections, so they are not repeated."""
        parts: list[str] = []

        history = state.get("experiment_history") or []
        if history:
            lines = [
                f"- {h.get('kind')} with {h.get('params')}: {h.get('outcome')}"
                for h in history[-4:]
            ]
            parts.append("Previous experiments in this run:\n" + "\n".join(lines))

        critiques = state.get("critiques") or []
        if critiques:
            latest = critiques[-1]
            objections = [
                f"- {o.claim_attacked}: {o.rationale}" for o in latest.verified_objections[:4]
            ]
            flags = latest.stat_flags.as_reasons()
            if objections or flags:
                parts.append(
                    "The Critic rejected the last attempt for these reasons. Address them:\n"
                    + "\n".join(objections + [f"- {f}" for f in flags])
                )

        return "\n\n".join(parts) if parts else ""
