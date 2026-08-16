"""Executor: run the experiment, and repair itself when it breaks.

Two execution paths share this node:

* **Registry** -- a vetted procedure runs against validated parameters. Python
  does the statistics; no model output is trusted with a number.
* **Sandbox** -- model-written code for anything the registry cannot express,
  executed under AST restrictions in an isolated subprocess.

Both fail in the same way -- an exception with a message -- and both are
repaired the same way: the error goes back to the model, which corrects and
retries, up to a configured limit. That loop is the assessment's "self-repair
of broken tools", and it is worth noting that the *registry* path benefits from
it too: a specification naming a column that turns out to be entirely null
fails at runtime, and the repair round fixes it without escalating to a full
redesign.

Every attempt, successful or not, is recorded in `experiment_history`. The
paper's methods section can then honestly say how many designs were tried.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..analysis.registry import ExperimentError, describe_columns, run_registry_experiment
from ..config import Role
from ..schemas import (
    AgentName,
    ArtifactKind,
    DataBundle,
    Dataset,
    ExperimentKind,
    ExperimentResult,
    ExperimentSpec,
    Level,
)
from ..tools.sandbox import run_code
from .base import AgentFailure, BaseAgent

log = logging.getLogger(__name__)

CODE_SYSTEM = """You write short Python analysis scripts.

Environment:
- A pandas DataFrame named `df` is already loaded. Do not read any files.
- Available: pandas, numpy, scipy, statsmodels, sklearn, math, statistics, json, plotly.
- No filesystem, no network, no os/sys/subprocess. Those imports are rejected before execution.
- Assign your findings to a dict named RESULT. Include p_value, effect_size, and n where \
applicable.

Write only the code. No explanation, no markdown fences."""

CODE_REPAIR = """Your code failed.

```python
{code}
```

Error:
{error}

Dataset columns:
{columns}

Return corrected code. Same requirements: assign to RESULT, no forbidden imports."""


class Executor(BaseAgent):
    name = AgentName.EXECUTOR
    fatal_on_failure = False

    async def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        spec: ExperimentSpec | None = state.get("experiment_spec")
        bundle: DataBundle | None = state.get("data_bundle")
        if spec is None or bundle is None or bundle.dataset is None:
            raise AgentFailure("no experiment specification or dataset to execute")

        dataset = bundle.dataset
        cycle = state.get("cycle", 0)
        started = time.perf_counter()

        if spec.kind == ExperimentKind.CUSTOM_CODE:
            result = await self._run_sandbox(spec, dataset, cycle=cycle)
        else:
            result = await self._run_registry(spec, dataset, cycle=cycle)

        result.duration_ms = int((time.perf_counter() - started) * 1000)
        self._report(result, cycle=cycle)

        history_entry = {
            "cycle": cycle,
            "kind": spec.kind.value,
            "params": spec.params,
            "outcome": (
                f"ok: {result.primary.test_name} p={result.primary.p_value}"
                if result.executed_ok and result.primary
                else f"failed: {result.error[:160]}"
            ),
            "repair_attempts": result.repair_attempts,
        }

        return {
            "experiment_result": result,
            "experiment_history": [history_entry],
            "phase": "uncertainty",
        }

    # ------------------------------------------------------------- registry

    async def _run_registry(
        self, spec: ExperimentSpec, dataset: Dataset, *, cycle: int
    ) -> ExperimentResult:
        """Run a vetted procedure, repairing the spec if it will not execute."""
        result = ExperimentResult(spec=spec)
        current = spec
        max_attempts = self.settings.max_code_repair_attempts

        for attempt in range(max_attempts + 1):
            try:
                self.say(
                    f"Running {current.kind.value} with {current.params}…", cycle=cycle
                )
                stats, figures = run_registry_experiment(
                    current.kind, dataset, current.params
                )
            except ExperimentError as exc:
                result.repair_attempts = attempt
                result.error = str(exc)
                if attempt >= max_attempts:
                    self.say(
                        f"Experiment failed after {attempt} repair attempt(s): {exc}",
                        level=Level.ERROR,
                        cycle=cycle,
                    )
                    return result

                self.say(
                    f"Execution failed ({str(exc)[:110]}); repairing the specification.",
                    level=Level.WARN,
                    cycle=cycle,
                )
                repaired = await self._repair_spec(current, dataset, str(exc))
                if repaired is None:
                    return result
                current = repaired
                continue
            except Exception as exc:  # noqa: BLE001 - a library blowing up is still a failure
                log.exception("registry procedure raised")
                result.error = f"{type(exc).__name__}: {exc}"
                result.repair_attempts = attempt
                return result

            result.spec = current
            result.stats = stats
            result.figures = figures
            result.executed_ok = True
            result.repair_attempts = attempt
            result.summary = self._summarise(stats)
            return result

        return result

    async def _repair_spec(
        self, spec: ExperimentSpec, dataset: Dataset, error: str
    ) -> ExperimentSpec | None:
        """Ask the Designer's model to fix a spec that would not run."""
        from .designer import REPAIR, SYSTEM

        try:
            return await self.router.structured(
                Role.REASONING,
                [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": REPAIR.format(
                            kind=spec.kind.value,
                            params=spec.params,
                            error=error,
                            columns=describe_columns(dataset),
                        ),
                    },
                ],
                ExperimentSpec,
                temperature=0.2,
                max_tokens=1200,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("spec repair failed: %s", exc)
            return None

    # -------------------------------------------------------------- sandbox

    async def _run_sandbox(
        self, spec: ExperimentSpec, dataset: Dataset, *, cycle: int
    ) -> ExperimentResult:
        """Execute model-written code, feeding failures back for repair."""
        result = ExperimentResult(spec=spec)
        code = spec.code
        max_attempts = self.settings.max_code_repair_attempts

        for attempt in range(max_attempts + 1):
            self.say(
                f"Executing generated analysis code (attempt {attempt + 1})…", cycle=cycle
            )
            outcome = run_code(
                code, dataset.data, timeout=self.settings.sandbox_timeout_seconds
            )
            result.repair_attempts = attempt

            if outcome.ok:
                result.executed_ok = True
                result.spec = spec.model_copy(update={"code": code})
                result.tables = outcome.result
                result.stats = self._stats_from_dict(outcome.result, spec)
                result.summary = (
                    f"Sandboxed analysis returned {len(outcome.result)} value(s)."
                )
                if outcome.stdout:
                    result.summary += f" stdout: {outcome.stdout[:200]}"
                return result

            # Static-analysis rejections and runtime errors are both repairable,
            # and the violation messages are already written for the model.
            error = (
                "; ".join(outcome.violations) if outcome.failed_validation else outcome.error
            )
            result.error = error
            result.traceback = outcome.traceback

            if attempt >= max_attempts:
                self.say(
                    f"Generated code failed after {attempt} repair attempt(s): {error[:140]}",
                    level=Level.ERROR,
                    cycle=cycle,
                )
                return result

            self.say(
                f"Code failed ({error[:110]}); showing the model its error and retrying.",
                level=Level.WARN,
                cycle=cycle,
            )
            repaired = await self._repair_code(code, error or outcome.traceback, dataset)
            if repaired is None:
                return result
            code = repaired

        return result

    async def _repair_code(self, code: str, error: str, dataset: Dataset) -> str | None:
        try:
            completion = await self.router.complete(
                Role.REASONING,
                [
                    {"role": "system", "content": CODE_SYSTEM},
                    {
                        "role": "user",
                        "content": CODE_REPAIR.format(
                            code=code[:3000],
                            error=error[:1500],
                            columns=describe_columns(dataset),
                        ),
                    },
                ],
                temperature=0.2,
                max_tokens=1200,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("code repair failed: %s", exc)
            return None

        text = completion.text.strip()
        # Models fence code despite being told not to.
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("python"):
                text = text[6:]
        return text.strip()

    @staticmethod
    def _stats_from_dict(payload: dict[str, Any], spec: ExperimentSpec) -> list:
        """Lift recognised keys out of a sandbox RESULT into a StatResult.

        Only known keys are read. Anything else stays in `tables` as raw
        output, because a number the system cannot name is a number it should
        not be building a claim on.
        """
        from ..analysis.stats import interpret_effect
        from ..schemas import StatResult

        def pick(*names: str) -> float | None:
            for name in names:
                value = payload.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value)
            return None

        p_value = pick("p_value", "p", "pvalue")
        effect = pick("effect_size", "r", "rho", "d", "correlation")
        n = pick("n", "sample_size", "count")
        if p_value is None and effect is None:
            return []

        effect_name = "r" if "r" in payload or "correlation" in payload else "effect"
        return [
            StatResult(
                test_name=f"sandboxed analysis ({spec.hypothesis.alternative[:60]})",
                p_value=p_value,
                effect_size=effect,
                effect_size_name=effect_name,
                effect_interpretation=interpret_effect(effect_name, effect),
                n=int(n) if n else 0,
                notes=["computed by model-written code in the sandbox"],
            )
        ]

    # --------------------------------------------------------------- output

    @staticmethod
    def _summarise(stats: list) -> str:
        if not stats:
            return "No statistical result was produced."
        primary = stats[0]
        parts = [primary.test_name]
        if primary.p_value is not None:
            parts.append(f"p = {primary.p_value:.4g}")
        if primary.effect_size is not None:
            parts.append(
                f"{primary.effect_size_name} = {primary.effect_size:.3f} "
                f"({primary.effect_interpretation})"
            )
        parts.append(f"n = {primary.n}")
        return "; ".join(parts)

    def _report(self, result: ExperimentResult, *, cycle: int) -> None:
        if result.executed_ok:
            self.say(result.summary, level=Level.SUCCESS, cycle=cycle)
            iterate, reason = result.forces_iteration()
            if iterate:
                # Said plainly at the moment it happens, because a null result
                # driving another cycle is the most interesting thing the
                # system does and it should be visible when it occurs.
                self.say(
                    f"This result will force another cycle: {reason}",
                    level=Level.WARN,
                    cycle=cycle,
                )

        self.publish(
            ArtifactKind.EXPERIMENT_RESULT,
            {
                "executed_ok": result.executed_ok,
                "kind": result.spec.kind.value,
                "summary": result.summary,
                "error": result.error,
                "repair_attempts": result.repair_attempts,
                "duration_ms": result.duration_ms,
                "stats": [
                    {
                        "test": s.test_name,
                        "statistic": s.statistic,
                        "p_value": s.p_value,
                        "p_value_corrected": s.p_value_corrected,
                        "effect_size": s.effect_size,
                        "effect_size_name": s.effect_size_name,
                        "effect_interpretation": s.effect_interpretation,
                        "ci_low": s.ci_low,
                        "ci_high": s.ci_high,
                        "n": s.n,
                        "significant": s.significant,
                        "assumptions": s.assumptions_checked,
                        "notes": s.notes,
                    }
                    for s in result.stats
                ],
            },
            message=result.summary or "Experiment execution finished",
            cycle=cycle,
        )

        for figure in result.figures:
            self.publish(
                ArtifactKind.FIGURE,
                {"figure": figure.figure_json, "title": figure.title, "caption": figure.caption},
                message=figure.title,
                cycle=cycle,
            )
