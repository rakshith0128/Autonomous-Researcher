"""Sandboxed execution of model-written analysis code.

The Experiment Designer has two paths: a typed registry of vetted statistical
procedures, and this -- free-form Python for the questions the registry cannot
express. The registry handles the common case reliably; this handles the rest,
and is where a model gets to be genuinely creative.

Letting an LLM write code that then runs on your machine deserves more than
good intentions, so there are three independent layers:

1. **Static analysis.** The AST is walked before anything executes. Imports are
   whitelisted, and the escape hatches that make Python sandboxes famously
   leaky -- ``eval``, ``exec``, ``__subclasses__``, ``__globals__``,
   attribute-based traversal out of the object graph -- are rejected outright.
2. **Process isolation.** Approved code runs in a separate interpreter via
   ``-I`` (isolated mode: no user site-packages, no inherited environment),
   in a scratch directory, with no network libraries importable.
3. **A hard timeout.** The child is killed on expiry, which also covers the
   infinite loop that static analysis deliberately does not try to detect.

None of this is a security boundary strong enough for hostile code -- a
determined attacker with arbitrary Python will eventually win, and the honest
mitigation is that the whole system already runs inside a disposable container.
What these layers reliably stop is the realistic failure: a confused model
importing ``os`` and deleting the working directory, or hanging the run.
"""

from __future__ import annotations

import ast
import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Modules the analysis code may import. Everything the registry uses, and
# nothing that touches the filesystem, the network, or the process table.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "pandas",
        "numpy",
        "scipy",
        "scipy.stats",
        "scipy.optimize",
        "scipy.signal",
        "statsmodels",
        "statsmodels.api",
        "statsmodels.formula.api",
        "statsmodels.stats.multitest",
        "sklearn",
        "sklearn.cluster",
        "sklearn.decomposition",
        "sklearn.ensemble",
        "sklearn.linear_model",
        "sklearn.metrics",
        "sklearn.model_selection",
        "sklearn.preprocessing",
        "math",
        "statistics",
        "json",
        "itertools",
        "collections",
        "functools",
        "re",
        "datetime",
        "random",
        "plotly",
        "plotly.graph_objects",
        "plotly.express",
    }
)

# Names that provide a route out of the sandbox regardless of imports.
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "__import__",
        "globals",
        "locals",
        "vars",
        "breakpoint",
        "memoryview",
        "exit",
        "quit",
    }
)

# Attribute traversal used to climb from any object back to builtins.
FORBIDDEN_ATTRS: frozenset[str] = frozenset(
    {
        "__subclasses__",
        "__globals__",
        "__code__",
        "__closure__",
        "__bases__",
        "__mro__",
        "__builtins__",
        "__loader__",
        "__spec__",
        "__reduce__",
        "__reduce_ex__",
        "func_globals",
        "gi_frame",
        "cr_frame",
    }
)


class SandboxViolation(ValueError):
    """Raised when code fails static analysis. Never executed."""


@dataclass
class SandboxResult:
    ok: bool = False
    result: dict = field(default_factory=dict)
    figures: list[dict] = field(default_factory=list)
    stdout: str = ""
    error: str = ""
    traceback: str = ""
    duration_ms: int = 0
    violations: list[str] = field(default_factory=list)

    @property
    def failed_validation(self) -> bool:
        return bool(self.violations)


def validate_code(code: str) -> list[str]:
    """Static analysis. Returns a list of violations; empty means approved.

    Violations are phrased for the *model*, not the operator: they are fed
    straight back into the repair prompt, so "module 'os' is not permitted"
    beats "SecurityError at node 14".
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error on line {exc.lineno}: {exc.msg}"]

    violations: list[str] = []

    for node in ast.walk(tree):
        # --- imports ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if alias.name not in ALLOWED_IMPORTS and root not in ALLOWED_IMPORTS:
                    violations.append(
                        f"module '{alias.name}' is not permitted; allowed: "
                        f"{', '.join(sorted(_roots()))}"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if module not in ALLOWED_IMPORTS and root not in ALLOWED_IMPORTS:
                violations.append(f"module '{module}' is not permitted")

        # --- dangerous builtins ---
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            violations.append(f"'{node.id}' is not available in the sandbox")

        # --- attribute-based escapes ---
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            violations.append(f"attribute '{node.attr}' is not accessible")

        # --- dynamic attribute access defeats the check above ---
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"getattr", "setattr", "delattr"}:
                violations.append(
                    f"'{node.func.id}' is not permitted; access attributes directly"
                )

    return sorted(set(violations))


def _roots() -> set[str]:
    return {name.split(".")[0] for name in ALLOWED_IMPORTS}


# Injected around the model's code. The contract is deliberately tiny: a
# DataFrame called `df` goes in, a dict called `RESULT` comes out.
_PREAMBLE = '''
import json, sys
import pandas as pd
import numpy as np

with open(sys.argv[1], "r", encoding="utf-8") as _f:
    _payload = json.load(_f)

df = pd.DataFrame(_payload["data"])
RESULT = {}
FIGURES = []

'''

_EPILOGUE = '''

def _jsonable(value):
    """Numpy scalars and pandas types are not JSON-serialisable, and a model
    returning np.float64 should not fail the whole experiment for it."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if (v != v or v in (float("inf"), float("-inf"))) else v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="list")
    if isinstance(value, pd.Series):
        return _jsonable(value.to_dict())
    return value

with open(sys.argv[2], "w", encoding="utf-8") as _f:
    json.dump({"result": _jsonable(RESULT), "figures": _jsonable(FIGURES)}, _f)
'''


def run_code(
    code: str,
    data: dict[str, list],
    *,
    timeout: int = 45,
) -> SandboxResult:
    """Validate, then execute analysis code against `data` in a child process.

    `data` is a columnar dict, matching `Dataset.data`, and arrives as `df`.
    """
    result = SandboxResult()

    violations = validate_code(code)
    if violations:
        result.violations = violations
        result.error = "code rejected by static analysis"
        log.warning("sandbox rejected code: %s", violations)
        return result

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ara-sandbox-") as tmp:
        workdir = Path(tmp)
        script = workdir / "analysis.py"
        payload = workdir / "input.json"
        output = workdir / "output.json"

        script.write_text(_PREAMBLE + code + _EPILOGUE, encoding="utf-8")
        payload.write_text(json.dumps({"data": data}), encoding="utf-8")

        try:
            completed = subprocess.run(
                # -I: isolated mode. Ignores PYTHON* env vars and the user site
                # directory, so the child cannot be steered by the environment.
                [sys.executable, "-I", str(script), str(payload), str(output)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
                check=False,
            )
        except subprocess.TimeoutExpired:
            result.error = f"execution exceeded {timeout}s and was terminated"
            result.duration_ms = int((time.perf_counter() - started) * 1000)
            return result

        result.duration_ms = int((time.perf_counter() - started) * 1000)
        result.stdout = (completed.stdout or "")[:8000]

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            result.traceback = stderr[-4000:]
            result.error = _last_exception_line(stderr)
            return result

        if not output.exists():
            result.error = "code completed without writing RESULT"
            return result

        try:
            parsed = json.loads(output.read_text(encoding="utf-8"))
        except ValueError as exc:
            result.error = f"RESULT was not JSON-serialisable: {exc}"
            return result

        result.result = parsed.get("result") or {}
        result.figures = parsed.get("figures") or []
        result.ok = True

    return result


def _last_exception_line(stderr: str) -> str:
    """The final traceback line is the part a model can act on."""
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    return lines[-1][:400] if lines else "execution failed with no output"
