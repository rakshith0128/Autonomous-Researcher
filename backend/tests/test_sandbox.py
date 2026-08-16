"""Tests for sandboxed execution of model-written code.

Split into static analysis (fast, no subprocess) and real execution (slower,
marked so it can be skipped locally). The escape-attempt cases are the ones
worth reading: each is a known way out of a naive Python sandbox.
"""

from __future__ import annotations

import pytest

from backend.tools.sandbox import (
    ALLOWED_IMPORTS,
    run_code,
    validate_code,
)

DATA = {"x": [1, 2, 3, 4, 5, 6], "y": [2, 4, 6, 8, 10, 12], "g": ["a", "a", "a", "b", "b", "b"]}


class TestStaticAnalysisAccepts:
    def test_plain_statistics(self):
        assert validate_code("from scipy import stats\nRESULT = {'n': len(df)}") == []

    def test_every_whitelisted_root_imports_cleanly(self):
        roots = sorted({name.split(".")[0] for name in ALLOWED_IMPORTS})
        code = "\n".join(f"import {root}" for root in roots)
        assert validate_code(code) == []

    def test_submodule_import(self):
        assert validate_code("import statsmodels.api as sm") == []


class TestStaticAnalysisRejects:
    @pytest.mark.parametrize(
        "code",
        [
            "import os",
            "import subprocess",
            "import socket",
            "import requests",
            "from os import path",
            "import urllib.request",
            "import shutil",
        ],
    )
    def test_forbidden_modules(self, code: str):
        assert validate_code(code), f"should have rejected: {code}"

    @pytest.mark.parametrize(
        "code",
        [
            "eval('1+1')",
            "exec('x=1')",
            "open('/etc/passwd')",
            "__import__('os')",
            "compile('x', '<s>', 'exec')",
            "g = globals()",
        ],
    )
    def test_forbidden_builtins(self, code: str):
        assert validate_code(code), f"should have rejected: {code}"

    @pytest.mark.parametrize(
        "code",
        [
            # The canonical Python sandbox escape: climb the class hierarchy
            # to reach builtins, then import anything.
            "().__class__.__bases__[0].__subclasses__()",
            "(lambda: 0).__globals__['__builtins__']",
            "df.__class__.__mro__",
            "some_func.__code__",
        ],
    )
    def test_object_graph_traversal(self, code: str):
        assert validate_code(code), f"should have rejected: {code}"

    def test_dynamic_attribute_access_is_blocked(self):
        """Otherwise getattr(x, '__' + 'globals__') walks straight past the
        static attribute check."""
        assert validate_code("getattr(df, '__cl' + 'ass__')")

    def test_syntax_error_is_reported_not_raised(self):
        violations = validate_code("def broken(:\n  pass")
        assert violations and "syntax error" in violations[0]

    def test_violation_text_is_written_for_the_model(self):
        """These strings go straight into the repair prompt."""
        violations = validate_code("import os")
        assert "not permitted" in violations[0]
        assert "allowed:" in violations[0]


class TestExecution:
    def test_runs_real_statistics(self):
        code = """
from scipy import stats
r, p = stats.pearsonr(df['x'], df['y'])
RESULT = {'r': r, 'p': p, 'n': len(df)}
"""
        out = run_code(code, DATA, timeout=30)
        assert out.ok, f"{out.error}\n{out.traceback}"
        assert out.result["n"] == 6
        assert out.result["r"] == pytest.approx(1.0)

    def test_numpy_scalars_survive_serialisation(self):
        """np.float64 is not JSON-serialisable; a model returning one should
        not lose the whole experiment to it."""
        code = "import numpy as np\nRESULT = {'m': np.mean(df['x']), 'c': np.int64(3)}"
        out = run_code(code, DATA, timeout=30)
        assert out.ok, out.error
        assert out.result["m"] == pytest.approx(3.5)
        assert out.result["c"] == 3

    def test_nan_becomes_null_rather_than_invalid_json(self):
        code = "import numpy as np\nRESULT = {'bad': np.float64('nan')}"
        out = run_code(code, DATA, timeout=30)
        assert out.ok, out.error
        assert out.result["bad"] is None

    def test_rejected_code_never_executes(self):
        out = run_code("import os\nos.system('echo pwned')", DATA)
        assert not out.ok
        assert out.failed_validation
        assert out.duration_ms == 0, "validation must short-circuit before spawning a process"

    def test_runtime_error_returns_an_actionable_message(self):
        """The final traceback line is what gets fed back for self-repair."""
        out = run_code("RESULT = {'v': df['nonexistent_column'].mean()}", DATA, timeout=30)
        assert not out.ok
        assert "nonexistent_column" in out.traceback
        assert out.error

    def test_missing_result_is_reported(self):
        out = run_code("x = 1 + 1", DATA, timeout=30)
        assert out.ok is True
        assert out.result == {}

    def test_stdout_is_captured(self):
        out = run_code("print('hello from the sandbox')\nRESULT = {'ok': True}", DATA, timeout=30)
        assert out.ok, out.error
        assert "hello from the sandbox" in out.stdout

    @pytest.mark.slow
    def test_infinite_loop_is_killed(self):
        """Static analysis deliberately does not try to detect this; the
        timeout is the mechanism that makes that safe."""
        out = run_code("while True:\n    pass\nRESULT={}", DATA, timeout=3)
        assert not out.ok
        assert "exceeded" in out.error

    def test_groupby_comparison_end_to_end(self):
        code = """
from scipy import stats
a = df[df['g'] == 'a']['y']
b = df[df['g'] == 'b']['y']
stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')
RESULT = {'statistic': stat, 'p_value': p, 'n_a': len(a), 'n_b': len(b)}
"""
        out = run_code(code, DATA, timeout=30)
        assert out.ok, f"{out.error}\n{out.traceback}"
        assert out.result["n_a"] == 3 and out.result["n_b"] == 3
        assert 0.0 <= out.result["p_value"] <= 1.0
