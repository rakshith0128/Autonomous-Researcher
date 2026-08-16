"""Keeps the two dependency manifests honest.

pyproject.toml drives local development; requirements.txt drives the container
image. They exist separately so that a code edit does not invalidate the Docker
layer that installs scipy. That is a real build-time saving and a real drift
risk, so the drift is tested rather than trusted -- a dependency that works
locally and is missing from the image fails here instead of on the deployed URL.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Normalise the handful of names that differ between PyPI and import form.
_NAME_RE = re.compile(r"^([A-Za-z0-9._-]+)")


def _canonical(name: str) -> str:
    return _NAME_RE.match(name.strip()).group(1).lower().replace("_", "-").replace(".", "-")


def _pyproject_deps() -> dict[str, str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    out = {}
    for spec in data["project"]["dependencies"]:
        # Strip extras: "uvicorn[standard]>=0.32" -> "uvicorn"
        bare = spec.split("[")[0]
        out[_canonical(bare)] = spec.strip()
    return out


def _requirements_deps() -> dict[str, str]:
    out = {}
    for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        bare = line.split("[")[0]
        out[_canonical(bare)] = line
    return out


def test_requirements_covers_every_pyproject_dependency():
    missing = sorted(set(_pyproject_deps()) - set(_requirements_deps()))
    assert not missing, (
        f"in pyproject.toml but not requirements.txt: {missing}. "
        "The container would start and then fail at import time."
    )


def test_requirements_adds_nothing_unexpected():
    extra = sorted(set(_requirements_deps()) - set(_pyproject_deps()))
    assert not extra, (
        f"in requirements.txt but not pyproject.toml: {extra}. "
        "Local development would not install these."
    )


def test_version_constraints_agree():
    pyproject, requirements = _pyproject_deps(), _requirements_deps()
    mismatched = [
        name
        for name in set(pyproject) & set(requirements)
        if pyproject[name].replace(" ", "") != requirements[name].replace(" ", "")
    ]
    assert not mismatched, f"version constraints differ between manifests: {mismatched}"


def test_dockerfile_targets_the_port_hugging_face_expects():
    """HF Spaces routes to 7860; any other port shows as a permanently
    building Space with no error message."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "EXPOSE 7860" in dockerfile
    assert "--port" in dockerfile and "7860" in dockerfile.split("CMD")[-1]


def test_dockerfile_runs_as_uid_1000():
    """HF Spaces containers run unprivileged; writing as root fails at runtime."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "useradd" in dockerfile and "-u 1000" in dockerfile
    assert "USER user" in dockerfile


def test_secrets_are_excluded_from_the_build_context():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert ".env" in ignored
    gitignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignored
