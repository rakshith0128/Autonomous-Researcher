"""Statistical helpers: effect sizes, intervals, and honest interpretation.

The assessment's iteration rule is "p-value > 0.05 or effect size trivial", so
effect size has to be a real computed quantity with a defensible label, not an
adjective a model chose. Every interpretation here uses conventional Cohen
thresholds, and the thresholds are stated in the output so a reader can
disagree with the convention rather than the arithmetic.

Bootstrap intervals are used in preference to parametric ones wherever the
distribution is unknown, which for scraped bibliometric data is always.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Conventional Cohen thresholds. Exposed rather than inlined so the paper can
# state which convention it is using.
THRESHOLDS: dict[str, tuple[float, float, float]] = {
    # name: (small, medium, large)
    "r": (0.1, 0.3, 0.5),
    "rho": (0.1, 0.3, 0.5),
    "tau": (0.07, 0.21, 0.35),
    "d": (0.2, 0.5, 0.8),
    "eta2": (0.01, 0.06, 0.14),
    "r2": (0.01, 0.09, 0.25),
    "cliffs_delta": (0.147, 0.33, 0.474),
}


def interpret_effect(name: str, value: float | None) -> str:
    """Label an effect size, or say plainly that there isn't one."""
    if value is None:
        return "unknown"
    thresholds = THRESHOLDS.get(name.lower())
    if thresholds is None:
        return "uninterpreted"

    small, medium, large = thresholds
    magnitude = abs(value)
    if magnitude < small:
        return "negligible"
    if magnitude < medium:
        return "small"
    if magnitude < large:
        return "medium"
    return "large"


def cohens_d(a: list[float], b: list[float]) -> float | None:
    """Standardised mean difference with a pooled standard deviation."""
    if len(a) < 2 or len(b) < 2:
        return None
    mean_a, mean_b = _mean(a), _mean(b)
    var_a, var_b = _variance(a), _variance(b)
    pooled = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)
    if pooled <= 0:
        return None
    return (mean_a - mean_b) / math.sqrt(pooled)


def cliffs_delta(a: list[float], b: list[float]) -> float | None:
    """Non-parametric effect size for Mann-Whitney.

    Reported alongside Cohen's d because when the data fails a normality check
    -- routine for citation counts, which are heavily skewed -- d is the wrong
    summary and quoting it alone would overstate the result.
    """
    if not a or not b:
        return None
    greater = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    return (greater - less) / (len(a) * len(b))


def bootstrap_ci(
    values: list[float],
    statistic,  # noqa: ANN001 - any callable over a resample
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap interval.

    Seeded so a rerun of the same data reproduces the same interval; the run
    manifest records the seed.
    """
    if len(values) < 3:
        return None, None
    try:
        import numpy as np

        rng = np.random.default_rng(seed)
        array = np.asarray(values, dtype=float)
        samples = [
            statistic(rng.choice(array, size=len(array), replace=True))
            for _ in range(n_resamples)
        ]
        alpha = (1.0 - confidence) / 2.0
        return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))
    except Exception as exc:  # noqa: BLE001 - an absent interval is not a failed test
        log.warning("bootstrap failed: %s", exc)
        return None, None


def correlation_ci(r: float, n: int, confidence: float = 0.95) -> tuple[float | None, float | None]:
    """Fisher z interval for a correlation coefficient."""
    if n < 4 or abs(r) >= 1.0:
        return None, None
    try:
        from scipy import stats as sp

        z = math.atanh(r)
        se = 1.0 / math.sqrt(n - 3)
        critical = sp.norm.ppf(1 - (1 - confidence) / 2)
        return math.tanh(z - critical * se), math.tanh(z + critical * se)
    except Exception:  # noqa: BLE001
        return None, None


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """FDR correction.

    Used whenever more than one hypothesis is tested in a cycle. The Critic
    checks for this explicitly, and testing six relationships then reporting
    the best raw p-value is the single most common way a plausible-looking
    result turns out to be noise.
    """
    if not p_values:
        return []
    try:
        from statsmodels.stats.multitest import multipletests

        return list(multipletests(p_values, method="fdr_bh")[1])
    except Exception:  # noqa: BLE001 - fall back to the manual computation
        indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
        total = len(p_values)
        corrected = [0.0] * total
        previous = 1.0
        for rank, (original_index, p) in enumerate(reversed(indexed), start=1):
            position = total - rank + 1
            value = min(previous, p * total / position)
            corrected[original_index] = value
            previous = value
        return corrected


def check_normality(values: list[float]) -> tuple[bool, float | None]:
    """Shapiro-Wilk. Returns (looks_normal, p_value)."""
    if len(values) < 3:
        return False, None
    try:
        from scipy import stats as sp

        # Shapiro is unreliable above ~5000 points and rejects trivially there.
        sample = values[:4999]
        _, p = sp.shapiro(sample)
        return bool(p > 0.05), float(p)
    except Exception:  # noqa: BLE001
        return False, None


def required_n_for_correlation(effect: float = 0.3, power: float = 0.8) -> int:
    """Approximate sample size needed to detect a correlation.

    Powers the Critic's "n too small" objection with a number rather than a
    feeling.
    """
    if abs(effect) >= 1.0 or effect == 0:
        return 0
    try:
        z_alpha, z_beta = 1.96, 0.84 if power <= 0.8 else 1.28
        z_r = 0.5 * math.log((1 + effect) / (1 - effect))
        return int(math.ceil(((z_alpha + z_beta) / z_r) ** 2 + 3))
    except (ValueError, ZeroDivisionError):
        return 0


@dataclass
class Describe:
    n: int
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "mean": round(self.mean, 4),
            "median": round(self.median, 4),
            "std": round(self.std, 4),
            "min": round(self.minimum, 4),
            "max": round(self.maximum, 4),
        }


def describe(values: list[float]) -> Describe | None:
    numeric = [v for v in values if isinstance(v, (int, float)) and not _is_nan(v)]
    if not numeric:
        return None
    ordered = sorted(numeric)
    mid = len(ordered) // 2
    median = (
        ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    )
    return Describe(
        n=len(numeric),
        mean=_mean(numeric),
        median=median,
        std=math.sqrt(_variance(numeric)),
        minimum=ordered[0],
        maximum=ordered[-1],
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def _is_nan(value: float) -> bool:
    return value != value
