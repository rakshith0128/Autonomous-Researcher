"""Typed experiment registry: real statistics from validated parameters.

The Experiment Designer emits JSON naming a procedure and its columns. Python
runs the actual test. The model never writes the statistics, which is the
whole point -- a model asked to "compute a p-value" will produce a number that
looks like a p-value.

Why a registry at all, when there is also a code sandbox? Reliability under an
audience. An agent writing free-form scipy fails often enough that a live demo
becomes a coin flip, while these six procedures cover most questions that
scraped bibliometric data can actually answer. The sandbox remains for
everything else, and the Designer chooses.

Every procedure returns `StatResult` carrying a p-value, an effect size with a
conventional label, an interval where one is computable, and the assumption
checks the Critic will ask about.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..schemas import Dataset, ExperimentKind, FigureSpec, StatResult
from .plots import group_comparison_chart, scatter_with_fit, timeseries_chart
from .stats import (
    benjamini_hochberg,
    check_normality,
    cliffs_delta,
    cohens_d,
    correlation_ci,
    describe,
    interpret_effect,
    required_n_for_correlation,
)

log = logging.getLogger(__name__)

MIN_ROWS = 8


class ExperimentError(ValueError):
    """Raised when a spec cannot be run against the supplied dataset.

    The message is written for the Experiment Designer, because it is fed back
    verbatim so it can correct the spec.
    """


def _numeric_column(dataset: Dataset, name: str) -> list[float]:
    """Extract a column as floats, dropping unusable entries."""
    if name not in dataset.data:
        raise ExperimentError(
            f"column {name!r} does not exist. Available columns: {', '.join(dataset.columns)}"
        )
    values: list[float] = []
    for raw in dataset.data[name]:
        if raw is None or isinstance(raw, bool):
            values.append(float("nan"))
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return values


def _paired(x: list[float], y: list[float]) -> tuple[list[float], list[float]]:
    """Drop rows where either value is missing, keeping the pairing intact."""
    pairs = [(a, b) for a, b in zip(x, y, strict=True) if a == a and b == b]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _require_rows(n: int, procedure: str) -> None:
    if n < MIN_ROWS:
        raise ExperimentError(
            f"{procedure} needs at least {MIN_ROWS} complete rows but only {n} are usable. "
            "Choose different columns or a different procedure."
        )


# --- procedures ------------------------------------------------------------


def run_correlation(dataset: Dataset, params: dict[str, Any]) -> tuple[list[StatResult], list[FigureSpec]]:
    """Pearson and Spearman between two numeric columns.

    Both are reported because they disagree in an informative way: a large
    Spearman with a small Pearson means a monotone but non-linear relationship,
    and citation data is full of those.
    """
    x_name, y_name = params.get("x"), params.get("y")
    if not x_name or not y_name:
        raise ExperimentError("correlation requires params 'x' and 'y' naming numeric columns")

    from scipy import stats as sp

    x, y = _paired(_numeric_column(dataset, x_name), _numeric_column(dataset, y_name))
    _require_rows(len(x), "correlation")
    if len({*x}) < 2 or len({*y}) < 2:
        raise ExperimentError(
            f"column {x_name if len({*x}) < 2 else y_name!r} is constant, so no correlation exists"
        )

    pearson_r, pearson_p = sp.pearsonr(x, y)
    spearman_r, spearman_p = sp.spearmanr(x, y)

    corrected = benjamini_hochberg([float(pearson_p), float(spearman_p)])
    normal_x, p_norm_x = check_normality(x)
    normal_y, p_norm_y = check_normality(y)
    low, high = correlation_ci(float(pearson_r), len(x))

    results = [
        StatResult(
            test_name=f"Pearson correlation ({x_name} vs {y_name})",
            statistic=float(pearson_r),
            p_value=float(pearson_p),
            p_value_corrected=corrected[0],
            correction_method="Benjamini-Hochberg FDR",
            effect_size=float(pearson_r),
            effect_size_name="r",
            effect_interpretation=interpret_effect("r", float(pearson_r)),
            ci_low=low,
            ci_high=high,
            n=len(x),
            comparisons_made=2,
            assumptions_checked={
                f"{x_name} normal": normal_x,
                f"{y_name} normal": normal_y,
            },
            notes=[
                f"Shapiro-Wilk p: {x_name}={p_norm_x}, {y_name}={p_norm_y}",
                f"n={len(x)}; detecting r=0.3 at 80% power needs "
                f"n>={required_n_for_correlation(0.3)}",
            ],
        ),
        StatResult(
            test_name=f"Spearman rank correlation ({x_name} vs {y_name})",
            statistic=float(spearman_r),
            p_value=float(spearman_p),
            p_value_corrected=corrected[1],
            correction_method="Benjamini-Hochberg FDR",
            effect_size=float(spearman_r),
            effect_size_name="rho",
            effect_interpretation=interpret_effect("rho", float(spearman_r)),
            n=len(x),
            comparisons_made=2,
            notes=["rank-based; robust to the skew typical of citation counts"],
        ),
    ]

    # If the data is non-normal, Spearman is the honest headline result, so it
    # is promoted ahead of Pearson rather than buried second.
    if not (normal_x and normal_y):
        results.reverse()
        results[0].notes.append(
            "promoted ahead of Pearson because a normality assumption failed"
        )

    figure = scatter_with_fit(
        x,
        y,
        x_label=x_name,
        y_label=y_name,
        title=f"{y_name} against {x_name}",
        caption=f"n={len(x)}. Line is least-squares; see the rank correlation for monotonicity.",
    )
    return results, [f for f in [figure] if f]


def run_group_comparison(
    dataset: Dataset, params: dict[str, Any]
) -> tuple[list[StatResult], list[FigureSpec]]:
    """Compare a numeric value across a categorical split.

    Chooses Welch's t-test or Mann-Whitney on the normality check rather than
    defaulting to t and hoping, and reports the matching effect size for
    whichever ran.
    """
    value_name, group_name = params.get("value"), params.get("group")
    if not value_name or not group_name:
        raise ExperimentError(
            "group_comparison requires params 'value' (numeric column) and 'group' (categorical column)"
        )
    if group_name not in dataset.data:
        raise ExperimentError(f"group column {group_name!r} does not exist")

    from scipy import stats as sp

    values = _numeric_column(dataset, value_name)
    labels = dataset.data[group_name]

    buckets: dict[str, list[float]] = {}
    for label, value in zip(labels, values, strict=True):
        if value != value or label is None:
            continue
        buckets.setdefault(str(label), []).append(value)

    usable = {k: v for k, v in buckets.items() if len(v) >= 3}
    if len(usable) < 2:
        raise ExperimentError(
            f"column {group_name!r} does not split the data into two groups of at least 3 "
            f"(found: { {k: len(v) for k, v in buckets.items()} })"
        )

    ordered = sorted(usable.items(), key=lambda kv: len(kv[1]), reverse=True)[:2]
    (name_a, group_a), (name_b, group_b) = ordered
    _require_rows(len(group_a) + len(group_b), "group comparison")

    normal_a, _ = check_normality(group_a)
    normal_b, _ = check_normality(group_b)
    parametric = normal_a and normal_b

    if parametric:
        statistic, p_value = sp.ttest_ind(group_a, group_b, equal_var=False)
        test_name = f"Welch t-test ({value_name} by {group_name})"
        effect, effect_name = cohens_d(group_a, group_b), "d"
    else:
        statistic, p_value = sp.mannwhitneyu(group_a, group_b, alternative="two-sided")
        test_name = f"Mann-Whitney U ({value_name} by {group_name})"
        effect, effect_name = cliffs_delta(group_a, group_b), "cliffs_delta"

    result = StatResult(
        test_name=test_name,
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size=effect,
        effect_size_name=effect_name,
        effect_interpretation=interpret_effect(effect_name, effect),
        n=len(group_a) + len(group_b),
        assumptions_checked={f"{name_a} normal": normal_a, f"{name_b} normal": normal_b},
        notes=[
            f"groups: {name_a} (n={len(group_a)}), {name_b} (n={len(group_b)})",
            "non-parametric test chosen because a normality check failed"
            if not parametric
            else "parametric test; both groups passed Shapiro-Wilk",
        ],
    )

    figure = group_comparison_chart(
        {name_a: group_a, name_b: group_b},
        value_label=value_name,
        title=f"{value_name} by {group_name}",
        caption="Boxes show median and quartiles; every observation is plotted.",
    )
    return [result], [f for f in [figure] if f]


def run_regression(dataset: Dataset, params: dict[str, Any]) -> tuple[list[StatResult], list[FigureSpec]]:
    """OLS with heteroskedasticity-robust standard errors."""
    outcome = params.get("outcome") or params.get("y")
    predictors = params.get("predictors") or params.get("features") or []
    if not outcome or not predictors:
        raise ExperimentError(
            "regression requires 'outcome' and a non-empty 'predictors' list of numeric columns"
        )
    if isinstance(predictors, str):
        predictors = [predictors]

    import numpy as np
    import statsmodels.api as sm

    y_all = _numeric_column(dataset, outcome)
    x_columns = [_numeric_column(dataset, name) for name in predictors]

    rows = [
        (y, *xs)
        for y, *xs in zip(y_all, *x_columns, strict=True)
        if y == y and all(x == x for x in xs)
    ]
    _require_rows(len(rows), "regression")
    if len(rows) <= len(predictors) + 1:
        raise ExperimentError(
            f"regression with {len(predictors)} predictors needs more than {len(predictors) + 1} "
            f"complete rows but has {len(rows)}"
        )

    y = np.array([r[0] for r in rows], dtype=float)
    X = sm.add_constant(np.array([r[1:] for r in rows], dtype=float))
    model = sm.OLS(y, X).fit(cov_type="HC3")

    results = [
        StatResult(
            test_name=f"OLS: {outcome} ~ {' + '.join(predictors)}",
            statistic=float(model.fvalue) if model.fvalue is not None else None,
            p_value=float(model.f_pvalue) if model.f_pvalue is not None else None,
            effect_size=float(model.rsquared),
            effect_size_name="r2",
            effect_interpretation=interpret_effect("r2", float(model.rsquared)),
            n=int(model.nobs),
            dof=float(model.df_resid),
            comparisons_made=len(predictors),
            notes=[
                f"adjusted R^2 = {model.rsquared_adj:.4f}",
                "HC3 robust standard errors",
                *[
                    f"{name}: beta={model.params[i + 1]:.4g}, p={model.pvalues[i + 1]:.4g}"
                    for i, name in enumerate(predictors)
                ],
            ],
        )
    ]

    figure = None
    if len(predictors) == 1:
        xs = [r[1] for r in rows]
        figure = scatter_with_fit(
            xs,
            list(y),
            x_label=predictors[0],
            y_label=outcome,
            title=f"{outcome} against {predictors[0]}",
            caption=f"OLS fit, n={len(rows)}, R^2={model.rsquared:.3f}",
        )
    return results, [f for f in [figure] if f]


def run_trend(dataset: Dataset, params: dict[str, Any]) -> tuple[list[StatResult], list[FigureSpec]]:
    """Monotonic trend over an ordered column, via Mann-Kendall.

    Rank-based, so it does not assume linearity or normality -- both wrong for
    publication counts over time.
    """
    time_name = params.get("time") or params.get("x")
    value_name = params.get("value") or params.get("y")
    if not time_name or not value_name:
        raise ExperimentError("trend requires 'time' and 'value' column names")

    from scipy import stats as sp

    t, v = _paired(_numeric_column(dataset, time_name), _numeric_column(dataset, value_name))
    _require_rows(len(t), "trend")

    ordered = sorted(zip(t, v, strict=True))
    times = [p[0] for p in ordered]
    values = [p[1] for p in ordered]

    tau, p_value = sp.kendalltau(times, values)
    slope, intercept, r_value, slope_p, _ = sp.linregress(times, values)

    results = [
        StatResult(
            test_name=f"Mann-Kendall trend ({value_name} over {time_name})",
            statistic=float(tau),
            p_value=float(p_value),
            effect_size=float(tau),
            effect_size_name="tau",
            effect_interpretation=interpret_effect("tau", float(tau)),
            n=len(times),
            notes=[
                f"Theil-Sen style slope estimate: {slope:.4g} per unit {time_name}",
                f"linear fit r={r_value:.3f}, p={slope_p:.4g}",
                "rank-based; assumes neither linearity nor normality",
            ],
        )
    ]

    buckets: dict[str, int] = {}
    for time_value in times:
        buckets[str(int(time_value))] = buckets.get(str(int(time_value)), 0) + 1
    figure = timeseries_chart(
        buckets,
        f"{value_name} over {time_name}",
        f"Kendall tau={tau:.3f}, p={p_value:.4g}, n={len(times)}",
    )
    return results, [f for f in [figure] if f]


def run_clustering(dataset: Dataset, params: dict[str, Any]) -> tuple[list[StatResult], list[FigureSpec]]:
    """KMeans with a k sweep, scored by silhouette."""
    columns = params.get("columns") or params.get("features") or []
    if isinstance(columns, str):
        columns = [columns]
    if len(columns) < 2:
        raise ExperimentError("clustering requires at least 2 numeric columns in 'columns'")

    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    matrix = [_numeric_column(dataset, name) for name in columns]
    rows = [r for r in zip(*matrix, strict=True) if all(v == v for v in r)]
    _require_rows(len(rows), "clustering")

    X = StandardScaler().fit_transform(np.array(rows, dtype=float))

    best = (None, -1.0, 0)
    for k in range(2, min(6, len(rows) // 2 + 1)):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(X, labels))
        if score > best[1]:
            best = (labels, score, k)

    labels, score, k = best
    if labels is None:
        raise ExperimentError("no clustering with k>=2 could be fitted to this data")

    return [
        StatResult(
            test_name=f"KMeans clustering over {', '.join(columns)}",
            statistic=score,
            effect_size=score,
            effect_size_name="silhouette",
            effect_interpretation=(
                "large" if score > 0.5 else "medium" if score > 0.35 else "negligible"
            ),
            n=len(rows),
            notes=[
                f"best k={k} by silhouette across k=2..5",
                "silhouette below ~0.35 indicates no meaningful cluster structure",
                # Clustering has no null hypothesis, so there is no p-value to
                # report. Saying so is better than inventing one.
                "no p-value: clustering is descriptive, not a hypothesis test",
            ],
        )
    ], []


def run_classification(
    dataset: Dataset, params: dict[str, Any]
) -> tuple[list[StatResult], list[FigureSpec]]:
    """Cross-validated classification against a stratified baseline.

    The baseline comparison is the point. Raw accuracy on an imbalanced target
    is meaningless -- 90% accuracy predicting a 90%-majority class is zero
    skill -- so the reported effect is the improvement over chance.
    """
    target = params.get("target") or params.get("outcome")
    features = params.get("features") or params.get("predictors") or []
    if not target or not features:
        raise ExperimentError("classification requires 'target' and a 'features' list")
    if isinstance(features, str):
        features = [features]
    if target not in dataset.data:
        raise ExperimentError(f"target column {target!r} does not exist")

    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    feature_columns = [_numeric_column(dataset, name) for name in features]
    raw_targets = dataset.data[target]

    rows = [
        (label, *values)
        for label, *values in zip(raw_targets, *feature_columns, strict=True)
        if label is not None and all(v == v for v in values)
    ]
    _require_rows(len(rows), "classification")

    y = np.array([str(r[0]) for r in rows])
    X = np.array([r[1:] for r in rows], dtype=float)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ExperimentError(f"target {target!r} has only one class, so nothing to classify")
    folds = int(min(5, counts.min()))
    if folds < 2:
        raise ExperimentError(
            f"smallest class in {target!r} has {counts.min()} example(s); need at least 2"
        )

    model_scores = cross_val_score(
        RandomForestClassifier(n_estimators=120, random_state=42), X, y, cv=folds
    )
    baseline_scores = cross_val_score(
        DummyClassifier(strategy="stratified", random_state=42), X, y, cv=folds
    )

    improvement = float(model_scores.mean() - baseline_scores.mean())
    from scipy import stats as sp

    _, p_value = sp.ttest_rel(model_scores, baseline_scores)

    return [
        StatResult(
            test_name=f"Random forest predicting {target} ({folds}-fold CV)",
            statistic=float(model_scores.mean()),
            p_value=float(p_value),
            effect_size=improvement,
            effect_size_name="accuracy_gain",
            effect_interpretation=(
                "large" if improvement > 0.2 else "medium" if improvement > 0.1
                else "small" if improvement > 0.03 else "negligible"
            ),
            ci_low=float(model_scores.mean() - model_scores.std()),
            ci_high=float(model_scores.mean() + model_scores.std()),
            n=len(rows),
            notes=[
                f"model accuracy {model_scores.mean():.3f} +/- {model_scores.std():.3f}",
                f"stratified baseline {baseline_scores.mean():.3f}",
                f"improvement over chance: {improvement:+.3f}",
                f"class balance: {dict(zip([str(c) for c in classes], [int(n) for n in counts], strict=True))}",
            ],
        )
    ], []


REGISTRY: dict[ExperimentKind, Callable[[Dataset, dict[str, Any]], tuple[list[StatResult], list[FigureSpec]]]] = {
    ExperimentKind.CORRELATION: run_correlation,
    ExperimentKind.GROUP_COMPARISON: run_group_comparison,
    ExperimentKind.REGRESSION: run_regression,
    ExperimentKind.TREND: run_trend,
    ExperimentKind.CLUSTERING: run_clustering,
    ExperimentKind.CLASSIFICATION: run_classification,
}


def describe_columns(dataset: Dataset) -> str:
    """Column summary handed to the Designer so it picks real columns.

    Including the summary statistics matters: a model that can see a column is
    constant, or almost entirely zero, stops proposing it as a predictor.
    """
    lines = []
    for name in dataset.columns:
        dtype = dataset.dtypes.get(name, "unknown")
        summary = describe([v for v in dataset.data[name] if isinstance(v, (int, float))])
        if summary and dtype in {"int", "float"}:
            lines.append(
                f"- {name} ({dtype}): n={summary.n}, mean={summary.mean:.3g}, "
                f"median={summary.median:.3g}, range [{summary.minimum:.3g}, {summary.maximum:.3g}]"
            )
        else:
            distinct = len({str(v) for v in dataset.data[name] if v is not None})
            example = next((v for v in dataset.data[name] if v is not None), "")
            lines.append(
                f"- {name} ({dtype}): {distinct} distinct values, e.g. {str(example)[:50]!r}"
            )
    return "\n".join(lines)


def run_registry_experiment(
    kind: ExperimentKind, dataset: Dataset, params: dict[str, Any]
) -> tuple[list[StatResult], list[FigureSpec]]:
    """Dispatch to a procedure, translating failures into Designer feedback."""
    procedure = REGISTRY.get(kind)
    if procedure is None:
        raise ExperimentError(f"no registry procedure for {kind}")
    return procedure(dataset, params)
