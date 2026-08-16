"""Uncertainty Quantifier: measure confidence, then refuse when it is too low.

Asking a model "how confident are you?" produces a number with no relationship
to correctness. Models are famously miscalibrated, and a self-reported 0.9 is
a stylistic choice rather than evidence. So confidence here is assembled from
four things that can actually be measured:

1. **Self-consistency.** The same judgement is sampled k times at a non-zero
   temperature. If the model reaches the same verdict every time, that is weak
   but real evidence of a stable conclusion; if it flip-flops, the conclusion
   is not there.

2. **Cross-model agreement.** The same judgement is put to a *different model
   family*. Two architectures trained by different labs agreeing is
   substantially stronger evidence than one model agreeing with itself -- which
   is all self-consistency can ever show.

3. **Statistical evidence.** Computed directly from the experiment: p-value,
   effect size, interval width, sample size. No model involved.

4. **Evidence quality.** How many independent sources, whether conflicts
   remain open, whether the data spans enough of the domain.

Below the abstention threshold the claim is **not asserted**. It moves to an
"Abstained -- Insufficient Evidence" section, and the UI says so out loud. A
system that visibly declines to answer is the most trustworthy thing in this
submission, and the cheapest to get wrong by omitting.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from pydantic import BaseModel, Field

from ..config import Role
from ..schemas import (
    AgentName,
    ArtifactKind,
    Claim,
    ConfidenceComponents,
    DataBundle,
    ExperimentResult,
    Level,
    ResearchQuestion,
    StatResult,
)
from .base import AgentFailure, BaseAgent

log = logging.getLogger(__name__)


class _Judgement(BaseModel):
    """One sampled verdict on whether a claim is supported."""

    supported: bool = Field(description="Does the evidence support this claim?")
    reasoning: str = ""


class _ClaimSet(BaseModel):
    claims: list[str] = Field(
        min_length=1, description="Specific factual statements this experiment supports"
    )


EXTRACT = """An experiment has finished. State the specific factual claims its results \
support -- the things a paper could assert.

Research question: {question}
Hypothesis tested: {hypothesis}
Prediction made in advance: {prediction}

Results:
{results}

Rules:
- Each claim must be a single, checkable statement.
- Include the direct statistical finding, and any secondary observations the numbers support.
- If the result is null, say so plainly as a claim ("no relationship was detected between X \
and Y"). A null result is a finding, not an absence of one.
- Do NOT overstate. If p > 0.05 the claim is that no effect was detected, not that no effect \
exists.
- 2 to 5 claims."""

JUDGE = """Judge whether this claim is supported by the evidence.

Claim: {claim}

Evidence:
{results}

Data: {sources} sources across {modalities} modalities, {rows} rows analysed.
{conflicts}

Answer supported=true only if the evidence genuinely establishes the claim. Consider whether \
the sample is large enough, whether the effect is meaningful and not merely detectable, and \
whether anything in the evidence contradicts it."""


#: How many claims get a second-model opinion per cycle.
#:
#: The cross-model check is the most informative confidence signal and the
#: most expensive: it is served by a reserved provider whose free tier is
#: metered at 20 requests *per day*. Spending one per claim would exhaust the
#: day's quota in a single run and leave later cycles with no second opinion at
#: all. Claims are scored in order, so this buys the check where it matters
#: most, and the remaining claims record `None` -- which `_combine` handles by
#: redistributing the weight rather than assuming agreement.
MAX_CROSS_MODEL_CHECKS = 2


class UncertaintyQuantifier(BaseAgent):
    name = AgentName.UNCERTAINTY
    fatal_on_failure = False

    async def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        result: ExperimentResult | None = state.get("experiment_result")
        bundle: DataBundle | None = state.get("data_bundle")
        question: ResearchQuestion | None = state.get("question")
        if result is None or bundle is None:
            raise AgentFailure("nothing to quantify uncertainty over")

        cycle = state.get("cycle", 0)
        rendered = _render_results(result)

        claims_text = await self._extract_claims(question, result, rendered)
        self.say(f"Extracted {len(claims_text)} candidate claim(s); scoring each.", cycle=cycle)

        statistical = _statistical_component(result)
        evidence = _evidence_component(bundle)

        claims: list[Claim] = []
        for index, text in enumerate(claims_text):
            components = ConfidenceComponents(
                statistical_evidence=statistical,
                evidence_quality=evidence,
            )
            components.self_consistency = await self._self_consistency(
                text, rendered, bundle, cycle=cycle
            )
            if index < MAX_CROSS_MODEL_CHECKS:
                components.cross_model_agreement = await self._cross_model(
                    text, rendered, bundle
                )

            confidence = _combine(components)
            claim = Claim(
                text=text,
                components=components,
                confidence=confidence,
                supporting_source_urls=[d.url for d in bundle.documents[:5]],
            )

            threshold = self.settings.confidence_abstain_threshold
            if confidence < threshold:
                claim.abstained = True
                claim.abstain_reason = (
                    f"confidence {confidence:.0%} is below the {threshold:.0%} threshold "
                    f"({components.explain()})"
                )
            claims.append(claim)

        asserted = [c for c in claims if not c.abstained]
        abstained = [c for c in claims if c.abstained]

        for claim in abstained:
            self.say(
                f"ABSTAINING — {claim.text[:110]} ({claim.confidence:.0%})",
                level=Level.WARN,
                cycle=cycle,
            )
        for claim in asserted:
            self.say(
                f"{claim.confidence:.0%} — {claim.text[:110]}",
                level=Level.SUCCESS,
                cycle=cycle,
            )

        overall = (
            sum(c.confidence for c in asserted) / len(asserted) if asserted else 0.0
        )

        self._report(claims, asserted, abstained, overall, cycle=cycle)

        return {
            "claims": asserted,
            "abstained_claims": abstained,
            "overall_confidence": round(overall, 3),
            "phase": "critique",
        }

    # -------------------------------------------------------------- claims

    async def _extract_claims(
        self, question: ResearchQuestion | None, result: ExperimentResult, rendered: str
    ) -> list[str]:
        try:
            batch = await self.router.structured(
                Role.REASONING,
                [
                    {
                        "role": "system",
                        "content": "You state precise, checkable claims from statistical results.",
                    },
                    {
                        "role": "user",
                        "content": EXTRACT.format(
                            question=question.text if question else "(unknown)",
                            hypothesis=result.spec.hypothesis.alternative,
                            prediction=result.spec.hypothesis.prediction,
                            results=rendered,
                        ),
                    },
                ],
                _ClaimSet,
                temperature=0.2,
                max_tokens=1000,
            )
            return [c.strip() for c in batch.claims if c.strip()][:5]
        except Exception as exc:  # noqa: BLE001 - fall back to a mechanical claim
            log.warning("claim extraction failed: %s", exc)
            primary = result.primary
            if primary is None:
                return ["The experiment did not produce an interpretable result."]
            direction = "a statistically significant" if primary.significant else "no significant"
            return [
                f"The analysis found {direction} result for {primary.test_name} "
                f"(p = {primary.p_value:.4g}, n = {primary.n})."
                if primary.p_value is not None
                else f"The analysis produced {primary.test_name}."
            ]

    # ---------------------------------------------------------- components

    async def _self_consistency(
        self, claim: str, rendered: str, bundle: DataBundle, *, cycle: int
    ) -> float:
        """Sample the same judgement k times; measure agreement.

        Agreement is mapped so that a unanimous verdict scores 1.0 and a
        perfect 50/50 split scores 0.0 -- a coin flip carries no information
        in either direction.
        """
        k = self.settings.self_consistency_samples
        prompt = self._judge_prompt(claim, rendered, bundle)

        samples = await self.router.structured_samples(
            Role.FAST, prompt, _Judgement, k=k, temperature=0.8, max_tokens=350
        )
        if not samples:
            return 0.0

        supported = sum(1 for s in samples if s.supported)
        agreement = max(supported, len(samples) - supported) / len(samples)
        consistent_verdict = supported > len(samples) / 2

        # A stable "not supported" is a confident refutation, not confidence in
        # the claim, so disagreement with the claim caps the score.
        scaled = (agreement - 0.5) * 2.0
        return round(max(0.0, scaled if consistent_verdict else scaled * 0.35), 3)

    async def _cross_model(
        self, claim: str, rendered: str, bundle: DataBundle
    ) -> float | None:
        """Put the same judgement to a different model family."""
        primary = self.router.provider_names[0] if self.router.provider_names else ""
        judgement, provider = await self.router.structured_cross_model(
            Role.FAST,
            self._judge_prompt(claim, rendered, bundle),
            _Judgement,
            exclude_provider=primary,
            temperature=0.1,
            max_tokens=350,
        )
        if judgement is None or not provider:
            # No second family configured. Returning None rather than a
            # neutral number keeps the absence visible in the report instead of
            # silently inflating confidence.
            return None
        return 1.0 if judgement.supported else 0.0

    def _judge_prompt(self, claim: str, rendered: str, bundle: DataBundle) -> list[dict[str, str]]:
        open_conflicts = bundle.open_conflicts
        conflicts = (
            f"Warning: {len(open_conflicts)} unresolved conflict(s) between sources, "
            f"e.g. {open_conflicts[0].subject}."
            if open_conflicts
            else "No unresolved source conflicts."
        )
        return [
            {"role": "system", "content": "You assess whether evidence supports a claim."},
            {
                "role": "user",
                "content": JUDGE.format(
                    claim=claim,
                    results=rendered,
                    sources=len(bundle.documents),
                    modalities=len(bundle.modalities),
                    rows=bundle.dataset.n_rows if bundle.dataset else 0,
                    conflicts=conflicts,
                ),
            },
        ]

    # --------------------------------------------------------------- output

    def _report(
        self,
        claims: list[Claim],
        asserted: list[Claim],
        abstained: list[Claim],
        overall: float,
        *,
        cycle: int,
    ) -> None:
        from ..analysis.plots import confidence_chart

        threshold = self.settings.confidence_abstain_threshold
        self.publish(
            ArtifactKind.CONFIDENCE_REPORT,
            {
                "threshold": threshold,
                "overall": round(overall, 3),
                "asserted": len(asserted),
                "abstained": len(abstained),
                "claims": [
                    {
                        "text": c.text,
                        "confidence": round(c.confidence, 3),
                        "abstained": c.abstained,
                        "abstain_reason": c.abstain_reason,
                        "components": {
                            "self_consistency": c.components.self_consistency,
                            "cross_model_agreement": c.components.cross_model_agreement,
                            "statistical_evidence": c.components.statistical_evidence,
                            "evidence_quality": c.components.evidence_quality,
                        },
                    }
                    for c in claims
                ],
            },
            message=(
                f"{len(asserted)} claim(s) asserted, {len(abstained)} abstained "
                f"(threshold {threshold:.0%})"
            ),
            cycle=cycle,
        )

        figure = confidence_chart([(c.text, c.confidence) for c in claims], threshold)
        if figure:
            self.publish(
                ArtifactKind.FIGURE,
                {"figure": figure.figure_json, "title": figure.title, "caption": figure.caption},
                message=figure.title,
                cycle=cycle,
            )


def _render_results(result: ExperimentResult) -> str:
    """Format an experiment's output for a model to read.

    Shared with the Critic so both agents reason about an identical rendering
    of the numbers. Two agents shown subtly different summaries of the same
    result will disagree for reasons that have nothing to do with the science.
    """
    if not result.executed_ok:
        return (
            f"The experiment FAILED to execute after {result.repair_attempts} repair "
            f"attempt(s).\nError: {result.error}"
        )

    lines = [f"Procedure: {result.spec.kind.value}", ""]
    for stat in result.stats:
        lines.append(f"### {stat.test_name}")
        if stat.statistic is not None:
            lines.append(f"- statistic: {stat.statistic:.6g}")
        if stat.p_value is not None:
            lines.append(f"- p-value: {stat.p_value:.6g}")
        if stat.p_value_corrected is not None:
            lines.append(
                f"- p-value corrected ({stat.correction_method}): {stat.p_value_corrected:.6g}"
            )
        if stat.effect_size is not None:
            lines.append(
                f"- effect size {stat.effect_size_name}: {stat.effect_size:.4g} "
                f"({stat.effect_interpretation})"
            )
        if stat.ci_low is not None and stat.ci_high is not None:
            lines.append(f"- 95% CI: [{stat.ci_low:.4g}, {stat.ci_high:.4g}]")
        lines.append(f"- n: {stat.n}")
        if stat.comparisons_made > 1:
            lines.append(f"- comparisons made: {stat.comparisons_made}")
        if stat.assumptions_checked:
            checks = ", ".join(
                f"{name}={'pass' if ok else 'FAIL'}"
                for name, ok in stat.assumptions_checked.items()
            )
            lines.append(f"- assumption checks: {checks}")
        for note in stat.notes:
            lines.append(f"- note: {note}")
        lines.append("")

    if result.tables:
        lines.append(f"Additional values: {result.tables}")
    return "\n".join(lines)


# --- measured components ---------------------------------------------------


def _statistical_component(result: ExperimentResult) -> float:
    """Confidence contributed by the statistics alone. No model involved."""
    if not result.executed_ok or result.primary is None:
        return 0.0

    primary: StatResult = result.primary
    score = 0.0

    # Significance, scaled rather than binary: p=0.049 and p=1e-9 are not the
    # same evidence, and a cliff edge at 0.05 is exactly the thinking the
    # Critic exists to attack.
    p = primary.p_value_corrected if primary.p_value_corrected is not None else primary.p_value
    if p is not None:
        if p < 0.001:
            score += 0.40
        elif p < 0.01:
            score += 0.33
        elif p < 0.05:
            score += 0.25
        elif p < 0.1:
            score += 0.08
    elif primary.effect_size is not None:
        # Descriptive procedures (clustering) have no p-value; judge on effect.
        score += 0.15

    score += {
        "large": 0.30,
        "medium": 0.22,
        "small": 0.12,
        "negligible": 0.0,
    }.get(primary.effect_interpretation.lower(), 0.05)

    # Sample size, saturating: 30 rows is workable, 200 is comfortable, and
    # beyond that more rows add little to *this* judgement.
    if primary.n > 0:
        score += 0.20 * min(math.log10(max(primary.n, 1)) / math.log10(200), 1.0)

    # A usable interval that excludes zero.
    if primary.ci_low is not None and primary.ci_high is not None:
        excludes_zero = (primary.ci_low > 0) == (primary.ci_high > 0)
        score += 0.10 if excludes_zero else 0.02

    if primary.comparisons_made > 1 and primary.p_value_corrected is None:
        score -= 0.10  # untested multiplicity

    if primary.assumptions_checked and not all(primary.assumptions_checked.values()):
        score -= 0.05

    return round(max(0.0, min(score, 1.0)), 3)


def _evidence_component(bundle: DataBundle) -> float:
    """Confidence contributed by the data's breadth and cleanliness."""
    score = 0.0
    score += 0.35 * min(len(bundle.modalities) / 4.0, 1.0)
    score += 0.25 * min(len(bundle.documents) / 8.0, 1.0)

    if bundle.dataset:
        score += 0.30 * min(bundle.dataset.n_rows / 60.0, 1.0)
        if bundle.dataset.cleaning.mappings_rejected:
            score -= 0.05

    score -= min(len(bundle.open_conflicts) * 0.07, 0.25)
    score -= min(len(bundle.acquisition_failures) * 0.05, 0.20)
    return round(max(0.0, min(score, 1.0)), 3)


def _combine(components: ConfidenceComponents) -> float:
    """Blend the four signals into one confidence.

    Statistical evidence carries the most weight because it is the only
    component computed without a model in the loop. When no second model family
    is available its weight is redistributed rather than defaulted, so a
    single-provider deployment does not get free confidence it did not earn.
    """
    weights = {
        "statistical": 0.40,
        "self_consistency": 0.20,
        "cross_model": 0.20,
        "evidence": 0.20,
    }

    if components.cross_model_agreement is None:
        # Redistribute proportionally across the components that exist.
        missing = weights.pop("cross_model")
        total = sum(weights.values())
        weights = {k: v + missing * (v / total) for k, v in weights.items()}
        cross = 0.0
    else:
        cross = components.cross_model_agreement

    score = (
        weights["statistical"] * components.statistical_evidence
        + weights["self_consistency"] * components.self_consistency
        + weights.get("cross_model", 0.0) * cross
        + weights["evidence"] * components.evidence_quality
    )
    return round(max(0.0, min(score, 1.0)), 3)
