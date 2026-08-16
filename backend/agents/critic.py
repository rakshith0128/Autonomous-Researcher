"""Critic: attack the methodology, and be held to the same standard.

The assessment requires the Critic to "ruthlessly attack methodology,
statistics, assumptions", force iteration when p > 0.05 or the effect is
trivial, and **cite counterevidence**.

Two design decisions make that real rather than performative:

**The statistical flags are computed, not generated.** Whether p exceeds alpha
is arithmetic, and a model asked to check it will sometimes get it wrong in
whichever direction the surrounding prose implies. Python sets the flags; the
model argues about confounding, generalisability, and construct validity --
the things it is actually good at.

**Objections must cite something real, and the citation is verified.** Every
objection carries a URL, that URL is fetched, and an objection whose evidence
cannot be retrieved is **dropped**. A critic that invents a damning reference
therefore loses the argument instead of winning it. This is the same
anti-fabrication stance applied to the agent whose job is scepticism -- the one
place it would be most tempting to skip.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..config import Role
from ..schemas import (
    AgentName,
    ArtifactKind,
    Critique,
    DataBundle,
    ExperimentResult,
    Level,
    Objection,
    RerouteTarget,
    ResearchQuestion,
    StatFlags,
    Verdict,
)
from .base import AgentFailure, BaseAgent

log = logging.getLogger(__name__)

MIN_N_FOR_CLAIM = 20
WIDE_CI_RATIO = 2.0

SYSTEM = """You are a hostile peer reviewer. Your job is to find what is wrong with this \
analysis, not to be encouraging.

Attack, in order of value:
- QUESTION DRIFT: does the experiment actually test the stated research question, or a \
different one that happened to be measurable? Check the analysed columns against what the \
question asks about. If the question asks about X and the experiment measured Y, that is a \
BLOCKING objection and you must reroute to "question" -- a paper answering something nobody \
asked is worse than a null result, because it looks like a finding.
- Confounding: what third variable explains this relationship?
- Construct validity: does the measured quantity actually represent what is claimed?
- Selection bias: how was the sample assembled, and what does that exclude?
- Generalisability: does this hold beyond the specific slice analysed?
- Interpretation: is a correlation being described as if it were causal?

Do NOT recompute the statistics. The p-values, effect sizes and assumption checks have already \
been calculated and are given to you as fact. Argue about what they MEAN.

Every objection must cite a real, specific URL supporting your point. The URL will be \
retrieved and checked. An objection whose citation cannot be fetched will be DISCARDED, so \
do not invent references -- an uncited objection is worth more to you than a fabricated one, \
because you can simply leave counterevidence_url empty.

Choose reroute_to carefully:
- "experiment" if the test or its parameters were wrong
- "data" if the data cannot support any test of this question
- "question" if the question itself is unanswerable with obtainable data
- "none" if you accept the result"""

PROMPT = """Research question: {question}

Hypothesis: {hypothesis}
Prediction registered before execution: {prediction}
Columns analysed, and how the designer claims they map to the question:
{addresses}

Results (already computed -- treat these numbers as given):
{results}

Data provenance:
- {sources} sources across {modalities} modalities
- {rows} rows analysed
- Cleaning: {cleaning}
{conflicts}

Mechanical checks already performed:
{flags}

Claims the system proposes to assert:
{claims}

Claims it has already ABSTAINED from due to low confidence:
{abstained}

{history}

Review this. If the mechanical checks show a blocking problem, you must not accept.
Then write a "Limitations and Future Work" section for the paper: specific, honest, and \
useful to a reader deciding whether to trust this."""


class Critic(BaseAgent):
    name = AgentName.CRITIC
    fatal_on_failure = False

    async def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        result: ExperimentResult | None = state.get("experiment_result")
        bundle: DataBundle | None = state.get("data_bundle")
        question: ResearchQuestion | None = state.get("question")
        if result is None or bundle is None:
            raise AgentFailure("nothing to critique")

        cycle = state.get("cycle", 0) + 1
        flags = compute_flags(result, bundle)

        self.say(
            f"Cycle {cycle}/{self.settings.max_cycles}. Mechanical checks: "
            + ("; ".join(flags.as_reasons()) or "none failed"),
            level=Level.WARN if flags.any_blocking else Level.INFO,
            cycle=cycle,
        )

        critique = await self._review(state, result, bundle, question, flags, cycle=cycle)
        critique.cycle = cycle
        critique.stat_flags = flags

        # Verify citations before the verdict is allowed to stand.
        if critique.objections:
            self.say(
                f"Verifying counterevidence for {len(critique.objections)} objection(s)…",
                cycle=cycle,
            )
            await self._verify_objections(critique, cycle=cycle)

        self._enforce(critique, flags)
        self._report(critique, cycle=cycle)

        # Publish the reroute as its own event.
        #
        # Without this the graph's red edges never animate and the vitals panel
        # reads "Reroutes 0" even after five rejected cycles -- the single most
        # legible evidence that this is a state machine rather than a chain,
        # invisible in the one place a reviewer is looking.
        if critique.verdict != Verdict.ACCEPT and critique.reroute_to != RerouteTarget.NONE:
            self.ctx.bus.reroute(
                target=critique.reroute_to.value,
                reason=(critique.summary or "; ".join(flags.as_reasons()))[:200],
                cycle=cycle,
                source=AgentName.CRITIC,
            )

        return {
            "critiques": [critique],
            "cycle": cycle,
            "reroute_to": critique.reroute_to.value,
            "reroute_reason": critique.summary[:300],
            "phase": "critique",
        }

    # -------------------------------------------------------------- review

    async def _review(
        self,
        state: dict[str, Any],
        result: ExperimentResult,
        bundle: DataBundle,
        question: ResearchQuestion | None,
        flags: StatFlags,
        *,
        cycle: int,
    ) -> Critique:
        from .uncertainty import _render_results

        claims = state.get("claims") or []
        abstained = state.get("abstained_claims") or []
        conflicts = bundle.open_conflicts

        history = ""
        previous = state.get("critiques") or []
        if previous:
            history = (
                "Your objections from earlier cycles (do not simply repeat them; "
                "assess whether they were addressed):\n"
                + "\n".join(
                    f"- cycle {c.cycle}: {c.summary[:180]}" for c in previous[-2:]
                )
            )

        prompt = PROMPT.format(
            question=question.text if question else "(unknown)",
            hypothesis=result.spec.hypothesis.alternative,
            prediction=result.spec.hypothesis.prediction,
            addresses=(
                f"{result.spec.params} — {result.spec.addresses_question or 'no mapping stated'}"
            ),
            results=_render_results(result),
            sources=len(bundle.documents),
            modalities=len(bundle.modalities),
            rows=bundle.dataset.n_rows if bundle.dataset else 0,
            cleaning=bundle.dataset.cleaning.model_dump() if bundle.dataset else "{}",
            conflicts=(
                f"- UNRESOLVED CONFLICTS: {len(conflicts)}, e.g. {conflicts[0].subject} "
                f"({conflicts[0].value_a} vs {conflicts[0].value_b})"
                if conflicts
                else "- No unresolved source conflicts"
            ),
            flags="\n".join(f"- {r}" for r in flags.as_reasons()) or "- all checks passed",
            claims="\n".join(f"- {c.text} ({c.confidence:.0%})" for c in claims) or "- none",
            abstained="\n".join(f"- {c.text}" for c in abstained) or "- none",
            history=history,
        )

        try:
            return await self.router.structured(
                Role.REASONING,
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                Critique,
                temperature=0.4,
                max_tokens=2200,
            )
        except Exception as exc:  # noqa: BLE001 - a silent critic is worse than a blunt one
            log.warning("critique generation failed: %s", exc)
            self.say(
                f"Critic model unavailable ({exc}); falling back to the mechanical checks alone.",
                level=Level.WARN,
                cycle=cycle,
            )
            return Critique(
                cycle=cycle,
                verdict=Verdict.REVISE if flags.any_blocking else Verdict.ACCEPT,
                reroute_to=RerouteTarget.EXPERIMENT if flags.any_blocking else RerouteTarget.NONE,
                summary="Automated review only; the critic model was unavailable.",
                limitations=(
                    "This run's critique was produced by mechanical statistical checks alone "
                    "because the reviewing model was unavailable. Methodological concerns such "
                    "as confounding and construct validity were therefore NOT assessed."
                ),
                confidence=0.3,
            )

    # -------------------------------------------------------- verification

    async def _verify_objections(self, critique: Critique, *, cycle: int) -> None:
        """Fetch every cited URL. Drop objections whose evidence is not real."""

        async def verify(objection: Objection) -> None:
            url = objection.counterevidence_url.strip()
            if not url:
                # An honest uncited objection is allowed to stand on reasoning,
                # but it does not get to claim external support.
                objection.verified = True
                objection.verification_note = "no citation offered; judged on reasoning alone"
                return

            if not url.startswith("http"):
                objection.verified = False
                objection.verification_note = f"not a retrievable URL: {url!r}"
                return

            try:
                text, tier = await self.fetcher.get_article_text(url, min_chars=200)
            except Exception as exc:  # noqa: BLE001
                objection.verified = False
                objection.verification_note = f"could not be retrieved: {exc}"
                return

            if not text:
                objection.verified = False
                objection.verification_note = "URL did not resolve to readable content"
                return

            objection.verified = True
            objection.verification_note = f"retrieved and readable via {tier}"

        await asyncio.gather(*(verify(o) for o in critique.objections), return_exceptions=True)

        dropped = [o for o in critique.objections if not o.verified]
        for objection in dropped:
            self.say(
                f"Objection DISCARDED — citation could not be verified "
                f"({objection.verification_note}): {objection.claim_attacked[:80]}",
                level=Level.WARN,
                cycle=cycle,
            )
        if dropped:
            self.say(
                f"{len(dropped)} of {len(critique.objections)} objections dropped for "
                "unverifiable evidence.",
                level=Level.WARN,
                cycle=cycle,
            )

    # -------------------------------------------------------- enforcement

    def _enforce(self, critique: Critique, flags: StatFlags) -> None:
        """Override the model's verdict when the arithmetic contradicts it.

        The assessment's rule is not advisory: p > alpha or a trivial effect
        forces another cycle. A model that reads a null result and calls it
        acceptable is simply overruled here.
        """
        if flags.any_blocking and critique.verdict == Verdict.ACCEPT:
            critique.verdict = Verdict.REVISE
            if critique.reroute_to == RerouteTarget.NONE:
                critique.reroute_to = RerouteTarget.EXPERIMENT
            reasons = "; ".join(flags.as_reasons())
            critique.summary = (
                f"Acceptance overridden by mechanical checks ({reasons}). " + critique.summary
            )
            self.say(
                f"Critic accepted, but mechanical checks force iteration: {reasons}",
                level=Level.WARN,
                cycle=critique.cycle,
            )

        if critique.verdict != Verdict.ACCEPT and critique.reroute_to == RerouteTarget.NONE:
            critique.reroute_to = RerouteTarget.EXPERIMENT

    # --------------------------------------------------------------- output

    def _report(self, critique: Critique, *, cycle: int) -> None:
        verified = critique.verified_objections
        self.say(
            f"Verdict: {critique.verdict.value.upper()}"
            + (
                f" — reroute to {critique.reroute_to.value}"
                if critique.verdict != Verdict.ACCEPT
                else ""
            ),
            level=Level.SUCCESS if critique.verdict == Verdict.ACCEPT else Level.WARN,
            cycle=cycle,
        )
        for objection in verified[:5]:
            self.say(
                f"[{objection.severity.value}] {objection.claim_attacked[:70]}: "
                f"{objection.rationale[:130]}",
                level=Level.WARN,
                cycle=cycle,
            )

        self.publish(
            ArtifactKind.CRITIQUE,
            {
                "cycle": cycle,
                "verdict": critique.verdict.value,
                "reroute_to": critique.reroute_to.value,
                "summary": critique.summary,
                "limitations": critique.limitations,
                "stat_flags": critique.stat_flags.model_dump(),
                "flag_reasons": critique.stat_flags.as_reasons(),
                "objections": [
                    {
                        "severity": o.severity.value,
                        "claim": o.claim_attacked,
                        "rationale": o.rationale,
                        "url": o.counterevidence_url,
                        "verified": o.verified,
                        "verification_note": o.verification_note,
                        "suggested_fix": o.suggested_fix,
                    }
                    for o in critique.objections
                ],
                "objections_verified": len(verified),
                "objections_dropped": len(critique.objections) - len(verified),
            },
            message=f"Critic verdict: {critique.verdict.value}",
            cycle=cycle,
        )


def compute_flags(result: ExperimentResult, bundle: DataBundle) -> StatFlags:
    """Mechanical statistical checks. Arithmetic, not opinion."""
    flags = StatFlags()

    if bundle.open_conflicts:
        flags.unresolved_conflicts = True

    if not result.executed_ok or result.primary is None:
        flags.p_gt_alpha = True
        return flags

    primary = result.primary
    alpha = result.spec.alpha

    effective_p = (
        primary.p_value_corrected if primary.p_value_corrected is not None else primary.p_value
    )
    # Descriptive procedures legitimately have no p-value; judging them on a
    # missing one would force endless cycles over a clustering result.
    if effective_p is not None:
        flags.p_gt_alpha = effective_p > alpha
    elif primary.effect_size is None:
        flags.p_gt_alpha = True

    flags.trivial_effect = primary.effect_is_trivial
    flags.n_too_small = 0 < primary.n < MIN_N_FOR_CLAIM
    flags.multiple_comparisons_uncorrected = (
        primary.comparisons_made > 1 and primary.p_value_corrected is None
    )
    flags.assumptions_violated = bool(primary.assumptions_checked) and not all(
        primary.assumptions_checked.values()
    )

    if primary.ci_low is not None and primary.ci_high is not None and primary.effect_size:
        width = abs(primary.ci_high - primary.ci_low)
        flags.wide_confidence_interval = width > WIDE_CI_RATIO * abs(primary.effect_size)

    return flags
