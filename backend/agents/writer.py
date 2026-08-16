"""Paper Writer: assemble the deliverable, then verify it against the evidence.

This is the last place a fabrication could enter the system, and the most
likely: writing a research paper is exactly the task that induces a model to
invent a supporting citation or restate a p-value slightly wrong.

Three defences, in order:

1. **Closed citation list.** The writer is given numbered references built from
   documents that were actually fetched, and told to cite `[n]` only.
2. **Injected numbers.** Every statistic is inserted from computed
   `StatResult` objects into a Results section the model does not author.
3. **Post-hoc verification.** The finished markdown is scanned: citations must
   appear in the provenance ledger, reported figures must match computed
   values, and cited URLs must resolve. Failures are stripped and *printed in
   the paper*.

That last part is deliberate. A verification section that admits "one citation
was removed because it was not in the evidence ledger" is far more convincing
than a clean paper with no audit trail -- it demonstrates the check runs.

If the cycle limit was reached without the Critic accepting, the paper says so
in its own header. No run is allowed to end by quietly implying success.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from ..analysis.verify import (
    format_reference_list,
    verify_citations,
    verify_numbers,
    verify_urls_live,
)
from ..config import Role
from ..schemas import (
    AgentName,
    ArtifactKind,
    Claim,
    Critique,
    DataBundle,
    DomainSelection,
    ExperimentResult,
    Level,
    Paper,
    PaperSection,
    Provenance,
    ResearchQuestion,
    Verdict,
)
from .base import AgentFailure, BaseAgent

log = logging.getLogger(__name__)


class _Narrative(BaseModel):
    """The prose the model is allowed to write. Numbers are not in here."""

    title: str = Field(description="A specific, informative paper title")
    abstract: str = Field(description="150-220 words summarising question, method, and finding")
    introduction: str = Field(description="Why this domain and question matter. 2-3 paragraphs.")
    methods: str = Field(description="How data was gathered and analysed. Be specific and honest.")
    discussion: str = Field(description="What the results mean, and what they do not establish.")


SYSTEM = """You write concise, honest scientific papers.

Absolute rules:

1. CITATIONS: cite only using bracketed numbers from the reference list you are given, \
like [1] or [2, 4]. Never write a URL. Never invent a reference. Never cite a number \
outside the list.

2. NUMBERS: do NOT state p-values, effect sizes, correlation coefficients or sample sizes \
in your prose. A Results section containing the exact computed values is inserted \
automatically. If you restate a statistic and get it slightly wrong, it will be flagged as \
unsupported. Refer to findings qualitatively ("a moderate positive association") and let the \
Results table carry the numbers.

3. HONESTY: if the result was null, say so directly. A null finding reported clearly is good \
science. Do not imply significance that was not found, and do not describe correlation as \
causation."""

PROMPT = """Write a short research paper.

Domain: {domain}
Why this domain was selected: {domain_rationale}

Research question: {question}
Why it required synthesis: {joins}

Hypothesis: {hypothesis}
Prediction registered in advance: {prediction}
Did the outcome match the prediction? {matched}

What the analysis found (qualitatively -- exact numbers are inserted separately):
{qualitative}

Data gathered:
- {sources} sources across {modalities} modalities: {modality_list}
- {rows} rows analysed after cleaning
- Cleaning performed: {cleaning}
{conflicts}

Claims the system is confident enough to assert:
{claims}

Claims it ABSTAINED from (confidence below threshold) -- mention in the discussion that the \
system declined to assert these:
{abstained}

The reviewer's objections:
{objections}

Reference list (cite by number ONLY):
{references}

Write the title, abstract, introduction, methods and discussion."""


class PaperWriter(BaseAgent):
    name = AgentName.WRITER
    fatal_on_failure = False

    async def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        bundle: DataBundle | None = state.get("data_bundle")
        question: ResearchQuestion | None = state.get("question")
        result: ExperimentResult | None = state.get("experiment_result")
        selection: DomainSelection | None = state.get("domain_selection")
        critiques: list[Critique] = state.get("critiques") or []

        cycle = state.get("cycle", 0)
        claims: list[Claim] = state.get("claims") or []
        abstained: list[Claim] = state.get("abstained_claims") or []

        if bundle is None or question is None:
            raise AgentFailure("insufficient state to write a paper")

        references = self._references(bundle)
        narrative = await self._write(
            state, bundle, question, result, selection, critiques, claims, abstained, references
        )

        paper = self._assemble(
            narrative, state, bundle, question, result, critiques, claims, abstained, references
        )

        markdown = paper.to_markdown()
        markdown, report = verify_citations(markdown, references)
        stats = result.stats if result else []
        report = verify_numbers(markdown, stats, report)
        report = await verify_urls_live([r.url for r in references], self.fetcher, report)

        self.say(f"Verification: {report.summary()}", level=Level.SUCCESS, cycle=cycle)
        for finding in report.findings[:6]:
            self.say(f"Verification failure — {finding}", level=Level.WARN, cycle=cycle)

        paper.sections.append(
            PaperSection(
                heading="Verification",
                body_markdown=report.as_markdown(),
                order=90,
            )
        )

        # Confidence is reduced when the paper's own citations did not check
        # out. A document that cited invented sources should not be presented
        # with the confidence its arithmetic alone would suggest.
        paper.overall_confidence = round(
            paper.overall_confidence * (0.6 + 0.4 * report.citation_integrity), 3
        )

        self._report(paper, report, cycle=cycle)

        return {
            "paper": paper,
            "overall_confidence": paper.overall_confidence,
            "phase": "done",
            "finished": True,
        }

    # ----------------------------------------------------------- references

    @staticmethod
    def _references(bundle: DataBundle) -> list[Provenance]:
        """Deduplicated provenance for everything actually retrieved."""
        seen: set[str] = set()
        references: list[Provenance] = []
        for document in bundle.documents:
            if document.url in seen:
                continue
            seen.add(document.url)
            references.append(document.provenance)
        return references

    # -------------------------------------------------------------- writing

    async def _write(
        self,
        state: dict[str, Any],
        bundle: DataBundle,
        question: ResearchQuestion,
        result: ExperimentResult | None,
        selection: DomainSelection | None,
        critiques: list[Critique],
        claims: list[Claim],
        abstained: list[Claim],
        references: list[Provenance],
    ) -> _Narrative:
        objections = []
        for critique in critiques[-2:]:
            objections += [
                f"- [{o.severity.value}] {o.claim_attacked}: {o.rationale}"
                for o in critique.verified_objections[:5]
            ]

        prompt = PROMPT.format(
            domain=state.get("domain_name", "unknown"),
            domain_rationale=selection.rationale if selection else "",
            question=question.text,
            joins=", ".join(question.proposal.required_joins),
            hypothesis=result.spec.hypothesis.alternative if result else "n/a",
            prediction=result.spec.hypothesis.prediction if result else "n/a",
            matched=self._prediction_matched(result),
            qualitative=self._qualitative(result),
            sources=len(bundle.documents),
            modalities=len(bundle.modalities),
            modality_list=", ".join(sorted(m.value for m in bundle.modalities)),
            rows=bundle.dataset.n_rows if bundle.dataset else 0,
            cleaning=bundle.dataset.cleaning.model_dump() if bundle.dataset else "{}",
            conflicts=(
                f"- {len(bundle.open_conflicts)} unresolved conflicts between sources"
                if bundle.open_conflicts
                else "- No unresolved source conflicts"
            ),
            claims="\n".join(f"- {c.text}" for c in claims) or "- none",
            abstained="\n".join(f"- {c.text}" for c in abstained) or "- none",
            objections="\n".join(objections) or "- none recorded",
            references=format_reference_list(references),
        )

        try:
            return await self.router.structured(
                Role.SYNTHESIS,
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                _Narrative,
                temperature=0.5,
                max_tokens=3000,
            )
        except Exception as exc:  # noqa: BLE001 - a mechanical paper beats no paper
            log.warning("narrative generation failed: %s", exc)
            self.say(
                f"Narrative model unavailable ({exc}); assembling a factual report instead.",
                level=Level.WARN,
            )
            return _Narrative(
                title=f"Analysis of {state.get('domain_name', 'an emerging domain')}",
                abstract=(
                    "This report was assembled mechanically because the writing model was "
                    "unavailable. It contains the computed results and provenance without "
                    "narrative interpretation."
                ),
                introduction=f"The system investigated: {question.text}",
                methods=(
                    f"Data was gathered from {len(bundle.documents)} sources across "
                    f"{len(bundle.modalities)} modalities and analysed with "
                    f"{result.spec.kind.value if result else 'no'} procedure."
                ),
                discussion="No narrative interpretation was produced.",
            )

    @staticmethod
    def _prediction_matched(result: ExperimentResult | None) -> str:
        """State plainly whether the pre-registered prediction held.

        Recorded before execution precisely so this question has an answer that
        cannot be rewritten after the fact.
        """
        if result is None or not result.executed_ok or result.primary is None:
            return "The experiment did not complete, so the prediction could not be evaluated."
        primary = result.primary
        if primary.significant and not primary.effect_is_trivial:
            return "Yes — a statistically significant, non-trivial effect was detected."
        if primary.significant:
            return "Partially — the result was statistically significant but the effect size is negligible."
        return "No — no statistically significant effect was detected. This is a null result."

    @staticmethod
    def _qualitative(result: ExperimentResult | None) -> str:
        """Describe the finding without numbers, so the model cannot copy them wrong."""
        if result is None or not result.executed_ok or result.primary is None:
            return "The experiment did not produce a usable result."
        primary = result.primary
        direction = ""
        if primary.effect_size is not None:
            direction = "positive" if primary.effect_size > 0 else "negative"
        strength = primary.effect_interpretation
        significance = (
            "statistically significant" if primary.significant else "not statistically significant"
        )
        return (
            f"- Procedure: {primary.test_name}\n"
            f"- The association is {significance}, with a {strength} {direction} effect.\n"
            f"- Sample size was {'adequate' if primary.n >= 30 else 'small'}."
        )

    # ------------------------------------------------------------ assembly

    def _assemble(
        self,
        narrative: _Narrative,
        state: dict[str, Any],
        bundle: DataBundle,
        question: ResearchQuestion,
        result: ExperimentResult | None,
        critiques: list[Critique],
        claims: list[Claim],
        abstained: list[Claim],
        references: list[Provenance],
    ) -> Paper:
        accepted = bool(critiques and critiques[-1].verdict == Verdict.ACCEPT)
        limitations = critiques[-1].limitations if critiques else ""

        if not accepted and critiques:
            limitations = (
                "**This run reached the cycle limit without the reviewer accepting the "
                "result.** The objections below were not resolved.\n\n" + limitations
            )

        sections = [
            PaperSection(heading="Introduction", body_markdown=narrative.introduction, order=10),
            PaperSection(
                heading="Domain Discovery",
                body_markdown=self._domain_section(state),
                order=20,
            ),
            PaperSection(heading="Methods", body_markdown=narrative.methods, order=30),
            PaperSection(
                heading="Data and Provenance",
                body_markdown=self._provenance_section(bundle, references),
                order=40,
            ),
            # Authored by code, not the model: these are the numbers.
            PaperSection(
                heading="Results",
                body_markdown=self._results_section(result),
                order=50,
            ),
            PaperSection(heading="Discussion", body_markdown=narrative.discussion, order=60),
        ]

        return Paper(
            title=narrative.title,
            abstract=narrative.abstract,
            domain=state.get("domain_name", ""),
            research_question=question.text,
            sections=sections,
            figures=result.figures if result else [],
            claims=claims,
            abstained_claims=abstained,
            limitations=limitations,
            references=references,
            unresolved_conflicts=bundle.open_conflicts,
            unresolved_objections=[
                f"[{o.severity.value}] {o.claim_attacked}: {o.rationale}"
                for c in critiques[-1:]
                for o in c.verified_objections
            ]
            if not accepted
            else [],
            overall_confidence=state.get("overall_confidence", 0.0),
            cycles_used=state.get("cycle", 0),
            accepted_by_critic=accepted,
        )

    @staticmethod
    def _domain_section(state: dict[str, Any]) -> str:
        selection: DomainSelection | None = state.get("domain_selection")
        if selection is None:
            return "_Domain discovery record unavailable._"

        lines = [
            "The domain was not specified by a human. Candidates were discovered from live "
            "arXiv, OpenAlex, GitHub and public-discussion signals, then ranked by a computed "
            "Emergence Index rather than by model judgement.",
            "",
            "| Candidate | Emergence Index | arXiv growth | OpenAlex growth | New repos |",
            "|---|---|---|---|---|",
        ]
        for candidate in selection.candidates[:6]:
            signals = candidate.signals
            lines.append(
                f"| {candidate.name} | {candidate.emergence_index:+.3f} | "
                f"{signals.arxiv_growth_ratio:.1f}x | {signals.openalex_growth_ratio:.1f}x | "
                f"{signals.github_repos_created_post_cutoff} |"
            )
        lines += [
            "",
            f"**Selected:** {selection.chosen_name}. {selection.rationale}",
        ]
        if selection.tiebreak_used:
            lines.append(
                f"\nThe two reviewers disagreed (spread {selection.reviewer_disagreement:.2f}), "
                "so a third tiebreak review was run with the disagreement made explicit."
            )
        return "\n".join(lines)

    @staticmethod
    def _provenance_section(bundle: DataBundle, references: list[Provenance]) -> str:
        lines = [
            f"{len(references)} sources were retrieved across "
            f"{len(bundle.modalities)} distinct modalities "
            f"({', '.join(sorted(m.value for m in bundle.modalities))}). "
            "Every source is content-hashed so this analysis can be re-checked.",
            "",
            "| # | Source | Modality | SHA-256 (first 12) | Retrieved |",
            "|---|---|---|---|---|",
        ]
        for i, reference in enumerate(references, 1):
            title = (reference.title or reference.url)[:70].replace("|", "\\|")
            lines.append(
                f"| {i} | {title} | {reference.modality.value} | "
                f"`{reference.sha256[:12]}` | {reference.retrieved_at:%Y-%m-%d} |"
            )

        if bundle.dataset:
            cleaning = bundle.dataset.cleaning
            lines += [
                "",
                f"After cleaning: {cleaning.rows_out} rows from {cleaning.rows_in} raw records "
                f"({cleaning.duplicates_removed} duplicates removed, "
                f"{cleaning.columns_mapped} column mappings validated, "
                f"{cleaning.mappings_rejected} rejected).",
            ]
            for note in cleaning.notes:
                lines.append(f"- {note}")

        if bundle.acquisition_failures:
            lines += ["", "**Sources that could not be acquired:**"]
            lines += [f"- {failure}" for failure in bundle.acquisition_failures]
        return "\n".join(lines)

    @staticmethod
    def _results_section(result: ExperimentResult | None) -> str:
        """Written entirely from computed values. No model output here."""
        if result is None or not result.executed_ok:
            reason = result.error if result else "no experiment was run"
            return f"_The experiment did not execute successfully: {reason}_"

        lines = [
            f"Procedure: `{result.spec.kind.value}` with parameters `{result.spec.params}`.",
            "",
            f"- **H0:** {result.spec.hypothesis.null}",
            f"- **H1:** {result.spec.hypothesis.alternative}",
            f"- **Pre-registered prediction:** {result.spec.hypothesis.prediction}",
            "",
            "| Test | Statistic | p | Effect | Interpretation | n |",
            "|---|---|---|---|---|---|",
        ]
        for stat in result.stats:
            p_display = (
                f"{stat.p_value_corrected:.4g} (corrected)"
                if stat.p_value_corrected is not None
                else (f"{stat.p_value:.4g}" if stat.p_value is not None else "n/a")
            )
            effect = (
                f"{stat.effect_size_name} = {stat.effect_size:.4g}"
                if stat.effect_size is not None
                else "n/a"
            )
            statistic = f"{stat.statistic:.4g}" if stat.statistic is not None else "n/a"
            lines.append(
                f"| {stat.test_name} | {statistic} | {p_display} | {effect} | "
                f"{stat.effect_interpretation} | {stat.n} |"
            )

        notes = [n for stat in result.stats for n in stat.notes]
        if notes:
            lines += ["", "**Analysis notes:**"] + [f"- {n}" for n in notes]

        if result.repair_attempts:
            lines += [
                "",
                f"_The experiment specification required {result.repair_attempts} automatic "
                "repair attempt(s) before it executed._",
            ]
        return "\n".join(lines)

    # --------------------------------------------------------------- output

    def _report(self, paper: Paper, report: Any, *, cycle: int) -> None:
        self.say(f"Paper complete: {paper.title}", level=Level.SUCCESS, cycle=cycle)
        self.publish(
            ArtifactKind.PAPER,
            {
                "title": paper.title,
                "abstract": paper.abstract,
                "markdown": paper.to_markdown(),
                "domain": paper.domain,
                "question": paper.research_question,
                "confidence": paper.overall_confidence,
                "cycles_used": paper.cycles_used,
                "accepted_by_critic": paper.accepted_by_critic,
                "claims": len(paper.claims),
                "abstained": len(paper.abstained_claims),
                "references": len(paper.references),
                "verification": report.model_dump(),
            },
            message=paper.title,
            cycle=cycle,
        )
