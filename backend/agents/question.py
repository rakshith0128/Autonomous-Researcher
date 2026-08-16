"""Question Generator: invent questions that cannot simply be looked up.

The assessment requires research questions that are "not directly searchable
(must require synthesis)". Most implementations assert this in a prompt and
hope. This one tests it: every generated question is put through a real search,
and a judge decides whether what came back actually answers it. Questions with
findable answers are disqualified and regenerated.

Two further constraints make the questions usable downstream:

* Each must declare **at least two distinct sources that have to be joined**.
  That is the operational definition of synthesis -- if one source answers it,
  it is a lookup.
* Each must name a **concrete measurable quantity**. A question no experiment
  can address is philosophy, and the Experiment Designer will choke on it three
  nodes later.

Rejected questions accumulate in run state across cycles, so when the Critic
sends work back here the generator can be shown exactly what has already failed
and why.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field

from ..config import Role
from ..schemas import (
    AgentName,
    ArtifactKind,
    DomainSelection,
    Level,
    PeerRating,
    QuestionProposalBatch,
    QuestionSet,
    ResearchQuestion,
    SearchabilityProbe,
)
from ..tools.arxiv import ArxivClient
from ..tools.search import SearchClient
from .base import AgentFailure, BaseAgent

log = logging.getLogger(__name__)

MAX_REGENERATION_ROUNDS = 2


# --- structured-output contracts used only by this agent -------------------


class _Verdict(BaseModel):
    """The searchability judge's ruling on one question."""

    directly_answers: bool = Field(
        description="True only if the search result states the answer outright"
    )
    reasoning: str = ""


class _Rating(BaseModel):
    question_id: str = ""
    novelty: float = Field(ge=0.0, le=1.0)
    feasibility: float = Field(ge=0.0, le=1.0)
    comment: str = ""


class _RatingBatch(BaseModel):
    ratings: list[_Rating] = Field(min_length=1)


class _SameQuestion(BaseModel):
    """Adjudication for a pair vector similarity cannot separate."""

    same: bool = Field(description="True only if these are the same question reworded")
    reasoning: str = ""

SYSTEM = """You design research questions for an autonomous analysis system.

A good question here has four properties:
1. It CANNOT be answered by finding one paper or one webpage. If a search would surface the \
answer directly, it is worthless.
2. Answering it requires JOINING at least two different data sources.
3. It names a concrete quantity that can be MEASURED from public data \
(paper counts, citations, benchmark scores, repository metrics, dates, author counts, \
extracted table values).
4. It is answerable with a statistical test on data gathered in minutes, not a lab experiment.

Avoid questions that are really opinions ("is X promising?"), predictions about the future, \
or that need proprietary data.

Prefer questions of the form: does measurable property A relate to measurable property B \
across this domain's literature?"""

PROMPT = """Domain: {domain}
{description}

Why this domain appears to be emerging: {why}

Measured evidence:
{evidence}

Representative recent papers in this domain:
{papers}

Generate {n} research questions about this domain.

For each, state the question, why it is non-trivial, which distinct data sources must be \
combined to answer it (at least two), and the concrete quantity that would have to be \
measured.{avoid}"""

JUDGE = """A research question was searched for. Decide whether the search result DIRECTLY \
answers it.

Question: {question}

Top search result:
Title: {title}
URL: {url}
Extract: {snippet}

Answer directly = the result states the answer, so no analysis would be needed.
Answer NOT directly = the result is merely related, background, or about a different question.

Be strict about what "directly answers" means. A paper on the same topic does not answer a \
question about a relationship the paper never measured."""

RATE = """Rate each research question on novelty and feasibility.

novelty: 1.0 = nobody has published this analysis; 0.0 = textbook knowledge.
feasibility: 1.0 = the data is clearly obtainable from public APIs and papers right now; \
0.0 = needs data that does not exist publicly.

Questions:
{questions}"""


class QuestionGenerator(BaseAgent):
    name = AgentName.QUESTION_GEN
    fatal_on_failure = True

    async def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        selection: DomainSelection | None = state.get("domain_selection")
        chosen = selection.chosen() if selection else None
        if chosen is None:
            raise AgentFailure("no domain was selected", fatal=True)

        cycle = state.get("cycle", 0)
        already_rejected: list[str] = state.get("rejected_questions") or []

        papers = await self._sample_papers(chosen.signals.term_used or chosen.name)

        questions: list[ResearchQuestion] = []
        rounds = 0
        newly_rejected: list[str] = []

        while rounds <= MAX_REGENERATION_ROUNDS:
            rounds += 1
            batch = await self._generate(
                chosen, papers, avoid=already_rejected + newly_rejected, n=4
            )

            candidates = [
                ResearchQuestion(id=uuid.uuid4().hex[:8], proposal=p) for p in batch.proposals
            ]

            self.say(f"Testing {len(candidates)} questions for triviality…", cycle=cycle)
            await self._screen(candidates, cycle=cycle)

            viable = [q for q in candidates if not q.disqualified]
            newly_rejected.extend(
                f"{q.text} (rejected: {q.disqualified_reason})"
                for q in candidates
                if q.disqualified
            )
            # Record rejections in vector memory so the *next* regeneration
            # round, and later cycles, can detect rewordings of them.
            if self.memory is not None and self.memory.available:
                for rejected in (q for q in candidates if q.disqualified):
                    self.memory.remember_rejection(rejected.text, rejected.disqualified_reason)
            questions.extend(candidates)

            if viable:
                break

            self.say(
                "Every question was directly answerable by search; regenerating with those "
                "marked as too easy.",
                level=Level.WARN,
                cycle=cycle,
            )

        viable = [q for q in questions if not q.disqualified]
        if not viable:
            raise AgentFailure(
                "could not produce a question that was not already answerable by search",
                fatal=True,
            )

        await self._rate(viable, cycle=cycle)
        viable.sort(key=lambda q: q.score, reverse=True)
        selected = viable[0]

        question_set = QuestionSet(
            questions=questions,
            selected_id=selected.id,
            regeneration_rounds=rounds - 1,
            rationale=(
                f"Highest combined novelty/feasibility ({selected.score:.2f}) among "
                f"{len(viable)} questions that survived the searchability screen."
            ),
            confidence=round(min(selected.score + 0.1, 1.0), 3),
        )

        self._report(question_set, cycle=cycle)

        return {
            "question_set": question_set,
            "question": selected,
            "rejected_questions": newly_rejected,
            "phase": "data",
        }

    # ----------------------------------------------------------- generation

    async def _sample_papers(self, term: str) -> str:
        """Ground question generation in real recent papers, not model memory."""
        arxiv = ArxivClient(self.fetcher)
        papers = await arxiv.search(term, max_results=8)
        self.tool("arxiv.search", ok=bool(papers), detail=f"{len(papers)} papers for {term!r}")
        if not papers:
            return "- (no recent papers retrieved; generate from the measured evidence alone)"
        return "\n".join(f"- {p.title}: {p.summary[:220]}" for p in papers[:8])

    async def _generate(
        self, chosen, papers: str, *, avoid: list[str], n: int
    ) -> QuestionProposalBatch:  # noqa: ANN001
        signals = chosen.signals
        evidence = (
            f"- arXiv: {signals.arxiv_recent_count} papers since 2024 "
            f"({signals.arxiv_growth_ratio:.1f}x the prior baseline)\n"
            f"- OpenAlex: {signals.openalex_recent_works} works, "
            f"citation velocity {signals.openalex_citation_velocity:.1f}/yr\n"
            f"- GitHub: {signals.github_repos_created_post_cutoff} repositories created since 2024"
        )

        avoid_text = ""
        if avoid:
            # Showing the generator its own rejected attempts is what stops it
            # proposing the same dead end on every cycle.
            recent = avoid[-6:]
            avoid_text = (
                "\n\nThese questions have ALREADY been tried and rejected. Do not repeat them "
                "or minor rewordings of them:\n" + "\n".join(f"- {r}" for r in recent)
            )

        try:
            return await self.router.structured(
                Role.REASONING,
                [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": PROMPT.format(
                            domain=chosen.name,
                            description=chosen.proposal.description,
                            why=chosen.proposal.why_emerging,
                            evidence=evidence,
                            papers=papers,
                            n=n,
                            avoid=avoid_text,
                        ),
                    },
                ],
                QuestionProposalBatch,
                temperature=0.75,
                max_tokens=2200,
            )
        except Exception as exc:
            raise AgentFailure(f"question generation failed: {exc}", fatal=True) from exc

    # ------------------------------------------------------------ screening

    async def _screen(self, questions: list[ResearchQuestion], *, cycle: int) -> None:
        """Disqualify questions a search can already answer."""
        await asyncio.gather(
            *(self._screen_one(q, cycle=cycle) for q in questions), return_exceptions=True
        )

    async def _screen_one(self, question: ResearchQuestion, *, cycle: int) -> None:
        # A question needing fewer than two joined sources is a lookup by
        # construction; no search needed to know that.
        if len(question.proposal.required_joins) < 2:
            question.disqualified = True
            question.disqualified_reason = "requires only one data source, so it is a lookup"
            return

        # Semantic negative memory: retrieval for recall, a model for precision.
        #
        # Listing rejected questions in the prompt catches literal repeats only;
        # by the third cycle the generator returns the same question reworded.
        # But similarity alone cannot separate a rewording from a genuinely new
        # question either -- measured, a different question scored 0.812 while a
        # true duplicate scored 0.810. So the vector store narrows the field and
        # a cheap model call decides the ambiguous cases.
        if self.memory is not None and self.memory.available:
            prior, ambiguous = self.memory.similar_rejection(question.text)
            if prior is not None:
                duplicate = True
                if ambiguous:
                    duplicate = await self._is_same_question(question.text, prior.text)
                if duplicate:
                    question.disqualified = True
                    question.disqualified_reason = (
                        f"already asked and rejected this run "
                        f"(similarity {prior.similarity:.2f}): {prior.text[:110]}"
                    )
                    self.say(
                        f"Rejected as a reworded repeat (similarity {prior.similarity:.2f}): "
                        f"{question.text[:80]}…",
                        level=Level.WARN,
                        cycle=cycle,
                    )
                    return

        search = SearchClient(self.fetcher)
        found, url, snippet = await search.is_directly_answerable(question.text)
        if not found:
            question.probe = SearchabilityProbe(
                query_used=question.text,
                directly_answered=False,
                reasoning="search returned nothing relevant",
            )
            return

        verdict = await self._judge(question.text, url, snippet)
        question.probe = SearchabilityProbe(
            query_used=question.text,
            directly_answered=verdict.directly_answers,
            evidence_url=url,
            evidence_snippet=snippet[:400],
            reasoning=verdict.reasoning,
        )

        if verdict.directly_answers:
            question.disqualified = True
            question.disqualified_reason = f"directly answerable by search ({url})"
            self.say(
                f"Rejected as trivial: {question.text[:90]}…",
                level=Level.WARN,
                cycle=cycle,
            )

    async def _is_same_question(self, candidate: str, prior: str) -> bool:
        """Adjudicate a topically-close pair that similarity cannot separate.

        Fails *open* -- an unreachable judge lets the question through. Wrongly
        discarding a valid question costs a whole regeneration round and the run
        never learns why; wrongly keeping a duplicate merely repeats work the
        Critic will reject again.
        """
        try:
            verdict = await self.router.structured(
                Role.FAST,
                [
                    {
                        "role": "system",
                        "content": (
                            "You decide whether two research questions ask the same thing. "
                            "Same topic is NOT the same question: two questions about "
                            "citation counts that relate them to different variables are "
                            "different questions."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question A: {prior}\n\nQuestion B: {candidate}\n\n"
                            "Would answering one also answer the other? Set same=true only "
                            "if they are the same question worded differently."
                        ),
                    },
                ],
                _SameQuestion,
                temperature=0.0,
                max_tokens=300,
            )
            return verdict.same
        except Exception as exc:  # noqa: BLE001
            log.warning("duplicate adjudication unavailable: %s", exc)
            return False

    async def _judge(self, question: str, url: str, snippet: str) -> _Verdict:
        try:
            return await self.router.structured(
                Role.FAST,
                [
                    {
                        "role": "system",
                        "content": "You judge whether a search result answers a question.",
                    },
                    {
                        "role": "user",
                        "content": JUDGE.format(
                            question=question, title="", url=url, snippet=snippet[:900]
                        ),
                    },
                ],
                _Verdict,
                temperature=0.0,
                max_tokens=400,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("searchability judge failed: %s", exc)
            # Failing open keeps a usable question rather than discarding it on
            # an infrastructure hiccup; the Critic still gets to object later.
            return _Verdict(directly_answers=False, reasoning=f"judge unavailable: {exc}")

    # -------------------------------------------------------------- rating

    async def _rate(self, questions: list[ResearchQuestion], *, cycle: int) -> None:
        rendered = "\n".join(
            f"{i + 1}. [{q.id}] {q.text}\n   joins: {', '.join(q.proposal.required_joins)}"
            f"\n   measures: {q.proposal.expected_measurable}"
            for i, q in enumerate(questions)
        )
        try:
            ratings = await self.router.structured(
                Role.REASONING,
                [
                    {"role": "system", "content": "You are a peer reviewer rating research questions."},
                    {"role": "user", "content": RATE.format(questions=rendered)},
                ],
                _RatingBatch,
                temperature=0.2,
                max_tokens=1200,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("peer rating failed: %s", exc)
            self.say("Peer rating unavailable; falling back to neutral scores.", level=Level.WARN)
            for question in questions:
                question.ratings.append(
                    PeerRating(rater_id="fallback", novelty=0.5, feasibility=0.5, comment="unrated")
                )
            return

        by_id = {q.id: q for q in questions}
        for i, rating in enumerate(ratings.ratings):
            target = by_id.get(rating.question_id) or (
                questions[i] if i < len(questions) else None
            )
            if target is not None:
                target.ratings.append(
                    PeerRating(
                        rater_id="peer-1",
                        novelty=rating.novelty,
                        feasibility=rating.feasibility,
                        comment=rating.comment,
                    )
                )

        for question in questions:
            if not question.ratings:
                question.ratings.append(
                    PeerRating(rater_id="fallback", novelty=0.5, feasibility=0.5)
                )

    # -------------------------------------------------------------- output

    def _report(self, question_set: QuestionSet, *, cycle: int) -> None:
        selected = question_set.selected()
        self.publish(
            ArtifactKind.QUESTION_SET,
            {
                "questions": [
                    {
                        "id": q.id,
                        "text": q.text,
                        "rationale": q.proposal.rationale,
                        "joins": q.proposal.required_joins,
                        "measurable": q.proposal.expected_measurable,
                        "novelty": round(q.mean_novelty, 2),
                        "feasibility": round(q.mean_feasibility, 2),
                        "score": round(q.score, 3),
                        "disqualified": q.disqualified,
                        "disqualified_reason": q.disqualified_reason,
                        "probe_url": q.probe.evidence_url if q.probe else "",
                        "selected": q.id == question_set.selected_id,
                    }
                    for q in question_set.questions
                ],
                "regeneration_rounds": question_set.regeneration_rounds,
            },
            message="Research questions generated and screened",
            cycle=cycle,
        )
        if selected:
            self.say(f"Selected question: {selected.text}", level=Level.SUCCESS, cycle=cycle)
            self.say(
                f"Requires joining: {', '.join(selected.proposal.required_joins)}", cycle=cycle
            )
