# Code Documentation

Complete technical documentation for the Autonomous Research Agent: what every module does,
why it is built that way, the failures that shaped it, and how to verify any of it yourself.

**Live system:** [https://autonomous-researcher-production-e552.up.railway.app](https://autonomous-researcher-production-6d38.up.railway.app)

---

# Tech Stack

Every technology choice here follows from a single rule: **the model proposes, Python decides.**
LangGraph enforces control flow so no agent can loop by choosing to; Pydantic enforces contracts
so malformed work cannot propagate; scipy computes the statistics so no number is ever generated;
and fastembed runs locally so retrieval has no quota to exhaust.

---

## Backend — Python 3.11

| Layer | Technology | Why this choice |
|---|---|---|
| **Orchestration** | LangGraph 0.2 + langchain-core 0.3 | A state machine with conditional edges and real cycles. The graph *is* the evidence that this is not a prompt chain — the Critic names where work is wrong and routing sends it there. |
| **API** | FastAPI 0.115 + Uvicorn 0.32 | Async throughout, with native SSE support for the live run console. |
| **Streaming** | sse-starlette 2.1 | Server-Sent Events survive free-tier reverse proxies better than WebSockets and reconnect trivially. |
| **Contracts** | Pydantic 2.9 + pydantic-settings 2.6 | Every agent takes a validated model in and returns one out. Malformed output fails validation before it can reach the next agent. |
| **LLM access** | openai 1.54 SDK | All four supported providers expose OpenAI-compatible endpoints, so provider support is a base URL and a model id — not an adapter per vendor. |
| **Retries** | tenacity 9.0 | Backoff policy for transient failures. |

## Data & storage

| Purpose | Technology | Notes |
|---|---|---|
| **Run ledger** | SQLite via aiosqlite 0.20 | Every event is persisted, which yields refresh-safe replay, the run gallery, and the trace view from a single write path. WAL mode keeps reads working during a live run. |
| **Vector memory** | ChromaDB 0.5 (ephemeral client) | Retrieval over acquired documents, plus semantic negative memory for rejected questions. Run-scoped, so runs never contaminate each other. |
| **Embeddings** | fastembed 0.4 — `BAAI/bge-small-en-v1.5` | Quantised ONNX running **locally**: no embedding API, no key, no quota. On a system where every other budget is a scarce free tier, that matters. |

## Data acquisition

**HTTP layer** — httpx 0.27 (async). One shared client with per-host circuit breakers, a retry
policy that distinguishes transient from permanent failures, response size guards, and politeness
throttling.

**Sources** — arXiv API · OpenAlex · GitHub REST · Tavily · Hacker News Algolia. All free, none
requiring a card.

**Extraction**

| Library | Role |
|---|---|
| PyMuPDF 1.24 | PDF text layer |
| pdfplumber 0.11 | Table recovery from PDFs |
| rapidocr-onnxruntime 1.3 + Pillow 11 | Figure OCR — pip-only, so there is no Tesseract system binary to fight in Docker |
| trafilatura 1.12 + beautifulsoup4 4.12 | Article extraction with boilerplate removal |
| feedparser 6.0 | arXiv Atom feeds |

## Analysis

| Library | Used for |
|---|---|
| **scipy 1.14** | Pearson/Spearman correlation, Welch t-test, Mann-Whitney U, Mann-Kendall trend, Shapiro-Wilk |
| **statsmodels 0.14** | OLS with HC3 robust standard errors, Benjamini-Hochberg FDR correction |
| **scikit-learn 1.5** | KMeans + silhouette, cross-validated classification against a stratified baseline |
| **pandas 2.2 / numpy 1.26** | Dataset assembly, bootstrap resampling |
| **plotly 5.24** | Figures built server-side, serialised to JSON, rendered client-side |

Every statistic in the final paper is computed here. No model output is trusted with a number.

## Frontend — React 18 + TypeScript 5.6

| Purpose | Technology |
|---|---|
| **Build** | Vite 6 |
| **Styling** | Tailwind CSS 4 (Vite plugin, CSS-first config) |
| **Live graph** | `@xyflow/react` 12 — nodes pulse as agents run, reroute edges light red |
| **Charts** | plotly.js-dist-min 2.35 + react-plotly.js — lazy-loaded, since it is ~3MB and unneeded until results exist |
| **Paper rendering** | react-markdown 9 + remark-gfm 4 |
| **Routing** | react-router-dom 6 |
| **Transitions** | framer-motion 11 |
| **Streaming** | Native `EventSource` |

## Infrastructure

**Container** — a multi-stage Dockerfile: Node 20 Alpine builds the React bundle, then
`python:3.11-slim` serves the API and the built assets from a single port. Runs as uid 1000 and
reads `$PORT` with a 7860 fallback, so one image runs unmodified on Railway, Render, or Hugging
Face Spaces.

**Hosting** — Railway for the backend (Docker, auto-deploy from GitHub); Vercel for the frontend
CDN in a split deployment; `render.yaml` is included as a documented fallback.

**LLM providers** — Groq (`openai/gpt-oss-120b`, `llama-3.3-70b-versatile`,
`llama-3.1-8b-instant`) and Google Gemini (`gemini-3.5-flash`, `gemini-3.5-flash-lite`,
`gemini-3.1-flash-lite`). Six models across two families, budgeted **per model** because free-tier
quotas differ by more than an order of magnitude between siblings.

## Testing & tooling

**pytest 8.3** with **pytest-asyncio** and **respx** for HTTP mocking — **280 tests, running fully
offline.** No network, no LLM calls, no credentials, so CI needs nothing configured. **ruff 0.7**
for linting.

---

## Deviations from the suggested stack, with justification

| Suggested | Used | Reason |
|---|---|---|
| Railway / Render / Fly.io | **Railway** | Of the three, only Railway and Render still have a genuinely free path. Fly.io removed its free allowances; Railway's trial needs no card and covers the required 7-day window at full CPU. |
| Hugging Face Spaces | **Railway** | Docker Spaces now require a paid PRO plan; only Static Spaces remain free, and those cannot run a Python backend. |
| Playwright (headless browser) | **Tiered fetcher** | Direct request → trafilatura extraction → reader proxy handles the pages this system encounters at a fraction of Playwright's ~500MB image cost. The tiers *are* the fallback Playwright would have provided. |
| Per-provider SDKs | **One OpenAI SDK** | Groq, Gemini, Cerebras and OpenRouter all expose OpenAI-compatible endpoints, so adding a provider is a base URL and a model name rather than a new adapter. |

## Contents

1. [Problem and approach](#1-problem-and-approach)
2. [System overview](#2-system-overview)
3. [Module reference](#3-module-reference)
4. [The nine agents in detail](#4-the-nine-agents-in-detail)
5. [How a run executes, step by step](#5-how-a-run-executes-step-by-step)
6. [Data acquisition](#6-data-acquisition)
7. [The Emergence Index](#7-the-emergence-index)
8. [Confidence and abstention](#8-confidence-and-abstention)
9. [Anti-fabrication](#9-anti-fabrication)
10. [Resilience on free tiers](#10-resilience-on-free-tiers)
11. [Testing](#11-testing)
12. [Development log: failures and fixes](#12-development-log-failures-and-fixes)
13. [Limitations](#13-limitations)

---

## 1. Problem and approach

The assessment asks for a system that runs with zero human intervention after startup:
discover an emerging post-2024 domain, formulate a non-trivial question, gather and clean data
from disparate sources, experiment, self-criticise, iterate up to five times, and produce a
mini research paper. It explicitly rules out RAG-only pipelines and single-agent prompt chains.

Three constraints did most of the design work.

**"No hardcoded domains."** The obvious workaround — ask a model to name trending fields —
just relocates the hardcoding into training data, and for a post-2024 cutoff that is precisely
the period the model knows least about. So the system pulls live evidence first and asks the
model only to *label the clusters it sees*. Ranking is then arithmetic over measured signals.

**"Not RAG-only, not a prompt chain."** This is a structural requirement, so the answer had to
be structural: a state machine with conditional edges and real cycles, where the Critic names
*where* work is wrong and the graph routes accordingly. Retrieval exists, but as one tool among
many rather than the architecture.

**"The AI lies to you constantly."** Taken literally. Nothing a model asserts is trusted where
a machine can check it: statistics are computed in Python, citations are verified against a
ledger of actual fetches, reported numbers are matched against computed values, and the
Critic's own evidence is fetched before its objections count.

### Guiding principle

> **The model proposes; Python decides.**

A model is excellent at naming a cluster, writing a hypothesis, proposing keyword vocabulary,
or arguing about confounding. It is unreliable at arithmetic, citation, and self-assessment.
Every interface in this system is drawn along that line.

---

## 2. System overview

```
15,703 lines Python · 1,947 lines TypeScript · 280 tests
```

| Layer | Location | Lines | Role |
|---|---|---|---|
| Contracts | `backend/schemas/` | 1,280 | Typed Pydantic models between every agent |
| LLM access | `backend/llm/` | 1,314 | Router, per-model budgets, structured-output repair |
| Tools | `backend/tools/` | 2,165 | arXiv, OpenAlex, GitHub, search, fetcher, PDF/OCR, sandbox |
| Analysis | `backend/analysis/` | 1,597 | Emergence Index, statistics, experiment registry, plots, verification |
| Agents | `backend/agents/` | 4,692 | The nine agents |
| Graph | `backend/graph/` | 430 | LangGraph assembly, state, routing, cycle cap |
| Memory | `backend/memory/` | 696 | SQLite ledger, Chroma vector memory |
| Runtime | `backend/runtime/` | 482 | Event bus, run orchestration |
| API | `backend/api/` | 241 | REST + SSE |
| Tests | `backend/tests/` | 2,224 | 280 tests, fully offline |

### Why typed contracts

Every agent takes a validated Pydantic model in and returns one out. This is the structural
difference between this system and a prompt chain: an agent **cannot** pass malformed work
downstream, because validation fails first and triggers a repair round.

It also made the late-stage feature work cheap. Adding question-derived columns to the dataset
required no changes to the graph, the registry, or the Designer — `Dataset` already accepted
arbitrary columns and everything downstream validated dynamically.

---

## 3. Module reference

### `backend/schemas/` — the contracts

| File | Contents |
|---|---|
| `common.py` | `Provenance` (URL + SHA-256 + timestamp), `Claim`, `ConfidenceComponents`, `Conflict`, `Modality`, `AgentName` |
| `domain.py` | `DomainProposal` (LLM-authored) vs `EmergenceSignals` (machine-measured) vs `DomainCandidate` |
| `question.py` | `QuestionProposal`, `SearchabilityProbe`, `PeerRating`, `ResearchQuestion` |
| `data.py` | `SourceDocument`, `ExtractedTable`, `ColumnMapping`, `CleaningReport`, `Dataset` |
| `experiment.py` | `Hypothesis`, `ExperimentSpec`, `StatResult`, `ExperimentResult`, `FigureSpec` |
| `critique.py` | `Objection`, `StatFlags`, `Critique`, `Verdict`, `RerouteTarget` |
| `paper.py` | `Paper` with markdown rendering, `RunManifest` |
| `events.py` | `RunEvent`, `EventType`, `ArtifactKind`, `RunSummary` |

The split that matters most is in `domain.py`: **LLM-authored** models carry no numbers used
in a decision; **machine-computed** models carry every number that does.

### `backend/llm/` — provider access

**`router.py`** — the only code that talks to a model. Agents request a *role*
(`REASONING`, `FAST`, `SYNTHESIS`), never a model name. All four supported providers expose
OpenAI-compatible endpoints, so provider support is a base URL and a model id rather than an
adapter per vendor.

Responsibilities: preflight by *generating* (not by reading `/models`, which lies), failover,
load balancing by remaining headroom, degradation ladder, structured-output enforcement with a
repair loop, and honouring provider retry hints.

**`budget.py`** — consumption tracking per `(provider, model)`. Sliding windows for
requests-per-minute, tokens-per-minute, requests-per-day, tokens-per-day. Reads are
non-destructive; eviction happens on write against the longest horizon.

**`structured.py`** — JSON recovery from models that promise JSON and deliver prose. Handles
markdown fences, leading and trailing commentary, trailing commas, bare lists where an object
was expected, single-key envelopes, and braces appearing in surrounding text. Failing that, it
shows the model its own output plus the validation error and asks again.

### `backend/tools/` — the outside world

| File | Purpose |
|---|---|
| `http.py` | Tiered fetcher: retry policy, per-host circuit breakers, size guards, politeness throttling, three-tier article extraction |
| `arxiv.py` | Publication counts, growth curves, full-text PDFs |
| `openalex.py` | Scholarly output, citation velocity |
| `github.py` | Repositories created post-cutoff, star velocity |
| `search.py` | Tavily with tiered fallback to Hacker News |
| `measure.py` | Concurrent multi-source measurement of one candidate domain |
| `pdf.py` | Text (PyMuPDF), tables (pdfplumber), figures (RapidOCR) |
| `sandbox.py` | AST-gated execution of model-written code |

### `backend/analysis/` — the arithmetic

| File | Purpose |
|---|---|
| `emergence.py` | Z-scored weighted ranking of candidate domains |
| `stats.py` | Effect sizes, bootstrap intervals, FDR correction, normality checks, power |
| `registry.py` | Six vetted statistical procedures |
| `plots.py` | Plotly figures serialised as JSON |
| `verify.py` | Citation and number verification against the evidence ledger |

### `backend/graph/` — control flow

`state.py` defines `ResearchState`, the single typed object carrying the whole run. `build.py`
assembles the LangGraph and owns every routing decision. Routing lives in one place by design:
an agent reports what happened, the graph decides what happens next. That separation is what
makes the iteration limit a guarantee rather than an instruction in a prompt.

---

## 4. The nine agents in detail

### 1. Domain Scout (`agents/scout.py`)

**Input:** nothing. **Output:** five measured, ranked `DomainCandidate`s.

1. Samples live evidence: arXiv category counts over the last 10 days, recent paper titles,
   real-time search results.
2. Asks a model to name the clusters *in that evidence*.
3. Measures each proposal against arXiv, OpenAlex, GitHub and public discussion — concurrently,
   since they are different hosts.
4. Ranks by Emergence Index (§7).

Search-term selection is defensive. Models asked for specificity reliably over-correct into
phrases so narrow that every source returns zero — *"indefinite causal order quantum
communication"* has no hits, while its core concept *"indefinite causal order"* has hundreds.
Each proposal's phrases are probed, plus shortened variants, and the first with real evidence
wins.

### 2. Peer Review Panel (`agents/panel.py`)

Growth measures whether a field is *rising*; it cannot measure whether a field is *researchable
in ten minutes with public data*. Two reviewers score every candidate independently, with
different prompts and — where a second provider is configured — different model families. When
they disagree beyond a threshold, a third tiebreak round runs **with the disagreement shown to
it**.

Final ordering is the product of panel score and a positive-shifted Emergence Index: a domain
must be both rising and studiable.

### 3. Question Generator (`agents/question.py`)

Produces 3–5 questions, each declaring ≥2 distinct sources that must be *joined*. Then:

- **Searchability probe.** Each question is actually searched. If a result answers it directly,
  it is rejected as trivial and regenerated.
- **Semantic negative memory.** Rejected questions enter vector memory. Similarity alone cannot
  separate a rewording from a genuinely new question (§12), so the vector store supplies recall
  and a cheap model call supplies precision on ambiguous pairs.
- **Peer rating** for novelty and feasibility.

The prompt states exactly what the pipeline can measure, so it stops proposing questions the
system structurally cannot serve.

### 4. Data Alchemist (`agents/alchemist.py`)

The largest agent, and the one with a machine-checked floor: **≥3 distinct modalities** or it
fails upward and the supervisor re-routes to a different question.

Acquires from five modalities, joins OpenAlex records to arXiv preprints by fuzzy title match,
derives question-specific features from abstracts (§6), aligns schemas (LLM proposes, Python
validates), drops zero-variance columns, and records source conflicts rather than resolving
them silently.

Documents already fetched this run are reused across reroutes — see §12 for why that matters.

### 5. Experiment Designer (`agents/designer.py`)

Pre-registers H₀, H₁ and a prediction **before** execution, so the Critic can catch the system
rationalising whatever it happens to find.

Validation before running: columns exist, types match, group columns actually have groups,
enough rows. Two guards are worth naming:

- **Tautology guard.** A derived column cannot be correlated against its own ingredients.
- **Identifier guard.** A column with near-unique values identifies rows rather than grouping
  them, and is rejected as a target or group.

And critically, it may declare a question **unanswerable** rather than substituting an
available column — see §12.

### 6. Executor (`agents/executor.py`)

Two paths: the typed registry, and AST-restricted sandboxed code for anything it cannot
express. Both repair themselves — the error goes back to the model, which corrects and retries
up to three times. Every attempt is recorded, so the paper can state how many designs were tried.

### 7. Uncertainty Quantifier (`agents/uncertainty.py`)

Extracts specific claims, scores each from four measured signals, and **abstains below 60%**
(§8).

### 8. Critic (`agents/critic.py`)

Statistical flags are computed in Python — whether *p* exceeds alpha is arithmetic, not
opinion. The model argues about confounding, construct validity, selection bias, and question
drift.

Every objection must cite a URL. **That URL is fetched.** Objections whose evidence cannot be
retrieved are discarded, so inventing a damning reference loses the argument rather than
winning it. If the mechanical checks show a blocking problem, an "accept" verdict is overridden.

### 9. Paper Writer (`agents/writer.py`)

Given a **closed reference list** and told to cite `[n]` only. Numbers are injected from
computed `StatResult` objects — the Results section is authored by code. Afterwards the
markdown is verified against the ledger and failures are printed *in the paper* (§9).

---

## 5. How a run executes, step by step

```
POST /api/runs
  └─ ResearchRunner: event bus, HTTP client, LLM router, vector memory
     └─ router.preflight()          one-token generation probe per model
        └─ graph.ainvoke(state)     wall-clock timeout wraps everything
```

Each node follows the same lifecycle in `agents/base.py`:

1. `node_enter` event → the UI lights that node
2. `execute()` runs
3. Failures are contained: `AgentFailure` records into state; `BudgetExhausted` ends the run
   honestly; anything unexpected is caught and logged rather than crashing the graph
4. `node_exit` event with timing

**Failure policy.** An agent that raises does *not* crash the run. Whether the run can continue
without that node's output is the graph's decision, not the node's — a dead Scout should end
the run, a dead OCR pass should not.

**Termination is guaranteed twice over:** the cycle cap in routing, and a wall-clock timeout
around the whole graph. A reviewer watching a live URL is never left with a spinner.

---

## 6. Data acquisition

### The five modalities

| Modality | Source | Extraction |
|---|---|---|
| `STRUCTURED_API` | OpenAlex | JSON records: year, citations, authors |
| `PDF` | arXiv | PyMuPDF text layer |
| `TABULAR` | PDF tables | pdfplumber, promoted when confidence ≥ 0.55 |
| `IMAGE` | PDF figures | RapidOCR — chart axis values appear nowhere else |
| `HTML` | Web article | Tiered extraction |

The floor is **distinct modalities**, not source count: three PDFs are not three sources.

### Derived features

Metadata alone cannot answer most interesting questions. A question about *"papers proposing
RL-based allocation versus heuristic methods"* needs a column saying which papers use
reinforcement learning — and no such column exists in bibliometric metadata.

Every row is built from an arXiv record carrying its **full abstract**. So:

1. One model call proposes keyword vocabulary from the question.
2. **Python applies it** across every abstract, producing deterministic binary columns.
3. Quality gates reject features that restate existing metadata, or that split the papers too
   unevenly to compare.

The model proposes vocabulary — what it is good at. Python decides membership, so the column is
a reproducible function of the text rather than a per-paper opinion.

### Cleaning and provenance

Every document carries `Provenance`: URL, SHA-256 of content, retrieval timestamp, modality.
Cleaning records rows in/out, duplicates removed, mappings validated and rejected, and columns
dropped for zero variance. Conflicts between sources are emitted as first-class `Conflict`
objects that depress confidence and must be addressed by the Critic.

---

## 7. The Emergence Index

Six measured signals per candidate:

| Signal | Weight | Meaning |
|---|---|---|
| arXiv growth ratio | 0.22 | Post-2024 volume vs equal-length prior window |
| arXiv relative slope | 0.18 | Month-over-month trend, normalised by the field's own mean |
| OpenAlex growth ratio | 0.22 | Same comparison across all published science |
| OpenAlex citation velocity | 0.12 | How fast recent work accrues citations |
| GitHub repos + star velocity | 0.20 | Repositories *created* after the cutoff, stars/day |
| Public attention | 0.06 | Recency tiebreaker |

Three statistical decisions:

**Log-scaling before standardising.** Growth ratios are heavy-tailed — a run might see 1.1,
2.4, 25.7 and 300. A raw z-score is dominated by the largest value and every other candidate
lands at roughly −0.5. `log1p` compresses the tail so comparison reflects order of magnitude.

**Missing measurements impute to the mean, not zero.** A rate-limited GitHub means that
candidate's code signal is *unknown*, not *absent*. Scoring it zero would punish a field for an
outage on our side.

**Completeness is reported.** A domain scored on two of four sources carries lower confidence.

### It measures acceleration, not size

In one run, Graph Neural Networks had **5,805 repositories** — far more than any other
candidate — and ranked third, because its growth was only 1.4×. Mechanistic Interpretability
won on 14.5× arXiv growth. That distinction is the point.

---

## 8. Confidence and abstention

Asking a model for a confidence score produces a number unrelated to correctness. Confidence is
assembled from four measurable signals:

| Signal | Weight | How it is obtained |
|---|---|---|
| Statistical evidence | 40% | p-value, effect size, interval width, n — computed, no model |
| Self-consistency | 20% | Same judgement sampled 3× at temperature > 0 |
| Cross-model agreement | 20% | Same judgement put to a different model family |
| Evidence quality | 20% | Independent sources, unresolved conflicts, dataset size |

**Statistical evidence is scaled, not binary.** p = 0.049 and p = 1e-9 are not the same
evidence, and a cliff edge at 0.05 is exactly the thinking the Critic exists to attack.

**Missing cross-model agreement is redistributed, not defaulted.** A single-provider deployment
does not receive confidence it did not earn.

Below 60%, the claim is not asserted — it moves to an **"Abstained — Insufficient Evidence"**
section of the paper, and the UI shows the refusal. In one run all three claims scored 42%, 58%
and 48%: the paper asserted nothing. That is the system working.

---

## 9. Anti-fabrication

A model writing a research paper *will* invent citations. Three layers, all mechanical:

**1. Closed citation list.** The writer receives numbered references built only from documents
actually fetched, and is told to cite `[n]`.

**2. Injected numbers.** Statistics are inserted from computed objects. The model is explicitly
told not to restate figures in prose.

**3. Post-hoc verification** (`analysis/verify.py`):

- Every URL and arXiv id is checked against the provenance ledger. Fabrications are **stripped**
  and replaced with a visible marker.
- Every `symbol = value` claim is matched against computed results within tolerance.
- Every cited URL is re-fetched.

Failures are **printed in the paper**. A verification section admitting *"one citation was
removed because it was not in the evidence ledger"* is more convincing than a clean paper with
no audit trail.

### Measured across three consecutive live runs

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Citations traced to fetched sources | 13/13 | 14/14 | 18/18 |
| Critic objections with verifiable evidence | 7/7 | 10/10 | 8/8 |
| Fabricated numbers caught | 1 | 2 | 1 |
| Dead URLs caught on re-check | 1 | 1 | 2 |

**45 of 45 citations verified. Zero fabrications survived.** Four invented numbers were caught,
including `arXiv:1805.05238` — a plausible-looking identifier for a paper this run never
fetched.

---

## 10. Resilience on free tiers

### Per-model budgets

Quotas are issued **per model**, and differ by more than an order of magnitude within one
provider: Gemini's `3.5-flash` allows 20 requests/day while its lite siblings allow 500.
Provider-level budgeting either starves the plentiful models or overspends the scarce one.

### Degradation ladder

```
1. preferred provider, preferred model for the role
2. next provider, same role         (ordered by remaining headroom, not fixed preference)
3. any provider, one tier cheaper   (REASONING → FAST)
4. wait out the shortest cooldown, retry once
5. BudgetExhausted — the run ends honestly
```

### Self-repair

- **Unsupported `response_format`** → drop the constraint, remember it, retry the same model
- **429 with a short retry hint** → sleep exactly that long and retry the *same* model
- **429 daily exhaustion** → bench the model for 30 minutes, keep its siblings
- **Model 404** → mark unavailable, stop asking
- **Malformed JSON** → show the model its own output and the validation error

### Circuit breakers

Per host, opening after three consecutive failures. A dead source degrades one branch of the
plan rather than stalling the run — and the Data Alchemist reads breaker state and re-plans
around what survives.

---

## 11. Testing

**280 tests, running fully offline.** No network, no LLM calls, no credentials — CI needs
nothing configured.

| File | Covers |
|---|---|
| `test_structured.py` | JSON recovery from every observed malformed shape |
| `test_budget.py` | Sliding windows, per-model isolation, penalty semantics |
| `test_router.py` | Quota calibration, retry hints, degradation |
| `test_http.py` | Retry policy, circuit breakers, size guards (mocked with respx) |
| `test_arxiv.py` | Growth measurement, partial-month trimming, slope |
| `test_emergence.py` | Weights, log-scaling, imputation, ranking |
| `test_sandbox.py` | Every classic Python sandbox escape |
| `test_verify.py` | Fabricated citations, unsupported numbers |
| `test_routing.py` | Cycle cap, reroute targets, compiled-graph termination |
| `test_vector.py` | Chunking, retrieval, duplicate-detection calibration |
| `test_features.py` | Feature derivation, quality gates |
| `test_events.py` | SSE wire format, replay, sequencing |
| `test_packaging.py` | Manifest drift, Dockerfile portability |

Most tests are **regression guards written after a real failure**, with the observed behaviour
recorded in the docstring. The calibration table in `test_vector.py`, for instance, exists
because measurement disproved the design.

```bash
.venv/Scripts/python -m pytest backend/tests -q
.venv/Scripts/python -m ruff check backend/
```

---

## 12. Development log: failures and fixes

The most instructive part of this project. Every entry was found by running the system, not by
reasoning about it, and each one was silent — producing plausible wrong output rather than an
error.

### Measurement bugs

**arXiv slope reported negative for a field growing 14×.** The current month is partial and the
oldest month in a capped sample is truncated; both drag the fit down. Now only fully-observed
months are used, and a slope needs ≥3 of them.

**OpenAlex returned 166,808 results for a niche field.** The default `search` parameter scores
loosely across full text. Measured alternatives:

| Query form | Results |
|---|---|
| `search=<term>` | 166,808 |
| `search="<term>"` | 7,927 |
| `title_and_abstract.search:<term>` | 11,881 |
| `title_and_abstract.search:"<term>"` | **2,745** |

A 60× difference between a meaningful signal and noise.

**Budget windows evicted each other.** The minute-window read pruned events the day-window read
still needed, so **daily quota ceilings never fired**. Reads are now non-destructive.

**Token ceiling wrong by 2.3×.** Configured at 14,000 TPM; the provider's own 429 stated
`Limit 6000`. The budget manager approved calls the provider then rejected — the exact failure
it exists to prevent.

### Provider bugs

**Model catalogues lie.** `gemini-2.5-flash` appears in `/models` and returns 404 on
generation. Preflight now *generates a token* per model.

**Hidden reasoning tokens count against `max_tokens`.** `gemini-3.5-flash` returned
`finish_reason=length` after **32 visible tokens** against an 800 ceiling. Ceilings are now
padded per provider.

**A 340ms wait ended a 15-minute run.** Groq replied *"Please try again in 340ms"*; the code
benched the model for 60 seconds, walked the failover chain, and killed the run with "every
provider is rate-limited". Retry hints are now honoured.

**Parallel sampling defeated the budget check.** Self-consistency fired k requests
simultaneously; all checked affordability before any was recorded, so every one passed and the
burst blew the limit. Now bounded by a semaphore.

### Scientific-validity bugs

**A tautology reported as a finding.** `citations_per_year` vs `citations` at ρ = 0.976,
p = 2e-15 — flawless statistics describing division. Nothing downstream would have caught it,
because the p-value is superb. Datasets now declare `derived_from`.

**Three consecutive papers answered the wrong question.** Questions asked about *research*
properties ("uses reinforcement learning", "releases code"); the dataset held only structural
metadata. The Designer correctly refused 3, 5 and 8 times — then the write-up fell back to
whatever trivia had run. One paper was titled *"The Relationship Between Author Count and Title
Length"* under a question about citation counts.

The abstracts were in memory the entire time. Fixed by deriving question-specific features from
them, plus a `DESIGN → QUESTION` reroute and question drift as the Critic's first-priority
attack.

**Vector similarity cannot detect duplicate questions.** Measured against a reference question:

| Probe | Similarity |
|---|---|
| Near-identical rewording | 0.984 |
| Paraphrase, same meaning | 0.828 |
| **Different question** | **0.812** |
| Unrelated | 0.512 |

A genuinely new question scores *above* a true duplicate. No threshold separates them. The
design became recall from the vector store, precision from a model.

### Performance bugs

**719 of 900 seconds spent re-indexing.** The Critic rerouted to the data phase twice, and each
visit re-downloaded, re-OCR'd and re-embedded the same four PDFs — 128s, 132s and 459s. Fixed
with content-hash dedupe, document reuse across reroutes, and index caps: **719s → 14s**.

**OCR triggered on "no tables found."** A table-less paper is not a scanned paper. 62s → 2.5s
per PDF.

### Interface bugs

**SSE `event:` field broke the browser.** Naming it makes the browser dispatch a custom event
type, so `EventSource.onmessage` never fires — a failure indistinguishable from a dead backend.

**`requestAnimationFrame` froze the feed.** Events were batched behind rAF, which does not fire
in a backgrounded tab. On a run lasting minutes, switching tabs is normal behaviour, and the
page would appear dead. Now on a timer.

**Reroutes were invisible.** Nothing published the `REROUTE` event, so the red edges never
animated and the counter read 0 after five rejected cycles — the clearest evidence of a state
machine, absent from the one place a reviewer looks.

**`max_cycles` was silently dropped.** LangGraph propagates only *declared* channels. Written
onto the state dict without being declared, it vanished between nodes and routing fell back to
a default — a run configured for 2 cycles ran to 3, then died on the recursion limit. **The
iteration cap was not actually enforced.**

### A recurring theme

Several bugs were **masked by graceful degradation**. When `embed_query` was missing, search
threw, the caller turned it into "no results", and every other signal reported a healthy store.
Later, an empty `vector.py` passed the whole test suite because the vector tests *skip* when
memory is unavailable.

The lesson: a fallback that cannot distinguish *"disabled"* from *"broken"* will eventually be
asked to.

---

## 13. Limitations

Stated plainly, because a system that reports its own uncertainty should do the same about
itself.

**Verification proves support, not truth.** The system confirms a claim is supported by the
data it gathered. A biased sample yields a well-verified wrong conclusion. The Critic raises
this as a limitation rather than fixing it.

**Sample sizes are small** — typically 14–60 papers, so most findings are underpowered. This is
why claims frequently abstain, and it is the honest outcome rather than a failure.

**Table extraction is not validated against the source.** A mis-parsed table produces numbers
that are wrong but internally consistent. Confidence is heuristic.

**Citation relevance is not checked.** Verification confirms a URL was fetched and resolves,
not that it supports the sentence it is attached to.

**The OpenAlex↔arXiv join is fuzzy**, matching titles at 0.82 similarity. A mismatch pairs the
wrong citation count to the wrong paper.

**OpenAlex's free budget is per-IP and daily.** When the Scout exhausts it, citation columns are
unavailable and the run falls back to arXiv metadata — and says so in the paper.

### What I would build next

1. **`cli verify <run_id>`** — re-fetch every source, compare hashes, recompute the statistics
   from the stored dataset, and confirm they match the paper. Turns "trust the run" into
   "reproduce the run".
2. **Citation relevance checking** — verify a fetched page actually supports the sentence.
3. **Larger samples** via paginated acquisition, to lift the statistical power floor.
