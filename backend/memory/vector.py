"""Vector memory: retrieval over acquired documents, and semantic negative memory.

Two jobs, both of which the system was measurably worse without.

**Retrieval over acquired documents (RAG).** The Data Alchemist downloads four
PDFs at roughly 60k characters each. That is a quarter of a million characters
of real evidence which cannot fit in any prompt, so before this module existed
the agents saw only titles and abstracts -- the full texts were gathered,
hashed, cited, and then never actually read. Chunking and embedding them lets
the Experiment Designer and the Paper Writer pull the passages that bear on
*this* question.

**Semantic negative memory.** Rejected questions were previously avoided by
listing them in the prompt and asking the generator not to repeat them. A
paraphrase walks straight through that. Cosine similarity does not.

A note on scope, because the assessment is explicit that "RAG-only" systems are
unacceptable: retrieval here is one tool among many inside a multi-agent state
machine. It informs agents that plan, execute statistics, criticise, and
iterate. It is not the architecture.

Storage is in-memory (Chroma's ephemeral client), scoped to a single run. Runs
are independent experiments and should not contaminate each other, and an
in-memory store needs no disk permissions on a free-tier container.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..schemas import Provenance, SourceDocument

log = logging.getLogger(__name__)

#: Chunk size in characters. Large enough to carry a whole argument, small
#: enough that a handful fit in a prompt alongside everything else an agent
#: needs. Roughly 200 tokens.
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150

#: Ceilings on how much gets embedded. Retrieval asks for the top 4-5 passages,
#: so indexing a full 70k-character paper end to end costs about 25 seconds and
#: improves the answer barely at all. A paper's opening sections carry its
#: question, method and headline results, which is what the agents query for.
MAX_CHUNKS_PER_DOCUMENT = 60
MAX_CHUNKS_PER_RUN = 260

# --- Duplicate-question thresholds, calibrated by measurement --------------
#
# Measured against the reference question "Does the presence of a GitHub
# repository correlate with citation counts?" using bge-small-en-v1.5:
#
#     near-identical rewording ........ 0.984
#     paraphrase, same meaning ........ 0.828
#     reworded, same meaning .......... 0.810
#     DIFFERENT question (author count) 0.812   <-- higher than a true duplicate
#     same domain, new question ....... 0.743
#     unrelated ....................... 0.512
#
# A genuinely new question scores *above* a true duplicate, so no single
# threshold can separate them: the embedding captures topical similarity, not
# semantic equivalence. A naive cut anywhere near 0.82 silently discards valid
# questions, which is the more expensive error -- the run loses a whole
# regeneration round and never learns why.
#
# So the vector store is used for **recall** and a model for **precision**:
# anything above the certain threshold is a duplicate outright, anything in the
# ambiguous band is adjudicated by a cheap LLM call, and anything below is
# allowed through untouched.
REJECTION_CERTAIN = 0.95
REJECTION_AMBIGUOUS = 0.78

DOCUMENTS = "documents"
REJECTIONS = "rejections"


@dataclass
class Retrieved:
    """One retrieved chunk."""

    text: str
    source_url: str = ""
    title: str = ""
    modality: str = ""
    similarity: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class _FastEmbedFunction:
    """Chroma embedding function backed by fastembed.

    fastembed runs a quantised ONNX model locally: no embedding API, no key, no
    per-call quota. Given that every other budget in this system is a scarce
    free-tier allowance, an embedding path that costs nothing and cannot be
    rate-limited is worth the ~90MB model download.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self._name = model_name

    def _embed(self, value: str | list[str]) -> list[list[float]]:
        texts = [value] if isinstance(value, str) else list(value)
        return [vector.tolist() for vector in self._model.embed(texts)]

    def __call__(self, input: str | list[str]) -> list[list[float]]:  # noqa: A002 - Chroma's API
        return self._embed(input)

    # Chroma calls `__call__` when *indexing* but `embed_query` /
    # `embed_documents` when *querying*, passing `input` as a keyword and
    # expecting a list of vectors back from both.
    #
    # Implementing only `__call__` indexes perfectly and then fails every
    # search -- and because the caller turns that exception into "no results",
    # retrieval silently returns nothing while the collection count, the
    # indexing log, and every other signal report a healthy store. Graceful
    # degradation hid this bug rather than surfacing it, which is worth
    # remembering: a fallback that cannot distinguish "empty" from "broken"
    # will eventually be asked to.
    def embed_query(self, input: str | list[str]) -> list[list[float]]:  # noqa: A002
        return self._embed(input)

    def embed_documents(self, input: str | list[str]) -> list[list[float]]:  # noqa: A002
        return self._embed(input)

    def name(self) -> str:
        """Chroma requires this for collection metadata."""
        return f"fastembed:{self._name}"


def chunk_text(text: str, *, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries.

    Overlap matters for scientific prose: a method and the number it produces
    frequently straddle a boundary, and a hard split leaves the retrieved chunk
    describing an approach without its result.
    """
    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))

        if end < len(text):
            # Prefer a paragraph break, then a sentence end, within the last
            # third of the window; otherwise accept the hard cut.
            window = text[start + (2 * size) // 3 : end]
            for pattern in ("\n\n", ". ", "\n"):
                position = window.rfind(pattern)
                if position != -1:
                    end = start + (2 * size) // 3 + position + len(pattern)
                    break

        chunk = text[start:end].strip()
        if len(chunk) > 60:  # skip slivers left by a boundary landing awkwardly
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


class VectorMemory:
    """Per-run vector store. Degrades to a no-op rather than failing a run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._client: Any = None
        self._collections: dict[str, Any] = {}
        self._embedder: Any = None
        self._available = False
        self._chunks_indexed = 0
        #: Content hashes already embedded. The Critic can reroute to the data
        #: phase several times per run, and without this the same documents are
        #: re-embedded on every visit -- measured at 128s, 132s and 459s across
        #: three visits, which is 719s of a 900s budget spent re-learning what
        #: was already known.
        self._indexed_hashes: set[str] = set()

    def initialise(self) -> bool:
        """Load the embedding model and open the store.

        Returns False if unavailable. Retrieval is an enhancement: losing it
        should degrade the depth of evidence the agents see, never end the run.
        """
        try:
            import chromadb

            self._embedder = _FastEmbedFunction()
            self._client = chromadb.EphemeralClient()
            self._available = True
            log.info("vector memory ready for run %s", self.run_id)
        except Exception as exc:  # noqa: BLE001 - optional capability
            log.warning("vector memory unavailable (%s); agents will see less evidence", exc)
            self._available = False
        return self._available

    @property
    def available(self) -> bool:
        return self._available

    @property
    def chunks_indexed(self) -> int:
        return self._chunks_indexed

    def _collection(self, namespace: str) -> Any:
        if namespace not in self._collections:
            self._collections[namespace] = self._client.get_or_create_collection(
                name=f"{self.run_id}-{namespace}",
                embedding_function=self._embedder,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[namespace]

    # ----------------------------------------------------------- documents

    def index_documents(self, documents: list[SourceDocument]) -> int:
        """Chunk and embed newly-acquired documents. Returns chunks indexed.

        Documents already embedded this run are skipped by content hash, and
        the total is capped. Embedding costs roughly 70ms per chunk on CPU, so
        an uncapped re-index of four full papers on every reroute is the single
        most expensive thing the system can do -- and it buys nothing, because
        retrieval only ever asks for the top handful.
        """
        if not self._available:
            return 0

        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []
        skipped = 0

        for document in documents:
            digest = document.provenance.sha256 or Provenance.hash_content(document.text or "")
            if digest in self._indexed_hashes:
                skipped += 1
                continue
            self._indexed_hashes.add(digest)

            body = document.text or ""
            # Tables carry numbers the prose often omits, so they are indexed
            # as their own chunks rather than being lost.
            for table in document.tables:
                header = " | ".join(table.columns)
                rows = "\n".join(" | ".join(str(c) for c in row) for row in table.rows[:12])
                body += f"\n\n[table from page {table.page}]\n{header}\n{rows}"

            for chunk_index, chunk in enumerate(chunk_text(body)[:MAX_CHUNKS_PER_DOCUMENT]):
                texts.append(chunk)
                metadatas.append(
                    {
                        "source_url": document.url,
                        "title": (document.provenance.title or "")[:200],
                        "modality": document.modality.value,
                        "ocr": bool(document.ocr_used),
                    }
                )
                ids.append(f"{digest[:8]}-c{chunk_index}")

        remaining = max(0, MAX_CHUNKS_PER_RUN - self._chunks_indexed)
        if len(texts) > remaining:
            log.info("capping index at %d chunks for this run", MAX_CHUNKS_PER_RUN)
            texts, metadatas, ids = texts[:remaining], metadatas[:remaining], ids[:remaining]

        if not texts:
            if skipped:
                log.info("all %d documents were already indexed", skipped)
            return 0

        try:
            self._collection(DOCUMENTS).add(documents=texts, metadatas=metadatas, ids=ids)
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to index documents: %s", exc)
            return 0

        self._chunks_indexed += len(texts)
        log.info(
            "indexed %d chunks from %d new documents (%d already indexed)",
            len(texts),
            len(documents) - skipped,
            skipped,
        )
        return len(texts)

    def search(self, query: str, *, k: int = 5, namespace: str = DOCUMENTS) -> list[Retrieved]:
        """Retrieve the chunks most relevant to a query."""
        if not self._available or not query.strip():
            return []

        try:
            collection = self._collection(namespace)
            count = collection.count()
            if count == 0:
                return []
            result = collection.query(query_texts=[query], n_results=min(k, count))
        except Exception as exc:  # noqa: BLE001
            log.warning("vector search failed: %s", exc)
            return []

        out: list[Retrieved] = []
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        for text, metadata, distance in zip(documents, metadatas, distances, strict=False):
            metadata = metadata or {}
            out.append(
                Retrieved(
                    text=text,
                    source_url=str(metadata.get("source_url", "")),
                    title=str(metadata.get("title", "")),
                    modality=str(metadata.get("modality", "")),
                    # Chroma returns cosine *distance*; similarity is 1 - d.
                    similarity=round(1.0 - float(distance), 4),
                    metadata=metadata,
                )
            )
        return out

    def context_for(self, query: str, *, k: int = 5, max_chars: int = 4000) -> str:
        """Retrieved evidence, formatted for a prompt.

        Every passage carries its source URL so an agent quoting it produces a
        citation that the verification pass can trace back to a real fetch.
        """
        hits = self.search(query, k=k)
        if not hits:
            return ""

        blocks: list[str] = []
        used = 0
        for hit in hits:
            block = (
                f"[source: {hit.title or hit.source_url} — {hit.source_url}]\n{hit.text}"
            )
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)

        return "\n\n---\n\n".join(blocks)

    # ---------------------------------------------------- negative memory

    def remember_rejection(self, question: str, reason: str = "") -> None:
        """Record a rejected question so paraphrases can be detected later."""
        if not self._available or not question.strip():
            return
        try:
            collection = self._collection(REJECTIONS)
            collection.add(
                documents=[question],
                metadatas=[{"reason": reason[:300]}],
                ids=[f"r{collection.count()}"],
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to record rejection: %s", exc)

    def similar_rejection(self, question: str) -> tuple[Retrieved | None, bool]:
        """Find the nearest rejected question. Returns (match, needs_adjudication).

        - `(hit, False)` above `REJECTION_CERTAIN`: a duplicate beyond doubt.
        - `(hit, True)` in the ambiguous band: topically close, but similarity
          alone cannot tell a rewording from a genuinely different question --
          the caller must adjudicate. See the calibration table above.
        - `(None, False)`: far enough away to allow.
        """
        hits = self.search(question, k=1, namespace=REJECTIONS)
        if not hits:
            return None, False

        best = hits[0]
        if best.similarity >= REJECTION_CERTAIN:
            return best, False
        if best.similarity >= REJECTION_AMBIGUOUS:
            return best, True
        return None, False

    def stats(self) -> dict[str, Any]:
        if not self._available:
            return {"available": False}
        counts = {}
        for namespace, collection in self._collections.items():
            try:
                counts[namespace] = collection.count()
            except Exception:  # noqa: BLE001
                counts[namespace] = -1
        return {"available": True, "chunks_indexed": self._chunks_indexed, "collections": counts}

    def close(self) -> None:
        self._collections.clear()
        self._client = None
