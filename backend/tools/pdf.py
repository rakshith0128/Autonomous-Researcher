"""PDF extraction: text, tables, and figures.

The assessment requires handling messy data -- "OCR on papers, table
extraction, schema alignment" -- and scientific PDFs are where all three go
wrong at once. Three extraction paths run against one document:

* **Text** via PyMuPDF. Fast and correct for the ~95% of arXiv PDFs that carry
  a real text layer.
* **Tables** via pdfplumber. Genuinely hard: a "table" in a physics paper is
  often ruled lines the parser cannot see, so results carry an explicit
  confidence and the paper's methods section is expected to admit it.
* **Figures** via PyMuPDF image extraction, then OCR. This is the *image*
  modality -- a chart's axis labels and a table rendered as a bitmap are
  frequently the only place a number appears.

OCR is imported lazily and only runs when the text layer is thin, because
onnxruntime is the single heaviest dependency in the image and most documents
never need it.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field

from ..schemas import ExtractedTable, Modality, Provenance, SourceDocument

log = logging.getLogger(__name__)

# Below this many characters per page, assume the text layer is missing or
# broken and fall back to OCR. Scanned papers typically yield near zero;
# a normal two-column page yields several thousand.
OCR_TEXT_THRESHOLD_PER_PAGE = 180

# Ignore small images: journal logos, ORCID icons, and equation fragments are
# extracted as images too, and OCR'ing them wastes seconds per document.
MIN_FIGURE_PIXELS = 40_000  # e.g. 200x200

# OCR is by far the most expensive operation in the pipeline -- roughly 10
# seconds per figure. Against a 5-10 minute run budget that buys very few
# figures, so the cap is low and a wall-clock budget backstops it.
MAX_OCR_FIGURES = 3
OCR_TIME_BUDGET_SECONDS = 25.0

MAX_PAGES = 40


@dataclass
class PdfExtraction:
    """Everything recovered from one PDF."""

    text: str = ""
    tables: list[ExtractedTable] = field(default_factory=list)
    figure_texts: list[str] = field(default_factory=list)
    page_count: int = 0
    ocr_used: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return bool(self.text.strip() or self.tables or self.figure_texts)


def extract_text(data: bytes, *, max_pages: int = MAX_PAGES) -> tuple[str, int, list[str]]:
    """Pull the text layer. Returns (text, page_count, warnings)."""
    warnings: list[str] = []
    try:
        import pymupdf
    except ImportError:  # pragma: no cover - dependency is declared
        return "", 0, ["pymupdf unavailable"]

    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            page_count = doc.page_count
            if page_count > max_pages:
                warnings.append(f"document has {page_count} pages; reading first {max_pages}")
            chunks = [doc[i].get_text() for i in range(min(page_count, max_pages))]
        return _normalise("\n".join(chunks)), page_count, warnings
    except Exception as exc:  # noqa: BLE001 - malformed PDFs are routine input
        log.warning("PDF text extraction failed: %s", exc)
        return "", 0, [f"text extraction failed: {exc}"]


def extract_tables(
    data: bytes, source_url: str, *, max_pages: int = 12, max_tables: int = 8
) -> list[ExtractedTable]:
    """Recover tables with pdfplumber.

    Confidence is heuristic and deliberately conservative. A table whose rows
    have inconsistent column counts is usually a mis-parse of a multi-column
    text block, and passing that downstream as clean data is worse than
    reporting nothing -- the Critic would rightly attack any statistic built
    on it.
    """
    tables: list[ExtractedTable] = []
    try:
        import pdfplumber
    except ImportError:  # pragma: no cover
        return tables

    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page_no, page in enumerate(pdf.pages[:max_pages], start=1):
                for raw in page.extract_tables() or []:
                    if len(raw) < 2:
                        continue
                    header = [_cell(c) for c in raw[0]]
                    rows = [[_cell(c) for c in row] for row in raw[1:]]
                    if not any(header) or not rows:
                        continue

                    tables.append(
                        ExtractedTable(
                            source_url=source_url,
                            page=page_no,
                            columns=header,
                            rows=rows,
                            extraction_method="pdfplumber.extract_tables",
                            confidence=_table_confidence(header, rows),
                        )
                    )
                    if len(tables) >= max_tables:
                        return tables
    except Exception as exc:  # noqa: BLE001
        log.warning("table extraction failed for %s: %s", source_url, exc)

    return tables


def _table_confidence(header: list[str], rows: list[list[str]]) -> float:
    """Score how likely this really is a table rather than mis-parsed text."""
    if not header or not rows:
        return 0.0

    width = len(header)
    consistent = sum(1 for row in rows if len(row) == width) / len(rows)
    named = sum(1 for h in header if h.strip()) / width

    cells = [c for row in rows for c in row]
    filled = sum(1 for c in cells if c.strip()) / len(cells) if cells else 0.0

    # A real data table is mostly rectangular, mostly populated, and has
    # labelled columns. Weighting consistency highest because a ragged row
    # count is the strongest single indicator of a mis-parse.
    return round(0.5 * consistent + 0.3 * named + 0.2 * filled, 3)


def extract_figure_text(
    data: bytes,
    *,
    max_figures: int = MAX_OCR_FIGURES,
    max_pages: int = 20,
    time_budget: float = OCR_TIME_BUDGET_SECONDS,
) -> list[str]:
    """OCR embedded figures to recover axis labels and bitmap tables."""
    import time as _time

    texts: list[str] = []
    try:
        import pymupdf
    except ImportError:  # pragma: no cover
        return texts

    reader = _ocr_reader()
    if reader is None:
        return texts

    deadline = _time.monotonic() + time_budget
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            for page_index in range(min(doc.page_count, max_pages)):
                for img in doc[page_index].get_images(full=True):
                    if len(texts) >= max_figures or _time.monotonic() > deadline:
                        return texts
                    try:
                        pix = pymupdf.Pixmap(doc, img[0])
                        if pix.width * pix.height < MIN_FIGURE_PIXELS:
                            continue
                        if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                        text = _ocr_image(reader, pix.tobytes("png"))
                        if text and len(text) > 20:
                            texts.append(text)
                    except Exception as exc:  # noqa: BLE001 - per-image failure is fine
                        log.debug("figure OCR skipped: %s", exc)
    except Exception as exc:  # noqa: BLE001
        log.warning("figure extraction failed: %s", exc)

    return texts


_OCR_SINGLETON: object | None = None
_OCR_TRIED = False


def _ocr_reader():  # noqa: ANN202 - third-party type
    """Load the OCR engine once, lazily.

    Model load costs several seconds and a few hundred MB of resident memory.
    Paying that on import would slow every run, including the majority that
    never touch a scanned page.
    """
    global _OCR_SINGLETON, _OCR_TRIED
    if _OCR_TRIED:
        return _OCR_SINGLETON
    _OCR_TRIED = True

    from ..config import get_settings

    if not get_settings().enable_ocr:
        log.info("OCR disabled by configuration; skipping the image modality")
        _OCR_SINGLETON = None
        return None

    try:
        from rapidocr_onnxruntime import RapidOCR

        _OCR_SINGLETON = RapidOCR()
        log.info("OCR engine loaded")
    except Exception as exc:  # noqa: BLE001 - OCR is an enhancement, never required
        log.warning("OCR unavailable (%s); continuing without it", exc)
        _OCR_SINGLETON = None
    return _OCR_SINGLETON


def _ocr_image(reader, png_bytes: bytes) -> str:  # noqa: ANN001
    try:
        result, _ = reader(png_bytes)
    except Exception as exc:  # noqa: BLE001
        log.debug("OCR call failed: %s", exc)
        return ""
    if not result:
        return ""
    return _normalise(" ".join(line[1] for line in result if len(line) > 1))


def parse_pdf(
    data: bytes,
    url: str,
    *,
    title: str = "",
    want_tables: bool = True,
    ocr_figures: str = "auto",
) -> SourceDocument:
    """Full extraction pipeline for one PDF, as a typed SourceDocument.

    `ocr_figures` controls the expensive path:

    * ``"auto"``  -- OCR only when the text layer is too thin to be real,
      i.e. an actually scanned document. This is the default and costs nothing
      on the ~95% of arXiv PDFs that carry proper text.
    * ``"force"`` -- always OCR. The Data Alchemist uses this on a single
      chosen paper to obtain the *image* modality, which counts toward the
      three-disparate-sources floor.
    * ``"never"`` -- skip entirely.

    An earlier version also triggered OCR when no tables were found, on the
    theory that a table-less paper might be scanned. That was wrong: plenty of
    papers simply have no tables, and it made every such document pay a
    60-second OCR penalty for nothing.
    """
    extraction = PdfExtraction()
    extraction.text, extraction.page_count, extraction.warnings = extract_text(data)

    if want_tables:
        extraction.tables = extract_tables(data, url)

    pages = max(extraction.page_count, 1)
    thin = len(extraction.text) / pages < OCR_TEXT_THRESHOLD_PER_PAGE

    if ocr_figures == "force" or (ocr_figures == "auto" and thin):
        extraction.figure_texts = extract_figure_text(data)
        extraction.ocr_used = bool(extraction.figure_texts)
        if thin:
            extraction.warnings.append(
                f"text layer thin ({len(extraction.text) // pages} chars/page); used OCR"
            )

    body = extraction.text
    if extraction.figure_texts:
        body += "\n\n## Figure and image text (OCR)\n" + "\n".join(extraction.figure_texts)

    return SourceDocument(
        provenance=Provenance(
            url=url,
            modality=Modality.PDF,
            sha256=Provenance.hash_content(data),
            byte_size=len(data),
            title=title,
            note=f"{extraction.page_count} pages",
        ),
        text=body,
        tables=extraction.tables,
        ocr_used=extraction.ocr_used,
        parse_warnings=extraction.warnings,
    )


def _cell(value: object) -> str:
    return _normalise(str(value)) if value is not None else ""


_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")


def _normalise(text: str) -> str:
    """Collapse the whitespace noise PDF extraction always produces."""
    text = text.replace("\x00", " ").replace("﻿", "")
    # Rejoin words split across a line break by hyphenation, which otherwise
    # corrupts every term the Data Alchemist tries to match on.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = _WS_RE.sub(" ", text)
    return _NL_RE.sub("\n\n", text).strip()
