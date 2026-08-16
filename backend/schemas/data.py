"""Data acquisition contracts.

The Data Alchemist is held to a machine-checked floor: at least
`min_distinct_modalities` *different* `Modality` values, or it fails upward and
the supervisor re-routes to a different question. Failure changing the plan is
the clearest evidence of real agency in the system, so it is a first-class
outcome here rather than an exception.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .common import Conflict, Modality, Provenance


class ExtractedTable(BaseModel):
    """A table recovered from a PDF or HTML page.

    `extraction_method` is retained because table extraction is the single
    least reliable step in the pipeline, and the paper's methods section is
    expected to be honest about how each number was obtained.
    """

    source_url: str
    page: int | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    extraction_method: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), len(self.columns))


class SourceDocument(BaseModel):
    """One acquired artefact, whatever its shape."""

    provenance: Provenance
    text: str = ""
    tables: list[ExtractedTable] = Field(default_factory=list)
    records: list[dict[str, Any]] = Field(
        default_factory=list, description="Parsed rows for tabular/API sources"
    )
    ocr_used: bool = False
    parse_warnings: list[str] = Field(default_factory=list)

    @property
    def modality(self) -> Modality:
        return self.provenance.modality

    @property
    def url(self) -> str:
        return self.provenance.url


class ColumnMapping(BaseModel):
    """One LLM-proposed schema alignment, before validation.

    The LLM suggests that column `source_column` in `source_url` means the same
    thing as canonical field `canonical_field`. Python then checks that the
    column exists and that its values actually parse as the declared type --
    a mapping that fails validation is dropped, not trusted.
    """

    source_url: str
    source_column: str
    canonical_field: str
    dtype: str = "string"
    unit: str = ""
    transform: str = ""
    validated: bool = False
    validation_error: str = ""


class CleaningReport(BaseModel):
    """What actually happened during cleaning, for the paper's methods section."""

    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    rows_dropped_missing: int = 0
    columns_mapped: int = 0
    mappings_rejected: int = 0
    unit_conversions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Dataset(BaseModel):
    """The analysis-ready table handed to the Experiment Designer.

    Held as columnar lists rather than a DataFrame so the whole graph state
    stays JSON-serialisable, which is what makes runs replayable and the
    reproducibility manifest complete.
    """

    name: str
    columns: list[str] = Field(default_factory=list)
    dtypes: dict[str, str] = Field(default_factory=dict)
    data: dict[str, list[Any]] = Field(default_factory=dict)
    row_provenance: list[str] = Field(
        default_factory=list, description="Source URL per row, parallel to the data lists"
    )
    derived_from: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Columns computed from other columns, e.g. citations_per_year <- "
            "[citations, year]. Used to reject tautological experiments."
        ),
    )
    cleaning: CleaningReport = Field(default_factory=CleaningReport)

    def are_dependent(self, a: str, b: str) -> bool:
        """Whether two columns are related by construction rather than by data.

        Correlating a derived column with its own source produces a perfect,
        highly significant, and entirely meaningless result -- observed live as
        `citations_per_year` vs `citations` at rho = 0.976, p = 2e-15. The
        statistics are impeccable; the finding is arithmetic.
        """
        if a == b:
            return True
        return b in self.derived_from.get(a, []) or a in self.derived_from.get(b, [])

    @property
    def n_rows(self) -> int:
        return len(next(iter(self.data.values()), []))

    def is_analysable(self, min_rows: int = 8) -> bool:
        """Below a handful of rows no statistical test is worth running, and
        the Critic will reject the result anyway -- better to fail early."""
        return self.n_rows >= min_rows and len(self.columns) >= 2


class DataBundle(BaseModel):
    """Everything the Alchemist gathered for one research question."""

    question_id: str
    documents: list[SourceDocument] = Field(default_factory=list)
    dataset: Dataset | None = None
    conflicts: list[Conflict] = Field(default_factory=list)
    mappings: list[ColumnMapping] = Field(default_factory=list)
    acquisition_failures: list[str] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    @property
    def modalities(self) -> set[Modality]:
        return {d.modality for d in self.documents}

    def meets_floor(self, min_modalities: int, min_sources: int) -> bool:
        return len(self.modalities) >= min_modalities and len(self.documents) >= min_sources

    def shortfall(self, min_modalities: int, min_sources: int) -> str:
        """Human-readable reason the floor was missed, for the event stream."""
        bits = []
        if len(self.modalities) < min_modalities:
            have = ", ".join(sorted(m.value for m in self.modalities)) or "none"
            bits.append(f"{len(self.modalities)}/{min_modalities} modalities (have: {have})")
        if len(self.documents) < min_sources:
            bits.append(f"{len(self.documents)}/{min_sources} sources")
        return "; ".join(bits)

    @property
    def open_conflicts(self) -> list[Conflict]:
        return [c for c in self.conflicts if not c.resolved]
