"""Typed contracts between agents.

Every agent in this system takes a validated Pydantic model in and returns a
validated Pydantic model out. That is the structural difference between this
and a prompt chain: an agent cannot pass malformed work downstream, because
validation fails first and triggers a repair round instead.
"""

from .common import (
    AgentName,
    Claim,
    ConfidenceComponents,
    Conflict,
    Modality,
    Provenance,
    TokenUsage,
    ToolCall,
)
from .critique import (
    Critique,
    Objection,
    RerouteTarget,
    Severity,
    StatFlags,
    Verdict,
)
from .data import (
    CleaningReport,
    ColumnMapping,
    DataBundle,
    Dataset,
    ExtractedTable,
    SourceDocument,
)
from .domain import (
    DomainCandidate,
    DomainProposal,
    DomainProposalBatch,
    DomainSelection,
    EmergenceSignals,
    PanelBallot,
    PanelVote,
)
from .events import (
    ArtifactKind,
    EventType,
    Level,
    RunEvent,
    RunStatus,
    RunSummary,
)
from .experiment import (
    ExperimentKind,
    ExperimentResult,
    ExperimentSpec,
    FigureSpec,
    Hypothesis,
    StatResult,
)
from .paper import Paper, PaperSection, RunManifest
from .question import (
    PeerRating,
    QuestionProposal,
    QuestionProposalBatch,
    QuestionSet,
    ResearchQuestion,
    SearchabilityProbe,
)

__all__ = [
    "AgentName",
    "ArtifactKind",
    "CleaningReport",
    "Claim",
    "ColumnMapping",
    "ConfidenceComponents",
    "Conflict",
    "Critique",
    "DataBundle",
    "Dataset",
    "DomainCandidate",
    "DomainProposal",
    "DomainProposalBatch",
    "DomainSelection",
    "EmergenceSignals",
    "EventType",
    "ExperimentKind",
    "ExperimentResult",
    "ExperimentSpec",
    "ExtractedTable",
    "FigureSpec",
    "Hypothesis",
    "Level",
    "Modality",
    "Objection",
    "PanelBallot",
    "PanelVote",
    "Paper",
    "PaperSection",
    "PeerRating",
    "Provenance",
    "QuestionProposal",
    "QuestionProposalBatch",
    "QuestionSet",
    "ResearchQuestion",
    "RerouteTarget",
    "RunEvent",
    "RunManifest",
    "RunStatus",
    "RunSummary",
    "SearchabilityProbe",
    "Severity",
    "SourceDocument",
    "StatFlags",
    "StatResult",
    "TokenUsage",
    "ToolCall",
    "Verdict",
]
