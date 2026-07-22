"""Core state and capability schemas for the deterministic agent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


AgentStatus = Literal["planning", "executing", "waiting", "done", "error"]
StepType = Literal["tool", "llm"]
StepStatus = Literal["pending", "running", "done", "failed"]
ChatRole = Literal["user", "assistant"]
UiEventCategory = Literal["system", "planner", "execution", "error", "state"]


def utc_now() -> datetime:
    """Return utc now."""

    return datetime.now(timezone.utc)


class RunJobAgentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folder_path: str | None = None
    role: str | None = None
    location: str | None = None
    ignore_location: bool | None = None
    remote_only: bool | None = None


class CreateJobFilesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_folder: str


class MatchCvArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_folder: str
    cvs_folder: str | None = None


class SearchWebArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    max_results: int | None = None
    output_path: str | None = None


class InspectPathArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class ListDirectoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class ReadTextFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class ReadJsonFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class ReadPdfTextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class ResolveJobFolderHintArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folder_hint: str


class ResolveLocalJobInputsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_folder: str


class ReadJobMetadataArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_folder: str


class DiscoverJobsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    location: str
    ignore_location: bool | None = None
    remote_only: bool | None = None
    sources: list[str] = Field(default_factory=list)
    max_results_per_source: int | None = None
    max_jobs: int | None = None
    max_company_attempts_per_source: int | None = None
    companies: dict[str, list[str]] | None = None


class CopyFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str
    destination_path: str


class WriteDocumentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    destination_path: str
    content: str


class ReadDocumentsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_path: str
    recursive: bool | None = None


class SummarizeDocumentsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_path: str
    instructions: str | None = None
    output_path: str | None = None
    recursive: bool | None = None


class EvaluateDocumentsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_path: str
    instructions: str
    output_path: str | None = None
    recursive: bool | None = None


class CleanJobDescriptionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_job_text: str


class GenerateApplicationMaterialsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cleaned_job_text: str


class BuildApplicationNotesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cleaned_job_text: str


class WriteJsonFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    destination_path: str
    payload: Any


class WriteSearchResultsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    destination_path: str
    query: str
    results: list[dict[str, Any]]


class GeneratedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    content: str


class WriteGeneratedDocumentsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str
    documents: list[GeneratedDocument]


class AnswerQuestionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str


class SummarizeTextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[dict[str, str]]
    instructions: str | None = None


class EvaluateTextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[dict[str, str]]
    instructions: str


class GenerateDocumentSetArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_documents: list[dict[str, str]]
    instructions: str


class RankCvArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_documents: list[dict[str, str]]
    cv_documents: list[dict[str, str]]
    instructions: str | None = None


class ArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str
    path: str
    step_id: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    working_memory: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)


class GoalState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_input: str = ""
    normalized_goal: str = ""


class PlanStepDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: int
    description: str
    type: StepType
    tool_name: str
    inputs: dict[str, Any] = Field(default_factory=dict)


class PlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_goal: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    plan: list[PlanStepDraft] = Field(default_factory=list)


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: int
    description: str
    type: StepType
    tool_name: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    status: StepStatus = "pending"
    output: dict[str, Any] | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    retry_count: int = 0


class AgentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: AgentStatus = "planning"
    goal: GoalState = Field(default_factory=GoalState)
    plan: list[PlanStep] = Field(default_factory=list)
    current_step: int = 0
    memory: MemoryState = Field(default_factory=MemoryState)
    last_error: str | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: ChatRole
    content: str
    turn_id: str
    timestamp: datetime = Field(default_factory=utc_now)


class UiEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    category: UiEventCategory
    message: str
    detail: str = ""
    expanded_text: str = ""
    timestamp: datetime = Field(default_factory=utc_now)
    step_id: int | None = None
