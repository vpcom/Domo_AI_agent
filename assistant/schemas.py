from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ToolName = Literal[
    "run_job_agent",
    "create_job_files",
    "match_cv",
    "copy_file",
    "write_document",
    "read_documents",
    "summarize_documents",
    "evaluate_documents",
]
ChatRole = Literal["user", "assistant"]
ContextSource = Literal["user", "inferred", "default", "workflow"]
ContextStatus = Literal["confirmed", "pending", "missing"]
ActivityCategory = Literal["decision", "workflow", "warning", "error"]
TurnIntent = Literal["respond", "clarify", "confirm", "execute"]
ConfirmationState = Literal["idle", "awaiting_confirmation", "confirmed"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunJobAgentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folder_path: str | None = None
    role: str | None = None
    location: str | None = None
    ignore_location: bool | None = None
    remote_only: bool | None = None


class MatchCvArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_folder: str
    cvs_folder: str | None = None


class CreateJobFilesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_folder: str


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


class ContextValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    value: str | bool | int | float | None = None
    source: ContextSource = "workflow"
    status: ContextStatus = "missing"


class ConversationContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: dict[str, ContextValue] = Field(default_factory=dict)
    parameters: dict[str, ContextValue] = Field(default_factory=dict)
    execution: dict[str, ContextValue] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: ChatRole
    content: str
    turn_id: str
    timestamp: datetime = Field(default_factory=utc_now)


class ActivityEventDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: ActivityCategory
    summary: str
    detail: str = ""
    raw_lines: list[str] = Field(default_factory=list)


class ActivityEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    timestamp: datetime = Field(default_factory=utc_now)
    category: ActivityCategory
    summary: str
    detail: str = ""
    run_id: str | None = None
    raw_lines: list[str] = Field(default_factory=list)
    turn_id: str | None = None


class PlannerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_message: str = ""
    turn_intent: TurnIntent
    action: ToolName | None = None
    arguments: dict[str, str | bool | int | float | None] = Field(
        default_factory=dict
    )
    steps: list["PlannerStep"] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    confidence: float | None = None
    reasoning: str = ""
    confirmation_required: bool = False
    activity_events: list[ActivityEventDraft] = Field(default_factory=list)


class PlannerStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ToolName
    arguments: dict[str, str | bool | int | float | None] = Field(
        default_factory=dict
    )


class PlannedToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: ToolName
    parameters: (
        RunJobAgentArgs
        | CreateJobFilesArgs
        | MatchCvArgs
        | CopyFileArgs
        | WriteDocumentArgs
        | ReadDocumentsArgs
        | SummarizeDocumentsArgs
        | EvaluateDocumentsArgs
    )
    request_id: str
    requires_approval: bool = False
    reason: str = ""


class ConversationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    messages: list[ChatMessage] = Field(default_factory=list)
    context: ConversationContext = Field(default_factory=ConversationContext)
    activity_events: list[ActivityEvent] = Field(default_factory=list)
    pending_tool_call: PlannedToolCall | None = None
    pending_tool_calls: list[PlannedToolCall] = Field(default_factory=list)
    confirmation_state: ConfirmationState = "idle"
    current_run_id: str | None = None
    is_executing: bool = False


class TurnResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_message: str = ""
    turn_intent: TurnIntent
    context_patch: list[ContextValue] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    confirmation_required: bool = False
    proposed_tool_call: PlannedToolCall | None = None
    proposed_tool_calls: list[PlannedToolCall] = Field(default_factory=list)
    activity_events: list[ActivityEvent] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_message: str
    context_patch: list[ContextValue] = Field(default_factory=list)
    activity_events: list[ActivityEvent] = Field(default_factory=list)


PlannerDecision.model_rebuild()
