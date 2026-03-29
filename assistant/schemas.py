from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ToolName = Literal["run_job_agent", "match_cv"]


class RunJobAgentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folder_path: str | None = None


class MatchCvArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_folder: str
    cvs_folder: str | None = None


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["tool", "respond"]
    tool_name: ToolName | None = None
    parameters: dict = Field(default_factory=dict)
    response: str = ""


class PlannedToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: ToolName
    parameters: RunJobAgentArgs | MatchCvArgs
    request_id: str
    requires_approval: bool = False
    reason: str = ""


class AgentOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["respond", "tool", "approval_required", "error"]
    message: str = ""
    tool_call: PlannedToolCall | None = None
