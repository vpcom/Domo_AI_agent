from dataclasses import dataclass
from typing import Any, Callable

from assistant.schemas import MatchCvArgs, RunJobAgentArgs
from workflows.match_cv_workflow import run_match_cv_workflow
from workflows.run_job_agent_workflow import run_job_agent_workflow


@dataclass(frozen=True)
class ToolSpec:
    name: str
    arg_model: type
    executor: Callable[..., Any]


WORKFLOWS = {
    "run_job_agent": run_job_agent_workflow,
    "match_cv": run_match_cv_workflow,
}


TOOLS = {
    "run_job_agent": ToolSpec(
        name="run_job_agent",
        arg_model=RunJobAgentArgs,
        executor=WORKFLOWS["run_job_agent"],
    ),
    "match_cv": ToolSpec(
        name="match_cv",
        arg_model=MatchCvArgs,
        executor=WORKFLOWS["match_cv"],
    ),
}
