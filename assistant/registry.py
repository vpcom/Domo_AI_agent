from dataclasses import dataclass
from typing import Any, Callable

from assistant.schemas import RunJobAgentArgs
from tools.job.run_job_agent import run_job_agent


@dataclass(frozen=True)
class ToolSpec:
    name: str
    arg_model: type
    executor: Callable[..., Any]


TOOLS = {
    "run_job_agent": ToolSpec(
        name="run_job_agent",
        arg_model=RunJobAgentArgs,
        executor=run_job_agent,
    ),
}
