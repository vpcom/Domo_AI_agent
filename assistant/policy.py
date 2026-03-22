from dataclasses import dataclass
from pathlib import Path

from assistant.schemas import PlannedToolCall, RunJobAgentArgs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOBS_ROOT = (PROJECT_ROOT / "data" / "jobs").resolve()
PROJECT_DATA_ROOT = (PROJECT_ROOT / "data").resolve()
PLACEHOLDER_PATH_FRAGMENTS = ("/path/to", "\\path\\to", "<path>", "example/path")
INSTRUCTION_WORDS = ("how", "explain", "help", "setup", "parameter", "parameters")
EXECUTION_WORDS = ("run", "execute", "process", "generate", "perform", "start")
SEARCH_WORDS = ("search", "find", "discover", "look for")
INSTRUCTION_PATTERNS = (
    "how to",
    "tell me how",
    "explain how",
    "what parameters",
    "which parameters",
    "setup help",
)


@dataclass(frozen=True)
class ToolPolicy:
    requires_approval: bool
    max_model_steps: int
    max_tool_steps: int


TOOL_POLICIES = {
    "run_job_agent": ToolPolicy(
        requires_approval=False,
        max_model_steps=1,
        max_tool_steps=1,
    )
}


def normalize_allowed_job_path(folder_path: str | None) -> str | None:
    if folder_path is None:
        return None

    stripped = folder_path.strip()
    if not stripped:
        return None

    lowered = stripped.lower()
    if any(fragment in lowered for fragment in PLACEHOLDER_PATH_FRAGMENTS):
        raise ValueError("Placeholder paths are not allowed.")

    candidate = Path(stripped).expanduser()
    if not candidate.is_absolute():
        direct_candidate = (PROJECT_ROOT / candidate).resolve()
        jobs_candidate = (JOBS_ROOT / candidate).resolve()
        direct_raw = direct_candidate / "job_description_raw.txt"
        jobs_raw = jobs_candidate / "job_description_raw.txt"
        direct_cleaned = direct_candidate / "cleaned_job_description.txt"
        jobs_cleaned = jobs_candidate / "cleaned_job_description.txt"

        if direct_raw.exists():
            candidate = direct_candidate
        elif jobs_raw.exists():
            candidate = jobs_candidate
        elif direct_cleaned.exists():
            candidate = direct_candidate
        elif jobs_cleaned.exists():
            candidate = jobs_candidate
        else:
            candidate = direct_candidate
    else:
        candidate = candidate.resolve()

    allowed_roots = (JOBS_ROOT, PROJECT_DATA_ROOT)
    if not any(root == candidate or root in candidate.parents for root in allowed_roots):
        raise ValueError(
            "Job folder must stay within the project's data roots."
        )

    return str(candidate)


def validate_semantics(user_input: str, tool_name: str) -> None:
    lowered = user_input.lower()

    if tool_name == "run_job_agent":
        if any(pattern in lowered for pattern in INSTRUCTION_PATTERNS):
            raise ValueError(
                "The request looks instructional, not executable. "
                "Respond with guidance instead of running the job tool."
            )

        if any(word in lowered for word in SEARCH_WORDS):
            return

        if any(word in lowered for word in INSTRUCTION_WORDS) and not any(
            word in lowered for word in EXECUTION_WORDS
        ):
            raise ValueError(
                "The request looks instructional, not executable. "
                "Respond with guidance instead of running the job tool."
            )


def plan_tool_call(
    tool_name: str, args: RunJobAgentArgs, user_input: str, request_id: str
) -> PlannedToolCall:
    validate_semantics(user_input, tool_name)

    normalized_path = normalize_allowed_job_path(args.folder_path)
    normalized_args = RunJobAgentArgs(folder_path=normalized_path)
    policy = TOOL_POLICIES[tool_name]

    return PlannedToolCall(
        tool_name=tool_name,
        parameters=normalized_args,
        request_id=request_id,
        requires_approval=policy.requires_approval and normalized_args.folder_path is not None,
        reason=(
            "This tool can fetch external content and write files under data/, "
            "so it requires approval before execution."
        ),
    )
