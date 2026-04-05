from dataclasses import dataclass
from pathlib import Path

from assistant.config import get_paths
from assistant.schemas import (
    CreateJobFilesArgs,
    MatchCvArgs,
    PlannedToolCall,
    RunJobAgentArgs,
)
from tools.job.job_folder_resolution import resolve_job_folder_hint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGURED_PATHS = get_paths()
JOBS_ROOT = CONFIGURED_PATHS["jobs_root"]
OUTPUTS_ROOT = CONFIGURED_PATHS["outputs_root"]
PROJECT_DATA_ROOT = CONFIGURED_PATHS["data_root"]
CVS_ROOT = CONFIGURED_PATHS["cvs_root"]
PLACEHOLDER_PATH_FRAGMENTS = ("/path/to", "\\path\\to", "<path>", "example/path")
INSTRUCTION_WORDS = ("how", "explain", "help", "setup", "parameter", "parameters")
EXECUTION_WORDS = ("run", "execute", "process", "generate", "perform", "start")
SEARCH_WORDS = ("search", "find", "discover", "look for")
CV_MATCH_WORDS = ("match cv", "best cv", "which cv fits", "fit this job")
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
    ),
    "match_cv": ToolPolicy(
        requires_approval=False,
        max_model_steps=1,
        max_tool_steps=1,
    ),
    "create_job_files": ToolPolicy(
        requires_approval=False,
        max_model_steps=1,
        max_tool_steps=1,
    ),
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

    candidate = resolve_job_folder_hint(stripped, PROJECT_ROOT, JOBS_ROOT)

    allowed_roots = (JOBS_ROOT, OUTPUTS_ROOT, PROJECT_DATA_ROOT)
    if not any(root == candidate or root in candidate.parents for root in allowed_roots):
        raise ValueError(
            "Job folder must stay within the project's data roots."
        )

    return str(candidate)


def normalize_allowed_cvs_path(cvs_folder: str | None) -> str:
    if cvs_folder is None:
        candidate = CVS_ROOT
        return str(candidate)

    stripped = cvs_folder.strip()
    if not stripped:
        return str(CVS_ROOT)

    lowered = stripped.lower()
    if any(fragment in lowered for fragment in PLACEHOLDER_PATH_FRAGMENTS):
        raise ValueError("Placeholder paths are not allowed.")

    candidate = Path(stripped).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if candidate != CVS_ROOT:
        raise ValueError(f"CV folder must be {CVS_ROOT}.")

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
        return

    if tool_name == "match_cv":
        if any(pattern in lowered for pattern in INSTRUCTION_PATTERNS):
            raise ValueError(
                "The request looks instructional, not executable. "
                "Respond with guidance instead of running the CV matching tool."
            )

        if any(word in lowered for word in CV_MATCH_WORDS):
            return

        if any(word in lowered for word in INSTRUCTION_WORDS) and not any(
            word in lowered for word in EXECUTION_WORDS
        ):
            raise ValueError(
                "The request looks instructional, not executable. "
                "Respond with guidance instead of running the CV matching tool."
            )
        return

    if tool_name == "create_job_files":
        if any(pattern in lowered for pattern in INSTRUCTION_PATTERNS):
            raise ValueError(
                "The request looks instructional, not executable. "
                "Respond with guidance instead of running the local job-file workflow."
            )

        if any(word in lowered for word in INSTRUCTION_WORDS) and not any(
            word in lowered for word in EXECUTION_WORDS
        ):
            raise ValueError(
                "The request looks instructional, not executable. "
                "Respond with guidance instead of running the local job-file workflow."
            )
        return


def plan_tool_call(
    tool_name: str,
    args: RunJobAgentArgs | CreateJobFilesArgs | MatchCvArgs,
    user_input: str,
    request_id: str,
) -> PlannedToolCall:
    validate_semantics(user_input, tool_name)

    if tool_name == "run_job_agent":
        normalized_path = normalize_allowed_job_path(args.folder_path)
        if normalized_path is not None and any(
            value is not None
            for value in (
                args.role,
                args.location,
                args.ignore_location,
                args.remote_only,
            )
        ):
            raise ValueError(
                "Search overrides are only supported for online job discovery, not local folder processing."
            )

        normalized_args = RunJobAgentArgs(
            folder_path=normalized_path,
            role=(args.role or "").strip() or None,
            location=(args.location or "").strip() or None,
            ignore_location=args.ignore_location,
            remote_only=args.remote_only,
        )
    elif tool_name == "create_job_files":
        normalized_job_path = normalize_allowed_job_path(args.job_folder)
        if normalized_job_path is None:
            raise ValueError("A job folder is required to create local job files.")
        normalized_args = CreateJobFilesArgs(job_folder=normalized_job_path)
    elif tool_name == "match_cv":
        normalized_job_path = normalize_allowed_job_path(args.job_folder)
        if normalized_job_path is None:
            raise ValueError("A job folder is required for CV matching.")
        normalized_args = MatchCvArgs(
            job_folder=normalized_job_path,
            cvs_folder=normalize_allowed_cvs_path(args.cvs_folder),
        )
    else:
        raise ValueError("Unknown tool requested.")

    policy = TOOL_POLICIES[tool_name]

    return PlannedToolCall(
        tool_name=tool_name,
        parameters=normalized_args,
        request_id=request_id,
        requires_approval=policy.requires_approval,
        reason=(
            "This tool can fetch external content and write files under data/, "
            "so it requires approval before execution."
        ),
    )
