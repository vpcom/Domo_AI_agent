from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from assistant.config import get_paths
from assistant.registry import TOOLS
from assistant.schemas import (
    CopyFileArgs,
    CreateJobFilesArgs,
    EvaluateDocumentsArgs,
    MatchCvArgs,
    PlannedToolCall,
    ReadDocumentsArgs,
    RunJobAgentArgs,
    SummarizeDocumentsArgs,
    WriteDocumentArgs,
)
from tools.job.job_folder_resolution import resolve_job_folder_hint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGURED_PATHS = get_paths()
JOBS_ROOT = CONFIGURED_PATHS["jobs_root"]
OUTPUTS_ROOT = CONFIGURED_PATHS["outputs_root"]
PROJECT_DATA_ROOT = CONFIGURED_PATHS["data_root"]
CVS_ROOT = CONFIGURED_PATHS["cvs_root"]
PLACEHOLDER_PATH_FRAGMENTS = ("/path/to", "\\path\\to", "<path>", "example/path")
SUPPORTED_DOCUMENT_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}
SUPPORTED_WRITE_EXTENSIONS = {".txt", ".md", ".markdown", ".json"}
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
    "copy_file": ToolPolicy(
        requires_approval=False,
        max_model_steps=1,
        max_tool_steps=1,
    ),
    "write_document": ToolPolicy(
        requires_approval=False,
        max_model_steps=1,
        max_tool_steps=1,
    ),
    "read_documents": ToolPolicy(
        requires_approval=False,
        max_model_steps=1,
        max_tool_steps=1,
    ),
    "summarize_documents": ToolPolicy(
        requires_approval=False,
        max_model_steps=1,
        max_tool_steps=1,
    ),
    "evaluate_documents": ToolPolicy(
        requires_approval=False,
        max_model_steps=1,
        max_tool_steps=1,
    ),
}


def filter_allowed_arguments(
    tool_name: str,
    raw_arguments: dict[str, Any] | None,
) -> dict[str, Any]:
    if tool_name not in TOOLS:
        raise ValueError("Unknown tool requested.")

    arguments = raw_arguments or {}
    if not isinstance(arguments, dict):
        raise ValueError("Planner arguments must be a JSON object.")

    allowed_keys = set(TOOLS[tool_name].argument_keys)
    return {
        key: value
        for key, value in arguments.items()
        if key in allowed_keys
    }


def missing_required_arguments(
    tool_name: str,
    arguments: dict[str, Any] | None,
) -> list[str]:
    if tool_name not in TOOLS:
        return []

    filtered_arguments = filter_allowed_arguments(tool_name, arguments)
    missing: list[str] = []
    for key in TOOLS[tool_name].required_keys:
        value = filtered_arguments.get(key)
        if value is None:
            missing.append(key)
        elif isinstance(value, str) and not value.strip():
            missing.append(key)
    return missing


def build_tool_args(
    tool_name: str,
    raw_arguments: dict[str, Any] | None,
) -> (
    RunJobAgentArgs
    | CreateJobFilesArgs
    | MatchCvArgs
    | CopyFileArgs
    | WriteDocumentArgs
    | ReadDocumentsArgs
    | SummarizeDocumentsArgs
    | EvaluateDocumentsArgs
):
    filtered_arguments = filter_allowed_arguments(tool_name, raw_arguments)

    try:
        return TOOLS[tool_name].arg_model.model_validate(filtered_arguments)
    except ValidationError as exc:
        first_error = exc.errors()[0]
        location = ".".join(str(part) for part in first_error.get("loc", ()))
        message = first_error.get("msg", "Invalid arguments.")
        if location:
            raise ValueError(
                f"Arguments for `{tool_name}` failed validation at `{location}`: {message}"
            ) from exc
        raise ValueError(
            f"Arguments for `{tool_name}` failed validation: {message}"
        ) from exc


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


def _resolve_project_data_path(path_value: str) -> Path:
    stripped = path_value.strip()
    if not stripped:
        raise ValueError("A path value is required.")

    lowered = stripped.lower()
    if any(fragment in lowered for fragment in PLACEHOLDER_PATH_FRAGMENTS):
        raise ValueError("Placeholder paths are not allowed.")

    candidate = Path(stripped).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not (
        candidate == PROJECT_DATA_ROOT
        or PROJECT_DATA_ROOT in candidate.parents
    ):
        raise ValueError("Document paths must stay within the project's data roots.")

    return candidate


def normalize_allowed_document_input_path(
    path_value: str | None,
    *,
    allow_missing_paths: set[str] | None = None,
) -> str:
    if path_value is None or not str(path_value).strip():
        raise ValueError("A document input path is required.")

    candidate = _resolve_project_data_path(str(path_value))
    normalized_allowed_missing_paths = {
        str(_resolve_project_data_path(item))
        for item in (allow_missing_paths or set())
    }

    if not candidate.exists() and str(candidate) not in normalized_allowed_missing_paths:
        raise ValueError(f"Project data path does not exist: {candidate}")

    if candidate.exists() and candidate.is_file() and candidate.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError(
            f"Unsupported document type: {candidate.suffix}. "
            f"Supported extensions: {sorted(SUPPORTED_DOCUMENT_EXTENSIONS)}"
        )
    if not candidate.exists() and candidate.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError(
            f"Unsupported document type: {candidate.suffix}. "
            f"Supported extensions: {sorted(SUPPORTED_DOCUMENT_EXTENSIONS)}"
        )

    return str(candidate)


def normalize_allowed_existing_project_data_path(
    path_value: str | None,
    *,
    require_file: bool,
) -> Path:
    if path_value is None or not str(path_value).strip():
        raise ValueError("A project data path is required.")

    candidate = _resolve_project_data_path(str(path_value))
    if not candidate.exists():
        raise ValueError(f"Project data path does not exist: {candidate}")

    if require_file and not candidate.is_file():
        raise ValueError(f"Expected a file path: {candidate}")

    return candidate


def normalize_allowed_document_output_path(
    path_value: str | None,
) -> str | None:
    if path_value is None:
        return None

    stripped = str(path_value).strip()
    if not stripped:
        return None

    candidate = _resolve_project_data_path(stripped)
    if candidate.suffix.lower() not in SUPPORTED_WRITE_EXTENSIONS:
        raise ValueError(
            f"Unsupported output document type: {candidate.suffix}. "
            f"Supported extensions: {sorted(SUPPORTED_WRITE_EXTENSIONS)}"
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

    if tool_name == "copy_file":
        return

    if tool_name == "write_document":
        return

    if tool_name == "read_documents":
        return

    if tool_name == "summarize_documents":
        return

    if tool_name == "evaluate_documents":
        return


def plan_tool_call(
    tool_name: str,
    args: (
        RunJobAgentArgs
        | CreateJobFilesArgs
        | MatchCvArgs
        | CopyFileArgs
        | WriteDocumentArgs
        | ReadDocumentsArgs
        | SummarizeDocumentsArgs
        | EvaluateDocumentsArgs
    ),
    user_input: str,
    request_id: str,
    allow_document_inputs: set[str] | None = None,
    skip_semantic_validation: bool = False,
) -> PlannedToolCall:
    if not skip_semantic_validation:
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
    elif tool_name == "copy_file":
        normalized_args = CopyFileArgs(
            source_path=str(
                normalize_allowed_existing_project_data_path(
                    args.source_path,
                    require_file=True,
                )
            ),
            destination_path=str(_resolve_project_data_path(args.destination_path)),
        )
    elif tool_name == "write_document":
        if not args.content.strip():
            raise ValueError("Document content cannot be empty.")
        destination_path = normalize_allowed_document_output_path(args.destination_path)
        if destination_path is None:
            raise ValueError("A destination path is required for writing a document.")
        normalized_args = WriteDocumentArgs(
            destination_path=destination_path,
            content=args.content,
        )
    elif tool_name == "read_documents":
        normalized_args = ReadDocumentsArgs(
            input_path=normalize_allowed_document_input_path(
                args.input_path,
                allow_missing_paths=allow_document_inputs,
            ),
            recursive=args.recursive,
        )
    elif tool_name == "summarize_documents":
        normalized_args = SummarizeDocumentsArgs(
            input_path=normalize_allowed_document_input_path(
                args.input_path,
                allow_missing_paths=allow_document_inputs,
            ),
            instructions=(args.instructions or "").strip() or None,
            output_path=normalize_allowed_document_output_path(args.output_path),
            recursive=args.recursive,
        )
    elif tool_name == "evaluate_documents":
        instructions = (args.instructions or "").strip()
        if not instructions:
            raise ValueError("Evaluation instructions are required.")
        normalized_args = EvaluateDocumentsArgs(
            input_path=normalize_allowed_document_input_path(
                args.input_path,
                allow_missing_paths=allow_document_inputs,
            ),
            instructions=instructions,
            output_path=normalize_allowed_document_output_path(args.output_path),
            recursive=args.recursive,
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
