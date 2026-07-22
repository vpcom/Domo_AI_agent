"""Validation and path-normalization helpers for planner and runtime code."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Any

from pydantic import ValidationError

from assistant.config import get_paths
from assistant.registry import LLM_TASKS, TOOLS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGURED_PATHS = get_paths()
JOBS_ROOT = CONFIGURED_PATHS["jobs_root"]
OUTPUTS_ROOT = CONFIGURED_PATHS["outputs_root"]
PLACEHOLDER_PATH_FRAGMENTS = (
    "/path/to", "\\path\\to", "<path>", "example/path")
OUTPUT_TIMESTAMP_PATTERN = re.compile(r"^\d{8}_\d{6}$")
STEP_REFERENCE_PATTERN = re.compile(r"^@step:(\d+)\.output\.(.+)$")
GOAL_REFERENCE_PATTERN = re.compile(r"^@goal:(user_input|normalized_goal)$")
MEMORY_REFERENCE_PATTERN = re.compile(r"^@memory:([A-Za-z0-9_.-]+)$")
SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
    ".csv",
}
SUPPORTED_WRITE_EXTENSIONS = {".txt", ".md", ".markdown", ".json"}


def build_timestamped_output_root() -> Path:
    """Build timestamped output root."""

    candidate_time = datetime.now().replace(microsecond=0)
    candidate = OUTPUTS_ROOT / candidate_time.strftime("%Y%m%d_%H%M%S")
    while candidate.exists():
        candidate_time += timedelta(seconds=1)
        candidate = OUTPUTS_ROOT / candidate_time.strftime("%Y%m%d_%H%M%S")
    return candidate


def filter_allowed_arguments(
    tool_name: str,
    raw_arguments: dict[str, Any] | None,
    *,
    step_type: str = "tool",
) -> dict[str, Any]:
    """Return filter allowed arguments."""

    spec = _get_spec(tool_name, step_type=step_type)
    arguments = raw_arguments or {}
    if not isinstance(arguments, dict):
        raise ValueError("Planner arguments must be a JSON object.")
    allowed_keys = set(spec.input_model.model_fields)
    return {key: value for key, value in arguments.items() if key in allowed_keys}


def missing_required_arguments(
    tool_name: str,
    arguments: dict[str, Any] | None,
    *,
    step_type: str = "tool",
) -> list[str]:
    """Return missing required arguments."""

    spec = _get_spec(tool_name, step_type=step_type)
    filtered_arguments = filter_allowed_arguments(
        tool_name,
        arguments,
        step_type=step_type,
    )
    missing: list[str] = []
    for key, field in spec.input_model.model_fields.items():
        if not field.is_required():
            continue
        value = filtered_arguments.get(key)
        if value is None:
            missing.append(key)
        elif isinstance(value, str) and not value.strip():
            missing.append(key)
    return missing


def build_tool_args(
    tool_name: str,
    raw_arguments: dict[str, Any] | None,
    *,
    step_type: str = "tool",
):
    """Build tool args."""

    spec = _get_spec(tool_name, step_type=step_type)
    filtered_arguments = filter_allowed_arguments(
        tool_name,
        raw_arguments,
        step_type=step_type,
    )
    try:
        return spec.input_model.model_validate(filtered_arguments)
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


def is_reference_string(value: Any) -> bool:
    """Return whether reference string."""

    if not isinstance(value, str):
        return False
    return bool(
        STEP_REFERENCE_PATTERN.fullmatch(value)
        or GOAL_REFERENCE_PATTERN.fullmatch(value)
        or MEMORY_REFERENCE_PATTERN.fullmatch(value)
    )


def validate_reference_string(value: str, *, current_step_id: int) -> None:
    """Validate reference string."""

    if "[" in value or "]" in value:
        raise ValueError(
            "Step references do not support list indexing or wildcards. "
            "Reference the whole list, for example `@step:0.output.result.results`."
        )

    if GOAL_REFERENCE_PATTERN.fullmatch(value):
        return
    if MEMORY_REFERENCE_PATTERN.fullmatch(value):
        return

    step_match = STEP_REFERENCE_PATTERN.fullmatch(value)
    if step_match is None:
        raise ValueError(
            "Only `@goal:<field>`, `@step:<id>.output.<path>`, and `@memory:<key>` references are allowed."
        )

    referenced_step_id = int(step_match.group(1))
    if referenced_step_id >= current_step_id:
        raise ValueError("Step references must target an earlier step.")


def validate_and_normalize_tool_inputs(
    tool_name: str,
    raw_inputs: dict[str, Any],
    *,
    output_root: Path | None = None,
    allow_references: bool,
) -> dict[str, Any]:
    """Validate and normalize tool inputs."""

    if allow_references and _contains_runtime_reference_value(raw_inputs):
        arguments = {
            key: value
            for key, value in filter_allowed_arguments(tool_name, raw_inputs).items()
            if value is not None
        }
    else:
        arguments = build_tool_args(
            tool_name, raw_inputs).model_dump(exclude_none=True)

    if tool_name == "run_job_agent":
        folder_path = arguments.get("folder_path")
        if _is_runtime_reference(folder_path, allow_references):
            return arguments
        arguments["folder_path"] = normalize_allowed_job_path(folder_path)
        return arguments

    if tool_name == "create_job_files":
        job_folder = arguments.get("job_folder")
        if _is_runtime_reference(job_folder, allow_references):
            return arguments
        arguments["job_folder"] = require_allowed_job_path(job_folder)
        return arguments

    if tool_name == "search_web":
        output_path = arguments.get("output_path")
        if output_path is not None and not _is_runtime_reference(output_path, allow_references):
            arguments["output_path"] = normalize_allowed_output_path(
                output_path,
                output_root=output_root,
            )
        return arguments

    if tool_name == "inspect_path":
        path_value = arguments.get("path")
        if not _is_runtime_reference(path_value, allow_references):
            arguments["path"] = str(
                _resolve_project_input_path(str(path_value)))
        return arguments

    if tool_name == "list_directory":
        path_value = arguments.get("path")
        if not _is_runtime_reference(path_value, allow_references):
            arguments["path"] = normalize_allowed_directory_path(path_value)
        return arguments

    if tool_name == "read_text_file":
        path_value = arguments.get("path")
        if not _is_runtime_reference(path_value, allow_references):
            arguments["path"] = normalize_allowed_text_file_path(path_value)
        return arguments

    if tool_name == "read_json_file":
        path_value = arguments.get("path")
        if not _is_runtime_reference(path_value, allow_references):
            arguments["path"] = normalize_allowed_file_with_extensions(
                path_value,
                allowed_extensions={".json"},
            )
        return arguments

    if tool_name == "read_pdf_text":
        path_value = arguments.get("path")
        if not _is_runtime_reference(path_value, allow_references):
            arguments["path"] = normalize_allowed_file_with_extensions(
                path_value,
                allowed_extensions={".pdf"},
            )
        return arguments

    if tool_name == "resolve_job_folder_hint":
        return arguments

    if tool_name in {"resolve_local_job_inputs", "read_job_metadata"}:
        job_folder = arguments.get("job_folder")
        if not _is_runtime_reference(job_folder, allow_references):
            arguments["job_folder"] = require_allowed_job_path(job_folder)
        return arguments

    if tool_name in {
        "discover_jobs",
        "clean_job_description",
        "generate_application_materials",
        "build_application_notes_from_job_description",
    }:
        return arguments

    if tool_name == "copy_file":
        source_path = arguments.get("source_path")
        if not _is_runtime_reference(source_path, allow_references):
            arguments["source_path"] = normalize_allowed_document_input_path(
                source_path)
        destination_path = arguments.get("destination_path")
        if not _is_runtime_reference(destination_path, allow_references):
            arguments["destination_path"] = normalize_allowed_output_path(
                destination_path,
                output_root=output_root,
            )
        return arguments

    if tool_name == "write_document":
        destination_path = arguments.get("destination_path")
        if not _is_runtime_reference(destination_path, allow_references):
            arguments["destination_path"] = normalize_allowed_output_path(
                destination_path,
                output_root=output_root,
            )
        return arguments

    if tool_name == "write_json_file":
        destination_path = arguments.get("destination_path")
        if not _is_runtime_reference(destination_path, allow_references):
            arguments["destination_path"] = normalize_allowed_output_path(
                destination_path,
                output_root=output_root,
            )
        return arguments

    if tool_name == "write_search_results":
        destination_path = arguments.get("destination_path")
        if not _is_runtime_reference(destination_path, allow_references):
            arguments["destination_path"] = normalize_allowed_output_path(
                destination_path,
                output_root=output_root,
            )
        return arguments

    if tool_name == "write_generated_documents":
        output_dir = arguments.get("output_dir")
        if not _is_runtime_reference(output_dir, allow_references):
            arguments["output_dir"] = normalize_allowed_output_dir(
                output_dir,
                output_root=output_root,
            )
        return arguments

    if tool_name == "read_documents":
        input_path = arguments.get("input_path")
        if not _is_runtime_reference(input_path, allow_references):
            arguments["input_path"] = normalize_allowed_document_input_path(
                input_path)
        return arguments

    raise ValueError(f"Unknown tool requested: {tool_name}")


def normalize_allowed_job_path(folder_path: str | None) -> str | None:
    """Return normalize allowed job path."""

    if folder_path is None:
        return None

    stripped = folder_path.strip()
    if not stripped:
        return None

    candidate = _resolve_project_input_path(stripped)
    allowed_roots = (JOBS_ROOT, OUTPUTS_ROOT)
    if not any(root == candidate or root in candidate.parents for root in allowed_roots):
        raise ValueError(
            "Job folders must stay within the configured jobs inputs root or outputs root."
        )
    return str(candidate)


def require_allowed_job_path(folder_path: str | None) -> str:
    """Return require allowed job path."""

    normalized = normalize_allowed_job_path(folder_path)
    if normalized is None:
        raise ValueError("A job folder is required.")
    if not Path(normalized).exists():
        raise ValueError(f"Job folder does not exist: {normalized}")
    return normalized


def normalize_allowed_document_input_path(path_value: str | None) -> str:
    """Return normalize allowed document input path."""

    if path_value is None or not str(path_value).strip():
        raise ValueError("A document input path is required.")

    candidate = _resolve_project_input_path(str(path_value))
    if not candidate.exists():
        raise ValueError(f"Project path does not exist: {candidate}")
    if candidate.is_file() and candidate.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError(
            f"Unsupported document type: {candidate.suffix}. "
            f"Supported extensions: {sorted(SUPPORTED_DOCUMENT_EXTENSIONS)}"
        )
    return str(candidate)


def normalize_allowed_directory_path(path_value: str | None) -> str:
    """Return normalize allowed directory path."""

    if path_value is None or not str(path_value).strip():
        raise ValueError("A folder path is required.")
    candidate = _resolve_project_input_path(str(path_value))
    if not candidate.exists():
        raise ValueError(f"Project path does not exist: {candidate}")
    if not candidate.is_dir():
        raise ValueError(f"Expected a folder path: {candidate}")
    return str(candidate)


def normalize_allowed_text_file_path(path_value: str | None) -> str:
    """Return normalize allowed text file path."""

    return normalize_allowed_file_with_extensions(
        path_value,
        allowed_extensions={
            ".txt",
            ".md",
            ".markdown",
            ".py",
            ".yaml",
            ".yml",
            ".toml",
            ".sh",
            ".csv",
        },
    )


def normalize_allowed_file_with_extensions(
    path_value: str | None,
    *,
    allowed_extensions: set[str],
) -> str:
    """Return normalize allowed file with extensions."""

    if path_value is None or not str(path_value).strip():
        raise ValueError("A file path is required.")
    candidate = _resolve_project_input_path(str(path_value))
    if not candidate.exists():
        raise ValueError(f"Project path does not exist: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"Expected a file path: {candidate}")
    if candidate.suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"Unsupported file type: {candidate.suffix}. "
            f"Supported extensions: {sorted(allowed_extensions)}"
        )
    return str(candidate)


def normalize_allowed_output_path(
    path_value: str | None,
    *,
    output_root: Path | None,
) -> str:
    """Return normalize allowed output path."""

    if path_value is None or not str(path_value).strip():
        raise ValueError("An output path is required.")

    candidate = _resolve_project_input_path(str(path_value))
    if candidate.suffix.lower() not in SUPPORTED_WRITE_EXTENSIONS:
        raise ValueError(
            f"Unsupported write type: {candidate.suffix}. "
            f"Supported extensions: {sorted(SUPPORTED_WRITE_EXTENSIONS)}"
        )
    if not (candidate == OUTPUTS_ROOT or OUTPUTS_ROOT in candidate.parents):
        raise ValueError(f"Writes must stay within {OUTPUTS_ROOT}.")
    return str(
        _normalize_output_path_under_timestamp_root(
            candidate, output_root=output_root)
    )


def normalize_allowed_output_dir(
    path_value: str | None,
    *,
    output_root: Path | None,
) -> str:
    """Return normalize allowed output directory."""

    if path_value is None or not str(path_value).strip():
        raise ValueError("An output directory is required.")

    candidate = _resolve_project_input_path(str(path_value))
    if candidate.suffix:
        raise ValueError("Output directory paths must not include a file extension.")
    if not (candidate == OUTPUTS_ROOT or OUTPUTS_ROOT in candidate.parents):
        raise ValueError(f"Writes must stay within {OUTPUTS_ROOT}.")
    return str(
        _normalize_output_path_under_timestamp_root(
            candidate, output_root=output_root)
    )


def _resolve_project_input_path(path_value: str) -> Path:
    """Resolve project input path."""

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

    if not (candidate == PROJECT_ROOT or PROJECT_ROOT in candidate.parents):
        raise ValueError("Paths must stay within the project root.")

    return candidate


def _normalize_output_path_under_timestamp_root(
    candidate: Path,
    *,
    output_root: Path | None,
) -> Path:
    """Return normalize output path under timestamp root."""

    relative_path = candidate.relative_to(OUTPUTS_ROOT)
    parts = relative_path.parts
    if not parts:
        raise ValueError("A file path under the outputs root is required.")

    if OUTPUT_TIMESTAMP_PATTERN.fullmatch(parts[0]):
        return candidate

    normalized_root = output_root or build_timestamped_output_root()
    return normalized_root / relative_path


def _get_spec(tool_name: str, *, step_type: str):
    """Return spec."""

    if step_type == "tool":
        if tool_name not in TOOLS:
            raise ValueError("Unknown tool requested.")
        return TOOLS[tool_name]
    if step_type == "llm":
        if tool_name not in LLM_TASKS:
            raise ValueError("Unknown llm task requested.")
        return LLM_TASKS[tool_name]
    raise ValueError(f"Unsupported step type: {step_type}")


def _is_runtime_reference(value: Any, allow_references: bool) -> bool:
    """Return whether runtime reference."""

    return allow_references and is_reference_string(value)


def _contains_runtime_reference_value(value: Any) -> bool:
    """Return whether a nested input payload contains a runtime reference."""

    if isinstance(value, dict):
        return any(_contains_runtime_reference_value(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_runtime_reference_value(item) for item in value)
    return is_reference_string(value)
