"""Static capability catalog exposed to the planner and direct-answer prompts."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


CapabilityGroup = Literal[
    "llm_tasks",
    "read_tools",
    "transform_tools",
    "write_tools",
    "commands",
]
CapabilityKind = Literal["llm_task", "tool", "command"]
CapabilityAccess = Literal["planner_visible",
                           "executor_only", "legacy_disabled", "disallowed"]
CapabilityApproval = Literal["auto", "conditional", "manual", "never"]

GROUP_LABELS: dict[CapabilityGroup, str] = {
    "llm_tasks": "LLM Tasks",
    "read_tools": "Read Tools",
    "transform_tools": "Transform Tools",
    "write_tools": "Write Tools",
    "commands": "Commands",
}
GROUP_ORDER: tuple[CapabilityGroup, ...] = (
    "llm_tasks",
    "read_tools",
    "transform_tools",
    "write_tools",
    "commands",
)
FORBIDDEN_ACTION_LINES: tuple[str, ...] = (
    "Do not delete, remove, erase, move, rename, or modify existing documents or files.",
    "Do not use account information, sign into accounts, or act through authenticated user sessions.",
    "Do not claim arbitrary shell, terminal, or account-based capabilities.",
)
_CAPABILITY_QUESTION_PATTERN = re.compile(
    r"\b(what can you do|what are your capabilities|what tools|what commands|what actions|available capabilities)\b",
    re.IGNORECASE,
)
_FORBIDDEN_REQUEST_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(delete|remove|erase|trash)\b", re.IGNORECASE),
        "I do not delete or remove documents or files.",
    ),
    (
        re.compile(r"\b(move|rename|relocate)\b", re.IGNORECASE),
        "I do not move or rename documents or files.",
    ),
    (
        re.compile(
            r"\b(edit|modify|change|update|overwrite|append)\b.*\b(file|document|pdf|json|txt|folder)\b",
            re.IGNORECASE,
        ),
        "I do not modify existing documents or files.",
    ),
    (
        re.compile(
            r"\b(sign in|signin|log in|login|use my account|using my account|with my account|password|credentials|token|cookie)\b",
            re.IGNORECASE,
        ),
        "I do not use account information or authenticated account access.",
    ),
)


@dataclass(frozen=True)
class CapabilityDefinition:
    name: str
    group: CapabilityGroup
    kind: CapabilityKind
    access: CapabilityAccess
    approval: CapabilityApproval
    description: str
    risks: tuple[str, ...]
    account_access: bool = False


CAPABILITY_DEFINITIONS: dict[str, CapabilityDefinition] = {
    "answer_question": CapabilityDefinition(
        name="answer_question",
        group="llm_tasks",
        kind="llm_task",
        access="planner_visible",
        approval="auto",
        description="Answer a direct user question based on the stated capabilities and current context.",
        risks=("llm",),
    ),
    "summarize_text": CapabilityDefinition(
        name="summarize_text",
        group="llm_tasks",
        kind="llm_task",
        access="planner_visible",
        approval="auto",
        description="Summarize documents that were already loaded into the plan.",
        risks=("llm",),
    ),
    "evaluate_text": CapabilityDefinition(
        name="evaluate_text",
        group="llm_tasks",
        kind="llm_task",
        access="planner_visible",
        approval="auto",
        description="Evaluate or rank loaded documents against explicit instructions.",
        risks=("llm",),
    ),
    "rank_cvs": CapabilityDefinition(
        name="rank_cvs",
        group="llm_tasks",
        kind="llm_task",
        access="planner_visible",
        approval="auto",
        description="Rank CV documents against job documents.",
        risks=("llm",),
    ),
    "inspect_path": CapabilityDefinition(
        name="inspect_path",
        group="read_tools",
        kind="tool",
        access="planner_visible",
        approval="auto",
        description="Inspect whether a project path exists and whether it is a file or folder.",
        risks=("read",),
    ),
    "list_directory": CapabilityDefinition(
        name="list_directory",
        group="read_tools",
        kind="tool",
        access="planner_visible",
        approval="auto",
        description="List the contents of a project folder.",
        risks=("read",),
    ),
    "read_text_file": CapabilityDefinition(
        name="read_text_file",
        group="read_tools",
        kind="tool",
        access="planner_visible",
        approval="auto",
        description="Read a UTF-8 text file from the project.",
        risks=("read",),
    ),
    "read_json_file": CapabilityDefinition(
        name="read_json_file",
        group="read_tools",
        kind="tool",
        access="planner_visible",
        approval="auto",
        description="Read and parse a JSON file from the project.",
        risks=("read",),
    ),
    "read_pdf_text": CapabilityDefinition(
        name="read_pdf_text",
        group="read_tools",
        kind="tool",
        access="planner_visible",
        approval="auto",
        description="Extract text from a PDF file in the project.",
        risks=("read",),
    ),
    "read_documents": CapabilityDefinition(
        name="read_documents",
        group="read_tools",
        kind="tool",
        access="planner_visible",
        approval="auto",
        description="Read supported local documents from a file or folder.",
        risks=("read",),
    ),
    "search_web": CapabilityDefinition(
        name="search_web",
        group="read_tools",
        kind="tool",
        access="planner_visible",
        approval="auto",
        description="Search the public web and return structured search results.",
        risks=("network", "read"),
    ),
    "resolve_job_folder_hint": CapabilityDefinition(
        name="resolve_job_folder_hint",
        group="read_tools",
        kind="tool",
        access="planner_visible",
        approval="auto",
        description="Resolve a job-folder hint to a concrete local folder path.",
        risks=("read",),
    ),
    "resolve_local_job_inputs": CapabilityDefinition(
        name="resolve_local_job_inputs",
        group="read_tools",
        kind="tool",
        access="planner_visible",
        approval="auto",
        description="Inspect a local job folder and resolve the available job-description inputs.",
        risks=("read",),
    ),
    "read_job_metadata": CapabilityDefinition(
        name="read_job_metadata",
        group="read_tools",
        kind="tool",
        access="planner_visible",
        approval="auto",
        description="Read or infer metadata for a local job folder.",
        risks=("read",),
    ),
    "discover_jobs": CapabilityDefinition(
        name="discover_jobs",
        group="read_tools",
        kind="tool",
        access="planner_visible",
        approval="auto",
        description="Discover matching jobs from configured public job-board sources using explicit search inputs.",
        risks=("network", "read"),
    ),
    "clean_job_description": CapabilityDefinition(
        name="clean_job_description",
        group="transform_tools",
        kind="tool",
        access="planner_visible",
        approval="auto",
        description="Clean raw job-posting text into a structured plain-text description.",
        risks=("llm",),
    ),
    "generate_application_materials": CapabilityDefinition(
        name="generate_application_materials",
        group="transform_tools",
        kind="tool",
        access="planner_visible",
        approval="auto",
        description="Generate application notes from a cleaned job description.",
        risks=("llm",),
    ),
    "build_application_notes_from_job_description": CapabilityDefinition(
        name="build_application_notes_from_job_description",
        group="transform_tools",
        kind="tool",
        access="planner_visible",
        approval="auto",
        description="Build the notes text block for a cleaned job description.",
        risks=("llm",),
    ),
    "write_document": CapabilityDefinition(
        name="write_document",
        group="write_tools",
        kind="tool",
        access="planner_visible",
        approval="conditional",
        description="Write a new text document into the outputs area without overwriting.",
        risks=("write",),
    ),
    "copy_file": CapabilityDefinition(
        name="copy_file",
        group="write_tools",
        kind="tool",
        access="planner_visible",
        approval="conditional",
        description="Copy a project file into the outputs area without overwriting.",
        risks=("read", "write"),
    ),
    "write_json_file": CapabilityDefinition(
        name="write_json_file",
        group="write_tools",
        kind="tool",
        access="planner_visible",
        approval="conditional",
        description="Write a new JSON file into the outputs area without overwriting.",
        risks=("write",),
    ),
    "write_search_results": CapabilityDefinition(
        name="write_search_results",
        group="write_tools",
        kind="tool",
        access="planner_visible",
        approval="conditional",
        description="Write previously collected search results into a new output file without overwriting.",
        risks=("write",),
    ),
    "write_generated_documents": CapabilityDefinition(
        name="write_generated_documents",
        group="write_tools",
        kind="tool",
        access="planner_visible",
        approval="conditional",
        description="Write multiple generated text documents into a new output directory without overwriting.",
        risks=("write",),
    ),
    "generate_document_set": CapabilityDefinition(
        name="generate_document_set",
        group="llm_tasks",
        kind="llm",
        access="planner_visible",
        approval="auto",
        description="Generate structured filename/content records for writing multiple output documents.",
        risks=("llm",),
    ),
    "run_job_agent": CapabilityDefinition(
        name="run_job_agent",
        group="commands",
        kind="command",
        access="planner_visible",
        approval="manual",
        description="Run the local job workflow or online job discovery workflow as a constrained internal command.",
        risks=("subprocess", "write"),
    ),
    "create_job_files": CapabilityDefinition(
        name="create_job_files",
        group="commands",
        kind="command",
        access="planner_visible",
        approval="manual",
        description="Generate application files for a local job folder via the constrained internal job command.",
        risks=("subprocess", "write"),
    ),
    "summarize_documents": CapabilityDefinition(
        name="summarize_documents",
        group="transform_tools",
        kind="tool",
        access="legacy_disabled",
        approval="never",
        description="Legacy composite summary tool. Do not use; plan with read_documents + summarize_text instead.",
        risks=("llm", "write"),
    ),
    "evaluate_documents": CapabilityDefinition(
        name="evaluate_documents",
        group="transform_tools",
        kind="tool",
        access="legacy_disabled",
        approval="never",
        description="Legacy composite evaluation tool. Do not use; plan with read_documents + evaluate_text instead.",
        risks=("llm", "write"),
    ),
    "match_cv": CapabilityDefinition(
        name="match_cv",
        group="transform_tools",
        kind="tool",
        access="legacy_disabled",
        approval="never",
        description="Legacy composite CV matching tool. Do not use; plan with read_documents + rank_cvs instead.",
        risks=("llm", "write"),
    ),
}


def build_capability_catalog_text() -> str:
    """Build capability catalog text."""

    grouped: dict[CapabilityGroup, list[CapabilityDefinition]] = {
        group: [] for group in GROUP_ORDER
    }
    for definition in CAPABILITY_DEFINITIONS.values():
        if definition.access != "planner_visible":
            continue
        grouped[definition.group].append(definition)

    lines: list[str] = []
    for group in GROUP_ORDER:
        items = sorted(grouped[group], key=lambda item: item.name)
        if not items:
            continue
        lines.append(f"{GROUP_LABELS[group]}:")
        for item in items:
            lines.append(f"- `{item.name}`: {item.description}")
        lines.append("")
    return "\n".join(lines).strip()


def build_forbidden_actions_text() -> str:
    """Build forbidden actions text."""

    return "\n".join(f"- {line}" for line in FORBIDDEN_ACTION_LINES)


def build_end_user_capability_summary() -> str:
    """Build end user capability summary."""

    sections = [
        "I can help only with these registered capabilities:",
        "",
        build_capability_catalog_text(),
        "",
        "I will not do these things:",
        build_forbidden_actions_text(),
    ]
    return "\n".join(sections).strip()


def is_capability_question(question: str) -> bool:
    """Return whether capability question."""

    return bool(_CAPABILITY_QUESTION_PATTERN.search(question.strip()))


def forbidden_request_response(question: str) -> str | None:
    """Return forbidden request response."""

    normalized = question.strip()
    for pattern, refusal in _FORBIDDEN_REQUEST_PATTERNS:
        if pattern.search(normalized):
            return (
                f"{refusal}\n\n"
                "I can still help with allowed actions such as reading files, analyzing documents, "
                "writing new output files without overwriting, searching the web, or running the registered job workflows."
            )
    return None
