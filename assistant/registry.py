"""Static capability registries for planner-visible tools and LLM tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from assistant.capabilities import CAPABILITY_DEFINITIONS
from assistant.llm_tasks import (
    answer_question,
    evaluate_text,
    generate_document_set,
    rank_cvs,
    summarize_text,
)
from assistant.schemas import (
    AnswerQuestionArgs,
    BuildApplicationNotesArgs,
    CleanJobDescriptionArgs,
    CopyFileArgs,
    CreateJobFilesArgs,
    DiscoverJobsArgs,
    EvaluateTextArgs,
    GenerateDocumentSetArgs,
    GenerateApplicationMaterialsArgs,
    InspectPathArgs,
    ListDirectoryArgs,
    ReadDocumentsArgs,
    ReadJobMetadataArgs,
    ReadJsonFileArgs,
    ReadPdfTextArgs,
    ReadTextFileArgs,
    RankCvArgs,
    ResolveJobFolderHintArgs,
    ResolveLocalJobInputsArgs,
    RunJobAgentArgs,
    SearchWebArgs,
    SummarizeTextArgs,
    WriteDocumentArgs,
    WriteGeneratedDocumentsArgs,
    WriteJsonFileArgs,
    WriteSearchResultsArgs,
)
from tools.atomic_tools import (
    build_application_notes_from_job_description,
    clean_job_description,
    discover_jobs,
    generate_application_materials,
    inspect_path,
    list_directory,
    read_job_metadata,
    read_json_file,
    read_pdf_text,
    read_text_file,
    resolve_job_folder_hint,
    resolve_local_job_inputs,
    write_json_file,
    write_generated_documents,
    write_search_results,
)
from tools.document_actions import copy_file, read_documents, write_document
from tools.job.create_job_files import create_job_files
from tools.job.run_job_agent import run_job_agent
from tools.web_search import search_web


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    function: Callable[..., dict]
    input_model: type
    description: str
    group: str
    kind: str
    approval: str
    risks: tuple[str, ...]
    account_access: bool = False
    allowed: bool = True


def _build_spec(name: str, function: Callable[..., dict], input_model: type) -> CapabilitySpec:
    """Build spec."""

    definition = CAPABILITY_DEFINITIONS[name]
    return CapabilitySpec(
        name=name,
        function=function,
        input_model=input_model,
        description=definition.description,
        group=definition.group,
        kind=definition.kind,
        approval=definition.approval,
        risks=definition.risks,
        account_access=definition.account_access,
        allowed=definition.access == "planner_visible",
    )


READ_TOOLS = {
    "inspect_path": _build_spec("inspect_path", inspect_path, InspectPathArgs),
    "list_directory": _build_spec("list_directory", list_directory, ListDirectoryArgs),
    "read_text_file": _build_spec("read_text_file", read_text_file, ReadTextFileArgs),
    "read_json_file": _build_spec("read_json_file", read_json_file, ReadJsonFileArgs),
    "read_pdf_text": _build_spec("read_pdf_text", read_pdf_text, ReadPdfTextArgs),
    "read_documents": _build_spec("read_documents", read_documents, ReadDocumentsArgs),
    "search_web": _build_spec("search_web", search_web, SearchWebArgs),
    "resolve_job_folder_hint": _build_spec(
        "resolve_job_folder_hint",
        resolve_job_folder_hint,
        ResolveJobFolderHintArgs,
    ),
    "resolve_local_job_inputs": _build_spec(
        "resolve_local_job_inputs",
        resolve_local_job_inputs,
        ResolveLocalJobInputsArgs,
    ),
    "read_job_metadata": _build_spec(
        "read_job_metadata",
        read_job_metadata,
        ReadJobMetadataArgs,
    ),
    "discover_jobs": _build_spec("discover_jobs", discover_jobs, DiscoverJobsArgs),
}

TRANSFORM_TOOLS = {
    "clean_job_description": _build_spec(
        "clean_job_description",
        clean_job_description,
        CleanJobDescriptionArgs,
    ),
    "generate_application_materials": _build_spec(
        "generate_application_materials",
        generate_application_materials,
        GenerateApplicationMaterialsArgs,
    ),
    "build_application_notes_from_job_description": _build_spec(
        "build_application_notes_from_job_description",
        build_application_notes_from_job_description,
        BuildApplicationNotesArgs,
    ),
}

WRITE_TOOLS = {
    "copy_file": _build_spec("copy_file", copy_file, CopyFileArgs),
    "write_document": _build_spec("write_document", write_document, WriteDocumentArgs),
    "write_json_file": _build_spec(
        "write_json_file",
        write_json_file,
        WriteJsonFileArgs,
    ),
    "write_search_results": _build_spec(
        "write_search_results",
        write_search_results,
        WriteSearchResultsArgs,
    ),
    "write_generated_documents": _build_spec(
        "write_generated_documents",
        write_generated_documents,
        WriteGeneratedDocumentsArgs,
    ),
}

COMMANDS = {
    "run_job_agent": _build_spec("run_job_agent", run_job_agent, RunJobAgentArgs),
    "create_job_files": _build_spec(
        "create_job_files",
        create_job_files,
        CreateJobFilesArgs,
    ),
}

TOOLS = {
    **READ_TOOLS,
    **TRANSFORM_TOOLS,
    **WRITE_TOOLS,
    **COMMANDS,
}

LLM_TASKS = {
    "answer_question": _build_spec(
        "answer_question",
        answer_question,
        AnswerQuestionArgs,
    ),
    "summarize_text": _build_spec(
        "summarize_text",
        summarize_text,
        SummarizeTextArgs,
    ),
    "evaluate_text": _build_spec(
        "evaluate_text",
        evaluate_text,
        EvaluateTextArgs,
    ),
    "generate_document_set": _build_spec(
        "generate_document_set",
        generate_document_set,
        GenerateDocumentSetArgs,
    ),
    "rank_cvs": _build_spec(
        "rank_cvs",
        rank_cvs,
        RankCvArgs,
    ),
}
