from dataclasses import dataclass
from typing import Any, Callable

from assistant.schemas import (
    CopyFileArgs,
    CreateJobFilesArgs,
    EvaluateDocumentsArgs,
    MatchCvArgs,
    ReadDocumentsArgs,
    RunJobAgentArgs,
    SummarizeDocumentsArgs,
    WriteDocumentArgs,
)
from workflows.document_workflows import (
    run_copy_file_workflow,
    run_evaluate_documents_workflow,
    run_read_documents_workflow,
    run_summarize_documents_workflow,
    run_write_document_workflow,
)
from workflows.create_job_files_workflow import run_create_job_files_workflow
from workflows.match_cv_workflow import run_match_cv_workflow
from workflows.run_job_agent_workflow import run_job_agent_workflow


@dataclass(frozen=True)
class ToolSpec:
    name: str
    arg_model: type
    argument_keys: tuple[str, ...]
    required_keys: tuple[str, ...]
    context_keys: tuple[str, ...]
    description: str
    aliases: tuple[str, ...]
    executor: Callable[..., Any]


# The agent can mostly execute these 3 workflows
WORKFLOWS = {
    "run_job_agent": run_job_agent_workflow,
    "create_job_files": run_create_job_files_workflow,
    "match_cv": run_match_cv_workflow,
    "copy_file": run_copy_file_workflow,
    "write_document": run_write_document_workflow,
    "read_documents": run_read_documents_workflow,
    "summarize_documents": run_summarize_documents_workflow,
    "evaluate_documents": run_evaluate_documents_workflow,
}

TOOLS = {
    "run_job_agent": ToolSpec(
        name="run_job_agent",
        arg_model=RunJobAgentArgs,
        argument_keys=("folder_path", "role", "location", "ignore_location", "remote_only"),
        required_keys=(),
        context_keys=("folder_path", "role", "location", "ignore_location", "remote_only"),
        description="Search online jobs or process an existing local job folder.",
        aliases=("job search", "search jobs", "find jobs", "discover jobs"),
        executor=WORKFLOWS["run_job_agent"],
    ),
    "match_cv": ToolSpec(
        name="match_cv",
        arg_model=MatchCvArgs,
        argument_keys=("job_folder", "cvs_folder"),
        required_keys=("job_folder",),
        context_keys=("job_folder", "cvs_folder"),
        description="Compare CV PDFs against a specific job folder.",
        aliases=("match cv", "best cv", "compare cv", "rank cvs"),
        executor=WORKFLOWS["match_cv"],
    ),
    "create_job_files": ToolSpec(
        name="create_job_files",
        arg_model=CreateJobFilesArgs,
        argument_keys=("job_folder",),
        required_keys=("job_folder",),
        context_keys=("job_folder",),
        description="Generate application files from an existing local job folder.",
        aliases=("create job files", "prepare documents", "generate application files"),
        executor=WORKFLOWS["create_job_files"],
    ),
    "copy_file": ToolSpec(
        name="copy_file",
        arg_model=CopyFileArgs,
        argument_keys=("source_path", "destination_path"),
        required_keys=("source_path", "destination_path"),
        context_keys=("source_path", "destination_path"),
        description="Copy a file within the project's data directories.",
        aliases=("copy file", "duplicate file", "copy document"),
        executor=WORKFLOWS["copy_file"],
    ),
    "write_document": ToolSpec(
        name="write_document",
        arg_model=WriteDocumentArgs,
        argument_keys=("destination_path", "content"),
        required_keys=("destination_path", "content"),
        context_keys=("destination_path",),
        description="Write a text document under the jobs or outputs directories and create parent folders if needed.",
        aliases=("write file", "write document", "save text", "create file"),
        executor=WORKFLOWS["write_document"],
    ),
    "read_documents": ToolSpec(
        name="read_documents",
        arg_model=ReadDocumentsArgs,
        argument_keys=("input_path", "recursive"),
        required_keys=("input_path",),
        context_keys=("input_path", "recursive"),
        description="Read text, markdown, or PDF documents from a file or folder.",
        aliases=("read documents", "read files", "open documents", "inspect documents"),
        executor=WORKFLOWS["read_documents"],
    ),
    "summarize_documents": ToolSpec(
        name="summarize_documents",
        arg_model=SummarizeDocumentsArgs,
        argument_keys=("input_path", "instructions", "output_path", "recursive"),
        required_keys=("input_path",),
        context_keys=("input_path", "instructions", "output_path", "recursive"),
        description="Summarize one or more documents and optionally write the result to a file.",
        aliases=("summarize documents", "summarise documents", "summarize files", "document summary", "analyze job ad"),
        executor=WORKFLOWS["summarize_documents"],
    ),
    "evaluate_documents": ToolSpec(
        name="evaluate_documents",
        arg_model=EvaluateDocumentsArgs,
        argument_keys=("input_path", "instructions", "output_path", "recursive"),
        required_keys=("input_path", "instructions"),
        context_keys=("input_path", "instructions", "output_path", "recursive"),
        description="Evaluate and rank documents against explicit instructions or criteria.",
        aliases=("evaluate documents", "rank documents", "sort documents", "score documents", "compare documents"),
        executor=WORKFLOWS["evaluate_documents"],
    ),
}
