from assistant.config import is_debug_enabled
from tools.document_actions import (
    copy_file,
    evaluate_documents,
    read_documents,
    summarize_documents,
    write_document,
)


def run_copy_file_workflow(source_path: str, destination_path: str):
    yield "Starting copy_file workflow...\n"
    yield f"Resolved source input: {source_path}\n"
    yield f"Resolved destination input: {destination_path}\n"
    if is_debug_enabled():
        yield "DEBUG: debug mode is enabled.\n"
    yield from copy_file(source_path=source_path, destination_path=destination_path)
    yield "Workflow finished.\n"


def run_write_document_workflow(destination_path: str, content: str):
    yield "Starting write_document workflow...\n"
    yield f"Resolved destination input: {destination_path}\n"
    if is_debug_enabled():
        yield "DEBUG: debug mode is enabled.\n"
    yield from write_document(destination_path=destination_path, content=content)
    yield "Workflow finished.\n"


def run_read_documents_workflow(input_path: str, recursive: bool | None = None):
    yield "Starting read_documents workflow...\n"
    yield f"Resolved input: {input_path}\n"
    if recursive is not None:
        yield f"Recursive: {recursive}\n"
    if is_debug_enabled():
        yield "DEBUG: debug mode is enabled.\n"
    yield from read_documents(input_path=input_path, recursive=recursive)
    yield "Workflow finished.\n"


def run_summarize_documents_workflow(
    input_path: str,
    instructions: str | None = None,
    output_path: str | None = None,
    recursive: bool | None = None,
):
    yield "Starting summarize_documents workflow...\n"
    yield f"Resolved input: {input_path}\n"
    if output_path:
        yield f"Resolved output: {output_path}\n"
    if instructions:
        yield f"Instructions: {instructions}\n"
    if recursive is not None:
        yield f"Recursive: {recursive}\n"
    if is_debug_enabled():
        yield "DEBUG: debug mode is enabled.\n"
    yield from summarize_documents(
        input_path=input_path,
        instructions=instructions,
        output_path=output_path,
        recursive=recursive,
    )
    yield "Workflow finished.\n"


def run_evaluate_documents_workflow(
    input_path: str,
    instructions: str,
    output_path: str | None = None,
    recursive: bool | None = None,
):
    yield "Starting evaluate_documents workflow...\n"
    yield f"Resolved input: {input_path}\n"
    if output_path:
        yield f"Resolved output: {output_path}\n"
    yield f"Instructions: {instructions}\n"
    if recursive is not None:
        yield f"Recursive: {recursive}\n"
    if is_debug_enabled():
        yield "DEBUG: debug mode is enabled.\n"
    yield from evaluate_documents(
        input_path=input_path,
        instructions=instructions,
        output_path=output_path,
        recursive=recursive,
    )
    yield "Workflow finished.\n"
