"""Create job files workflow module for the Domo assistant.

Create job files workflow workflow definitions for the Domo assistant.
"""

from assistant.config import is_debug_enabled
from tools.job.create_job_files import create_job_files


def run_create_job_files_workflow(job_folder: str):
    """Run create job files workflow."""

    yield "Starting create_job_files workflow...\n"
    yield f"Resolved job input: {job_folder}\n"
    if is_debug_enabled():
        yield "DEBUG: debug mode is enabled.\n"

    for chunk in create_job_files(job_folder=job_folder):
        yield chunk

    yield "Workflow finished.\n"
