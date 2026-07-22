"""Thin structured wrapper for the local job-file generation command."""

from __future__ import annotations

from tools.job.run_job_agent import run_job_agent


def create_job_files(job_folder: str) -> dict:
    """Create job files."""

    return run_job_agent(folder_path=job_folder)
