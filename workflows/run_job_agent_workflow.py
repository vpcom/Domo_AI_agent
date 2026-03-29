from assistant.config import get_config_path, get_display_path, is_debug_enabled
from tools.job.run_job_agent import run_job_agent


def run_job_agent_workflow(folder_path: str | None = None):
    yield "Starting job workflow...\n"
    if folder_path:
        yield f"Resolved job input: {folder_path}\n"
    else:
        yield (
            "Running default online job search from "
            f"{get_display_path(get_config_path())}\n"
        )
    if is_debug_enabled():
        yield "DEBUG: debug mode is enabled.\n"

    for chunk in run_job_agent(folder_path=folder_path):
        yield chunk

    yield "Workflow finished.\n"
