from assistant.config import get_config_path, get_display_path, is_debug_enabled
from tools.job.run_job_agent import run_job_agent


def run_job_agent_workflow(
    folder_path: str | None = None,
    role: str | None = None,
    location: str | None = None,
    ignore_location: bool | None = None,
    remote_only: bool | None = None,
):
    yield "Starting job workflow...\n"
    if folder_path:
        yield f"Resolved job input: {folder_path}\n"
    else:
        yield (
            "Running default online job search from "
            f"{get_display_path(get_config_path())}\n"
        )
        overrides = []
        if role:
            overrides.append(f"role={role}")
        if location:
            overrides.append(f"location={location}")
        if ignore_location is not None:
            overrides.append(f"ignore_location={ignore_location}")
        if remote_only is not None:
            overrides.append(f"remote_only={remote_only}")
        if overrides:
            yield f"Search overrides: {', '.join(overrides)}\n"
    if is_debug_enabled():
        yield "DEBUG: debug mode is enabled.\n"

    for chunk in run_job_agent(
        folder_path=folder_path,
        role=role,
        location=location,
        ignore_location=ignore_location,
        remote_only=remote_only,
    ):
        yield chunk

    yield "Workflow finished.\n"
