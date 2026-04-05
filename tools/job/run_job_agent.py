from pathlib import Path
import json
import os
import subprocess
import sys

from assistant.audit import log_event
from assistant.config import get_job_workflow_config, get_prompt_override_fields

SUBPROCESS_TIMEOUT_SECONDS = get_job_workflow_config()["subprocess_timeout_seconds"]


def run_job_agent(
    folder_path: str | None = None,
    role: str | None = None,
    location: str | None = None,
    ignore_location: bool | None = None,
    remote_only: bool | None = None,
):
    """
    Runs the job workflow.
    Keeps legacy behavior for now via subprocess.
    """

    base_path = Path(__file__).resolve().parent
    project_root = base_path.parents[1]
    project_python = project_root / ".venv" / "bin" / "python"
    python_executable = str(
        project_python) if project_python.exists() else sys.executable
    cmd = [python_executable, "-m", "tools.job.main"]
    raw_search_overrides = {
        "role": role,
        "location": location,
        "ignore_location": ignore_location,
        "remote_only": remote_only,
    }
    allowed_override_fields = set(get_prompt_override_fields("run_job_agent"))
    search_overrides = {
        key: value
        for key, value in raw_search_overrides.items()
        if key in allowed_override_fields and value is not None
    }

    if folder_path:
        if search_overrides:
            raise ValueError(
                "Search overrides are only supported when running online job discovery."
            )
        cmd.append(folder_path)

    log_event("job_subprocess_spawned", command=cmd, cwd=str(project_root))
    env = os.environ.copy()
    if search_overrides:
        env["DOMO_JOB_SEARCH_OVERRIDES_JSON"] = json.dumps(search_overrides)
    process = subprocess.Popen(
        cmd,
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        yield "Error: no subprocess output available.\n"
        return

    try:
        for line in process.stdout:
            yield line

        process.wait(timeout=SUBPROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        log_event("job_subprocess_timeout", command=cmd,
                  timeout=SUBPROCESS_TIMEOUT_SECONDS)
        yield (
            "Job workflow timed out and was terminated after "
            f"{SUBPROCESS_TIMEOUT_SECONDS} seconds.\n"
        )
        return

    log_event("job_subprocess_finished", command=cmd,
              returncode=process.returncode)
