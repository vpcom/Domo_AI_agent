from pathlib import Path
import subprocess
import sys

from assistant.audit import log_event

SUBPROCESS_TIMEOUT_SECONDS = 300


def run_job_agent(folder_path: str | None = None):
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

    if folder_path:
        cmd.append(folder_path)

    log_event("job_subprocess_spawned", command=cmd, cwd=str(project_root))
    process = subprocess.Popen(
        cmd,
        cwd=project_root,
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
