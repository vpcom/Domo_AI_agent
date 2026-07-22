"""Structured wrapper around the constrained internal job workflow command."""

from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys

from assistant.config import get_job_workflow_config, get_prompt_override_fields


SUBPROCESS_TIMEOUT_SECONDS = get_job_workflow_config()[
    "subprocess_timeout_seconds"]


def run_job_agent(
    folder_path: str | None = None,
    role: str | None = None,
    location: str | None = None,
    ignore_location: bool | None = None,
    remote_only: bool | None = None,
) -> dict:
    """Run the job agent workflow and stream the output."""

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
        raise RuntimeError("No subprocess output available.")

    lines: list[str] = []
    try:
        for line in process.stdout:
            lines.append(line)
        process.wait(timeout=SUBPROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        raise RuntimeError(
            "Job workflow timed out and was terminated after "
            f"{SUBPROCESS_TIMEOUT_SECONDS} seconds."
        ) from exc

    if process.returncode not in (0, None):
        raise RuntimeError(
            f"Job workflow failed with return code {process.returncode}.\n"
            f"{''.join(lines).strip()}"
        )

    output_root = _extract_output_root(lines)
    artifacts: list[dict] = []
    if output_root:
        artifacts.append(
            {
                "name": Path(output_root).name,
                "kind": "folder",
                "path": output_root,
                "metadata": {"source": "job_workflow"},
            }
        )

    return {
        "result": {
            "command": cmd,
            "output_lines": lines,
            "output_root": output_root,
        },
        "metadata": {
            "display_text": "".join(lines).strip(),
            "artifacts": artifacts,
        },
    }


def _extract_output_root(lines: list[str]) -> str | None:
    """Extract output root."""

    for line in reversed(lines):
        prefix = "Done. Output written to: "
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None
