from assistant.config import get_display_path, get_paths, is_debug_enabled
from tools.job.match_cv import match_cv


def run_match_cv_workflow(job_folder: str, cvs_folder: str | None = None):
    resolved_cvs_input = cvs_folder or get_display_path(get_paths()["cvs_root"])
    yield "Starting CV matching workflow...\n"
    yield f"Resolved job input: {job_folder}\n"
    yield f"Resolved CV input: {resolved_cvs_input}\n"
    if is_debug_enabled():
        yield "DEBUG: debug mode is enabled.\n"

    for chunk in match_cv(job_folder=job_folder, cvs_folder=cvs_folder):
        yield chunk

    yield "Workflow finished.\n"
