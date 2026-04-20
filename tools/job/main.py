import sys
from datetime import datetime, timedelta
from pathlib import Path

from assistant.config import get_paths

ROOT = Path(__file__).resolve().parents[2]

from tools.job.clean_job_description import run as clean_job_description
from tools.job.discover_jobs import (
    build_job_folder_path,
    discover_jobs_from_config,
    load_inputs_config,
    save_discovered_job,
)
from tools.job.export_job_pdf import run as export_job_pdf
from tools.job.generate_application_materials import run as generate_application_materials
from tools.job.job_folder_resolution import resolve_job_folder_hint
from tools.job.local_job_inputs import (
    CLEANED_DESCRIPTION_FILE,
    METADATA_FILE,
    ResolvedLocalJobInputs,
    resolve_local_job_inputs,
)
from tools.job.filesystem import save_json, save_text
from tools.job.models import JobState

PATHS = get_paths()
JOBS_ROOT = PATHS["jobs_root"]
OUTPUTS_ROOT = PATHS["outputs_root"]


def build_state(output_folder: Path) -> JobState:
    return JobState(
        folder=output_folder,
        raw_file=output_folder / "job_description_raw.txt",
        metadata_file=output_folder / METADATA_FILE,
        cleaned_file=output_folder / CLEANED_DESCRIPTION_FILE,
        pdf_file=output_folder / "job_description.pdf",
        info_file=output_folder / "info.txt",
    )


def run_state(state: JobState) -> None:
    print(f"[job] start raw_folder={state.raw_file.parent}")
    if not state.raw_file.exists():
        raise FileNotFoundError(f"Missing input file: {state.raw_file}")

    clean_job_description(state)
    export_job_pdf(state)
    generate_application_materials(state)
    print(f"[job] completed output={state.folder}")


def run_state_from_cleaned(state: JobState, cleaned_source_file: Path) -> None:
    print(f"[job] start cleaned_folder={cleaned_source_file.parent}")
    if not cleaned_source_file.exists():
        raise FileNotFoundError(f"Missing input file: {cleaned_source_file}")

    cleaned_text = cleaned_source_file.read_text(encoding="utf-8")
    save_text(state.cleaned_file, cleaned_text)
    print(f"[job] staged cleaned_file={state.cleaned_file}")

    export_job_pdf(state)
    generate_application_materials(state)
    print(f"[job] completed output={state.folder}")


def resolve_job_folder(job_folder_arg: str) -> tuple[Path, ResolvedLocalJobInputs]:
    job_folder = resolve_job_folder_hint(job_folder_arg, ROOT, JOBS_ROOT)

    if not job_folder.exists():
        raise FileNotFoundError(
            f"Job folder does not exist: {job_folder}\n"
            "Pass an existing folder containing `job_description_raw.txt` or "
            f"`{CLEANED_DESCRIPTION_FILE}`, `job_description.txt`, `job description.txt`, "
            "`job_description.pdf`, or `job description.pdf`, or use batch mode with no argument."
        )

    if not job_folder.is_dir():
        raise ValueError(f"Job path is not a folder: {job_folder}")

    resolved_inputs = resolve_local_job_inputs(job_folder)
    if resolved_inputs is not None:
        return job_folder, resolved_inputs

    raise FileNotFoundError(
        f"Folder does not contain `job_description_raw.txt`, `{CLEANED_DESCRIPTION_FILE}`, "
        f"`job_description.txt`, `job description.txt`, `job_description.pdf`, or `job description.pdf`: {job_folder}"
    )


def _build_run_output_root() -> Path:
    candidate_time = datetime.now().replace(microsecond=0)
    candidate = OUTPUTS_ROOT / candidate_time.strftime("%Y%m%d_%H%M%S")
    while candidate.exists():
        candidate_time += timedelta(seconds=1)
        candidate = OUTPUTS_ROOT / candidate_time.strftime("%Y%m%d_%H%M%S")
    return candidate


def _stage_resolved_inputs(
    state: JobState,
    resolved_inputs: ResolvedLocalJobInputs,
) -> None:
    if resolved_inputs.mode == "raw":
        if resolved_inputs.raw_text is None:
            raise ValueError("Resolved raw inputs are missing raw text.")
        save_text(state.raw_file, resolved_inputs.raw_text)

    if resolved_inputs.metadata is not None:
        save_json(state.metadata_file, resolved_inputs.metadata)


def run_single(job_folder_arg: str) -> None:
    job_folder, resolved_inputs = resolve_job_folder(job_folder_arg)
    run_output_root = _build_run_output_root()
    output_folder = run_output_root / job_folder.name
    print(f"[run] mode=single job_folder={job_folder} input_mode={resolved_inputs.mode}")
    print(f"[run] output_folder={output_folder}")
    state = build_state(output_folder)
    _stage_resolved_inputs(state, resolved_inputs)
    if resolved_inputs.mode == "raw":
        run_state(state)
    else:
        run_state_from_cleaned(state, resolved_inputs.source_file)
    print(f"Done. Output written to: {run_output_root}")


def run_batch() -> None:
    print("[run] mode=batch discovery")
    config = load_inputs_config()
    print(
        "[run] inputs "
        f"role={config.get('role')} "
        f"location={config.get('location')} "
        f"sources={config.get('sources', [])} "
        f"max_jobs={config.get('max_jobs', 1)} "
        f"max_results_per_source={config.get('max_results_per_source', 5)} "
        f"max_company_attempts_per_source={config.get('max_company_attempts_per_source', 'all')}"
    )
    jobs = discover_jobs_from_config(config)
    run_output_root = _build_run_output_root()
    date_prefix = datetime.now().strftime("%Y%m%d")

    print(f"[run] discovered {len(jobs)} matching job(s)")
    print(f"[run] batch_output_root={run_output_root}")

    for index, job in enumerate(jobs, start=1):
        job_folder = build_job_folder_path(job, run_output_root, date_prefix)
        raw_file = job_folder / "job_description_raw.txt"
        metadata_file = job_folder / METADATA_FILE
        print(
            f"[run] preparing job {index}/{len(jobs)} "
            f"company={job['company']} title={job['title']} "
            f"location={job['location']} source={job['source']}"
        )
        print(f"[run] job_folder={job_folder}")
        save_discovered_job(raw_file, metadata_file, job)

        state = build_state(job_folder)
        run_state(state)

    print(f"Done. Output written to: {run_output_root}")


def main() -> None:
    if len(sys.argv) == 1:
        run_batch()
        return

    if len(sys.argv) != 2:
        raise ValueError(
            'Usage: python -m tools.job.main "data/inputs/jobs/YYYYMMDD - Company - Role"\n'
            '   or: python -m tools.job.main "company-name"\n'
            '   or: python -m tools.job.main "data/inputs/jobs/some-folder-with-cleaned-job-description"\n'
            "   or: python -m tools.job.main"
        )

    run_single(sys.argv[1])


if __name__ == "__main__":
    main()
