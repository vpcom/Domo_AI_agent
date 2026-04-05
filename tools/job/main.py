import sys
from datetime import datetime
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
    PDF_DESCRIPTION_FILE,
    RAW_DESCRIPTION_FILE,
    ensure_local_job_inputs,
    find_cleaned_job_description_file,
)
from tools.job.models import JobState

PATHS = get_paths()
DATA_ROOT = PATHS["data_root"]
JOBS_ROOT = PATHS["jobs_root"]
OUTPUTS_ROOT = PATHS["outputs_root"]


def build_state(job_folder: Path, output_folder: Path) -> JobState:
    return JobState(
        folder=output_folder,
        raw_file=job_folder / RAW_DESCRIPTION_FILE,
        metadata_file=job_folder / "job_metadata.json",
        cleaned_file=output_folder / CLEANED_DESCRIPTION_FILE,
        pdf_file=output_folder / "job_description.pdf",
        notes_file=output_folder / "application_notes.txt",
        summary_file=output_folder / "summary.txt",
        skills_file=output_folder / "skills.txt",
        cv_file=output_folder / "sample_cv.txt",
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
    state.cleaned_file.parent.mkdir(parents=True, exist_ok=True)
    state.cleaned_file.write_text(cleaned_text, encoding="utf-8")
    print(f"[job] staged cleaned_file={state.cleaned_file}")

    export_job_pdf(state)
    generate_application_materials(state)
    print(f"[job] completed output={state.folder}")


def resolve_job_folder(job_folder_arg: str) -> tuple[Path, str]:
    job_folder = resolve_job_folder_hint(job_folder_arg, ROOT, JOBS_ROOT)

    if not job_folder.exists():
        raise FileNotFoundError(
            f"Job folder does not exist: {job_folder}\n"
            f"Pass an existing folder containing `{RAW_DESCRIPTION_FILE}` or "
            f"`{CLEANED_DESCRIPTION_FILE}`, `job_description.txt`, `job description.txt`, "
            f"`{PDF_DESCRIPTION_FILE}`, or `job description.pdf`, or use batch mode with no argument."
        )

    if not job_folder.is_dir():
        raise ValueError(f"Job path is not a folder: {job_folder}")

    raw_file = job_folder / RAW_DESCRIPTION_FILE
    cleaned_file = find_cleaned_job_description_file(job_folder)
    if raw_file.exists():
        return job_folder, "raw"
    if cleaned_file is not None:
        return job_folder, "cleaned"
    ensure_local_job_inputs(job_folder)
    if raw_file.exists():
        return job_folder, "raw"

    raise FileNotFoundError(
        f"Folder does not contain `{RAW_DESCRIPTION_FILE}`, `{CLEANED_DESCRIPTION_FILE}`, "
        f"`job_description.txt`, `job description.txt`, `{PDF_DESCRIPTION_FILE}`, or `job description.pdf`: {job_folder}"
    )


def run_single(job_folder_arg: str) -> None:
    job_folder, input_mode = resolve_job_folder(job_folder_arg)
    output_folder = OUTPUTS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[run] mode=single job_folder={job_folder} input_mode={input_mode}")
    print(f"[run] output_folder={output_folder}")
    state = build_state(job_folder, output_folder)
    if input_mode == "raw":
        run_state(state)
    else:
        cleaned_source_file = find_cleaned_job_description_file(job_folder)
        if cleaned_source_file is None:
            raise FileNotFoundError(f"Missing cleaned job description in {job_folder}")
        run_state_from_cleaned(state, cleaned_source_file)
    print(f"Done. Output written to: {state.folder}")


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
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_root = OUTPUTS_ROOT / run_timestamp
    date_prefix = datetime.now().strftime("%Y%m%d")

    print(f"[run] discovered {len(jobs)} matching job(s)")
    print(f"[run] batch_output_root={run_output_root}")

    for index, job in enumerate(jobs, start=1):
        job_folder = build_job_folder_path(job, JOBS_ROOT, date_prefix)
        raw_file = job_folder / "job_description_raw.txt"
        metadata_file = job_folder / "job_metadata.json"
        print(
            f"[run] preparing job {index}/{len(jobs)} "
            f"company={job['company']} title={job['title']} "
            f"location={job['location']} source={job['source']}"
        )
        print(f"[run] job_folder={job_folder}")
        save_discovered_job(raw_file, metadata_file, job)

        output_folder = run_output_root / job_folder.name
        print(f"[run] output_folder={output_folder}")
        state = build_state(job_folder, output_folder)
        run_state(state)

    print(f"Done. Processed {len(jobs)} jobs.")


def main() -> None:
    if len(sys.argv) == 1:
        run_batch()
        return

    if len(sys.argv) != 2:
        raise ValueError(
            'Usage: python -m tools.job.main "data/jobs/YYYYMMDD - Company - Role"\n'
            '   or: python -m tools.job.main "company-name"\n'
            '   or: python -m tools.job.main "data/jobs/some-folder-with-cleaned-job-description"\n'
            "   or: python -m tools.job.main"
        )

    run_single(sys.argv[1])


if __name__ == "__main__":
    main()
