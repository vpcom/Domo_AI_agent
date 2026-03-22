import sys
from pathlib import Path


def get_job_folder_from_args() -> Path:
    if len(sys.argv) != 2:
        raise ValueError(
            'Usage: python main.py "jobs/20260302 - Company name - Job title"'
        )

    job_folder = Path(sys.argv[1])

    if not job_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {job_folder}")

    if not job_folder.is_dir():
        raise ValueError(f"Path is not a folder: {job_folder}")

    return job_folder


def read_required_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Input file is empty: {path}")

    return content


def save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
