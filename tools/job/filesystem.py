import sys
from pathlib import Path
import json
import shutil


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
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}") from exc


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}") from exc


def copy_file_no_overwrite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
    shutil.copy2(source, destination)


def ensure_new_file_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
