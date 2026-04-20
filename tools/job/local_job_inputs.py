from dataclasses import dataclass
import json
from pathlib import Path

from tools.job.job_folder_resolution import parse_folder_date
from tools.job.pdf_utils import extract_pdf_text
from tools.job.text_normalization import normalize_job_posting_text

RAW_DESCRIPTION_FILE = "job_description_raw.txt"
RAW_DESCRIPTION_SOURCE_FILES = (
    "job_description.txt",
    "job description.txt",
)
CLEANED_DESCRIPTION_FILE = "cleaned_job_description.txt"
LEGACY_CLEANED_DESCRIPTION_FILE = "job_description_cleaned.txt"
PDF_DESCRIPTION_FILE = "job_description.pdf"
PDF_DESCRIPTION_SOURCE_FILES = (
    PDF_DESCRIPTION_FILE,
    "job description.pdf",
)
METADATA_FILE = "job_metadata.json"


@dataclass(frozen=True)
class ResolvedLocalJobInputs:
    mode: str
    raw_text: str | None
    cleaned_text: str | None
    metadata: dict | None
    source_file: Path


def _find_first_existing_file(job_folder: Path, file_names: tuple[str, ...]) -> Path | None:
    for file_name in file_names:
        candidate = job_folder / file_name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def find_cleaned_job_description_file(job_folder: Path) -> Path | None:
    return _find_first_existing_file(
        job_folder,
        (
            CLEANED_DESCRIPTION_FILE,
            LEGACY_CLEANED_DESCRIPTION_FILE,
        ),
    )


def find_pdf_job_description_file(job_folder: Path) -> Path | None:
    return _find_first_existing_file(job_folder, PDF_DESCRIPTION_SOURCE_FILES)


def find_source_text_job_description_file(job_folder: Path) -> Path | None:
    return _find_first_existing_file(job_folder, RAW_DESCRIPTION_SOURCE_FILES)


def infer_local_pdf_metadata(job_folder: Path, description: str) -> dict:
    folder_name = job_folder.name
    parts = [part.strip() for part in folder_name.split(" - ") if part.strip()]
    folder_date = parse_folder_date(folder_name)

    company = ""
    title = ""
    if folder_date and len(parts) >= 3:
        company = parts[1]
        title = " - ".join(parts[2:])
    elif len(parts) >= 2:
        company = parts[0]
        title = " - ".join(parts[1:])
    else:
        title = folder_name

    return {
        "company": company,
        "title": title,
        "location": "",
        "description": description,
        "source": "local_pdf",
        "url": "",
    }


def _read_metadata_file(job_folder: Path) -> dict | None:
    metadata_file = job_folder / METADATA_FILE
    if not metadata_file.exists() or not metadata_file.is_file():
        return None
    return json.loads(metadata_file.read_text(encoding="utf-8"))


def resolve_local_job_inputs(job_folder: Path) -> ResolvedLocalJobInputs | None:
    raw_file = job_folder / RAW_DESCRIPTION_FILE
    if raw_file.exists():
        raw_text = normalize_job_posting_text(raw_file.read_text(encoding="utf-8"))
        if not raw_text:
            raise ValueError(f"Input file is empty: {raw_file}")
        metadata = _read_metadata_file(job_folder) or infer_local_pdf_metadata(
            job_folder,
            raw_text,
        )
        return ResolvedLocalJobInputs(
            mode="raw",
            raw_text=raw_text,
            cleaned_text=None,
            metadata=metadata,
            source_file=raw_file,
        )

    cleaned_file = find_cleaned_job_description_file(job_folder)
    if cleaned_file is not None:
        cleaned_text = normalize_job_posting_text(
            cleaned_file.read_text(encoding="utf-8")
        )
        if not cleaned_text:
            raise ValueError(f"Input file is empty: {cleaned_file}")
        metadata = _read_metadata_file(job_folder) or infer_local_pdf_metadata(
            job_folder,
            cleaned_text,
        )
        return ResolvedLocalJobInputs(
            mode="cleaned",
            raw_text=None,
            cleaned_text=cleaned_text,
            metadata=metadata,
            source_file=cleaned_file,
        )

    text_source_file = find_source_text_job_description_file(job_folder)
    pdf_source_file = find_pdf_job_description_file(job_folder)

    if text_source_file is not None:
        print(f"[job] bootstrapping raw job input from text_file={text_source_file}")
        raw_text = normalize_job_posting_text(
            text_source_file.read_text(encoding="utf-8")
        )
    elif pdf_source_file is not None:
        print(f"[job] bootstrapping raw job input from pdf_file={pdf_source_file}")
        raw_text = normalize_job_posting_text(extract_pdf_text(pdf_source_file))
    else:
        return None

    if not raw_text:
        source_file = text_source_file or pdf_source_file
        raise ValueError(f"No readable text could be extracted from {source_file}")

    source_file = text_source_file or pdf_source_file
    metadata = _read_metadata_file(job_folder) or infer_local_pdf_metadata(job_folder, raw_text)
    return ResolvedLocalJobInputs(
        mode="raw",
        raw_text=raw_text,
        cleaned_text=None,
        metadata=metadata,
        source_file=source_file,
    )


def ensure_local_job_inputs(job_folder: Path) -> None:
    resolve_local_job_inputs(job_folder)
