"""Atomic read, transform, and write capabilities exposed to the agent."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from assistant.config import get_paths
from tools.job.clean_job_description import clean_job_description as clean_job_description_text
from tools.job.discover_jobs import build_source_companies, discover_jobs as discover_jobs_with_config
from tools.job.filesystem import save_json, save_text
from tools.job.generate_application_materials import (
    build_application_notes_from_job_description as build_application_notes_text,
    generate_application_materials as generate_application_materials_for_job,
)
from tools.job.job_folder_resolution import resolve_job_folder_hint as resolve_job_folder_path
from tools.job.local_job_inputs import resolve_local_job_inputs as resolve_job_folder_inputs
from tools.job.pdf_utils import extract_pdf_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS = get_paths()
JOBS_ROOT = PATHS["jobs_root"]


def inspect_path(path: str) -> dict:
    """Return inspect path."""

    target = Path(path)
    exists = target.exists()
    result = {
        "path": str(target),
        "exists": exists,
        "is_file": target.is_file() if exists else False,
        "is_dir": target.is_dir() if exists else False,
        "name": target.name,
        "suffix": target.suffix if exists else "",
        "size_bytes": target.stat().st_size if exists and target.is_file() else None,
    }
    display = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True)
    return {
        "result": result,
        "metadata": {
            "display_text": display,
            "artifacts": [],
        },
    }


def list_directory(path: str) -> dict:
    """Return list directory."""

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Folder does not exist: {target}")
    if not target.is_dir():
        raise ValueError(f"Path is not a folder: {target}")

    entries = []
    for child in sorted(target.iterdir(), key=lambda item: item.name.lower()):
        entries.append(
            {
                "name": child.name,
                "path": str(child),
                "kind": "dir" if child.is_dir() else "file",
                "size_bytes": child.stat().st_size if child.is_file() else None,
            }
        )

    lines = [f"Listed {len(entries)} item(s) in {target}"]
    for entry in entries:
        lines.append(f"- [{entry['kind']}] {entry['name']}")
    return {
        "result": {
            "path": str(target),
            "entries": entries,
        },
        "metadata": {
            "display_text": "\n".join(lines),
            "artifacts": [],
        },
    }


def read_text_file(path: str) -> dict:
    """Return read text file."""

    target = Path(path)
    content = target.read_text(encoding="utf-8")
    return {
        "result": {
            "path": str(target),
            "content": content,
        },
        "metadata": {
            "display_text": content,
            "artifacts": [],
        },
    }


def read_json_file(path: str) -> dict:
    """Return read json file."""

    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    return {
        "result": {
            "path": str(target),
            "payload": payload,
        },
        "metadata": {
            "display_text": json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
            "artifacts": [],
        },
    }


def read_pdf_text(path: str) -> dict:
    """Return read pdf text."""

    target = Path(path)
    content = extract_pdf_text(target)
    return {
        "result": {
            "path": str(target),
            "content": content,
        },
        "metadata": {
            "display_text": content,
            "artifacts": [],
        },
    }


def resolve_job_folder_hint(folder_hint: str) -> dict:
    """Resolve job folder hint."""

    resolved = resolve_job_folder_path(folder_hint, PROJECT_ROOT, JOBS_ROOT)
    return {
        "result": {
            "folder_hint": folder_hint,
            "resolved_path": str(resolved),
            "exists": resolved.exists(),
        },
        "metadata": {
            "display_text": f"Resolved `{folder_hint}` to `{resolved}`.",
            "artifacts": [],
        },
    }


def resolve_local_job_inputs(job_folder: str) -> dict:
    """Resolve local job inputs."""

    target = Path(job_folder)
    resolved = resolve_job_folder_inputs(target)
    if resolved is None:
        raise FileNotFoundError(
            f"No supported local job inputs found in: {target}")

    result = {
        "job_folder": str(target),
        "mode": resolved.mode,
        "raw_text": resolved.raw_text,
        "cleaned_text": resolved.cleaned_text,
        "metadata": resolved.metadata,
        "source_file": str(resolved.source_file),
    }
    display_lines = [
        f"Resolved local job inputs for {target}",
        f"Mode: {resolved.mode}",
        f"Source file: {resolved.source_file}",
    ]
    return {
        "result": result,
        "metadata": {
            "display_text": "\n".join(display_lines),
            "artifacts": [],
        },
    }


def read_job_metadata(job_folder: str) -> dict:
    """Return read job metadata."""

    target = Path(job_folder)
    resolved = resolve_job_folder_inputs(target)
    if resolved is None or resolved.metadata is None:
        raise FileNotFoundError(
            f"No job metadata could be resolved for: {target}")

    return {
        "result": {
            "job_folder": str(target),
            "metadata": resolved.metadata,
        },
        "metadata": {
            "display_text": json.dumps(
                resolved.metadata,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ),
            "artifacts": [],
        },
    }


def discover_jobs(
    role: str,
    location: str,
    ignore_location: bool | None = None,
    remote_only: bool | None = None,
    sources: list[str] | None = None,
    max_results_per_source: int | None = None,
    max_jobs: int | None = None,
    max_company_attempts_per_source: int | None = None,
    companies: dict[str, list[str]] | None = None,
) -> dict:
    """Return discover jobs."""

    config = {
        "role": role,
        "location": location,
        "ignore_location": bool(ignore_location),
        "remote_only": bool(remote_only),
        "sources": list(sources or ["greenhouse", "lever", "ashby"]),
        "max_results_per_source": max_results_per_source or 5,
        "max_jobs": max_jobs or 1,
        "max_company_attempts_per_source": max_company_attempts_per_source,
        "companies": companies or {},
    }
    source_companies = build_source_companies(config)
    jobs = discover_jobs_with_config(
        config["role"].lower(),
        config["location"].lower(),
        config["ignore_location"],
        config["remote_only"],
        config["sources"],
        config["max_results_per_source"],
        config["max_jobs"],
        source_companies,
        config["max_company_attempts_per_source"],
    )
    lines = [f"Discovered {len(jobs)} job(s)."]
    for job in jobs:
        lines.append(
            f"- {job.get('company', '')} | {job.get('title', '')} | {job.get('location', '')} | {job.get('source', '')}"
        )
    return {
        "result": {
            "config": config,
            "jobs": jobs,
        },
        "metadata": {
            "display_text": "\n".join(lines),
            "artifacts": [],
        },
    }


def clean_job_description(raw_job_text: str) -> dict:
    """Return clean job description."""

    cleaned = clean_job_description_text(raw_job_text)
    return {
        "result": {
            "cleaned_text": cleaned,
        },
        "metadata": {
            "display_text": cleaned,
            "artifacts": [],
        },
    }


def generate_application_materials(cleaned_job_text: str) -> dict:
    """Return generate application materials."""

    materials = generate_application_materials_for_job(cleaned_job_text)
    info = str(materials.get("info", "")).strip()
    return {
        "result": materials,
        "metadata": {
            "display_text": info,
            "artifacts": [],
        },
    }


def build_application_notes_from_job_description(cleaned_job_text: str) -> dict:
    """Build application notes from job description."""

    info = build_application_notes_text(cleaned_job_text)
    return {
        "result": {
            "info": info,
        },
        "metadata": {
            "display_text": info,
            "artifacts": [],
        },
    }


def write_json_file(destination_path: str, payload: Any) -> dict:
    """Return write json file."""

    destination = Path(destination_path)
    save_json(destination, payload)
    return {
        "result": {
            "destination_path": str(destination),
            "payload": payload,
        },
        "metadata": {
            "display_text": f"Wrote JSON file to `{destination}`.",
            "artifacts": [
                {
                    "name": destination.name,
                    "kind": "file",
                    "path": str(destination),
                    "metadata": {
                        "content_type": "json",
                    },
                }
            ],
        },
    }


def write_search_results(
    destination_path: str,
    query: str,
    results: list[dict[str, Any]],
) -> dict:
    """Return write search results."""

    destination = Path(destination_path)
    content = _render_search_results(query, results)
    save_text(destination, content if content.endswith(
        "\n") else content + "\n")
    return {
        "result": {
            "destination_path": str(destination),
            "query": query,
            "results": results,
        },
        "metadata": {
            "display_text": f"Wrote search results to `{destination}`.",
            "artifacts": [
                {
                    "name": destination.name,
                    "kind": "file",
                    "path": str(destination),
                    "metadata": {
                        "content_type": "search_results",
                    },
                }
            ],
        },
    }


def write_generated_documents(
    output_dir: str,
    documents: list[dict[str, Any]],
) -> dict:
    """Write a generated set of text documents under one output directory."""

    destination_dir = Path(output_dir)
    artifacts = []
    written_documents = []

    if not documents:
        raise ValueError("At least one generated document is required.")

    used_filenames: set[str] = set()
    for index, document in enumerate(documents, start=1):
        filename = _safe_generated_filename(
            str(document.get("filename", "")),
            index=index,
        )
        filename = _dedupe_filename(filename, used_filenames)
        used_filenames.add(filename)
        content = str(document.get("content", "")).strip()
        if not content:
            raise ValueError(f"Generated document `{filename}` has empty content.")

        destination = destination_dir / filename
        save_text(destination, content if content.endswith("\n") else content + "\n")
        written_documents.append(
            {
                "filename": filename,
                "destination_path": str(destination),
                "characters_written": len(content),
            }
        )
        artifacts.append(
            {
                "name": filename,
                "kind": "file",
                "path": str(destination),
                "metadata": {
                    "content_type": "generated_document",
                    "characters_written": len(content),
                },
            }
        )

    return {
        "result": {
            "output_dir": str(destination_dir),
            "documents": written_documents,
        },
        "metadata": {
            "display_text": (
                f"Wrote {len(written_documents)} generated document(s) "
                f"to `{destination_dir}`."
            ),
            "artifacts": artifacts,
        },
    }


def _safe_generated_filename(filename: str, *, index: int) -> str:
    """Return a safe basename for a generated text document."""

    name = Path(filename.strip()).name
    if not name or name in {".", ".."}:
        name = f"document-{index}.txt"

    path_name = Path(name)
    suffix = path_name.suffix.lower()
    if suffix not in {".txt", ".md", ".markdown"}:
        suffix = ".txt"

    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path_name.stem).strip(".-_")
    if not safe_stem:
        safe_stem = f"document-{index}"
    return f"{safe_stem[:80]}{suffix}"


def _dedupe_filename(filename: str, used_filenames: set[str]) -> str:
    """Return a filename that is unique within the generated document set."""

    if filename not in used_filenames:
        return filename

    path_name = Path(filename)
    stem = path_name.stem
    suffix = path_name.suffix
    counter = 2
    while True:
        candidate = f"{stem}-{counter}{suffix}"
        if candidate not in used_filenames:
            return candidate
        counter += 1


def _render_search_results(query: str, results: list[dict[str, Any]]) -> str:
    """Return render search results."""

    lines = [f"Found {len(results)} result(s) for: {query}", ""]
    for index, result in enumerate(results, start=1):
        title = str(result.get("title", "")).strip() or "(untitled)"
        url = str(result.get("url", "")).strip()
        lines.append(f"{index}. {title}")
        if url:
            lines.append(f"   URL: {url}")
    return "\n".join(lines).strip()
