from datetime import date, datetime
from pathlib import Path
import re


def normalize_search_text(value: str) -> str:
    lowered = value.lower().replace("/", " ").replace("-", " ")
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def parse_folder_date(folder_name: str) -> date | None:
    match = re.match(r"(?P<date>\d{8})\b", folder_name)
    if not match:
        return None

    try:
        return datetime.strptime(match.group("date"), "%Y%m%d").date()
    except ValueError:
        return None


def find_best_matching_job_folder(
    folder_hint: str,
    jobs_root: Path,
    today: date | None = None,
) -> Path | None:
    if not jobs_root.exists() or not jobs_root.is_dir():
        return None

    normalized_hint = normalize_search_text(folder_hint)
    if not normalized_hint:
        return None

    today = today or date.today()
    hint_tokens = normalized_hint.split()
    candidates: list[tuple[int, int, int, str, Path]] = []

    for folder in jobs_root.iterdir():
        if not folder.is_dir():
            continue

        normalized_name = normalize_search_text(folder.name)
        if not normalized_name:
            continue

        exact_phrase = normalized_hint in normalized_name
        token_matches = sum(1 for token in hint_tokens if token in normalized_name)
        if not exact_phrase and token_matches == 0:
            continue

        if exact_phrase:
            match_quality = 3
        elif token_matches == len(hint_tokens):
            match_quality = 2
        else:
            match_quality = 1

        folder_date = parse_folder_date(folder.name)
        date_distance = abs((folder_date - today).days) if folder_date else 10**6
        candidates.append(
            (
                -match_quality,
                date_distance,
                -token_matches,
                folder.name.lower(),
                folder,
            )
        )

    if not candidates:
        return None

    candidates.sort()
    return candidates[0][-1]


def resolve_job_folder_hint(
    folder_hint: str,
    project_root: Path,
    jobs_root: Path,
    today: date | None = None,
) -> Path:
    candidate = Path(folder_hint).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    direct_candidate = (project_root / candidate).resolve()
    if direct_candidate.exists():
        return direct_candidate

    jobs_candidate = (jobs_root / candidate).resolve()
    if jobs_candidate.exists():
        return jobs_candidate

    if candidate.parent == Path("."):
        fuzzy_match = find_best_matching_job_folder(candidate.name, jobs_root, today=today)
        if fuzzy_match is not None:
            return fuzzy_match.resolve()

    return direct_candidate
