from pathlib import Path
import json
import re

from assistant.config import get_paths, is_debug_enabled
from integrations.ollama_client import call_llm
from tools.job.job_folder_resolution import find_best_matching_job_folder
from tools.job.local_job_inputs import (
    CLEANED_DESCRIPTION_FILE,
    LEGACY_CLEANED_DESCRIPTION_FILE,
    RAW_DESCRIPTION_FILE,
    ensure_local_job_inputs,
    find_cleaned_job_description_file,
)
from tools.job.pdf_utils import extract_pdf_text


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_first_json_object(text: str) -> dict:
    cleaned = _strip_code_fences(text)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = None
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(cleaned):
        if start is None:
            if char == "{":
                start = index
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:index + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    break

    return {}


def _coerce_score(value) -> float | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = _normalize_whitespace(str(value))
        if not text:
            return None

        direct_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*10\b", text)
        if direct_match:
            score = float(direct_match.group(1))
        else:
            try:
                score = float(text)
            except ValueError:
                return None

    if 0.0 <= score <= 10.0:
        return score
    return None


def _extract_score_from_text(text: str) -> float | None:
    cleaned = _normalize_whitespace(_strip_code_fences(text))
    patterns = [
        r"([0-9]+(?:\.[0-9]+)?)\s*/\s*10\b",
        r"([0-9]+(?:\.[0-9]+)?)\s+out of\s+10\b",
        r"\bscore(?:d)?\b[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)\b",
        r"\brating\b[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            return _coerce_score(match.group(1))
    return None


def _normalize_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_normalize_whitespace(str(item)) for item in value if str(item).strip()]
    text = _normalize_whitespace(str(value))
    return [text] if text else []


def _extract_fit_summary(parsed: dict, raw_response: str) -> str:
    summary_keys = [
        "fit_summary",
        "summary",
        "reason",
        "rationale",
        "analysis",
        "explanation",
    ]
    for key in summary_keys:
        value = _normalize_whitespace(str(parsed.get(key, "")))
        if value:
            return value

    strengths = _normalize_list(parsed.get("strengths"))
    weaknesses = _normalize_list(parsed.get("weaknesses"))
    if strengths or weaknesses:
        parts = []
        if strengths:
            parts.append(f"Strengths: {', '.join(strengths)}.")
        if weaknesses:
            parts.append(f"Weaknesses: {', '.join(weaknesses)}.")
        return " ".join(parts)

    return _normalize_whitespace(_strip_code_fences(raw_response))


def _repair_llm_evaluation(raw_response: str) -> tuple[dict, str | None]:
    repair_prompt = f"""You are extracting structured data from a CV-to-job evaluation.

Convert the evaluation below into JSON. If the evaluation does not contain an explicit score,
infer a reasonable score from 1 to 10 based on the overall tone and evidence.

Return ONLY valid JSON:
{{
  "score": number (1-10),
  "strengths": ["..."],
  "weaknesses": ["..."],
  "fit_summary": "short explanation"
}}

Evaluation:
{raw_response}
"""

    try:
        repaired_response = call_llm(repair_prompt)
    except Exception:
        return {}, None

    return _extract_first_json_object(repaired_response), repaired_response


def _interpret_llm_evaluation(response: str) -> dict:
    parsed = _extract_first_json_object(response)

    score = _coerce_score(parsed.get("score"))
    if score is None:
        score = _extract_score_from_text(response)

    fit_summary = _extract_fit_summary(parsed, response)
    repair_response = None
    repair_parsed = None

    if score is None:
        repair_parsed, repair_response = _repair_llm_evaluation(response)
        repaired_score = _coerce_score(repair_parsed.get("score"))
        if repaired_score is None and repair_response is not None:
            repaired_score = _extract_score_from_text(repair_response)
        if repaired_score is not None:
            score = repaired_score

        repaired_summary = _extract_fit_summary(repair_parsed, response)
        if repaired_summary:
            fit_summary = repaired_summary

    return {
        "score": score if score is not None else 0.0,
        "fit_summary": fit_summary,
        "parsed": parsed,
        "repair_response": repair_response,
        "repair_parsed": repair_parsed,
    }

def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()

    project_root = Path(__file__).resolve().parents[2]
    configured_paths = get_paths()
    jobs_root = configured_paths["jobs_root"]
    outputs_root = configured_paths["outputs_root"]
    data_root = configured_paths["data_root"]
    cvs_root = configured_paths["cvs_root"]

    # If the provided relative path exists directly under the project root, use it.
    direct = project_root / path
    if direct.exists():
        return direct.resolve()

    # Common data roots where job folders live
    candidates = [
        jobs_root / path,
        outputs_root / path,
        data_root / path,
        cvs_root / path,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    # Fuzzy search: look for directories under data/jobs whose name contains the provided name
    try:
        name = path.name
        fuzzy_match = find_best_matching_job_folder(name, jobs_root)
        if fuzzy_match is not None:
            return fuzzy_match.resolve()
    except Exception:
        pass

    # Fallback: return the path relative to project root (may not exist)
    return (project_root / path).resolve()


def match_cv(job_folder: str, cvs_folder: str | None = None):
    job_path = Path(job_folder)
    default_cvs_path = get_paths()["cvs_root"]
    cvs_path = Path(cvs_folder) if cvs_folder else default_cvs_path

    job_path = _resolve_path(job_path)
    cvs_path = _resolve_path(cvs_path)

    # Informational: show which folder was resolved (helps when callers pass just a folder name)
    yield f"Resolved job folder: {job_path}\n"

    # Load job text first so we can extract a short job title for logging

    ensure_local_job_inputs(job_path)

    cleaned_file = find_cleaned_job_description_file(job_path)
    raw_file = job_path / RAW_DESCRIPTION_FILE

    if cleaned_file is not None:
        job_text = cleaned_file.read_text(encoding="utf-8")
    elif raw_file.exists():
        job_text = raw_file.read_text(encoding="utf-8")
    else:
        yield (
            "Error: missing job description file. Expected "
            f"`{LEGACY_CLEANED_DESCRIPTION_FILE}`, `{CLEANED_DESCRIPTION_FILE}`, "
            f"`{RAW_DESCRIPTION_FILE}`, `job_description.txt`, `job description.txt`, "
            "`job_description.pdf`, or `job description.pdf`.\n"
        )
        return

    # Derive a short job title from the first non-empty line of the job text, or fallback to folder name
    job_title = None
    try:
        for line in job_text.splitlines():
            line = line.strip()
            if line:
                job_title = line
                break
    except Exception:
        job_title = None

    if not job_title:
        job_title = job_path.name

    yield f"Loading job description: {job_title}\n"
    if is_debug_enabled():
        yield "DEBUG: debug mode enabled for match_cv (extra details will be shown)\n"

    yield "Loading CVs...\n"

    if not cvs_path.exists() or not cvs_path.is_dir():
        yield f"Error: CV folder not found: {cvs_path}\n"
        return

    pdf_files = sorted(
        file for file in cvs_path.iterdir() if file.is_file() and file.suffix.lower() == ".pdf"
    )
    if not pdf_files:
        yield f"Error: no PDF CVs found in {cvs_path}\n"
        return

    results = []

    for file in pdf_files:
        # Announce the CV we're evaluating
        yield f"Evaluating CV: {file.name}\n"

        score = 0.0
        fit_summary = ""
        raw_response = None
        parsed = None
        repair_response = None
        repair_parsed = None

        try:
            cv_text = extract_pdf_text(file)
            response = call_llm(
                f"""Compare this CV with this job description.

Job description:
{job_text}

CV:
{cv_text}

Return ONLY valid JSON. Do not add markdown, commentary, or code fences.
{{
  "score": number (1-10),
  "strengths": ["..."],
  "weaknesses": ["..."],
  "fit_summary": "short explanation"
}}"""
            )
            raw_response = response
            interpreted = _interpret_llm_evaluation(response)
            score = interpreted["score"]
            fit_summary = interpreted["fit_summary"]
            parsed = interpreted["parsed"]
            repair_response = interpreted["repair_response"]
            repair_parsed = interpreted["repair_parsed"]
        except json.JSONDecodeError:
            score = 0.0
            fit_summary = "LLM response could not be parsed as JSON."
        except Exception as exc:
            score = 0.0
            fit_summary = f"Evaluation failed: {exc}"

        results.append({
            "file": file.name,
            "score": score,
            "summary": fit_summary,
        })

        # Emit the numeric score and a short reason so it's visible in the stream
        yield f"  Score: {score:.2f}\n"
        if fit_summary:
            yield f"  Reason: {fit_summary}\n"
        else:
            yield f"  Reason: (no summary provided)\n"

        # Optional debug output: raw LLM response and parsed object
        if is_debug_enabled():
            if raw_response is not None:
                # Truncate long raw responses for readability
                snippet = raw_response[:1000] + \
                    ("..." if len(raw_response) > 1000 else "")
                yield f"  DEBUG: raw_llm_response: {snippet}\n"
            if parsed is not None:
                try:
                    parsed_snip = json.dumps(parsed, indent=2)[:1000]
                    yield f"  DEBUG: parsed_json: {parsed_snip}\n"
                except Exception:
                    yield f"  DEBUG: parsed_json unavailable\n"
            if repair_response is not None:
                snippet = repair_response[:1000] + \
                    ("..." if len(repair_response) > 1000 else "")
                yield f"  DEBUG: repaired_llm_response: {snippet}\n"
            if repair_parsed is not None:
                try:
                    repaired_snip = json.dumps(repair_parsed, indent=2)[:1000]
                    yield f"  DEBUG: repaired_parsed_json: {repaired_snip}\n"
                except Exception:
                    yield f"  DEBUG: repaired_parsed_json unavailable\n"

    if not results:
        yield "Error: no CV analysis results were produced.\n"
        return

    best = max(results, key=lambda x: x["score"])
    yield f"Best CV selected: {best['file']} (score: {best['score']})\n"

    best_cv_path = cvs_path / best["file"]
    try:
        best_cv_text = extract_pdf_text(best_cv_path)
    except Exception as exc:
        best_cv_text = f"[Error extracting PDF text: {exc}]"
    (job_path / "best_cv.txt").write_text(best_cv_text, encoding="utf-8")

    analysis = {
        "best_cv": best["file"],
        "results": results,
    }
    (job_path / "cv_match_analysis.json").write_text(
        json.dumps(analysis, indent=2),
        encoding="utf-8",
    )

    summary_text = f"Best CV: {best['file']} (score: {best['score']})\n\n"
    for r in results:
        summary_text += f"{r['file']} -> score: {r['score']}\n"
        summary_text += f"{r['summary']}\n\n"

    (job_path / "cv_match_summary.txt").write_text(summary_text, encoding="utf-8")

    yield "CV matching completed.\n"
