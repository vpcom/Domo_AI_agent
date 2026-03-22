import json

from integrations.ollama_client import call_llm
from tools.job.models import JobState
from tools.job.prompts import build_generation_prompt


def parse_json_response(raw_text: str) -> dict:
    raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(raw_text, strict=False)
    except json.JSONDecodeError:
        pass

    cleaned = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start: end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        try:
            return json.loads(candidate, strict=False)

        except json.JSONDecodeError as error:
            print("\n--- RAW MODEL RESPONSE ---\n")
            print(raw_text)
            print("\n--- END RAW MODEL RESPONSE ---\n")

            raise ValueError(
                f"Could not parse model output as JSON: {error}"
            ) from error

    print("\n--- RAW MODEL RESPONSE ---\n")
    print(raw_text)
    print("\n--- END RAW MODEL RESPONSE ---\n")

    raise ValueError("No JSON object found in model response.")


def build_application_notes_from_job_description(cleaned_job_text: str) -> str:
    return generate_application_materials(cleaned_job_text)["notes"]


def _normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _normalize_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        lines = [line.strip(" -\t") for line in value.splitlines()]
        return [line for line in lines if line]
    return [str(value).strip()]


def generate_application_materials(cleaned_job_text: str) -> dict:
    print(f"[notes] generating materials from cleaned characters={len(cleaned_job_text)}")
    generation_prompt = build_generation_prompt(cleaned_job_text)
    raw_response = call_llm(generation_prompt)
    data = parse_json_response(raw_response)

    summary = _normalize_text(data.get("summary", ""))
    skills = _normalize_list(data.get("skills", []))
    cv_summary = _normalize_text(data.get("cv_summary", ""))
    key_strengths = _normalize_list(data.get("key_strengths", []))
    cv_base_texts = _normalize_text(data.get("cv_base_texts", ""))
    cover_letter = _normalize_text(data.get("cover_letter", ""))

    skills_block = "\n".join(f"- {skill}" for skill in skills)
    strengths_block = "\n".join(f"- {strength}" for strength in key_strengths)

    notes = f"""SUMMARY
{summary}

KEY SKILLS
{skills_block}

CV SUMMARY
{cv_summary}

KEY STRENGTHS
{strengths_block}

CV BASE TEXTS
{cv_base_texts}

COVER LETTER
{cover_letter}
"""

    sample_cv = f"""CV SUMMARY
{cv_summary}

CORE SKILLS
{skills_block}

KEY STRENGTHS
{strengths_block}

CV BASE TEXTS
{cv_base_texts}
"""

    return {
        "notes": notes,
        "summary": summary,
        "skills": skills_block,
        "sample_cv": sample_cv,
    }


# AGENT TOOL ENTRYPOINT
def run(state: JobState) -> None:
    print(f"[notes] reading cleaned_file={state.cleaned_file}")
    cleaned_text = state.cleaned_file.read_text(encoding="utf-8")
    materials = generate_application_materials(cleaned_text)

    state.notes_file.parent.mkdir(parents=True, exist_ok=True)
    state.notes_file.write_text(materials["notes"], encoding="utf-8")
    state.summary_file.write_text(materials["summary"], encoding="utf-8")
    state.skills_file.write_text(materials["skills"], encoding="utf-8")
    state.cv_file.write_text(materials["sample_cv"], encoding="utf-8")
    print(f"[notes] wrote notes_file={state.notes_file}")
    print(f"[notes] wrote summary_file={state.summary_file}")
    print(f"[notes] wrote skills_file={state.skills_file}")
    print(f"[notes] wrote cv_file={state.cv_file}")
