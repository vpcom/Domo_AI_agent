from integrations.ollama_client import call_llm
from tools.job.models import JobState
from tools.job.prompts import build_cleaning_prompt
from tools.job.text_normalization import normalize_job_posting_text


def clean_job_description(raw_job_text: str) -> str:
    normalized_raw_text = normalize_job_posting_text(raw_job_text)
    cleaning_prompt = build_cleaning_prompt(normalized_raw_text)
    cleaned = call_llm(cleaning_prompt).strip()
    return normalize_job_posting_text(cleaned)


def run(state: JobState) -> None:
    print(f"[clean] reading raw_file={state.raw_file}")
    raw_text = state.raw_file.read_text(encoding="utf-8")
    print(f"[clean] raw characters={len(raw_text)}")

    cleaned = clean_job_description(raw_text)
    print(f"[clean] cleaned characters={len(cleaned)}")

    state.cleaned_file.parent.mkdir(parents=True, exist_ok=True)
    state.cleaned_file.write_text(cleaned, encoding="utf-8")
    print(f"[clean] wrote cleaned_file={state.cleaned_file}")
