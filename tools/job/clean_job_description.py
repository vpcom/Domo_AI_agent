from integrations.ollama_client import call_llm
from tools.job.models import JobState
from tools.job.prompts import build_cleaning_prompt


def clean_job_description(raw_job_text: str) -> str:
    cleaning_prompt = build_cleaning_prompt(raw_job_text)
    return call_llm(cleaning_prompt).strip()


def run(state: JobState) -> None:
    print(f"[clean] reading raw_file={state.raw_file}")
    raw_text = state.raw_file.read_text(encoding="utf-8")
    print(f"[clean] raw characters={len(raw_text)}")

    cleaned = clean_job_description(raw_text)
    print(f"[clean] cleaned characters={len(cleaned)}")

    state.cleaned_file.parent.mkdir(parents=True, exist_ok=True)
    state.cleaned_file.write_text(cleaned, encoding="utf-8")
    print(f"[clean] wrote cleaned_file={state.cleaned_file}")
