"""Planning helpers for the deterministic agent runtime."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from assistant.capabilities import (
    build_capability_catalog_text,
    build_forbidden_actions_text,
)
from assistant.config import get_display_path, get_paths
from assistant.policy import (
    build_timestamped_output_root,
    build_tool_args,
    is_reference_string,
    missing_required_arguments,
    STEP_REFERENCE_PATTERN,
    validate_and_normalize_tool_inputs,
    validate_reference_string,
)
from assistant.registry import LLM_TASKS, TOOLS
from assistant.schemas import PlanDraft, PlanStepDraft
from integrations.ollama_client import call_llm


PATHS = get_paths()
OUTPUTS_ROOT_DISPLAY = get_display_path(PATHS["outputs_root"])
JOBS_ROOT_DISPLAY = get_display_path(PATHS["jobs_root"])
DOCUMENTS_ROOT_DISPLAY = get_display_path(PATHS["documents_root"])
CVS_ROOT_DISPLAY = get_display_path(PATHS["cvs_root"])
EMBEDDED_SOURCE_DISALLOWED_TOOLS = {
    "inspect_path",
    "list_directory",
    "read_text_file",
    "read_json_file",
    "read_pdf_text",
    "read_documents",
    "resolve_job_folder_hint",
    "resolve_local_job_inputs",
    "read_job_metadata",
}
LOCAL_PATH_HINT_PATTERN = re.compile(
    r"(?<!\w)(?:/[\w./-]+|\.{1,2}/[\w./-]+|data/[\w./-]+)"
)


class PlanningError(ValueError):
    def __init__(self, message: str, *, trace: list[dict[str, str]]) -> None:
        super().__init__(message)
        self.trace = trace


def plan_goal(user_input: str) -> tuple[PlanDraft, Path, list[dict[str, str]]]:
    """Build, validate, and return a plan draft for the current user goal."""

    prompt = build_planner_prompt(user_input)
    trace: list[dict[str, str]] = []
    validation_error: str | None = None

    for _ in range(2):
        trace.append(
            {
                "message": "Planner prompt prepared.",
                "detail": "",
                "expanded_text": prompt,
            }
        )
        response = call_llm(prompt)
        trace.append(
            {
                "message": "Planner raw response received.",
                "detail": "",
                "expanded_text": response,
            }
        )

        try:
            draft = _parse_plan_response(response)
            output_root = build_timestamped_output_root()
            validated_steps = validate_plan_draft(
                draft,
                output_root=output_root,
                user_input=user_input,
            )
        except ValueError as exc:
            validation_error = str(exc)
            trace.append(
                {
                    "message": "Planner validation failed.",
                    "detail": validation_error,
                    "expanded_text": "",
                }
            )
            prompt = build_repair_prompt(
                user_input, response, validation_error)
            continue

        return (
            PlanDraft(
                normalized_goal=draft.normalized_goal.strip(),
                confidence=draft.confidence,
                plan=validated_steps,
            ),
            output_root,
            trace,
        )

    raise PlanningError(
        validation_error or "Planner did not return a valid plan.",
        trace=trace,
    )


def validate_plan_draft(
    draft: PlanDraft,
    *,
    output_root: Path,
    user_input: str | None = None,
) -> list[PlanStepDraft]:
    """Validate plan draft."""

    if not draft.normalized_goal.strip():
        raise ValueError("Planner must provide a non-empty normalized goal.")
    if not draft.plan:
        raise ValueError("Planner must provide at least one step.")
    _validate_embedded_source_plan(draft, user_input=user_input)

    validated_steps: list[PlanStepDraft] = []
    for index, step in enumerate(draft.plan):
        if step.step_id != index:
            raise ValueError("Step ids must be sequential and start at 0.")
        if not step.description.strip():
            raise ValueError(f"Step {step.step_id} is missing a description.")
        if not step.tool_name.strip():
            raise ValueError(f"Step {step.step_id} is missing a tool_name.")

        _validate_step_references(step.inputs, current_step_id=step.step_id)
        _validate_known_step_reference_paths(step.inputs, validated_steps)

        if step.type == "tool":
            if step.tool_name not in TOOLS or not TOOLS[step.tool_name].allowed:
                raise ValueError(
                    f"Unknown or disallowed tool: {step.tool_name}")
            _validate_tool_input_shapes(step.tool_name, step.inputs, validated_steps)
            missing = missing_required_arguments(step.tool_name, step.inputs)
            if missing:
                raise ValueError(
                    f"Tool step `{step.tool_name}` is missing required inputs: {', '.join(missing)}"
                )
            normalized_inputs = validate_and_normalize_tool_inputs(
                step.tool_name,
                step.inputs,
                output_root=output_root,
                allow_references=True,
            )
        elif step.type == "llm":
            if step.tool_name not in LLM_TASKS or not LLM_TASKS[step.tool_name].allowed:
                raise ValueError(
                    f"Unknown or disallowed llm task: {step.tool_name}")
            missing = missing_required_arguments(
                step.tool_name,
                step.inputs,
                step_type="llm",
            )
            if missing:
                raise ValueError(
                    f"LLM step `{step.tool_name}` is missing required inputs: {', '.join(missing)}"
                )
            _validate_llm_input_shapes(step.tool_name, step.inputs)
            if _contains_reference(step.inputs):
                normalized_inputs = dict(step.inputs)
            else:
                normalized_inputs = build_tool_args(
                    step.tool_name,
                    step.inputs,
                    step_type="llm",
                ).model_dump(exclude_none=True)
        else:
            raise ValueError(f"Unsupported step type: {step.type}")

        validated_steps.append(
            PlanStepDraft(
                step_id=step.step_id,
                description=step.description.strip(),
                type=step.type,
                tool_name=step.tool_name.strip(),
                inputs=normalized_inputs,
            )
        )

    return validated_steps


def build_planner_prompt(user_input: str) -> str:
    """Build planner prompt."""

    capability_catalog = build_capability_catalog_text()
    capability_contracts = _build_capability_contract_text()
    forbidden_actions = build_forbidden_actions_text()
    input_source_guidance = _build_input_source_guidance(user_input)
    return f"""
You are planning work for a deterministic local agent.
Return ONLY valid JSON with this exact top-level shape:
{{
  "normalized_goal": "short normalized goal",
  "confidence": 0.0,
  "plan": [
    {{
      "step_id": 0,
      "description": "short step description",
      "type": "tool" | "llm",
      "tool_name": "registry key",
      "inputs": {{}}
    }}
  ]
}}

Rules:
- Build the full plan before execution.
- Never execute tools yourself.
- Never add fields outside the schema.
- Every plan item must include exactly these required keys:
  `step_id`, `description`, `type`, `tool_name`, and `inputs`.
- `type` must be exactly `"tool"` or `"llm"`.
- `tool_name` must be a separate field and must contain the exact capability registry key.
- Never encode a capability name inside `type`.
- Never omit `tool_name`.
- Never omit `inputs`; use an empty object only for capabilities with no required inputs.
- Set `confidence` between 0.0 and 1.0.
- Use high confidence only when the request is clear and all required inputs are fully specified.
- For direct questions, explanations, and other read-only requests, prefer a plan that can execute immediately.
- For writing new output files, use high confidence only when the output path and content inputs are explicit.
- `write_document` always requires `inputs.destination_path` and `inputs.content`.
- If the user asks for multiple generated files whose filenames or contents depend
  on earlier results, use `generate_document_set` followed by
  `write_generated_documents`.
- Never plan a delete, remove, erase, move, rename, or modify-existing-file action.
- Never plan account usage, sign-in flows, or authenticated account actions.
- If the user requests a forbidden action, produce one `llm` step using `answer_question` that briefly refuses and points to allowed alternatives.
- The agent has access only to these registered capabilities:
{capability_catalog}
- Exact capability contracts you must follow:
{capability_contracts}
- These actions are forbidden:
{forbidden_actions}
- Input-source policy:
{input_source_guidance}
- For direct questions or general explanations, use one `llm` step with `tool_name="answer_question"` and `inputs={{"question": "@goal:user_input"}}`.
- For document summarization, use:
  1. `read_documents` or atomic file-reading tools
  2. `summarize_text` with `documents` referencing the previous step result.
- For document evaluation or ranking, use:
  1. `read_documents` or atomic file-reading tools
  2. `evaluate_text` with `documents` referencing the previous step result.
- For CV matching, use:
  1. `read_documents` on the job folder
  2. `read_documents` on the CV folder
  3. `rank_cvs` using both previous outputs
- For web lookups, use `search_web` directly. Do not add an LLM step to parse,
  classify, explain, or rewrite the search query first.
- When writing search results, prefer `write_search_results` instead of writing through `search_web`.
- If the user already pasted source text into the prompt, do not create a synthetic read step such as `read_text`.
- If the user already pasted source text into the prompt, do not start with `read_text_file`, `read_documents`, or another local-file read unless the user explicitly provided a real local path.
- For output files, write only under `{OUTPUTS_ROOT_DISPLAY}` and never overwrite existing files.
- Existing job folders should usually be under `{JOBS_ROOT_DISPLAY}`.
- Existing document folders should usually be under `{DOCUMENTS_ROOT_DISPLAY}`.
- CV folders should usually be under `{CVS_ROOT_DISPLAY}`.
- Allowed references are only:
  - `@goal:user_input`
  - `@goal:normalized_goal`
  - `@step:<id>.output.result.<path>`
  - `@memory:<key>`
- Step references must only point to earlier steps.
- Step references do not support list indexing, wildcards, or projections such as
  `results[0]`, `results[*]`, or `.title`. Reference the whole list and pass it
  to an LLM task when extraction is needed.
- To answer a web lookup from search results, use `summarize_text` with
  `documents="@step:<search_step>.output.result.results"` and targeted
  instructions.
- To save a generated web-search summary, add one `write_document` step with
  `content="@step:<summary_step>.output.result.summary"`.
- To create one generated file per search result, use:
  1. `search_web`
  2. `generate_document_set` with `source_documents="@step:<search_step>.output.result.results"`
  3. `write_generated_documents` with `documents="@step:<document_set_step>.output.result.documents"`
- Valid web-search answer example:
  {{
    "step_id": 0,
    "description": "Search the web",
    "type": "tool",
    "tool_name": "search_web",
    "inputs": {{
      "query": "best Bruce Lee movies"
    }}
  }},
  {{
    "step_id": 1,
    "description": "Summarize the search results",
    "type": "llm",
    "tool_name": "summarize_text",
    "inputs": {{
      "documents": "@step:0.output.result.results",
      "instructions": "List and summarize the top 5 Bruce Lee movies."
    }}
  }},
  {{
    "step_id": 2,
    "description": "Write the summarized results",
    "type": "tool",
    "tool_name": "write_document",
    "inputs": {{
      "destination_path": "data/outputs/bruce_lee_movie_summaries.txt",
      "content": "@step:1.output.result.summary"
    }}
  }}
- Valid generated multi-file example:
  {{
    "step_id": 0,
    "description": "Search the web",
    "type": "tool",
    "tool_name": "search_web",
    "inputs": {{
      "query": "best Bruce Lee movies"
    }}
  }},
  {{
    "step_id": 1,
    "description": "Generate one summary document per movie",
    "type": "llm",
    "tool_name": "generate_document_set",
    "inputs": {{
      "source_documents": "@step:0.output.result.results",
      "instructions": "Create 5 text documents, one per Bruce Lee movie. Each filename must be a safe descriptive .txt basename and each content must summarize that movie."
    }}
  }},
  {{
    "step_id": 2,
    "description": "Write the generated movie summary files",
    "type": "tool",
    "tool_name": "write_generated_documents",
    "inputs": {{
      "output_dir": "data/outputs/bruce_lee_movie_summaries",
      "documents": "@step:1.output.result.documents"
    }}
  }}
- Valid embedded-text example:
  {{
    "step_id": 0,
    "description": "Clean the pasted job ad",
    "type": "tool",
    "tool_name": "clean_job_description",
    "inputs": {{
      "raw_job_text": "@goal:user_input"
    }}
  }},
  {{
    "step_id": 1,
    "description": "Build the application notes",
    "type": "tool",
    "tool_name": "build_application_notes_from_job_description",
    "inputs": {{
      "cleaned_job_text": "@step:0.output.result.cleaned_text"
    }}
  }},
  {{
    "step_id": 2,
    "description": "Write the generated note",
    "type": "tool",
    "tool_name": "write_document",
    "inputs": {{
      "destination_path": "data/outputs/example/note.md",
      "content": "@step:1.output.result.info"
    }}
  }}
- Invalid example you must never output:
  {{
    "step_id": 1,
    "description": "Read the pasted text",
    "type": "tool",
    "tool_name": "read_text",
    "inputs": {{
      "text": "@goal:user_input"
    }}
  }}

LATEST USER REQUEST
{user_input}
"""


def build_repair_prompt(
    user_input: str,
    previous_response: str,
    validation_error: str,
) -> str:
    """Build repair prompt."""

    capability_catalog = build_capability_catalog_text()
    capability_contracts = _build_capability_contract_text()
    input_source_guidance = _build_input_source_guidance(user_input)
    return f"""
Your previous plan was invalid.
Validation error: {validation_error}

Return ONLY corrected JSON using the required schema.
Do not add commentary.
Remember:
- Every plan item must include `step_id`, `description`, `type`, `tool_name`,
  and `inputs`.
- `type` must be exactly `"tool"` or `"llm"`.
- `tool_name` must hold the exact capability name.
- Never omit `tool_name`; for summarization use `tool_name="summarize_text"`.
- Never omit required `inputs` for the chosen capability.
- `write_document` must always include `inputs.destination_path` and
  `inputs.content`.
- If a requested write depends on generated results, set `content` to the
  previous LLM output reference, such as `@step:1.output.result.summary`.
- If the user asks for multiple generated files whose filenames or contents depend
  on earlier results, use `generate_document_set` then
  `write_generated_documents`.
- Use only exact capability names from this registry list:
{capability_catalog}
- Follow these exact capability contracts:
{capability_contracts}
- Do not use forms like `"write_document_as_tool"` or `"write_search_results_as_tool"`.
- Do not invent synthetic read steps such as `"read_text"` for prompt content.
- If the user already pasted source text, pass it directly with `@goal:user_input` into the next real step.
- Do not use list indexing, wildcards, or projections in step references, such as
  `results[0]`, `results[*]`, or `.title`.
- For extracting an answer from web search results, pass
  `@step:<search_step>.output.result.results` as `summarize_text.documents`.
Input-source policy:
{input_source_guidance}

LATEST USER REQUEST
{user_input}

PREVIOUS INVALID RESPONSE
{previous_response}
"""


def _parse_plan_response(response: str) -> PlanDraft:
    """Parse and normalize a planner response before strict validation."""

    try:
        parsed = json.loads(response)
        if not isinstance(parsed, dict):
            raise ValueError("Planner response was not a JSON object.")
    except Exception:
        parsed = _extract_json_object(response)
        if parsed is None:
            raise ValueError("Planner response was not valid JSON.")

    try:
        return PlanDraft.model_validate(_normalize_plan_payload(parsed))
    except Exception as exc:
        raise ValueError(
            f"Planner response did not match the plan schema: {exc}"
        ) from exc


def _normalize_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize common planner shape mistakes before schema validation."""

    normalized = dict(payload)
    raw_plan = normalized.get("plan")
    if isinstance(raw_plan, list):
        normalized["plan"] = [
            _normalize_step_payload(step)
            if isinstance(step, dict)
            else step
            for step in raw_plan
        ]
    return normalized


def _normalize_step_payload(step: dict[str, Any]) -> dict[str, Any]:
    """Repair common malformed planner step encodings."""

    normalized = dict(step)
    raw_type = normalized.get("type")
    raw_tool_name = normalized.get("tool_name")

    if "inputs" not in normalized:
        for fallback_key in ("parameters", "args", "arguments"):
            fallback_value = normalized.get(fallback_key)
            if isinstance(fallback_value, dict):
                normalized["inputs"] = fallback_value
                break

    step_type = raw_type.strip() if isinstance(raw_type, str) else ""
    tool_name = raw_tool_name.strip() if isinstance(raw_tool_name, str) else ""
    inferred_tool_name = tool_name or _infer_capability_name_from_step(
        step_type, normalized)
    inferred_type = _infer_step_type(step_type, inferred_tool_name)

    if inferred_type is not None:
        normalized["type"] = inferred_type
    if inferred_tool_name:
        normalized["tool_name"] = inferred_tool_name

    return normalized


def _infer_capability_name_from_step(step_type: str, step: dict[str, Any]) -> str:
    """Infer a capability name when the planner used the wrong field."""

    candidates: list[str] = []
    for key in ("tool_name", "name", "tool", "task", "capability"):
        value = step.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    if step_type:
        candidates.append(step_type)
        for suffix in ("_as_tool", "_as_llm", "_tool", "_llm"):
            if step_type.endswith(suffix):
                candidates.append(step_type[: -len(suffix)])

    for candidate in candidates:
        if candidate in TOOLS or candidate in LLM_TASKS:
            return candidate

    inferred_from_inputs = _infer_capability_name_from_inputs(step_type, step)
    if inferred_from_inputs:
        return inferred_from_inputs

    return ""


def _infer_capability_name_from_inputs(step_type: str, step: dict[str, Any]) -> str:
    """Infer a capability from distinctive input keys when the name is omitted."""

    inputs = step.get("inputs")
    if step_type != "llm" or not isinstance(inputs, dict):
        return ""

    text = " ".join(
        str(value)
        for value in (
            step.get("description", ""),
            inputs.get("instructions", ""),
        )
    ).lower()
    if "documents" in inputs and any(
        marker in text
        for marker in ("summarize", "summary", "list", "extract", "identify")
    ):
        return "summarize_text"

    if "documents" in inputs and any(
        marker in text for marker in ("evaluate", "rank", "score")
    ):
        return "evaluate_text"

    return ""


def _infer_step_type(step_type: str, tool_name: str) -> str | None:
    """Infer the canonical step type from either field."""

    if tool_name in TOOLS:
        return "tool"
    if tool_name in LLM_TASKS:
        return "llm"
    if step_type in {"tool", "llm"}:
        return step_type
    return None


def _extract_json_object(raw_text: str) -> dict | None:
    """Extract json object."""

    cleaned = raw_text.strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(cleaned[start: end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _validate_step_references(value: object, *, current_step_id: int) -> None:
    """Validate step references."""

    if isinstance(value, dict):
        for item in value.values():
            _validate_step_references(item, current_step_id=current_step_id)
        return

    if isinstance(value, list):
        for item in value:
            _validate_step_references(item, current_step_id=current_step_id)
        return

    if not isinstance(value, str):
        return
    if "{{" in value or "}}" in value:
        raise ValueError("Legacy placeholder syntax is not allowed.")
    if value.startswith("@") or is_reference_string(value):
        validate_reference_string(value, current_step_id=current_step_id)


KNOWN_RESULT_FIELDS = {
    "answer_question": {"text"},
    "read_text_file": {"path", "content"},
    "read_documents": {"input_path", "documents"},
    "search_web": {"query", "results"},
    "summarize_text": {"summary", "documents"},
    "evaluate_text": {"report", "parsed"},
    "generate_document_set": {"documents", "source_documents"},
    "rank_cvs": {"best_cv", "results"},
    "clean_job_description": {"cleaned_text"},
    "build_application_notes_from_job_description": {"info"},
    "write_document": {"destination_path", "characters_written"},
    "write_json_file": {"destination_path", "payload"},
    "write_search_results": {"destination_path", "query", "results"},
    "write_generated_documents": {"output_dir", "documents"},
}


def _validate_known_step_reference_paths(
    value: object,
    previous_steps: list[PlanStepDraft],
) -> None:
    """Reject references to known result fields that a previous step cannot return."""

    if isinstance(value, dict):
        for item in value.values():
            _validate_known_step_reference_paths(item, previous_steps)
        return

    if isinstance(value, list):
        for item in value:
            _validate_known_step_reference_paths(item, previous_steps)
        return

    if not isinstance(value, str):
        return

    step_match = STEP_REFERENCE_PATTERN.fullmatch(value)
    if step_match is None:
        return

    referenced_step_id = int(step_match.group(1))
    if referenced_step_id >= len(previous_steps):
        return

    output_path = step_match.group(2)
    if not output_path.startswith("result."):
        return

    result_field = output_path.removeprefix("result.").split(".", 1)[0]
    tool_name = previous_steps[referenced_step_id].tool_name
    known_fields = KNOWN_RESULT_FIELDS.get(tool_name)
    if known_fields is not None and result_field not in known_fields:
        raise ValueError(
            f"Step reference `{value}` points to `{result_field}`, but "
            f"`{tool_name}` result fields are: {', '.join(sorted(known_fields))}."
        )


def _contains_reference(value: object) -> bool:
    """Return contains reference."""

    if isinstance(value, dict):
        return any(_contains_reference(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_reference(item) for item in value)
    return isinstance(value, str) and value.startswith("@")


def _is_list_or_reference(value: object) -> bool:
    """Return whether a value is a list now or a runtime list reference later."""

    return isinstance(value, list) or (
        isinstance(value, str) and is_reference_string(value)
    )


def _validate_llm_input_shapes(tool_name: str, inputs: dict[str, Any]) -> None:
    """Validate LLM input shapes that may include runtime references."""

    if tool_name in {"summarize_text", "evaluate_text"}:
        documents = inputs.get("documents")
        if not _is_list_or_reference(documents):
            raise ValueError(
                f"LLM step `{tool_name}` input `documents` must be a list or runtime reference."
            )
        return

    if tool_name == "generate_document_set":
        source_documents = inputs.get("source_documents")
        if not _is_list_or_reference(source_documents):
            raise ValueError(
                "LLM step `generate_document_set` input `source_documents` must be a list or runtime reference."
            )
        return

    if tool_name == "rank_cvs":
        for key in ("job_documents", "cv_documents"):
            value = inputs.get(key)
            if not _is_list_or_reference(value):
                raise ValueError(
                    f"LLM step `rank_cvs` input `{key}` must be a list or runtime reference."
                )


def _validate_tool_input_shapes(
    tool_name: str,
    inputs: dict[str, Any],
    previous_steps: list[PlanStepDraft],
) -> None:
    """Validate tool input shapes that are too semantic for schema validation."""

    if tool_name == "write_generated_documents":
        documents = inputs.get("documents")
        if not _is_list_or_reference(documents):
            raise ValueError(
                "Tool step `write_generated_documents` input `documents` must be a list or runtime reference."
            )
        return

    if tool_name != "search_web":
        return

    query = inputs.get("query")
    if not isinstance(query, str):
        return

    step_match = STEP_REFERENCE_PATTERN.fullmatch(query)
    if step_match is None:
        return

    referenced_step_id = int(step_match.group(1))
    if referenced_step_id >= len(previous_steps):
        return

    referenced_tool = previous_steps[referenced_step_id].tool_name
    if referenced_tool == "answer_question":
        raise ValueError(
            "`search_web.query` must not reference an `answer_question` step. "
            "Use `@goal:user_input` or a literal query directly."
        )


def _build_capability_contract_text() -> str:
    """Return compact contract guidance for the planner's most common capabilities."""

    return "\n".join(
        (
            "- `answer_question(inputs={\"question\": ...})`",
            "- `read_text_file(inputs={\"path\": ...})` -> output at `@step:<id>.output.result.content`",
            "- `read_documents(inputs={\"input_path\": ..., \"recursive\": false})` -> output at `@step:<id>.output.result.documents`",
            "- `clean_job_description(inputs={\"raw_job_text\": ...})` -> output at `@step:<id>.output.result.cleaned_text`",
            "- `build_application_notes_from_job_description(inputs={\"cleaned_job_text\": ...})` -> output at `@step:<id>.output.result.info`",
            "- `write_document(inputs={\"destination_path\": ..., \"content\": ...})`",
            "- `search_web(inputs={\"query\": ...})` -> output at `@step:<id>.output.result.query` and `@step:<id>.output.result.results`",
            "- `write_search_results(inputs={\"destination_path\": ..., \"query\": ..., \"results\": ...})`",
            "- `summarize_text(inputs={\"documents\": ...})` -> output at `@step:<id>.output.result.summary`",
            "- `evaluate_text(inputs={\"documents\": ..., \"instructions\": ...})` -> output at `@step:<id>.output.result.report`",
            "- `generate_document_set(inputs={\"source_documents\": ..., \"instructions\": ...})` -> output at `@step:<id>.output.result.documents`",
            "- `write_generated_documents(inputs={\"output_dir\": ..., \"documents\": ...})`",
            "- `rank_cvs(inputs={\"job_documents\": ..., \"cv_documents\": ...})` -> output at `@step:<id>.output.result.results` and `@step:<id>.output.result.best_cv`",
        )
    )


def _validate_embedded_source_plan(
    draft: PlanDraft,
    *,
    user_input: str | None,
) -> None:
    """Reject local-file read plans when the source text already lives in the prompt."""

    if not user_input or not _request_contains_embedded_source_text(user_input):
        return
    if _request_contains_explicit_local_path(user_input):
        return

    first_step = draft.plan[0]
    if (
        first_step.type == "tool"
        and first_step.tool_name in EMBEDDED_SOURCE_DISALLOWED_TOOLS
    ):
        raise ValueError(
            "The user request already contains source text. Do not start by reading a local file or folder. Use `@goal:user_input` directly."
        )

    if not any(_contains_goal_user_input_reference(step.inputs) for step in draft.plan):
        raise ValueError(
            "The user request already contains source text. At least one step must consume `@goal:user_input` directly."
        )


def _contains_goal_user_input_reference(value: object) -> bool:
    """Return whether a value contains a direct `@goal:user_input` reference."""

    if isinstance(value, dict):
        return any(_contains_goal_user_input_reference(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_goal_user_input_reference(item) for item in value)
    return value == "@goal:user_input"


def _build_input_source_guidance(user_input: str) -> str:
    """Describe whether the planner should use prompt text or local files as input."""

    if _request_contains_embedded_source_text(user_input):
        return (
            "- The user request already contains source text or document content.\n"
            "- Treat the prompt itself as the input source when possible.\n"
            "- Use `@goal:user_input` for raw prompt content instead of inventing local file paths.\n"
            "- Do not create synthetic read steps such as `read_text` for prompt content.\n"
            "- Do not assume there is a file in `data/inputs/...` unless the user explicitly gave a real path."
        )
    return (
        "- Use file-reading tools only when the user explicitly refers to an existing local path, file, or folder.\n"
        "- Do not invent local file paths."
    )


def _request_contains_embedded_source_text(user_input: str) -> bool:
    """Heuristically detect when the user pasted source content into the request."""

    normalized = user_input.strip()
    if "```" in normalized:
        return True
    if len(normalized) >= 400 and "\n" in normalized:
        return True
    lines = [line for line in normalized.splitlines() if line.strip()]
    if len(lines) >= 8:
        return True
    lowered = normalized.lower()
    return any(
        marker in lowered
        for marker in (
            "here is the job",
            "job description",
            "job ad",
            "posting text",
            "content:",
        )
    )


def _request_contains_explicit_local_path(user_input: str) -> bool:
    """Return whether the user explicitly supplied a local path in the request."""

    return bool(LOCAL_PATH_HINT_PATTERN.search(user_input))
