# Changelog

All notable changes to Domo are documented here.

## 0.1.1 - 2026-07-22

### Added

- Added `generate_document_set`, an LLM task that produces structured generated documents with `filename` and `content` fields.
- Added `write_generated_documents`, a deterministic write tool for creating multiple generated text files under a safe output directory.
- Added planner guidance for dynamic multi-file workflows such as “write one summary file per search result.”
- Added validation for planner/runtime contracts, including unsupported wildcard references, invalid result-field references, and invalid LLM document input shapes.
- Added regression tests for planner repair, search-result summarization, generated document sets, and multi-file writing.

### Changed

- Planner prompts now require every step to include `step_id`, `description`, `type`, `tool_name`, and `inputs`.
- Web lookup plans now use `search_web` directly instead of inserting an LLM query-parsing step.
- Search results can be passed into LLM document tasks by normalizing result records into document-like records.
- Generated output directories are normalized under the timestamped `data/outputs/...` root.

### Fixed

- Fixed failures caused by references to non-existent result fields such as `result.content`, `result.answer`, and wildcard paths like `results[*].title`.
- Fixed planner acceptance of `rank_cvs` inputs with strings where document lists were required.
- Fixed repeated malformed planner responses that omitted `tool_name` on LLM summarization steps.

## 0.1.0 - Initial

### Added

- Initial local Streamlit assistant with chat, agent-state, and activity panes.
- Deterministic plan execution with `AgentState`, step tracking, working memory, artifacts, and structured errors.
- Core web search, document reading, document writing, JSON writing, search-result writing, and job workflow capabilities.
- Local LLM-backed tasks for direct answers, text summarization, document evaluation, and CV ranking.
- Path normalization and write safeguards for project-local reads and `data/outputs/...` writes.
