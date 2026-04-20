# Domo is your personal AI agent on your computer

This AI agent is designed to serve as a fully private, locally running personal assistant that helps users manage and understand their digital workspace without relying on external services. Its core aim is to provide a free, secure alternative to cloud-based tools by enabling intelligent operations on local documents (such as searching for information, summarizing files or entire folders, classifying and organizing content, and ranking documents by relevance) while also augmenting its capabilities with internet-based research. By combining local data access with AI-driven reasoning, it allows users to efficiently navigate and act on their information while keeping full control over their data.

<figure>
  <img src="img/domo_demo.png" alt="Domo is your personal assistant on your computer" />
  <figcaption>Domo is your personal assistant on your computer</figcaption>
</figure>

## Run

Start the app with:

```bash
./run_app.sh
```

Run the job workflow directly with:

```bash
.venv/bin/python -m tools.job.main
```

Process a single local job folder with:

```bash
.venv/bin/python -m tools.job.main data/inputs/jobs/my-job
```

Notes:

- Open the local URL shown by Streamlit, typically `http://localhost:8051`.
- The Streamlit assistant now uses a session-scoped chat UI with three panes:
  - chat for clarification and confirmation
  - a context panel showing retained parameter values, their source, and their status
  - an activity panel showing agent decisions and workflow progress, with raw workflow output available per run
- Domo always asks for confirmation in chat before executing a workflow.
- This repo must be run with the local [`.venv`](/Users/z/dev/domo/domo/.venv), not a global Streamlit install.
- `./run_app.sh` uses `.venv/bin/streamlit` explicitly, which avoids `ModuleNotFoundError` from Anaconda/global Python.
- If you prefer the raw command, use `.venv/bin/streamlit run app/streamlit_app.py --server.port 8051`.
- Use the `streamlit` CLI to run the app, not `python app/streamlit_app.py`.
- Start Ollama on `http://localhost:11434` before using the assistant or the job workflow.
- `python -m tools.job.main` with no argument runs the default ATS search using `config.yaml`.
- `python -m tools.job.main <folder>` accepts a folder containing `job_description_raw.txt`, `cleaned_job_description.txt`, `job_description.txt`, `job description.txt`, `job_description.pdf`, or `job description.pdf`.
- You can also pass a company-name-style hint like `checkr`; the workflow will search `data/inputs/jobs/` and pick the closest dated matching folder.
- Input folders now live under `data/inputs/...`. The agent may read project files and `data/`, but it may only create new files under `data/outputs/`.
- Any assistant-managed write under `data/outputs/` is normalized under `data/outputs/YYYYMMDD_HHMMSS/`, for example `data/outputs/20260322_134649/`.
- The assistant policy now refuses to overwrite existing files and does not delete files.
- Generated artifacts are written under timestamped folders in the configured outputs folder, which defaults to `data/outputs/YYYYMMDD_HHMMSS/`.
- Main application settings now live in `config.yaml`, including paths, debug mode, Ollama settings, and job-search parameters.
- The assistant can also inspect project files, inspect `data/`, and search the internet through the `search_web` workflow.
- In the assistant, `run_job_agent` is the general job workflow: it runs online job search when no `folder_path` is provided, or processes a local job folder when `folder_path` is provided.
- `create_job_files` is the dedicated local-folder workflow for generating output files from an existing job folder, a company-name hint that resolves under `data/inputs/jobs/`, or a staged `data/outputs/YYYYMMDD_HHMMSS/` working folder created from pasted job text.
- The assistant can also override `job_search.role`, `job_search.location`, `job_search.ignore_location`, and `job_search.remote_only` for a single run based on your prompt.
- The allowed per-prompt overrides are listed explicitly in `config.yaml` under `prompt_overrides`; currently only `run_job_agent` has overrideable fields, while `create_job_files` and `match_cv` are empty.

## Structure

```text
domo/
├── README.md
├── config.yaml
├── pyproject.toml
├── requirements.txt
├── run_app.sh
├── app/
│   └── streamlit_app.py
├── assistant/
│   ├── audit.py
│   ├── domo_agent.py
│   ├── policy.py
│   ├── registry.py
│   ├── router.py
│   ├── runtime.py
│   └── schemas.py
├── integrations/
│   └── ollama_client.py
├── img/
│   ├── domo_blue.webp
│   └── domo_yellow.webp
├── tools/
│   ├── job/
│   │   ├── __init__.py
│   │   ├── clean_job_description.py
│   │   ├── create_job_files.py
│   │   ├── discover_jobs.py
│   │   ├── export_job_pdf.py
│   │   ├── filesystem.py
│   │   ├── generate_application_materials.py
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── prompts.py
│   │   └── run_job_agent.py
└── data/
    ├── inputs/
    │   ├── cvs/
    │   ├── documents/
    │   └── jobs/
    └── outputs/
        └── logs/
```

## Current job workflow

- `assistant/runtime.py` now manages a conversational session, retains structured context values, prepares confirmations, and records structured activity events before dispatching tools.
- `assistant/policy.py` validates tool arguments, allows read-only inspection of project files and `data/`, and restricts new file creation to `data/outputs/` without overwriting.
- `tools/job/run_job_agent.py` launches the online job-search workflow through the project `.venv`.
- `tools/job/create_job_files.py` launches the local job-folder workflow through the same processor.
- `tools/job/main.py` supports three modes:
  - no argument: search ATS sources using the parameters in `config.yaml`
  - folder containing `job_description_raw.txt`: clean the raw job ad, generate a PDF, and generate application materials
  - folder containing `cleaned_job_description.txt`: skip cleaning and generate the remaining outputs from the cleaned text
  - folder containing `job_description.txt` or `job description.txt`: normalize the text in memory, stage the derived files under the run output folder, then continue through the normal flow
  - folder containing `job_description.pdf` or `job description.pdf`: extract the PDF text in memory, stage the derived files under the run output folder, then continue through the normal flow
- Main configuration lives in `config.yaml`, including:
  - `debug.enabled`
  - `paths`
  - `ollama`
  - `job_workflow`
  - `prompt_overrides`
  - `job_search`
- Job search parameters in `config.yaml` include:
  - `role`
  - `location`
  - `ignore_location`
  - `remote_only`
  - `sources`
  - `companies`
  - `max_jobs`
  - `max_results_per_source`
  - `max_company_attempts_per_source`
- Batch search stages discovered job inputs directly under the configured outputs folder, which defaults to `data/outputs/`.
- Local folder mode can resolve a company-name hint such as `checkr` to the closest dated matching folder under the jobs directory.
- The assistant workflow `create_job_files` uses that local-folder mode directly and writes the generated files under the outputs directory.
- The assistant can also use `search_web` for general internet lookup, and `read_documents` / `summarize_documents` / `evaluate_documents` against project files or `data/`.
- Generated artifacts are written under timestamped folders in the configured outputs folder, which defaults to `data/outputs/`.
- Current generated files are:
  - `job_description_raw.txt`
  - `job_metadata.json`
  - `cleaned_job_description.txt`
  - `job_description.pdf`
  - `info.txt`
- CV matching writes `best_cv.txt`, `cv_match_analysis.json`, and `cv_match_summary.txt` into a new output folder under `data/outputs/`.
- The workflow can search and prepare documents, but it does not currently submit applications through portals or monitor application status.

## License

This project is licensed under the MIT License.
