# Domo is your personal AI agent on your computer

This AI agent is designed to serve as a fully private, locally running personal assistant that helps users manage and understand their digital workspace without relying on external services. Its core aim is to provide a free, secure alternative to cloud-based tools by enabling intelligent operations on local documents (such as searching for information, summarizing files or entire folders, classifying and organizing content, and ranking documents by relevance) while also augmenting its capabilities with internet-based research. By combining local data access with AI-driven reasoning, it allows users to efficiently navigate and act on their information while keeping full control over their data.

<figure>
  <img src="img/Domo UI.png" alt="Domo is your personal assistant on your computer" />
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
.venv/bin/python -m tools.job.main data/jobs/my-job
```

Notes:

- Open the local URL shown by Streamlit, typically `http://localhost:8051`.
- This repo must be run with the local [`.venv`](/Users/z/dev/domo/domo/.venv), not a global Streamlit install.
- `./run_app.sh` uses `.venv/bin/streamlit` explicitly, which avoids `ModuleNotFoundError` from Anaconda/global Python.
- If you prefer the raw command, use `.venv/bin/streamlit run app/streamlit_app.py --server.port 8051`.
- Use the `streamlit` CLI to run the app, not `python app/streamlit_app.py`.
- Start Ollama on `http://localhost:11434` before using the assistant or the job workflow.
- `python -m tools.job.main` with no argument runs the default ATS search using `config.yaml`.
- `python -m tools.job.main <folder>` accepts a folder containing `job_description_raw.txt`, `cleaned_job_description.txt`, `job_description.txt`, `job description.txt`, `job_description.pdf`, or `job description.pdf`.
- You can also pass a company-name-style hint like `checkr`; the workflow will search `data/jobs/` and pick the closest dated matching folder.
- Generated artifacts are written under the configured outputs folder, which defaults to `data/outputs/`.
- Main application settings now live in `config.yaml`, including paths, debug mode, Ollama settings, and job-search parameters.
- In the assistant, `run_job_agent` is the online search workflow and `create_job_files` is the local-folder workflow for generating output files from an existing job folder or company-name hint.
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
    ├── jobs/
    ├── logs/
    └── outputs/
```

## Current job workflow

- `assistant/runtime.py` plans whether to answer directly or call a tool.
- `assistant/policy.py` validates tool arguments and restricts job folder paths to the project data roots.
- `tools/job/run_job_agent.py` launches the online job-search workflow through the project `.venv`.
- `tools/job/create_job_files.py` launches the local job-folder workflow through the same processor.
- `tools/job/main.py` supports three modes:
  - no argument: search ATS sources using the parameters in `config.yaml`
  - folder containing `job_description_raw.txt`: clean the raw job ad, generate a PDF, and generate application materials
  - folder containing `cleaned_job_description.txt`: skip cleaning and generate the remaining outputs from the cleaned text
  - folder containing `job_description.txt` or `job description.txt`: normalize the text into `job_description_raw.txt`, create `job_metadata.json` if needed, then continue through the normal flow
  - folder containing `job_description.pdf` or `job description.pdf`: extract the PDF text into `job_description_raw.txt`, create `job_metadata.json` if needed, then continue through the normal flow
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
- Batch search writes discovered job inputs under the configured jobs folder, which defaults to `data/jobs/`.
- Local folder mode can resolve a company-name hint such as `checkr` to the closest dated matching folder under the jobs directory.
- The assistant workflow `create_job_files` uses that local-folder mode directly and writes the six generated job files under the outputs directory.
- Generated artifacts are written under timestamped folders in the configured outputs folder, which defaults to `data/outputs/`.
- Current generated files are:
  - `cleaned_job_description.txt`
  - `job_description.pdf`
  - `application_notes.txt`
  - `summary.txt`
  - `skills.txt`
  - `sample_cv.txt`
- The workflow can search and prepare documents, but it does not currently submit applications through portals or monitor application status.
