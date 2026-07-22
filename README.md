# Domo is your personal AI agent on your computer

This AI agent is designed to serve as a fully private, locally running personal assistant that helps users manage and understand their digital workspace without relying on external services. Its core aim is to provide a free, secure alternative to cloud-based tools by enabling intelligent operations on local documents (such as searching for information, summarizing files or entire folders, classifying and organizing content, and ranking documents by relevance) while also augmenting its capabilities with internet-based research. By combining local data access with AI-driven reasoning, it allows users to efficiently navigate and act on their information while keeping full control over their data.

<figure>
  <img src="img/domo_demo.png" alt="Domo is your personal assistant on your computer" />
  <figcaption>Domo is your personal assistant on your computer</figcaption>
</figure>

## Run

Start Ollama before using the assistant.

Start the app with:

```bash
./run_app.sh
```

Run a workflow directly with:

```bash
.venv/bin/python -m tools.<name_of_tool>.main
```

Run tests with:

```bash
PYTHONPATH=. pytest
```

Notes:

- Open the local URL shown by Streamlit, typically `http://localhost:8051`.
- The Streamlit assistant uses a session-scoped UI with three panes:
  - chat history
  - a derived agent-state panel
  - an activity log
- The core agent is now a deterministic plan executor with a fixed `AgentState`:
  - `status`
  - `goal`
  - `plan`
  - `current_step`
  - `memory`
  - `last_error`
- Domo prepares a full plan first, waits for approval, and then executes step by step.
- The planner uses a fixed capability registry. Important capability groups include:
  - read tools such as web search and local document readers
  - LLM tasks such as direct answers, summarization, evaluation, CV ranking, and generated document sets
  - write tools such as single-document writes, JSON writes, search-result writes, and generated multi-document writes
- For requests like “create a file per result,” Domo can generate structured `filename`/`content` records and write them with `write_generated_documents`.
- Chat history and UI logs are stored outside the core agent state.
- This repo must be run with the local [`.venv`](/Users/z/dev/domo/domo/.venv), not a global Streamlit install.
- `./run_app.sh` uses `.venv/bin/streamlit` explicitly, which avoids `ModuleNotFoundError` from Anaconda/global Python.
- If you prefer the raw command, use `.venv/bin/streamlit run app/streamlit_app.py --server.port 8051`.
- Use the `streamlit` CLI to run the app, not `python app/streamlit_app.py`.
- See [CHANGELOG.md](CHANGELOG.md) for release history.

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
│   ├── controller.py
│   ├── domo_agent.py
│   ├── llm_tasks.py
│   ├── planner.py
│   ├── policy.py
│   ├── registry.py
│   ├── runtime.py
│   └── schemas.py
├── integrations/
│   └── ollama_client.py
├── tools/
├── workflows/
└── data/
    ├── inputs/
    └── outputs/
        └── logs/
```

## License

This project is licensed under the MIT License.
