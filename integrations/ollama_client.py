from datetime import datetime, timedelta

import requests

from assistant.config import get_ollama_config

_failure_count = 0
_circuit_open_until: datetime | None = None


def call_llm(prompt: str, model: str | None = None) -> str:
    global _failure_count, _circuit_open_until

    settings = get_ollama_config()
    base_url = str(settings["base_url"]).rstrip("/")
    generate_path = "/" + str(settings["generate_path"]).lstrip("/")
    url = f"{base_url}{generate_path}"
    selected_model = model or settings["model"]
    now = datetime.utcnow()

    if _circuit_open_until is not None and now < _circuit_open_until:
        raise RuntimeError(
            "Ollama circuit is open after repeated failures. "
            f"Wait a minute, make sure Ollama is running on {base_url}, then retry."
        )

    last_error = None
    for _ in range(int(settings["max_retries"]) + 1):
        try:
            response = requests.post(
                url,
                json={
                    "model": selected_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=int(settings["timeout_seconds"]),
            )
            response.raise_for_status()
            _failure_count = 0
            _circuit_open_until = None
            return response.json()["response"]
        except requests.RequestException as exc:
            last_error = exc

    _failure_count += 1
    if _failure_count >= int(settings["failure_threshold"]):
        _circuit_open_until = now + timedelta(
            seconds=int(settings["circuit_open_seconds"])
        )

    raise RuntimeError(
        f"Ollama is not reachable at {base_url}. "
        "Start Ollama and make sure the local API is available, then retry."
    ) from last_error
