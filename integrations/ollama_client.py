from datetime import datetime, timedelta

import requests

FAILURE_THRESHOLD = 3
MAX_RETRIES = 1
CIRCUIT_OPEN_SECONDS = 60

_failure_count = 0
_circuit_open_until: datetime | None = None


def call_llm(prompt: str, model: str = "mistral") -> str:
    global _failure_count, _circuit_open_until

    url = "http://localhost:11434/api/generate"
    now = datetime.utcnow()

    if _circuit_open_until is not None and now < _circuit_open_until:
        raise RuntimeError(
            "Ollama circuit is open after repeated failures. "
            "Wait a minute, make sure Ollama is running on http://localhost:11434, then retry."
        )

    last_error = None
    for _ in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=120,
            )
            response.raise_for_status()
            _failure_count = 0
            _circuit_open_until = None
            return response.json()["response"]
        except requests.RequestException as exc:
            last_error = exc

    _failure_count += 1
    if _failure_count >= FAILURE_THRESHOLD:
        _circuit_open_until = now + timedelta(seconds=CIRCUIT_OPEN_SECONDS)

    raise RuntimeError(
        "Ollama is not reachable at http://localhost:11434. "
        "Start Ollama and make sure the local API is available, then retry."
    ) from last_error
