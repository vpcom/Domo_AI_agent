"""Interface-layer logging helpers for deterministic agent runs."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import Any

from assistant.config import get_paths
from assistant.schemas import AgentState


LOG_DIR = get_paths()["logs_root"]
LOG_FILE = LOG_DIR / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOGGER = logging.getLogger("domo.audit")
if not _LOGGER.handlers:
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False

_STATE_LOGGER = logging.getLogger("domo.state")
if not _STATE_LOGGER.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _STATE_LOGGER.addHandler(handler)
    _STATE_LOGGER.setLevel(logging.INFO)
    _STATE_LOGGER.propagate = False


def log_event(event_type: str, **data: Any) -> None:
    """Log a structured event to the audit log."""

    payload = {"event": event_type, **data}
    _LOGGER.info(json.dumps(payload, ensure_ascii=True, sort_keys=True))


def log_state_snapshot(
    state: AgentState,
    *,
    reason: str,
    turn_id: str | None = None,
) -> None:
    """Log a snapshot of the current agent or conversation state."""

    payload = {
        "event": "state_changed",
        "reason": reason,
        "session_id": state.session_id,
        "turn_id": turn_id,
        "state": state.model_dump(mode="json"),
    }
    _STATE_LOGGER.info(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
    )
