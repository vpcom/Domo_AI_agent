import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from assistant.config import get_paths
from assistant.schemas import ActivityEvent


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


def log_event(event_type: str, **data: Any) -> None:
    payload = {"event": event_type, **data}
    _LOGGER.info(json.dumps(payload, ensure_ascii=True, sort_keys=True))


def log_activity_event(
    activity_event: ActivityEvent,
    *,
    session_id: str,
    turn_id: str | None = None,
) -> None:
    payload = {
        "event": "activity",
        "session_id": session_id,
        "turn_id": turn_id or activity_event.turn_id,
        "run_id": activity_event.run_id,
        "activity": activity_event.model_dump(mode="json"),
    }
    _LOGGER.info(json.dumps(payload, ensure_ascii=True, sort_keys=True))
