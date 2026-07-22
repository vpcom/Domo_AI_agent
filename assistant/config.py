"""Config module for the Domo assistant.

Config helpers and runtime support for the Domo assistant.
"""

from functools import lru_cache
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_app_config() -> dict:
    """Load application configuration from disk."""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing configuration file: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Configuration file must contain a YAML object: {CONFIG_PATH}"
        )

    return loaded


def get_config_path() -> Path:
    """Return the configuration file path."""

    return CONFIG_PATH


def get_display_path(path: Path | None = None) -> str:
    """Return a human-readable display path for the application."""

    target = path or CONFIG_PATH
    try:
        return str(target.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(target)


def _resolve_path(value: str) -> Path:
    """Resolve path."""

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _get_section(name: str):
    """Return section."""

    config = load_app_config()
    if name not in config:
        raise ValueError(
            f"Missing required config section `{name}` in {CONFIG_PATH.name}"
        )
    return config[name]


def get_paths() -> dict[str, Path]:
    """Return canonical application paths from configuration."""

    configured_paths = _get_section("paths")
    return {
        "data_root": _resolve_path(configured_paths["data_root"]),
        "inputs_root": _resolve_path(configured_paths["inputs_root"]),
        "jobs_root": _resolve_path(configured_paths["jobs_root"]),
        "documents_root": _resolve_path(configured_paths["documents_root"]),
        "outputs_root": _resolve_path(configured_paths["outputs_root"]),
        "cvs_root": _resolve_path(configured_paths["cvs_root"]),
        "logs_root": _resolve_path(configured_paths["logs_root"]),
    }


def get_job_search_config() -> dict:
    """Return job search configuration settings."""

    return dict(_get_section("job_search"))


def get_job_workflow_config() -> dict:
    """Return job workflow configuration."""

    return dict(_get_section("job_workflow"))


def get_prompt_override_fields(tool_name: str) -> list[str]:
    """Return configured prompt override field names."""

    configured = _get_section("prompt_overrides")
    values = configured.get(tool_name, [])
    return [str(value) for value in values]


def get_ollama_config() -> dict:
    """Return Ollama client configuration."""

    return dict(_get_section("ollama"))


def is_debug_enabled() -> bool:
    """Return whether debug mode is currently enabled."""

    return bool(_get_section("debug").get("enabled", False))
