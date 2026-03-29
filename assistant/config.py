from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "domo_config.yaml"

DEFAULT_CONFIG = {
    "debug": {
        "enabled": False,
    },
    "paths": {
        "data_root": "data",
        "jobs_root": "data/jobs",
        "outputs_root": "data/outputs",
        "cvs_root": "data/cvs",
        "logs_root": "data/logs",
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "generate_path": "/api/generate",
        "model": "mistral",
        "timeout_seconds": 120,
        "max_retries": 1,
        "failure_threshold": 3,
        "circuit_open_seconds": 60,
    },
    "job_workflow": {
        "subprocess_timeout_seconds": 300,
    },
    "job_search": {
        "role": "Full Stack Engineer",
        "location": "Zurich",
        "sources": [
            "greenhouse",
            "lever",
            "ashby",
        ],
        "companies": {
            "greenhouse": [
                "stripe",
                "shopify",
                "notion",
            ],
            "lever": [
                "figma",
                "segment",
            ],
            "ashby": [
                "openai",
                "scaleai",
            ],
        },
        "max_jobs": 2,
        "max_results_per_source": 5,
        "max_company_attempts_per_source": 15,
    },
}


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = {key: deepcopy(value) for key, value in base.items()}
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged

    return deepcopy(override)


@lru_cache(maxsize=1)
def load_app_config() -> dict:
    config = deepcopy(DEFAULT_CONFIG)

    if CONFIG_PATH.exists():
        with CONFIG_PATH.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}

        if not isinstance(loaded, dict):
            raise ValueError(
                f"Configuration file must contain a YAML object: {CONFIG_PATH}"
            )

        config = _deep_merge(config, loaded)

    return config


def get_config_path() -> Path:
    return CONFIG_PATH


def get_display_path(path: Path | None = None) -> str:
    target = path or CONFIG_PATH
    try:
        return str(target.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(target)


def _resolve_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def get_paths() -> dict[str, Path]:
    configured_paths = load_app_config()["paths"]
    return {
        "data_root": _resolve_path(configured_paths["data_root"]),
        "jobs_root": _resolve_path(configured_paths["jobs_root"]),
        "outputs_root": _resolve_path(configured_paths["outputs_root"]),
        "cvs_root": _resolve_path(configured_paths["cvs_root"]),
        "logs_root": _resolve_path(configured_paths["logs_root"]),
    }


def get_job_search_config() -> dict:
    return deepcopy(load_app_config()["job_search"])


def get_job_workflow_config() -> dict:
    return deepcopy(load_app_config()["job_workflow"])


def get_ollama_config() -> dict:
    return deepcopy(load_app_config()["ollama"])


def is_debug_enabled() -> bool:
    return bool(load_app_config()["debug"].get("enabled", False))
