"""Configuration loading for the research sandbox.

Values come from ``config/settings.yaml``. Optional, non-secret overrides may
be supplied through a ``.env`` file (see ``.env.example``). No API keys are
read, required or supported: the app cannot reach a venue.
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# Environment variables that may override scalar settings, mapped to the
# dotted path they override. Deliberately tiny and secret-free.
ENV_OVERRIDES: dict[str, str] = {
    "HIVEMIND_SEED": "data.seed",
    "HIVEMIND_N_MARKETS": "data.n_markets",
    "HIVEMIND_INITIAL_CAPITAL": "backtest.initial_capital",
    "HIVEMIND_KILL_SWITCH": "risk.kill_switch",
    "HIVEMIND_DUCKDB_PATH": "data.duckdb_path",
}


def _coerce(value: str) -> Any:
    """Convert an environment string into a bool, int, float or str."""
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def _set_path(config: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``config["a"]["b"] = value`` for a dotted path ``"a.b"``."""
    keys = dotted.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_settings(path: str | Path | None = None, use_env: bool = True) -> dict[str, Any]:
    """Load settings from YAML and apply optional environment overrides.

    Args:
        path: Settings file. Defaults to ``config/settings.yaml``.
        use_env: When True, load ``.env`` and apply :data:`ENV_OVERRIDES`.

    Returns:
        A nested dictionary of configuration values.
    """
    settings_path = Path(path) if path else DEFAULT_SETTINGS_PATH
    with settings_path.open("r", encoding="utf-8") as handle:
        settings: dict[str, Any] = yaml.safe_load(handle) or {}

    if use_env:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        for env_key, dotted in ENV_OVERRIDES.items():
            raw = os.environ.get(env_key)
            if raw is not None and raw != "":
                _set_path(settings, dotted, _coerce(raw))

    # Safety rail: this project is simulation-only and never places orders.
    settings.setdefault("app", {})["live_trading_enabled"] = False
    return settings


@lru_cache(maxsize=1)
def _cached_settings() -> dict[str, Any]:
    return load_settings()


def get_settings() -> dict[str, Any]:
    """Return a deep copy of the process-wide settings (safe to mutate)."""
    return copy.deepcopy(_cached_settings())


def section(settings: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one top-level settings section, or an empty dict if absent."""
    value = settings.get(name, {})
    return value if isinstance(value, dict) else {}


def resolve_path(relative: str | Path) -> Path:
    """Resolve a settings path relative to the project root."""
    candidate = Path(relative)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
