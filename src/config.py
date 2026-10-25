"""配置加载：读取 config.yaml，并支持环境变量覆盖。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

_ENV_OVERRIDES = {
    "base_url": "LLM_BASE_URL",
    "api_key": "LLM_API_KEY",
    "model": "LLM_MODEL",
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"未找到配置文件：{cfg_path}。请复制 config.example.yaml 为 config.yaml 并填写。"
        )
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    _apply_env_overrides(cfg)
    _resolve_paths(cfg)
    return cfg


def _apply_env_overrides(cfg: dict[str, Any]) -> None:
    llm = cfg.setdefault("llm", {})
    for key, env_name in _ENV_OVERRIDES.items():
        value = os.getenv(env_name)
        if value:
            llm[key] = value


def _resolve_paths(cfg: dict[str, Any]) -> None:
    paths = cfg.setdefault("paths", {})
    for key in ("corpus_dir", "index_dir", "eval_set", "results_dir", "log_dir"):
        value = paths.get(key)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            paths[key] = str(PROJECT_ROOT / path)


def require_llm_config(cfg: dict[str, Any]) -> None:
    llm = cfg.get("llm", {})
    missing = [key for key in ("base_url", "api_key", "model") if not llm.get(key)]
    if missing:
        raise ValueError(f"config.yaml 的 llm 段缺少：{', '.join(missing)}")
