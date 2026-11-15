"""Configuration loader and path resolution utilities for CCTB-Net."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_scalar(val: str) -> Any:
    try:
        return yaml.safe_load(val)
    except Exception:
        return val


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    if not overrides:
        return cfg
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            continue
        key, raw_val = item.split("=", 1)
        parts = [p.strip() for p in key.split(".") if p.strip()]
        if not parts:
            continue
        cur = out
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = _parse_scalar(raw_val)
    return out


def load_config(path: str | Path, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base_rel = raw.pop("_base_", None)
    if base_rel:
        base_path = (cfg_path.parent / base_rel).resolve()
        base_cfg = load_config(base_path)
        cfg = _deep_merge(base_cfg, raw)
    else:
        cfg = raw
    return apply_overrides(cfg, overrides)


def resolve_path(rel_or_abs: str | Path) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def get_profile_dirs(cfg: Dict[str, Any]) -> Dict[str, Path]:
    profile = str(cfg.get("project", {}).get("profile", "default"))
    exp_name = str(cfg.get("project", {}).get("name", profile))
    seed = int(cfg.get("project", {}).get("seed", 42))
    data_root = resolve_path(cfg["paths"]["data_root"])
    out_root = resolve_path(cfg["paths"]["output_root"])
    return {
        "network_raw": resolve_path(cfg["paths"]["network_dir"]),
        "linux_raw": resolve_path(cfg["paths"]["linux_dir"]),
        "interim_dir": data_root / "interim" / profile,
        "processed_dir": data_root / "processed" / profile,
        "run_dir": out_root / "runs" / exp_name / f"seed_{seed}",
    }
