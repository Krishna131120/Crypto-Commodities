from __future__ import annotations

from pathlib import Path
from typing import List

from ml.horizons import normalize_profile, available_profiles


def timeframe_dir(asset_type: str, symbol: str, timeframe: str) -> Path:
    return Path("models") / asset_type / symbol.upper() / timeframe


def horizon_dir(asset_type: str, symbol: str, timeframe: str, horizon: str) -> Path:
    normalized = normalize_profile(horizon)
    return timeframe_dir(asset_type, symbol, timeframe) / normalized


def summary_path(asset_type: str, symbol: str, timeframe: str, horizon: str) -> Path:
    return horizon_dir(asset_type, symbol, timeframe, horizon) / "summary.json"


def list_horizon_dirs(asset_type: str, symbol: str, timeframe: str) -> List[Path]:
    base = timeframe_dir(asset_type, symbol, timeframe)
    if not base.exists():
        return []
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if subdirs:
        return sorted(subdirs)
    legacy_summary = base / "summary.json"
    if legacy_summary.exists():
        return [base]
    return []


def ensure_horizon_dirs(asset_type: str, symbol: str, timeframe: str) -> None:
    base = timeframe_dir(asset_type, symbol, timeframe)
    base.mkdir(parents=True, exist_ok=True)
    for profile in available_profiles():
        (base / profile).mkdir(parents=True, exist_ok=True)

