"""
Shared utilities for user-selectable prediction horizon profiles.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from ml.targets import TargetConfig


DEFAULT_HORIZON_PROFILE = "short"

PROFILE_BASES: Dict[str, TargetConfig] = {
    "intraday": TargetConfig(
        horizon=1,
        smoothing_window=2,
        directional_threshold=0.0005,
        neutral_band=0.0002,
        quantile_bins=5,
        quantile_window=90,
        include_directional=True,
        include_quantiles=True,
        include_smoothed_return=True,
    ),
    "short": TargetConfig(
        horizon=4,
        smoothing_window=4,
        directional_threshold=0.0012,
        neutral_band=0.0006,
        quantile_bins=5,
        quantile_window=140,
        include_directional=True,
        include_quantiles=True,
        include_smoothed_return=True,
    ),
    "long": TargetConfig(
        horizon=30,
        smoothing_window=10,
        directional_threshold=0.003,
        neutral_band=0.0015,
        quantile_bins=5,
        quantile_window=300,
        include_directional=True,
        include_quantiles=True,
        include_smoothed_return=True,
    ),
}

PROFILE_METADATA: Dict[str, Dict[str, Any]] = {
    "intraday": {
        "label": "Intraday",
        "description": "Same-day horizon (1-2 bars) suited for quick scalp decisions.",
    },
    "short": {
        "label": "Short-Term",
        "description": "Swing horizon of roughly 3-5 days to capture short moves.",
    },
    "long": {
        "label": "Long",
        "description": "30-day position horizon targeting bigger directional shifts.",
    },
}


def available_profiles() -> List[str]:
    """Return sorted list of supported horizon profile names."""
    return sorted(PROFILE_BASES.keys())


def describe_profile(name: str) -> str:
    """Return human description for a profile."""
    return PROFILE_METADATA.get(name, {}).get("description", "")


def normalize_profile(name: Optional[str]) -> str:
    """Map arbitrary string to a supported profile, falling back to default."""
    if not name:
        return DEFAULT_HORIZON_PROFILE
    key = name.strip().lower()
    return key if key in PROFILE_BASES else DEFAULT_HORIZON_PROFILE


def get_profile_config(name: Optional[str]) -> Tuple[str, TargetConfig]:
    """Return (normalized_name, cloned TargetConfig) for the requested profile."""
    key = normalize_profile(name)
    return key, replace(PROFILE_BASES[key])


def build_profile_report(name: str, config: TargetConfig) -> Dict[str, Any]:
    """Create a metadata payload describing the active target configuration."""
    meta = PROFILE_METADATA.get(name, {})
    return {
        "name": name,
        "label": meta.get("label", name.title()),
        "description": meta.get("description"),
        "horizon_bars": config.horizon,
        "smoothing_window": config.smoothing_window,
        "directional_threshold": config.directional_threshold,
        "neutral_band": config.neutral_band,
        "quantile_bins": config.quantile_bins,
        "quantile_window": config.quantile_window,
    }


