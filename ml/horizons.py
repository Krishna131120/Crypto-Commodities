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
        "description": "Same-day horizon (1-2 bars) suited for quick scalp decisions. Fast entries/exits, small position sizes.",
        "trading_style": "Scalping - many small trades, quick in/out",
        "typical_hold_time": "Hours to 1 day",
        "position_size_pct": 0.05,  # 5% max per symbol
        "stop_loss_pct": 0.08,  # 8% stop (widened from 5% to avoid premature exits from crypto volatility)
        "min_confidence": 0.15,  # Higher confidence needed for quick trades (increased from 0.12)
    },
    "short": {
        "label": "Short-Term",
        "description": "Swing horizon of roughly 3-5 days to capture short moves. Medium position sizes, moderate frequency.",
        "trading_style": "Swing trading - hold for several days",
        "typical_hold_time": "3-5 days",
        "position_size_pct": 0.10,  # 10% max per symbol (default)
        "stop_loss_pct": 0.06,  # 6% stop (wider to avoid premature exits from crypto volatility, still protects capital)
        "min_confidence": 0.10,  # Standard confidence threshold
    },
    "long": {
        "label": "Long-Term",
        "description": "30-day position horizon targeting bigger directional shifts. Larger positions, fewer trades, ride big trends.",
        "trading_style": "Trend following - hold for weeks/months",
        "typical_hold_time": "Weeks to months",
        "position_size_pct": 0.18,  # 18% max per symbol (larger for rare trades)
        "stop_loss_pct": 0.07,  # 7% stop (wider for long-term positions, avoids crypto volatility, still limits major losses)
        "min_confidence": 0.08,  # Lower threshold (bigger moves are rarer but more reliable)
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
        "trading_style": meta.get("trading_style", ""),
        "typical_hold_time": meta.get("typical_hold_time", ""),
        "position_size_pct": meta.get("position_size_pct", 0.10),
        "stop_loss_pct": meta.get("stop_loss_pct", 0.02),
        "min_confidence": meta.get("min_confidence", 0.10),
    }


def get_horizon_risk_config(horizon_profile: str) -> Dict[str, float]:
    """
    Return risk configuration parameters for a given horizon profile.
    
    Returns a dict with:
    - max_notional_per_symbol_pct: Maximum position size as % of equity
    - default_stop_loss_pct: Stop-loss distance from entry
    - min_confidence: Minimum confidence threshold for trading
    
    This allows each horizon to have its own risk parameters.
    """
    meta = PROFILE_METADATA.get(normalize_profile(horizon_profile), {})
    return {
        "max_notional_per_symbol_pct": meta.get("position_size_pct", 0.10),
        "default_stop_loss_pct": meta.get("stop_loss_pct", 0.06),
        "min_confidence": meta.get("min_confidence", 0.10),
    }


def print_horizon_summary() -> None:
    """Print a user-friendly summary of all available horizons and their trading behavior."""
    print("=" * 80)
    print("AVAILABLE HORIZON PROFILES")
    print("=" * 80)
    print()
    
    for profile_name in available_profiles():
        meta = PROFILE_METADATA.get(profile_name, {})
        config = PROFILE_BASES[profile_name]
        
        print(f"📊 {meta.get('label', profile_name.title())} ({profile_name})")
        print(f"   Description: {meta.get('description', 'N/A')}")
        print(f"   Trading Style: {meta.get('trading_style', 'N/A')}")
        print(f"   Typical Hold Time: {meta.get('typical_hold_time', 'N/A')}")
        print(f"   Prediction Window: {config.horizon} bars")
        print(f"   Position Size: Up to {meta.get('position_size_pct', 0.10)*100:.0f}% of equity per symbol")
        print(f"   Stop-Loss: {meta.get('stop_loss_pct', 0.02)*100:.1f}% from entry")
        print(f"   Min Confidence: {meta.get('min_confidence', 0.10)*100:.0f}%")
        print(f"   Directional Threshold: {config.directional_threshold*100:.2f}%")
        print(f"   Neutral Band: ±{config.neutral_band*100:.2f}%")
        print()
        print("   How SELLING works:")
        print("   - LONG signal → BUY BTCUSD (up to position size limit)")
        print("   - FLAT/HOLD signal → SELL entire position (if you're long)")
        print("   - SHORT signal → SELL to open short (if shorting enabled)")
        print("   - When SHORT flips to FLAT/LONG → BUY to close short position")
        print()
        print("-" * 80)
        print()


