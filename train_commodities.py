"""
Commodity-specific training module.

This module provides commodity-focused training functions that are completely
separate from crypto training logic. This ensures:
- No mixing of crypto and commodity training parameters
- Commodity-specific optimizations and configurations
- Independent evolution of commodity training logic
- Clear separation of concerns

All commodity training goes through this module, not train_models.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from train_models import train_for_symbol
from ml.horizons import DEFAULT_HORIZON_PROFILE, normalize_profile
from ml.json_logger import get_training_logger


def train_commodity_symbols(
    commodities_symbols: List[str],
    timeframe: str = "1d",
    output_dir: str = "models",
    horizon_profiles: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """
    Train models for commodity symbols only.
    
    This is the commodity-specific entry point that ensures no mixing
    with crypto training logic.
    
    Args:
        commodities_symbols: List of commodity symbols (e.g., ["GC=F", "CL=F"])
        timeframe: Timeframe string (default: "1d")
        output_dir: Base output directory for models
        horizon_profiles: Optional dict mapping symbol to horizon profile
        
    Returns:
        List of training summary dictionaries
    """
    if not commodities_symbols:
        return []
    
    summaries: List[Dict] = []
    output_path = Path(output_dir)
    
    # Default horizon profile for commodities
    default_commodity_horizon = DEFAULT_HORIZON_PROFILE
    
    for symbol in commodities_symbols:
        # Get horizon profile for this symbol (if specified)
        profile_name = (horizon_profiles or {}).get(symbol, default_commodity_horizon)
        profile_name = normalize_profile(profile_name)
        
        # Commodity-specific model directory
        model_dir = output_path / "commodities" / symbol / timeframe / profile_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"\n{'='*80}")
            print(f"TRAINING: commodities/{symbol}/{timeframe}")
            print(f"Horizon profile: {profile_name.title()}")
            print(f"{'='*80}")
            
            # Train using the core training function (which handles asset_type internally)
            summary = train_for_symbol(
                asset_type="commodities",  # Explicitly set to commodities
                symbol=symbol,
                timeframe=timeframe,
                output_dir=model_dir,
                horizon_profile=profile_name,
                verbose=True,
            )
            
            summaries.append(summary)
            print(json.dumps(summary, indent=2))
            
        except Exception as exc:
            error_msg = f"commodities/{symbol} ({profile_name}): {exc}"
            print(f"[SKIP] {error_msg}")
            skip_logger = get_training_logger("commodities", symbol, timeframe)
            skip_logger.error(
                f"Training skipped: {error_msg}",
                category="TRAINING",
                symbol=symbol,
                asset_type="commodities",
            )
            summaries.append({
                "symbol": symbol,
                "asset_type": "commodities",
                "status": "failed",
                "error": str(exc),
            })
    
    return summaries


def train_commodity_symbol(
    symbol: str,
    timeframe: str = "1d",
    horizon: str = DEFAULT_HORIZON_PROFILE,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Train models for a single commodity symbol.
    
    Convenience function for training a single commodity symbol.
    
    Args:
        symbol: Commodity symbol (e.g., "GC=F")
        timeframe: Timeframe string (default: "1d")
        horizon: Horizon profile name (default: from DEFAULT_HORIZON_PROFILE)
        output_dir: Optional output directory (defaults to models/commodities/...)
        
    Returns:
        Training summary dictionary
    """
    if output_dir is None:
        output_dir = Path("models") / "commodities" / symbol / timeframe / normalize_profile(horizon)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return train_for_symbol(
        asset_type="commodities",
        symbol=symbol,
        timeframe=timeframe,
        output_dir=output_dir,
        horizon_profile=normalize_profile(horizon),
        verbose=True,
    )

