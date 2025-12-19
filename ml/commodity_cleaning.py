"""
Commodity-specific data cleaning and normalization utilities.

Commodities have different characteristics than crypto:
- Different price scales (gold ~$2000, crude oil ~$70, silver ~$25)
- Trading hours (not 24/7 like crypto)
- Different volatility patterns
- Supply/demand dynamics, weather, geopolitical factors

This module provides thorough cleaning and normalization specifically
optimized for commodity data.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Commodity-specific price ranges (for validation)
# These are reasonable bounds to detect data errors
COMMODITY_PRICE_RANGES = {
    "GC=F": (500, 5000),      # Gold: $500-$5000 per oz
    "CL=F": (10, 200),        # Crude Oil: $10-$200 per barrel
    "SI=F": (5, 100),         # Silver: $5-$100 per oz
    "PL=F": (200, 3000),      # Platinum: $200-$3000 per oz
    "NG=F": (1, 20),          # Natural Gas: $1-$20 per MMBtu
    "HG=F": (1, 10),          # Copper: $1-$10 per lb
    "ZC=F": (200, 1000),      # Corn: $200-$1000 per bushel (cents)
    "ZS=F": (500, 2000),      # Soybeans: $500-$2000 per bushel (cents)
    "ZW=F": (300, 1200),      # Wheat: $300-$1200 per bushel (cents)
    "KC=F": (50, 300),        # Coffee: $50-$300 per lb (cents)
    "SB=F": (5, 50),          # Sugar: $5-$50 per lb (cents)
}

# Typical daily volatility by commodity (for outlier detection)
COMMODITY_VOLATILITY = {
    "GC=F": 0.015,   # Gold: ~1.5% daily
    "CL=F": 0.025,   # Crude Oil: ~2.5% daily
    "SI=F": 0.020,   # Silver: ~2% daily
    "PL=F": 0.018,   # Platinum: ~1.8% daily
    "NG=F": 0.030,   # Natural Gas: ~3% daily (more volatile)
    "HG=F": 0.022,   # Copper: ~2.2% daily
    "ZC=F": 0.020,   # Corn: ~2% daily
    "ZS=F": 0.020,   # Soybeans: ~2% daily
    "ZW=F": 0.022,   # Wheat: ~2.2% daily
    "KC=F": 0.025,   # Coffee: ~2.5% daily
    "SB=F": 0.025,   # Sugar: ~2.5% daily
}


def detect_price_outliers(
    df: pd.DataFrame,
    symbol: str,
    method: str = "iqr",
    multiplier: float = 3.0,
) -> pd.Series:
    """
    Detect price outliers in commodity data.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Commodity symbol (for price range validation)
        method: 'iqr' (interquartile range) or 'zscore' (standard deviation)
        multiplier: Multiplier for IQR or Z-score threshold
        
    Returns:
        Boolean Series indicating outlier rows
    """
    if df.empty:
        return pd.Series([], dtype=bool, index=df.index)
    
    outliers = pd.Series(False, index=df.index)
    
    # Method 1: Price range validation (hard bounds)
    if symbol in COMMODITY_PRICE_RANGES:
        price_min, price_max = COMMODITY_PRICE_RANGES[symbol]
        # Check all OHLC columns
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                price_outliers = (df[col] < price_min) | (df[col] > price_max)
                outliers |= price_outliers
    
    # Method 2: Statistical outlier detection on returns
    if "close" in df.columns:
        returns = df["close"].pct_change().dropna()
        if len(returns) > 10:  # Need enough data
            if method == "iqr":
                q1 = returns.quantile(0.25)
                q3 = returns.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower_bound = q1 - multiplier * iqr
                    upper_bound = q3 + multiplier * iqr
                    return_outliers = (returns < lower_bound) | (returns > upper_bound)
                    # Map back to original index (shift by 1 since returns are shifted)
                    return_outlier_indices = returns[return_outliers].index
                    outliers.loc[return_outlier_indices] = True
            elif method == "zscore":
                mean_return = returns.mean()
                std_return = returns.std()
                if std_return > 0:
                    z_scores = np.abs((returns - mean_return) / std_return)
                    return_outliers = z_scores > multiplier
                    return_outlier_indices = returns[return_outliers].index
                    outliers.loc[return_outlier_indices] = True
    
    # Method 3: Detect extreme intraday moves (high-low range)
    if all(col in df.columns for col in ["high", "low", "close"]):
        intraday_range_pct = (df["high"] - df["low"]) / df["close"]
        if symbol in COMMODITY_VOLATILITY:
            typical_vol = COMMODITY_VOLATILITY[symbol]
            # Flag if intraday range > 5x typical daily volatility
            extreme_moves = intraday_range_pct > (typical_vol * 5.0)
            outliers |= extreme_moves
    
    return outliers


def clean_commodity_candles(
    df: pd.DataFrame,
    symbol: str,
    remove_outliers: bool = True,
    fill_gaps: bool = True,
    validate_ohlc: bool = True,
    min_volume_threshold: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Comprehensive cleaning of commodity OHLCV data.
    
    Args:
        df: Raw OHLCV DataFrame with timestamp index
        symbol: Commodity symbol (for symbol-specific validation)
        remove_outliers: Whether to remove detected outliers
        fill_gaps: Whether to forward-fill small gaps (trading hours)
        validate_ohlc: Whether to validate OHLC relationships (high >= low, etc.)
        min_volume_threshold: Minimum volume threshold (None = no filtering)
        
    Returns:
        Tuple of (cleaned_df, cleaning_stats)
    """
    if df.empty:
        return df, {"rows_removed": 0, "outliers_removed": 0, "invalid_ohlc_removed": 0, "low_volume_removed": 0}
    
    original_len = len(df)
    cleaned = df.copy()
    stats = {
        "rows_removed": 0,
        "outliers_removed": 0,
        "invalid_ohlc_removed": 0,
        "low_volume_removed": 0,
        "gaps_filled": 0,
    }
    
    # Step 1: Validate OHLC relationships
    if validate_ohlc and all(col in cleaned.columns for col in ["open", "high", "low", "close"]):
        invalid_mask = (
            (cleaned["high"] < cleaned["low"]) |
            (cleaned["high"] < cleaned["open"]) |
            (cleaned["high"] < cleaned["close"]) |
            (cleaned["low"] > cleaned["open"]) |
            (cleaned["low"] > cleaned["close"])
        )
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            cleaned = cleaned.loc[~invalid_mask]
            stats["invalid_ohlc_removed"] = invalid_count
    
    # Step 2: Remove negative or zero prices
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col in cleaned.columns:
            invalid_prices = cleaned[col] <= 0
            if invalid_prices.any():
                cleaned = cleaned.loc[~invalid_prices]
                stats["rows_removed"] += int(invalid_prices.sum())
    
    # Step 3: Remove negative volumes (keep zero volumes, they're valid for low-liquidity periods)
    if "volume" in cleaned.columns:
        invalid_volumes = cleaned["volume"] < 0
        if invalid_volumes.any():
            cleaned = cleaned.loc[~invalid_volumes]
            stats["rows_removed"] += int(invalid_volumes.sum())
    
    # Step 4: Volume filtering (optional)
    if min_volume_threshold is not None and "volume" in cleaned.columns:
        low_volume_mask = cleaned["volume"] < min_volume_threshold
        low_volume_count = int(low_volume_mask.sum())
        if low_volume_count > 0:
            # Don't remove all low-volume rows, just flag them
            # Instead, we'll mark them but keep them (some commodities have natural low-volume periods)
            stats["low_volume_removed"] = 0  # We're not removing, just tracking
    
    # Step 5: Outlier detection and removal
    if remove_outliers:
        outlier_mask = detect_price_outliers(cleaned, symbol, method="iqr", multiplier=3.0)
        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0:
            # Only remove if outlier count is reasonable (< 5% of data)
            outlier_pct = outlier_count / len(cleaned) if len(cleaned) > 0 else 0
            if outlier_pct < 0.05:  # Less than 5% outliers
                cleaned = cleaned.loc[~outlier_mask]
                stats["outliers_removed"] = outlier_count
            else:
                # Too many outliers - might be a data issue, log but don't remove
                stats["outliers_removed"] = 0
                # Could log a warning here
    
    # Step 6: Handle gaps (forward-fill for small gaps, but preserve trading hours structure)
    if fill_gaps and cleaned.index.is_monotonic_increasing:
        # For commodities, we want to preserve trading hours
        # So we only forward-fill if the gap is small (e.g., 1-2 periods)
        # Large gaps indicate market closure (weekends, holidays)
        if len(cleaned) > 1:
            # Calculate time differences
            if isinstance(cleaned.index, pd.DatetimeIndex):
                time_diffs = cleaned.index.to_series().diff()
                # Forward-fill OHLCV for small gaps (same day or next day)
                # But don't fill across large gaps (weekends, holidays)
                median_gap = time_diffs.median()
                small_gap_threshold = median_gap * 2  # 2x median gap
                
                # For small gaps, forward-fill price data
                for col in price_cols + (["volume"] if "volume" in cleaned.columns else []):
                    if col in cleaned.columns:
                        cleaned[col] = cleaned[col].ffill(limit=1)
                        stats["gaps_filled"] += int(cleaned[col].isna().sum() == 0)
    
    # Step 7: Final validation - ensure no NaN in critical columns
    critical_cols = ["open", "high", "low", "close"]
    missing_critical = cleaned[critical_cols].isna().any(axis=1)
    if missing_critical.any():
        cleaned = cleaned.loc[~missing_critical]
        stats["rows_removed"] += int(missing_critical.sum())
    
    # Step 8: Sort by timestamp (ensure chronological order)
    if not cleaned.index.is_monotonic_increasing:
        cleaned = cleaned.sort_index()
    
    stats["rows_removed"] = original_len - len(cleaned)
    
    return cleaned, stats


def normalize_commodity_features(
    df: pd.DataFrame,
    symbol: str,
    price_aware: bool = True,
    log_transform: bool = False,
) -> pd.DataFrame:
    """
    Normalize commodity features with price-aware scaling.
    
    Commodities have different price scales, so we need to:
    1. Normalize price-based features relative to current price level
    2. Use percentage-based features where possible
    3. Apply log transforms for highly skewed distributions
    
    Args:
        df: Feature DataFrame (after feature generation)
        symbol: Commodity symbol (for symbol-specific handling)
        price_aware: Whether to apply price-aware normalization
        log_transform: Whether to apply log transform to highly skewed features
        
    Returns:
        Normalized DataFrame
    """
    if df.empty:
        return df
    
    normalized = df.copy()
    
    # Get current price level for price-aware normalization
    if price_aware and "close" in df.columns:
        current_price = df["close"].iloc[-1] if len(df) > 0 else 1.0
        
        # Normalize price-based features by current price level
        # This makes features comparable across different price levels
        price_based_features = [
            col for col in df.columns
            if any(term in col.lower() for term in ["sma", "ema", "price", "atr", "bb"])
            and col not in ["close", "open", "high", "low"]
        ]
        
        for col in price_based_features:
            if col in normalized.columns and current_price > 0:
                # Convert absolute price differences to percentages
                if normalized[col].abs().max() > current_price * 0.1:  # If values are large relative to price
                    normalized[col] = normalized[col] / current_price
    
    # Log transform for highly skewed features (optional)
    if log_transform:
        # Identify highly skewed features
        numeric_cols = normalized.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in normalized.columns:
                col_data = normalized[col].dropna()
                if len(col_data) > 10:
                    # Check skewness
                    skewness = col_data.skew()
                    # Apply log transform if highly skewed and all positive
                    if abs(skewness) > 2.0 and (col_data > 0).all():
                        normalized[col] = np.log1p(normalized[col])  # log1p handles zeros
    
    return normalized


def get_commodity_cleaning_config(symbol: str) -> Dict[str, any]:
    """
    Get commodity-specific cleaning configuration.
    
    Args:
        symbol: Commodity symbol
        
    Returns:
        Dictionary with cleaning parameters
    """
    # Default config
    config = {
        "remove_outliers": True,
        "fill_gaps": True,
        "validate_ohlc": True,
        "min_volume_threshold": None,
        "outlier_method": "iqr",
        "outlier_multiplier": 3.0,
        "price_aware_normalization": True,
        "log_transform": False,
    }
    
    # Symbol-specific overrides
    if symbol in ["NG=F", "CL=F"]:
        # Energy commodities: more volatile, allow larger moves
        config["outlier_multiplier"] = 3.5
    elif symbol in ["GC=F", "SI=F", "PL=F"]:
        # Precious metals: more stable, stricter outlier detection
        config["outlier_multiplier"] = 2.5
    elif symbol in ["ZC=F", "ZS=F", "ZW=F"]:
        # Agricultural: seasonal patterns, allow some volatility
        config["outlier_multiplier"] = 3.0
        config["log_transform"] = True  # Agricultural prices can be skewed
    
    return config

