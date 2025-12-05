"""
Target engineering utilities for flexible label generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class TargetConfig:
    """
    Configuration for generating multiple supervised targets.

    Attributes:
        horizon: Prediction horizon in bars.
        smoothing_window: Rolling window (in bars) used to smooth noisy returns.
        directional_threshold: Minimum absolute return treated as a directional signal.
        neutral_band: Optional +/- band around zero for "hold" labels (set to None for binary).
        quantile_bins: Number of quantile buckets for ordinal classification.
        quantile_window: Rolling lookback window used for quantile estimation.
        include_directional: Whether to emit directional labels.
        include_quantiles: Whether to emit quantile bins.
        include_smoothed_return: Whether to emit the smoothed regression target.
    """

    horizon: int = 1
    smoothing_window: int = 5
    directional_threshold: float = 0.0005
    neutral_band: Optional[float] = 0.0002
    quantile_bins: int = 5
    quantile_window: int = 252
    include_directional: bool = True
    include_quantiles: bool = True
    include_smoothed_return: bool = True


def _compute_directional_labels(
    returns: pd.Series, threshold: float, neutral_band: Optional[float]
) -> pd.Series:
    """
    Map returns to {-1, 0, 1} labels using configurable thresholds.
    """
    up_mask = returns >= threshold
    if neutral_band is None:
        down_mask = ~up_mask
        labels = pd.Series(np.where(up_mask, 1, -1), index=returns.index, dtype=int)
    else:
        down_mask = returns <= -threshold
        neutral_mask = (~up_mask) & (~down_mask) | returns.abs() < neutral_band
        labels = pd.Series(np.zeros(len(returns), dtype=int), index=returns.index)
        labels.loc[up_mask] = 1
        labels.loc[down_mask] = -1
        labels.loc[neutral_mask] = 0
    return labels


def _compute_quantile_bins(
    returns: pd.Series, bins: int, window: int
) -> pd.Series:
    """
    Compute rolling quantile bins to capture magnitude regimes.
    """
    if bins < 2:
        raise ValueError("quantile_bins must be >= 2")

    def _rolling_qcut(window_series: pd.Series) -> float:
        try:
            return pd.qcut(
                window_series,
                q=bins,
                labels=False,
                duplicates="drop",
            ).iloc[-1]
        except ValueError:
            return np.nan

    quantile_labels = (
        returns.rolling(window, min_periods=max(10, bins)).apply(_rolling_qcut, raw=False)
    )
    quantile_labels = quantile_labels.replace([np.inf, -np.inf], np.nan)
    quantile_labels = quantile_labels.bfill().ffill()
    if quantile_labels.isna().any():
        quantile_labels = quantile_labels.fillna(0)
    return quantile_labels.astype(int)


def generate_targets(
    df: pd.DataFrame,
    config: Optional[TargetConfig] = None,
) -> Dict[str, str]:
    """
    Augment the dataset with additional target representations.

    Returns:
        Mapping of generated column names to a short description.
    """
    if config is None:
        config = TargetConfig()

    if "close" not in df.columns:
        raise KeyError("Dataset must contain a 'close' column before target generation.")

    target_meta: Dict[str, str] = {}

    df["target"] = df["close"].shift(-config.horizon)
    df["target_return"] = df["target"] / df["close"] - 1.0
    target_meta["target_return"] = f"{config.horizon}-step ahead close-to-close return"

    if config.include_smoothed_return and config.smoothing_window > 1:
        smoothed = (
            df["target_return"]
            .rolling(config.smoothing_window, min_periods=1)
            .mean()
            .fillna(0.0)
        )
        df["target_return_smoothed"] = smoothed
        target_meta[
            "target_return_smoothed"
        ] = f"Rolling mean of target_return ({config.smoothing_window} bars)"
    else:
        df["target_return_smoothed"] = df["target_return"]
        target_meta["target_return_smoothed"] = "Alias of target_return (no smoothing)"

    if config.include_directional:
        directional_source = df["target_return_smoothed"]
        df["target_direction"] = _compute_directional_labels(
            directional_source,
            threshold=config.directional_threshold,
            neutral_band=config.neutral_band,
        )
        target_meta[
            "target_direction"
        ] = (
            f"Directional label using +/-{config.directional_threshold:.4f} threshold "
            f"and neutral band {config.neutral_band}"
        )

    if config.include_quantiles:
        quantile_source = df["target_return"]
        quantile_bins = _compute_quantile_bins(
            quantile_source,
            bins=config.quantile_bins,
            window=min(config.quantile_window, len(df)),
        )
        df["target_quantile_bin"] = quantile_bins
        target_meta[
            "target_quantile_bin"
        ] = (
            f"{config.quantile_bins} quantile bins over rolling window "
            f"{config.quantile_window} (capped at dataset length)"
        )

    return target_meta


