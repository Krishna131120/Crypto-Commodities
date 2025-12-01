"""
Dataset assembly utilities for training price-prediction models.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from ml.context_features import (
    ContextFeatureConfig,
    build_context_features,
    context_config_to_dict,
)
from ml.targets import TargetConfig, generate_targets

FEATURE_SCHEMA_VERSION = 2
REQUIRED_ACTION_FEATURES = [
    "SMA_5",
    "SMA_10",
    "SMA_50",
    "SMA_200",
    "RSI_14",
    "MACD_histogram",
    "ATR_14",
    "Volume_Ratio",
]


class DatasetNotFoundError(FileNotFoundError):
    """Raised when no candle data is available for a symbol/timeframe."""


def _get_feature_calculator():
    from fetchers import FeatureCalculator

    return FeatureCalculator


def _iter_symbol_dirs(asset_type: str, symbol: str, timeframe: str) -> Iterable[Path]:
    asset_root = config.BASE_DATA_DIR / asset_type
    if not asset_root.exists():
        return []
    for source_dir in asset_root.iterdir():
        if not source_dir.is_dir():
            continue
        candidate = source_dir / symbol / timeframe
        if (candidate / "data.json").exists():
            yield candidate


def load_candles(asset_type: str, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load OHLCV candles for a given asset/symbol/timeframe as a Pandas DataFrame.
    """
    for directory in _iter_symbol_dirs(asset_type, symbol, timeframe):
        data_file = directory / "data.json"
        with open(data_file, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not raw:
            continue
        df = pd.DataFrame(raw)
        required = {"open", "high", "low", "close", "volume", "timestamp"}
        missing = required.difference(df.columns)
        if missing:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").set_index("timestamp")
        return df.astype(float, errors="ignore")
    raise DatasetNotFoundError(
        f"No data.json found for {asset_type}/{symbol}/{timeframe}"
    )


def build_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Compute a rich feature matrix using the FeatureCalculator.
    """
    FeatureCalculator = _get_feature_calculator()
    calculator = FeatureCalculator(df)
    calculator.compute_all()
    feature_version = getattr(calculator, "SCHEMA_VERSION", 1)
    feature_series: Dict[str, pd.Series] = {}
    for name, series in calculator.series_cache.items():
        feature_series[name] = series
    feature_frame = pd.DataFrame(feature_series)
    feature_frame = feature_frame.reindex(df.index)
    return feature_frame, int(feature_version)


def _validate_dataset_schema(dataset: pd.DataFrame):
    missing = [col for col in REQUIRED_ACTION_FEATURES if col not in dataset.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if dataset["target"].isna().any() or dataset["target_return"].isna().any():
        raise ValueError("Targets contain NaN values after alignment.")
    if not dataset.index.is_monotonic_increasing:
        raise ValueError("Dataset index must be sorted chronologically.")


def _compute_split_indices(
    total: int, train_ratio: float, val_ratio: float, gap_days: int
) -> Dict[str, Tuple[int, int]]:
    if total <= 0:
        raise ValueError("Cannot compute split indices for empty dataset.")
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("Train/validation ratios must be positive.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("Train + validation ratios must leave room for test data.")

    test_ratio = 1.0 - train_ratio - val_ratio
    train_end = max(1, int(total * train_ratio))
    val_start = min(total, train_end + gap_days)
    remaining = max(1, total - val_start)
    val_size = max(1, int(remaining * (val_ratio / (val_ratio + test_ratio))))
    val_end = min(total, val_start + val_size)
    test_start = min(total, val_end + gap_days)
    test_end = total

    if test_start >= total:
        test_start = max(val_end, total - 1)

    return {
        "train": (0, min(train_end, total)),
        "val": (val_start, min(val_end, total)),
        "test": (test_start, test_end),
    }


def _compute_split_boundaries(
    index: pd.Index, train_ratio: float, val_ratio: float, gap_days: int
) -> Dict[str, Dict[str, str]]:
    if index.empty:
        raise ValueError("Cannot compute boundaries for empty index.")
    slices = _compute_split_indices(len(index), train_ratio, val_ratio, gap_days)

    def _segment(name: str) -> Dict[str, str]:
        start_idx, end_idx = slices[name]
        end_idx = max(start_idx, end_idx - 1)
        return {
            "start": index[start_idx].isoformat(),
            "end": index[end_idx].isoformat(),
        }

    return {
        "train": _segment("train"),
        "val": _segment("val"),
        "test": _segment("test"),
    }


def assemble_dataset(
    asset_type: str,
    symbol: str,
    timeframe: str = "1d",
    horizon: int = 1,
    train_ratio: float = 0.75,
    val_ratio: float = 0.125,
    gap_days: int = 0,
    expected_feature_version: Optional[int] = FEATURE_SCHEMA_VERSION,
    verbose: bool = False,
    logger=None,
    target_config: Optional[TargetConfig] = None,
    context_config: Optional[ContextFeatureConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Dict[str, str]]]]:
    """
    Assemble a supervised learning dataset for a symbol with advanced targets/context.
    
    Args:
        asset_type: Type of asset (crypto/commodities)
        symbol: Symbol name
        timeframe: Timeframe string
        horizon: Number of periods ahead to predict (overridden by TargetConfig if provided)
        target_config: Optional TargetConfig describing additional labels
        context_config: Optional ContextFeatureConfig controlling macro/spread features
        verbose: Print diagnostic information
    
    Returns:
        Tuple of (dataset, metadata)
    """
    candles = load_candles(asset_type, symbol, timeframe)
    if logger:
        logger.info(
            f"Loaded {len(candles)} raw candles",
            category="DATA",
            symbol=symbol,
            asset_type=asset_type,
            data={"rows": len(candles), "timeframe": timeframe}
        )
    if verbose:
        print(f"[DATA] Loaded {len(candles)} raw candles for {symbol}")
    
    features, feature_version = build_feature_frame(candles)
    if expected_feature_version is not None and feature_version != expected_feature_version:
        raise ValueError(
            f"Feature schema mismatch (expected {expected_feature_version}, got {feature_version})"
        )
    if logger:
        logger.info(
            f"Computed {len(features.columns)} features",
            category="DATA",
            symbol=symbol,
            asset_type=asset_type,
            data={"feature_count": len(features.columns)}
        )
    if verbose:
        print(f"[DATA] Computed {len(features.columns)} features")
    
    base = candles.copy()
    base["asset_type"] = asset_type
    base["symbol"] = symbol
    base["timeframe"] = timeframe
    dataset = base.join(features)
    
    target_config = target_config or TargetConfig(horizon=horizon)
    target_meta = generate_targets(dataset, config=target_config)

    # Handle missing features with strictly backward-looking fills
    feature_cols = [
        col for col in dataset.columns
        if col not in base.columns and col not in {"target", "target_return"}
    ]
    initial_nans = dataset[feature_cols].isna().sum().sum()
    dataset[feature_cols] = dataset[feature_cols].ffill()

    # Drop rows that still contain NaNs after forward-fill (typically the warm-up period)
    residual_mask = dataset[feature_cols].isna().any(axis=1)
    residual_rows_dropped = int(residual_mask.sum())
    if residual_rows_dropped:
        dataset = dataset.loc[~residual_mask]

    # Replace any remaining stray NaNs (e.g., non-numeric columns) with zeros
    dataset[feature_cols] = dataset[feature_cols].fillna(0.0)

    # Remove constant or near-constant features to avoid degenerate splits
    numeric_feature_frame = dataset[feature_cols].select_dtypes(include=[np.number])
    constant_features = [
        col for col in numeric_feature_frame.columns
        if col not in REQUIRED_ACTION_FEATURES
        and numeric_feature_frame[col].nunique(dropna=False) <= 1
    ]
    if constant_features:
        dataset = dataset.drop(columns=constant_features)
        feature_cols = [col for col in feature_cols if col not in constant_features]
    
    # Enrich with contextual features
    context_config = context_config or ContextFeatureConfig()
    context_frame, context_meta = build_context_features(
        base_candles=base,
        asset_type=asset_type,
        symbol=symbol,
        timeframe=timeframe,
        config=context_config,
    )
    if not context_frame.empty:
        dataset = dataset.join(context_frame, how="left")
    
    # Drop rows where target or target_return is NaN (last row due to shift(-1))
    rows_before = len(dataset)
    dataset = dataset.dropna(subset=["target", "target_return"])
    rows_after = len(dataset)

    _validate_dataset_schema(dataset)
    split_boundaries = _compute_split_boundaries(
        dataset.index, train_ratio=train_ratio, val_ratio=val_ratio, gap_days=gap_days
    )
    
    if logger:
        logger.info(
            f"Dataset assembled: {rows_after} rows, {len(feature_cols)} features",
            category="DATA",
            symbol=symbol,
            asset_type=asset_type,
            data={
                "rows_before": rows_before,
                "rows_after": rows_after,
                "rows_dropped": rows_before - rows_after,
                "feature_count": len(feature_cols),
                "initial_nans": int(initial_nans),
                "residual_rows_dropped": residual_rows_dropped,
                "constant_features_removed": constant_features,
                "feature_version": feature_version,
                "split_boundaries": split_boundaries,
            }
        )
    if verbose:
        print(f"[DATA] Feature NaNs handled: {initial_nans} -> 0")
        print(f"[DATA] Rows before dropna: {rows_before}, after: {rows_after} (dropped {rows_before - rows_after} rows with NaN target)")
        print(f"[DATA] Final dataset size: {len(dataset)} rows, {len(feature_cols)} features")
    
    latest_close = float(base["close"].iloc[-1])
    latest_timestamp = base.index[-1].isoformat()
    metadata = {
        "feature_version": feature_version,
        "split_boundaries": split_boundaries,
        "target_meta": target_meta,
        "target_config": asdict(target_config),
        "context_meta": context_meta,
        "context_config": context_config_to_dict(context_config),
        "latest_close": latest_close,
        "latest_timestamp": latest_timestamp,
    }
    return dataset, metadata


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    gap_days: int = 0,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split for time-series datasets with optional gap to prevent data leakage.
    Optimized to use all available data efficiently.
    
    Args:
        df: Dataset to split
        train_ratio: Proportion for training (default 0.7 = 70%)
        val_ratio: Proportion for validation (default 0.15 = 15%)
        gap_days: Number of rows to skip between train/val and val/test (prevents leakage)
        verbose: Print split information
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if df.empty:
        raise ValueError("Dataset is empty; cannot split.")
    total = len(df)
    slices = _compute_split_indices(total, train_ratio, val_ratio, gap_days)

    train = df.iloc[slice(*slices["train"])]
    val = df.iloc[slice(*slices["val"])]
    test = df.iloc[slice(*slices["test"])]

    if len(train) == 0:
        raise ValueError("Train split would be empty; provide more data.")
    if len(val) == 0:
        raise ValueError("Validation split would be empty; provide more data.")
    if len(test) == 0:
        raise ValueError("Test split would be empty; provide more data.")
    
    used_rows = slices["test"][1]
    unused_rows = total - used_rows
    
    if verbose:
        print(f"[SPLIT] Total: {total} rows")
        print(f"[SPLIT] Train: {len(train)} rows ({len(train)/total*100:.1f}%)")
        print(f"[SPLIT] Val: {len(val)} rows ({len(val)/total*100:.1f}%)")
        print(f"[SPLIT] Test: {len(test)} rows ({len(test)/total*100:.1f}%)")
        if gap_days > 0:
            print(f"[SPLIT] Gap: {gap_days} rows between splits (prevents data leakage)")
        if unused_rows > 0:
            print(f"[SPLIT] Unused rows: {unused_rows} ({unused_rows/total*100:.1f}%)")
        print(f"[SPLIT] Total used: {used_rows} rows ({used_rows/total*100:.1f}%)")
    
    return train, val, test


def extract_xy(df: pd.DataFrame, target_column: str = "target") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset frame into features and target vector.
    """
    excluded = {
        "target",
        "target_return",
        "symbol",
        "asset_type",
        "timeframe",
        "schema_version",
        "source",
        "fallback_status",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    feature_cols = [col for col in df.columns if col not in excluded]
    X = df[feature_cols].select_dtypes(include=[np.number]).astype(float)
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")
    y = df[target_column].astype(float)
    return X, y


