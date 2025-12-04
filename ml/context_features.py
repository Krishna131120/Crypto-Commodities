"""
Contextual feature engineering: macro indicators, spreads, volatility regimes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import config

try:
    import yfinance as yf  # type: ignore
except ImportError:  # pragma: no cover
    yf = None  # Fallback if yfinance is unavailable


BASE_CONTEXT_DIR = Path("data/context")
MACRO_STORE = BASE_CONTEXT_DIR / "macro"
VOL_STORE = BASE_CONTEXT_DIR / "volatility"
SPREAD_STORE = BASE_CONTEXT_DIR / "spreads"
CACHE_DIR = BASE_CONTEXT_DIR / "cache"


MACRO_SERIES = {
    "dxy_index": {"ticker": "DX-Y.NYB"},
    "vix_index": {"ticker": "^VIX"},
    "us10y_yield": {"ticker": "^TNX"},
    "us2y_yield": {"ticker": "^UST2Y"},
    "tips_etf": {"ticker": "TIP"},
}

VOL_SERIES = {
    "gvz_gold_vol": {"ticker": "^GVZ"},
    "ovx_crude_vol": {"ticker": "^OVX"},
}

COMMODITY_RELATIONSHIPS = {
    "GC=F": ["SI=F", "PL=F"],
    "CL=F": ["BZ=F"],
    "SI=F": ["GC=F"],
}


@dataclass
class ContextFeatureConfig:
    include_macro: bool = True
    include_spreads: bool = True
    include_volatility_indices: bool = True
    include_regime_features: bool = True
    include_intraday_aggregates: bool = True
    intraday_timeframes: Tuple[str, ...] = ("4h", "1h")
    intraday_lookback: int = 30


def _ensure_dirs():
    for directory in {BASE_CONTEXT_DIR, MACRO_STORE, VOL_STORE, SPREAD_STORE, CACHE_DIR}:
        directory.mkdir(parents=True, exist_ok=True)


def _normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index).tz_convert("UTC") if index.tz else pd.DatetimeIndex(index, tz="UTC")
    return idx


def _load_local_series(path: Path) -> Optional[pd.Series]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return pd.Series(df["value"].to_numpy(), index=pd.DatetimeIndex(df["timestamp"], tz="UTC"))
    except Exception:
        return None


def _save_series(path: Path, series: pd.Series):
    payload = pd.DataFrame({"timestamp": series.index.tz_convert("UTC"), "value": series.values})
    path.parent.mkdir(parents=True, exist_ok=True)
    payload.to_csv(path, index=False)


def _download_series(name: str, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    """
    Download series from yfinance, suppressing errors (context features are optional).
    """
    if yf is None:
        return None
    try:
        # Suppress yfinance errors - these are optional context features
        import warnings
        import logging
        yf_logger = logging.getLogger("yfinance")
        old_level = yf_logger.level
        yf_logger.setLevel(logging.CRITICAL)  # Suppress ERROR and WARNING
        
        try:
            data = yf.download(
                ticker,
                start=start.tz_convert(None),
                end=(end + pd.Timedelta(days=2)).tz_convert(None),
                progress=False,
                quiet=True,  # Suppress progress output
            )
            if data.empty:
                return None
            series = data["Adj Close"].copy()
            series.index = pd.DatetimeIndex(series.index).tz_localize("UTC")
            path = CACHE_DIR / f"{name}.csv"
            _save_series(path, series)
            return series
        finally:
            yf_logger.setLevel(old_level)  # Restore original log level
    except Exception:
        # Silently fail - context features are optional
        return None


def _load_external_series(name: str, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    cache_path = CACHE_DIR / f"{name}.csv"
    series = _load_local_series(cache_path)
    if series is None or series.index.min() > start or series.index.max() < end:
        downloaded = _download_series(name, ticker, start, end)
        if downloaded is not None:
            series = downloaded
    if series is None:
        index = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D", tz="UTC")
        return pd.Series(np.nan, index=index, name=name)
    return series


def _align_series(series: pd.Series, index: pd.DatetimeIndex, method: str = "ffill") -> pd.Series:
    reindexed = series.reindex(index).fillna(method=method)
    return reindexed


def _compute_macro_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    features = {}
    start = index.min() - pd.Timedelta(days=30)
    end = index.max() + pd.Timedelta(days=30)
    for name, meta in MACRO_SERIES.items():
        raw = _load_external_series(name, meta["ticker"], start, end)
        aligned = _align_series(raw, index)
        features[name] = aligned
        features[f"{name}_return_5"] = aligned.pct_change(5)
        features[f"{name}_zscore_60"] = (aligned - aligned.rolling(60).mean()) / aligned.rolling(60).std()
    if "us10y_yield" in features and "tips_etf" in features:
        features["real_yield_spread"] = features["us10y_yield"] - features["tips_etf"].pct_change(5) * 100
    if "us10y_yield" in features and "us2y_yield" in features:
        features["yield_curve_2s10s"] = features["us10y_yield"] - features["us2y_yield"]
    return pd.DataFrame(features, index=index)


def _load_symbol_candles(asset_type: str, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    root = config.BASE_DATA_DIR / asset_type
    if not root.exists():
        return None
    for source_dir in root.iterdir():
        candidate = source_dir / symbol / timeframe / "data.json"
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not data:
                continue
            df = pd.DataFrame(data)
            if "timestamp" not in df.columns or "close" not in df.columns:
                continue
            idx = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index(idx)
            return df[["close", "volume"]].astype(float)
    return None


def _compute_spread_features(
    asset_type: str,
    symbol: str,
    timeframe: str,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    references = COMMODITY_RELATIONSHIPS.get(symbol, [])
    if asset_type != "commodities" or not references:
        return pd.DataFrame(index=index)
    base_df = _load_symbol_candles(asset_type, symbol, timeframe)
    if base_df is None:
        return pd.DataFrame(index=index)
    out = {}
    for ref_symbol in references:
        ref_df = _load_symbol_candles(asset_type, ref_symbol, timeframe)
        if ref_df is None:
            continue
        ref_aligned = ref_df["close"].reindex(base_df.index).fillna(method="ffill")
        base_close = base_df["close"].reindex(base_df.index).fillna(method="ffill")
        spread = base_close - ref_aligned
        ratio = base_close / ref_aligned.replace(0, np.nan)
        out[f"spread_{symbol}_{ref_symbol}"] = spread
        out[f"ratio_{symbol}_{ref_symbol}"] = ratio
        out[f"spread_zscore_{symbol}_{ref_symbol}"] = (
            (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
        )
    if not out:
        return pd.DataFrame(index=index)
    return pd.DataFrame(out).reindex(index).fillna(method="ffill").fillna(method="bfill")


def _compute_vol_index_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    start = index.min() - pd.Timedelta(days=30)
    end = index.max() + pd.Timedelta(days=30)
    features = {}
    for name, meta in VOL_SERIES.items():
        raw = _load_external_series(name, meta["ticker"], start, end)
        aligned = _align_series(raw, index)
        features[name] = aligned
        features[f"{name}_pct_change"] = aligned.pct_change(5)
        features[f"{name}_zscore"] = (aligned - aligned.rolling(60).mean()) / aligned.rolling(60).std()
    return pd.DataFrame(features, index=index)


def _compute_regime_features(base_df: pd.DataFrame) -> pd.DataFrame:
    close = base_df["close"]
    returns = close.pct_change().fillna(0.0)
    volatility = returns.rolling(20).std().fillna(0.0)
    trend = returns.rolling(60).mean() / (volatility.replace(0, np.nan))
    regime = pd.DataFrame(index=base_df.index)
    regime["trend_strength_60"] = trend.fillna(0.0)
    regime["volatility_cluster_20"] = volatility / volatility.rolling(60).mean()
    regime["mean_reversion_flag"] = ((returns.rolling(5).sum().abs() < 0.001)).astype(int)
    regime["volatility_regime_state"] = pd.cut(
        volatility,
        bins=[-np.inf, volatility.quantile(0.33), volatility.quantile(0.66), np.inf],
        labels=[0, 1, 2],
    ).astype(int)
    regime["drawdown_lookback_90"] = close / close.rolling(90).max() - 1.0
    return regime


def _compute_intraday_aggregates(
    asset_type: str,
    symbol: str,
    timeframes: Iterable[str],
    lookback: int,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    aggregates = {}
    for tf in timeframes:
        df = _load_symbol_candles(asset_type, symbol, tf)
        if df is None or df.empty:
            continue
        df = df.resample("1D").agg({"close": "last", "volume": "sum"})
        df["return"] = df["close"].pct_change()
        df["return_vol"] = df["return"].rolling(lookback).std()
        for col in ["return", "return_vol", "volume"]:
            name = f"intraday_{tf}_{col}"
            aggregates[name] = df[col]
    if not aggregates:
        return pd.DataFrame(index=daily_index)
    frame = pd.DataFrame(aggregates)
    frame.index = pd.DatetimeIndex(frame.index, tz="UTC")
    return frame.reindex(daily_index).fillna(method="ffill").fillna(method="bfill")


def build_context_features(
    base_candles: pd.DataFrame,
    asset_type: str,
    symbol: str,
    timeframe: str,
    config: Optional[ContextFeatureConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    """
    Construct contextual features aligned to the base candle index.

    Returns:
        Tuple of (feature_frame, metadata)
    """
    if config is None:
        config = ContextFeatureConfig()

    _ensure_dirs()
    index = _normalize_index(base_candles.index)
    feature_blocks: List[pd.DataFrame] = []
    metadata: Dict[str, Dict[str, str]] = {}

    if config.include_macro:
        macro = _compute_macro_features(index)
        feature_blocks.append(macro)
        metadata["macro"] = {"columns": list(macro.columns)}

    if config.include_volatility_indices:
        vol = _compute_vol_index_features(index)
        feature_blocks.append(vol)
        metadata["volatility_indices"] = {"columns": list(vol.columns)}

    if config.include_spreads:
        spreads = _compute_spread_features(asset_type, symbol, timeframe, index)
        if not spreads.empty:
            feature_blocks.append(spreads)
            metadata["spreads"] = {"columns": list(spreads.columns)}

    if config.include_regime_features:
        regime = _compute_regime_features(base_candles)
        feature_blocks.append(regime)
        metadata["regime"] = {"columns": list(regime.columns)}

    if config.include_intraday_aggregates:
        intraday = _compute_intraday_aggregates(
            asset_type,
            symbol,
            config.intraday_timeframes,
            config.intraday_lookback,
            index,
        )
        if not intraday.empty:
            feature_blocks.append(intraday)
            metadata["intraday"] = {
                "columns": list(intraday.columns),
                "timeframes": list(config.intraday_timeframes),
            }

    if not feature_blocks:
        return pd.DataFrame(index=index), metadata

    combined = pd.concat(feature_blocks, axis=1)
    combined = combined.reindex(index).fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return combined, metadata


def context_config_to_dict(config_obj: ContextFeatureConfig) -> Dict[str, str]:
    """
    Helper to serialize config for metadata logging.
    """
    return {k: str(v) for k, v in asdict(config_obj).items()}


