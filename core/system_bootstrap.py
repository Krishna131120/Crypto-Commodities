"""
Utility helpers to make sure data ingestion, feature generation, and model training
run automatically before MCP tools are invoked.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import fetchers
from fetchers import update_feature_store
from ml.horizons import normalize_profile, DEFAULT_HORIZON_PROFILE
from train_models import train_symbols
from core.model_paths import summary_path as build_summary_path, horizon_dir, ensure_horizon_dirs


_STATE_LOCK = threading.Lock()
_SYMBOL_LOCKS: Dict[str, threading.Lock] = {}
_SYMBOL_STATUS: Dict[str, Dict[str, Any]] = {}

_BOOTSTRAP_THREAD: Optional[threading.Thread] = None
_BOOTSTRAP_THREAD_LOCK = threading.Lock()

_LIVE_THREAD: Optional[threading.Thread] = None
_LIVE_THREAD_LOCK = threading.Lock()


def _get_symbol_key(asset_type: str, symbol: str, timeframe: str, horizon: Optional[str]) -> str:
    normalized = normalize_profile(horizon or DEFAULT_HORIZON_PROFILE)
    return f"{asset_type.lower()}::{symbol.upper()}::{timeframe}::{normalized}"


def _get_symbol_lock(key: str) -> threading.Lock:
    with _STATE_LOCK:
        lock = _SYMBOL_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SYMBOL_LOCKS[key] = lock
        return lock


def _features_path(asset_type: str, symbol: str, timeframe: str) -> Path:
    return Path("data") / "features" / asset_type / symbol / timeframe / "features.json"


def _find_data_dir(asset_type: str, symbol: str, timeframe: str) -> Optional[Path]:
    """Locate the directory that contains data.json for the symbol/timeframe."""
    base = Path("data") / "json" / "raw" / asset_type
    if not base.exists():
        return None

    candidates: List[Path] = []
    for source_dir in base.iterdir():
        if not source_dir.is_dir():
            continue
        # Try both original symbol casing and uppercase
        possible_names = [symbol, symbol.upper()]
        for sym in possible_names:
            candidate = source_dir / sym / timeframe
            if candidate.exists():
                candidates.append(candidate)
    if candidates:
        # Prefer directory that matches exact casing first
        exact = [c for c in candidates if c.parts[-2] == symbol]
        return exact[0] if exact else candidates[0]
    return None


def _ingest_historical(asset_type: str, symbol: str, timeframe: str, years: float) -> None:
    print(f"[BOOTSTRAP] Ingesting historical data for {asset_type}/{symbol}/{timeframe} ({years}y)")
    if asset_type == "crypto":
        fetchers.ingest_all_historical(
            crypto_symbols=[symbol],
            commodities_symbols=[],
            timeframe=timeframe,
            years=years,
        )
    else:
        fetchers.ingest_all_historical(
            crypto_symbols=[],
            commodities_symbols=[symbol],
            timeframe=timeframe,
            years=years,
        )


def _generate_features(asset_type: str, symbol: str, timeframe: str, data_dir: Path) -> None:
    print(f"[BOOTSTRAP] Generating features for {asset_type}/{symbol}/{timeframe}")
    update_feature_store(asset_type, symbol, timeframe, data_dir)


def _train_models(asset_type: str, symbol: str, timeframe: str, horizon_profile: str) -> None:
    print(f"[BOOTSTRAP] Training models for {asset_type}/{symbol}/{timeframe} ({horizon_profile})")
    
    # Use commodity-specific training for commodities, crypto-specific for crypto
    if asset_type == "commodities":
        from train_commodities import train_commodity_symbols
        horizon_map = {symbol: horizon_profile}
        train_commodity_symbols(
            commodities_symbols=[symbol],
            timeframe=timeframe,
            horizon_profiles=horizon_map,
        )
    else:
        # Crypto training
        horizon_map = {asset_type: horizon_profile}
        crypto_symbols = [symbol] if asset_type == "crypto" else []
        commodity_symbols = [symbol] if asset_type != "crypto" else []
        from train_models import train_symbols
        train_symbols(
            crypto_symbols=crypto_symbols,
            commodities_symbols=commodity_symbols,
            timeframe=timeframe,
            horizon_profiles=horizon_map,
        )


def ensure_symbol_ready(
    asset_type: str,
    symbol: str,
    timeframe: str,
    horizon_profile: Optional[str] = None,
    years: float = 5.0,
) -> Dict[str, Any]:
    """
    Ensure data, features, and models exist for the requested symbol/timeframe.
    Returns a status dict describing the result.
    """
    symbol = symbol.upper()
    horizon_profile = normalize_profile(horizon_profile or DEFAULT_HORIZON_PROFILE)
    key = _get_symbol_key(asset_type, symbol, timeframe, horizon_profile)

    # Fast path: already ready
    feature_path = _features_path(asset_type, symbol, timeframe)
    summary_path = build_summary_path(asset_type, symbol, timeframe, horizon_profile)
    if feature_path.exists() and summary_path.exists():
        status = {
            "ready": True,
            "symbol": symbol,
            "asset_type": asset_type,
            "timeframe": timeframe,
            "horizon": horizon_profile,
        }
        _SYMBOL_STATUS[key] = status
        return status

    lock = _get_symbol_lock(key)
    with lock:
        # Re-check after acquiring lock
        feature_path = _features_path(asset_type, symbol, timeframe)
        summary_path = build_summary_path(asset_type, symbol, timeframe, horizon_profile)
        if feature_path.exists() and summary_path.exists():
            status = {"ready": True, "symbol": symbol, "asset_type": asset_type, "timeframe": timeframe}
            _SYMBOL_STATUS[key] = status
            return status

        status: Dict[str, Any] = {
            "ready": False,
            "symbol": symbol,
            "asset_type": asset_type,
            "timeframe": timeframe,
            "horizon": horizon_profile,
            "message": "Preparing data/models",
        }
        _SYMBOL_STATUS[key] = status

        try:
            data_dir = _find_data_dir(asset_type, symbol, timeframe)
            if data_dir is None or not data_dir.exists():
                _ingest_historical(asset_type, symbol, timeframe, years)
                data_dir = _find_data_dir(asset_type, symbol, timeframe)
                if data_dir is None or not data_dir.exists():
                    raise FileNotFoundError(
                        f"Historical data directory not found for {asset_type}/{symbol}/{timeframe}"
                    )

            if not feature_path.exists():
                _generate_features(asset_type, symbol, timeframe, data_dir)
                if not feature_path.exists():
                    raise FileNotFoundError(
                        f"Features not generated for {asset_type}/{symbol}/{timeframe}"
                    )

            ensure_horizon_dirs(asset_type, symbol, timeframe)
            if not summary_path.exists():
                _train_models(asset_type, symbol, timeframe, horizon_profile)
                if not summary_path.exists():
                    raise FileNotFoundError(
                        f"Model summary not found after training for {asset_type}/{symbol}/{timeframe}"
                    )

            status.update({"ready": True, "message": "Symbol prepared"})
            _SYMBOL_STATUS[key] = status
            return status
        except Exception as exc:
            status.update({"ready": False, "error": str(exc)})
            _SYMBOL_STATUS[key] = status
            print(f"[BOOTSTRAP] Failed to prepare {asset_type}/{symbol}/{timeframe}: {exc}")
            return status


def start_live_feeds_async(
    crypto_symbols: Optional[List[str]] = None,
    commodities_symbols: Optional[List[str]] = None,
    timeframe: str = "1d",
) -> Optional[threading.Thread]:
    """Start live feeds in a daemon thread (only once)."""
    crypto_symbols = crypto_symbols or config.CRYPTO_SYMBOLS or []
    commodities_symbols = commodities_symbols or config.COMMODITIES_SYMBOLS or []

    if not crypto_symbols and not commodities_symbols:
        return None

    with _LIVE_THREAD_LOCK:
        global _LIVE_THREAD
        if _LIVE_THREAD and _LIVE_THREAD.is_alive():
            return _LIVE_THREAD

        def _run_live():
            try:
                fetchers.start_live_feeds_with_fallback(
                    crypto_symbols=crypto_symbols,
                    commodities_symbols=commodities_symbols,
                    crypto_timeframe=timeframe,
                    commodities_timeframe="1d",
                )
            except Exception as exc:
                print(f"[BOOTSTRAP] Live feeds exited: {exc}")

        _LIVE_THREAD = threading.Thread(
            target=_run_live,
            name="LiveFeedsThread",
            daemon=True,
        )
        _LIVE_THREAD.start()
        return _LIVE_THREAD


def kickoff_background_bootstrap(start_live: bool = True, timeframe: str = "1d") -> Optional[threading.Thread]:
    """
    Start a background thread that prepares configured symbols and optionally launches live feeds.
    Safe to call multiple times; only the first call creates the thread.
    """
    with _BOOTSTRAP_THREAD_LOCK:
        global _BOOTSTRAP_THREAD
        if _BOOTSTRAP_THREAD and _BOOTSTRAP_THREAD.is_alive():
            return _BOOTSTRAP_THREAD

        def _bootstrap():
            crypto_symbols = config.CRYPTO_SYMBOLS or []
            commodities_symbols = config.COMMODITIES_SYMBOLS or []

            if not crypto_symbols and not commodities_symbols:
                print("[BOOTSTRAP] No symbols configured; skipping automatic preparation.")
                return

            for symbol in crypto_symbols:
                ensure_symbol_ready("crypto", symbol, timeframe)

            for symbol in commodities_symbols:
                ensure_symbol_ready("commodities", symbol, timeframe)

            if start_live:
                start_live_feeds_async(
                    crypto_symbols=crypto_symbols,
                    commodities_symbols=commodities_symbols,
                    timeframe=timeframe,
                )

        _BOOTSTRAP_THREAD = threading.Thread(
            target=_bootstrap,
            name="BootstrapThread",
            daemon=True,
        )
        _BOOTSTRAP_THREAD.start()
        return _BOOTSTRAP_THREAD


def get_symbol_status() -> Dict[str, Dict[str, Any]]:
    """Return a copy of the current bootstrap status map."""
    with _STATE_LOCK:
        return dict(_SYMBOL_STATUS)

