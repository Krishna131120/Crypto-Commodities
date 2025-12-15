"""
Unified pipeline runner: prune data/feature directories to only the requested
symbols, then run ingestion (historical/live/both) so that only those symbols
produce features.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Set, Optional, List
from datetime import datetime
import json
import os
import tempfile

import fetchers
from fetchers import update_feature_store
from ml.horizons import available_profiles as available_horizon_profiles
from train_models import train_symbols

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data" / "json" / "raw"
FEATURE_ROOT = PROJECT_ROOT / "data" / "features"
CHECKPOINT_FILE = PROJECT_ROOT / "pipeline_checkpoint.json"


def _atomic_write_json(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _load_checkpoint() -> Dict[str, Any]:
    if not CHECKPOINT_FILE.exists():
        return {"stages": {}}
    try:
        return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"stages": {}}


def _update_checkpoint(
    checkpoint: Dict[str, Any],
    stage: str,
    status: str,
    data: Optional[Dict[str, Any]] = None,
):
    checkpoint.setdefault("stages", {})
    checkpoint["stages"][stage] = {
        "status": status,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    if data:
        checkpoint["stages"][stage]["data"] = data
    _atomic_write_json(CHECKPOINT_FILE, checkpoint)


def _normalize_symbols(symbols: Optional[Iterable[str]]) -> Set[str]:
    if not symbols:
        return set()
    return {s.strip().upper() for s in symbols if s.strip()}


def prune_asset_data(asset_type: str, keep_symbols: Set[str]):
    asset_root = DATA_ROOT / asset_type
    if not asset_root.exists() or not keep_symbols:
        return
    print(f"[PRUNE] {asset_type}: keeping {sorted(keep_symbols)}")
    for source_dir in asset_root.iterdir():
        if not source_dir.is_dir():
            continue
        for symbol_dir in source_dir.iterdir():
            if symbol_dir.name.upper() not in keep_symbols:
                shutil.rmtree(symbol_dir, ignore_errors=True)
                print(f"  [REMOVE] {symbol_dir}")
        # If source dir empty, remove it
        if not any(source_dir.iterdir()):
            shutil.rmtree(source_dir, ignore_errors=True)


def prune_features(asset_type: str, keep_symbols: Set[str]):
    feat_root = FEATURE_ROOT / asset_type
    if not feat_root.exists():
        return
    if not keep_symbols:
        return
    for symbol_dir in feat_root.iterdir():
        if not symbol_dir.is_dir():
            continue
        if symbol_dir.name.upper() not in keep_symbols:
            shutil.rmtree(symbol_dir, ignore_errors=True)
            print(f"  [REMOVE FEATURES] {symbol_dir}")


def run_ingestion(mode: str,
                  crypto_symbols: Optional[List[str]],
                  commodities_symbols: Optional[List[str]],
                  timeframe: str,
                  years: float) -> bool:
    crypto_symbols = crypto_symbols or None
    commodities_symbols = commodities_symbols or None
    live_pending = False
    if mode == "both":
        fetchers.ingest_all_historical(
            crypto_symbols=crypto_symbols,
            commodities_symbols=commodities_symbols,
            timeframe=timeframe,
            years=years
        )
        live_pending = True
    elif mode == "historical":
        fetchers.ingest_all_historical(
            crypto_symbols=crypto_symbols,
            commodities_symbols=commodities_symbols,
            timeframe=timeframe,
            years=years
        )
    elif mode == "live":
        fetchers.start_live_feeds_with_fallback(
            crypto_symbols=crypto_symbols,
            commodities_symbols=commodities_symbols,
            crypto_timeframe=timeframe,
            commodities_timeframe="1d"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return live_pending


def regenerate_features(asset_type: str, symbols: Set[str], timeframe: str) -> int:
    """
    Regenerate features for given symbols.
    Returns the number of symbols successfully updated.
    """
    if not symbols:
        return 0
    asset_root = DATA_ROOT / asset_type
    if not asset_root.exists():
        return 0
    
    updated_count = 0
    for source_dir in asset_root.iterdir():
        if not source_dir.is_dir():
            continue
        for symbol_dir in source_dir.iterdir():
            symbol_name = symbol_dir.name.upper()
            if symbol_name not in symbols:
                continue
            timeframe_path = symbol_dir / timeframe
            if not timeframe_path.exists():
                continue
            try:
                update_feature_store(
                    asset_type=asset_type,
                    symbol=symbol_dir.name,
                    timeframe=timeframe,
                    data_directory=timeframe_path
                )
                print(f"[FEATURE] Updated {asset_type}/{symbol_dir.name}/{timeframe}")
                updated_count += 1
            except Exception as exc:
                print(f"[FEATURE ERROR] {asset_type}/{symbol_dir.name}: {exc}")
    
    return updated_count


def _prompt_train_choice() -> bool:
    while True:
        choice = input("\nTrain models for these symbols now? (y/n): ").strip().lower()
        if choice in {"y", "yes"}:
            return True
        if choice in {"n", "no"}:
            return False
        print("Please enter 'y' or 'n'.")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline runner: prune logs and ingest data+features for specific symbols."
    )
    parser.add_argument(
        "--mode",
        choices=["historical", "live", "both"],
        default=None,
        help="Ingestion mode."
    )
    parser.add_argument("--crypto-symbols", nargs="+", help="Crypto symbols (e.g., SOL-USDT ETH-USDT)")
    parser.add_argument("--commodities-symbols", nargs="+", help="Commodity symbols (e.g., CL=F SI=F)")
    parser.add_argument("--timeframe", default="1d", help="Historical timeframe (default 1d)")
    parser.add_argument("--years", type=float, default=None, help="Years of historical data (default 5)")
    horizon_choices = available_horizon_profiles()
    parser.add_argument(
        "--crypto-horizon",
        choices=horizon_choices,
        help="Prediction horizon profile for crypto models when training.",
    )
    parser.add_argument(
        "--commodities-horizon",
        choices=horizon_choices,
        help="Prediction horizon profile for commodity models when training.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="After ingestion, train prediction models for the selected symbols.",
    )

    args = parser.parse_args()
    crypto_symbols = _normalize_symbols(args.crypto_symbols)
    commodities_symbols = _normalize_symbols(args.commodities_symbols)

    timeframe = args.timeframe or "1d"
    # Default to a longer history window if not specified.
    years = args.years if args.years is not None else 10.0
    mode = args.mode or "both"
    train_requested = args.train
    live_pending = False

    horizon_map: Dict[str, str] = {}
    if not crypto_symbols and not commodities_symbols:
        print("\nNo symbols provided via CLI. Entering interactive mode...\n")
        c_syms, d_syms, years_input, mode_input, horizon_preferences, auto_train = fetchers.get_user_input()
        if c_syms is None and d_syms is None:
            print("No symbols were provided. Exiting.")
            return
        crypto_symbols = _normalize_symbols(c_syms)
        commodities_symbols = _normalize_symbols(d_syms)
        years = years_input
        horizon_map = horizon_preferences or {}
        mode = mode_input
        train_requested = auto_train
        if not train_requested and mode != "live":
            train_requested = _prompt_train_choice()
    elif mode == "both":
        train_requested = True

    if not horizon_map:
        horizon_map = {
            asset: profile
            for asset, profile in (
                ("crypto", args.crypto_horizon),
                ("commodities", args.commodities_horizon),
            )
            if profile
        }

    if not crypto_symbols and not commodities_symbols:
        parser.error("Please provide at least one crypto or commodity symbol.")

    checkpoint = _load_checkpoint()
    _update_checkpoint(
        checkpoint,
        "init",
        "completed",
        {
            "mode": mode,
            "timeframe": timeframe,
            "years": years,
            "train_requested": train_requested,
            "crypto_symbols": sorted(crypto_symbols),
            "commodities_symbols": sorted(commodities_symbols),
            "horizon_profiles": horizon_map,
        },
    )

    try:
        _update_checkpoint(checkpoint, "prune", "in_progress")
        prune_asset_data("crypto", crypto_symbols)
        prune_asset_data("commodities", commodities_symbols)
        prune_features("crypto", crypto_symbols)
        prune_features("commodities", commodities_symbols)
        _update_checkpoint(checkpoint, "prune", "completed")
    except Exception as exc:
        _update_checkpoint(checkpoint, "prune", "failed", {"error": str(exc)})
        raise

    try:
        _update_checkpoint(checkpoint, "ingestion", "in_progress")
        live_pending = run_ingestion(
            mode=mode,
            crypto_symbols=sorted(crypto_symbols),
            commodities_symbols=sorted(commodities_symbols),
            timeframe=timeframe,
            years=max(years, 0.1)
        )
        _update_checkpoint(checkpoint, "ingestion", "completed")
    except Exception as exc:
        _update_checkpoint(checkpoint, "ingestion", "failed", {"error": str(exc)})
        raise

    try:
        _update_checkpoint(checkpoint, "features", "in_progress")
        regenerate_features("crypto", crypto_symbols, timeframe)
        regenerate_features("commodities", commodities_symbols, "1d")
        _update_checkpoint(checkpoint, "features", "completed")
    except Exception as exc:
        _update_checkpoint(checkpoint, "features", "failed", {"error": str(exc)})
        raise

    if train_requested and mode in {"historical", "both"}:
        print("\n" + "=" * 80)
        print("TRAINING MODELS")
        print("=" * 80)
        try:
            _update_checkpoint(checkpoint, "training", "in_progress")
            train_symbols(
                crypto_symbols=sorted(crypto_symbols),
                commodities_symbols=sorted(commodities_symbols),
                timeframe=timeframe,
                horizon_profiles=horizon_map or None,
            )
            _update_checkpoint(checkpoint, "training", "completed")
        except Exception as exc:
            _update_checkpoint(checkpoint, "training", "failed", {"error": str(exc)})
            print(f"[TRAIN ERROR] {exc}")
    else:
        _update_checkpoint(checkpoint, "training", "skipped", {"reason": "Not requested"})

    _update_checkpoint(checkpoint, "pipeline", "completed")

    if live_pending:
        print("\n" + "=" * 80)
        print("STARTING LIVE FEEDS")
        print("=" * 80)
        
        # Show summary of what will be monitored
        total_symbols = len(crypto_symbols) + len(commodities_symbols)
        if total_symbols == 1:
            symbol_name = list(crypto_symbols)[0] if crypto_symbols else list(commodities_symbols)[0]
            print(f"\n📊 Monitoring single symbol: {symbol_name}")
            print(f"   Live price updates will be reflected in:")
            print(f"   - data/json/raw/crypto/binance/{symbol_name}/{timeframe}/data.json")
            print(f"   - models/crypto/{symbol_name}/{timeframe}/summary.json (if model trained)")
            print()
        
        fetchers.start_live_feeds_with_fallback(
            sorted(crypto_symbols),
            sorted(commodities_symbols),
            timeframe,
            "1d",
        )


if __name__ == "__main__":
    main()

