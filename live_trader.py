"""
Live trading loop: continuously run model predictions and execute trades on Alpaca paper account.

This script:
1. Discovers crypto symbols with trained models
2. Loads latest features for each symbol
3. Runs InferencePipeline.predict() to get consensus
4. Executes trades via ExecutionEngine
5. Runs continuously with configurable interval
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from core.model_paths import horizon_dir, list_horizon_dirs
from ml.horizons import DEFAULT_HORIZON_PROFILE, normalize_profile, print_horizon_summary
from ml.inference import InferencePipeline
from ml.risk import RiskManagerConfig
from trading.execution_engine import ExecutionEngine, TradingRiskConfig
from trading.symbol_universe import all_enabled, find_by_data_symbol


def load_feature_row(asset_type: str, symbol: str, timeframe: str) -> Optional[pd.Series]:
    """
    Load the latest feature row from features.json.
    
    Returns None if features file doesn't exist or is invalid.
    """
    feature_path = Path("data/features") / asset_type / symbol / timeframe / "features.json"
    if not feature_path.exists():
        return None
    
    try:
        payload = json.loads(feature_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "features" in payload:
            # Convert features dict to Series
            features_dict = payload["features"]
            # Extract just the values (skip status/reason fields if present)
            clean_features = {}
            for name, value_data in features_dict.items():
                if isinstance(value_data, dict):
                    clean_features[name] = value_data.get("value")
                else:
                    clean_features[name] = value_data
            return pd.Series(clean_features)
        elif isinstance(payload, list):
            # Legacy format: list of feature dicts
            if payload:
                return pd.Series(payload[-1])
        return None
    except Exception as exc:
        print(f"[ERROR] Failed to load features for {symbol}: {exc}")
        return None


def get_current_price_from_features(asset_type: str, symbol: str, timeframe: str) -> Optional[float]:
    """
    Resolve the latest tradable price for a symbol.

    Priority:
    1. Alpaca last trade for the mapped trading symbol (primary source).
    2. Local Binance/Yahoo data.json (fallback).
    """
    # 1) Try Alpaca first (primary live source)
    try:
        from trading.alpaca_client import AlpacaClient
        from trading.symbol_universe import find_by_data_symbol as _find

        asset_mapping = _find(symbol)
        if asset_mapping:
            client = AlpacaClient()
            last_trade = client.get_last_trade(asset_mapping.trading_symbol)
            if last_trade:
                price = last_trade.get("price") or last_trade.get("p")
                if price:
                    return float(price)
    except Exception:
        # If Alpaca data is unavailable, fall back to local data.
        pass

    # 2) Fallback: local data.json
    data_path = Path("data/json/raw") / asset_type / "binance" / symbol / timeframe / "data.json"
    if not data_path.exists():
        data_path = Path("data/json/raw") / asset_type / "yahoo_chart" / symbol / timeframe / "data.json"

    if data_path.exists():
        try:
            payload = json.loads(data_path.read_text(encoding="utf-8"))
            if isinstance(payload, list) and payload:
                latest = payload[-1]
                return float(latest.get("close", 0))
            elif isinstance(payload, dict) and "close" in payload:
                return float(payload["close"])
        except Exception:
            pass

    return None


def discover_tradable_symbols(asset_type: str = "crypto", timeframe: str = "1d") -> List[Dict[str, Any]]:
    """
    Discover which symbols have trained models and are in our trading universe.
    
    Returns list of dicts with:
    - asset: AssetMapping
    - model_dir: Path to model directory
    - horizon: str (horizon profile name)
    """
    tradable = []
    universe = all_enabled()
    
    for asset in universe:
        if asset.asset_type != asset_type:
            continue
        
        # Check if we have trained models for this symbol
        horizon_dirs = list_horizon_dirs(asset_type, asset.data_symbol, timeframe)
        if not horizon_dirs:
            continue
        
        # Prefer the horizon profile specified in asset mapping, fallback to default
        preferred_horizon = asset.horizon_profile or DEFAULT_HORIZON_PROFILE
        preferred_horizon = normalize_profile(preferred_horizon)
        
        # Try preferred horizon first, then any available
        model_dir = None
        used_horizon = None
        
        for horizon_path in horizon_dirs:
            horizon_name = horizon_path.name
            summary_path = horizon_path / "summary.json"
            if summary_path.exists():
                if horizon_name == preferred_horizon:
                    model_dir = horizon_path
                    used_horizon = horizon_name
                    break
                elif model_dir is None:
                    model_dir = horizon_path
                    used_horizon = horizon_name
        
        if model_dir and (model_dir / "summary.json").exists():
            # Check if model is marked as tradable (robustness check passed)
            try:
                import json
                with open(model_dir / "summary.json", "r") as f:
                    summary = json.load(f)
                is_tradable = summary.get("tradable", True)  # Default to True for backward compatibility
                if not is_tradable:
                    # Skip non-tradable models (failed robustness checks)
                    continue
            except Exception:
                # If we can't read the summary, skip it (might be corrupted)
                continue
            
            tradable.append({
                "asset": asset,
                "model_dir": model_dir,
                "horizon": used_horizon,
            })
    
    return tradable


def run_trading_cycle(
    execution_engine: ExecutionEngine,
    tradable_symbols: List[Dict[str, Any]],
    dry_run: bool = False,
    verbose: bool = True,
    update_data: bool = True,
    regenerate_features_flag: bool = True,
) -> Dict[str, Any]:
    """
    Run one complete trading cycle: fetch data -> regenerate features -> predict + execute for all symbols.
    
    Args:
        execution_engine: Execution engine for placing trades
        tradable_symbols: List of symbols with trained models
        dry_run: If True, don't send real orders
        verbose: Print detailed progress
        update_data: If True, fetch latest live data from Alpaca/Binance before each cycle
        regenerate_features_flag: If True, regenerate features after updating data
    
    Returns summary dict with counts of successes/failures.
    """
    cycle_start = datetime.utcnow()
    results = {
        "cycle_start": cycle_start.isoformat() + "Z",
        "symbols_processed": 0,
        "symbols_traded": 0,
        "symbols_skipped": 0,
        "errors": [],
        "details": [],
    }
    
    # Step 1: Update live data for all symbols (if enabled)
    # PRIMARY: Get live price from Alpaca and update last candle's close price
    # This allows intraday trading with fresh prices even when daily candle isn't complete
    if update_data:
        if verbose:
            print("[UPDATE] Fetching latest live prices from Alpaca...")
        try:
            from trading.alpaca_client import AlpacaClient
            from trading.symbol_universe import find_by_data_symbol
            from fetchers import load_json_file, save_json_file, get_data_path
            from pathlib import Path
            import json
            
            # Get unique symbols
            unique_symbols = list(set(info["asset"].data_symbol for info in tradable_symbols))
            updated_count = 0
            client = AlpacaClient()
            
            for symbol in unique_symbols:
                try:
                    # Get trading symbol for Alpaca
                    asset_mapping = find_by_data_symbol(symbol)
                    if not asset_mapping:
                        continue
                    
                    # Get live price from Alpaca
                    last_trade = client.get_last_trade(asset_mapping.trading_symbol)
                    if not last_trade:
                        if verbose:
                            print(f"  [SKIP] {symbol}: No live price from Alpaca")
                        continue
                    
                    live_price = last_trade.get("price") or last_trade.get("p")
                    if not live_price:
                        continue
                    
                    live_price = float(live_price)
                    
                    # Load existing data.json
                    data_paths = [
                        get_data_path("crypto", symbol, "1d", None, "alpaca").parent / "data.json",
                        get_data_path("crypto", symbol, "1d", None, "binance").parent / "data.json",
                    ]
                    
                    data_file = None
                    for path in data_paths:
                        if path.exists():
                            data_file = path
                            break
                    
                    if not data_file or not data_file.exists():
                        if verbose:
                            print(f"  [SKIP] {symbol}: No existing data.json found")
                        continue
                    
                    # Load existing candles
                    existing_candles = load_json_file(data_file)
                    if not existing_candles:
                        if verbose:
                            print(f"  [SKIP] {symbol}: No existing candles")
                        continue
                    
                    # Update the last candle's close price with live price
                    last_candle = existing_candles[-1].copy()
                    last_timestamp = last_candle.get("timestamp", "")
                    
                    # Update close price (and high/low if live price exceeds them)
                    last_candle["close"] = live_price
                    if live_price > last_candle.get("high", 0):
                        last_candle["high"] = live_price
                    if live_price < last_candle.get("low", float("inf")) or last_candle.get("low", 0) == 0:
                        last_candle["low"] = live_price
                    last_candle["source"] = last_candle.get("source", "alpaca")
                    # Mark as live-updated
                    last_candle["live_updated"] = True
                    
                    # Replace last candle in the list
                    existing_candles[-1] = last_candle
                    
                    # Save updated data
                    save_json_file(data_file, existing_candles, append=False)
                    updated_count += 1
                    
                    if verbose:
                        print(f"  [OK] {symbol}: Updated last candle close to ${live_price:.2f} (Alpaca live)")
                        
                except Exception as sym_exc:
                    if verbose:
                        print(f"  [WARN] {symbol}: Failed to update with live price ({sym_exc})")
            
            if verbose:
                print(f"[UPDATE] Live prices updated for {updated_count}/{len(unique_symbols)} symbol(s) from Alpaca")
        except Exception as exc:
            if verbose:
                print(f"[WARN] Failed to update live prices from Alpaca: {exc}")
            # Continue anyway - we'll use existing data
    
    # Step 2: Regenerate features for all symbols (if enabled)
    if regenerate_features_flag:
        if verbose:
            print("[FEATURES] Regenerating features with latest data...")
        try:
            from pipeline_runner import regenerate_features
            unique_symbols = list(set(info["asset"].data_symbol for info in tradable_symbols))
            regenerate_features("crypto", set(unique_symbols), "1d")
            if verbose:
                print(f"[FEATURES] Features regenerated for {len(unique_symbols)} symbol(s)")
        except Exception as exc:
            if verbose:
                print(f"[WARN] Failed to regenerate features: {exc}")
            # Continue anyway - we'll use existing features
    
    # Step 3: Run predictions and execute trades for each symbol
    for symbol_info in tradable_symbols:
        asset = symbol_info["asset"]
        model_dir = symbol_info["model_dir"]
        horizon = symbol_info["horizon"]
        data_symbol = asset.data_symbol
        
        try:
            # Load latest features (now freshly regenerated)
            feature_row = load_feature_row("crypto", data_symbol, "1d")
            if feature_row is None or feature_row.empty:
                if verbose:
                    print(f"[SKIP] {data_symbol}: No features available")
                results["symbols_skipped"] += 1
                results["details"].append({
                    "symbol": data_symbol,
                    "status": "skipped",
                    "reason": "no_features",
                })
                continue
            
            # Get current price (from Alpaca - always fresh)
            current_price = get_current_price_from_features("crypto", data_symbol, "1d")
            if current_price is None or current_price <= 0:
                if verbose:
                    print(f"[SKIP] {data_symbol}: No valid price available")
                results["symbols_skipped"] += 1
                results["details"].append({
                    "symbol": data_symbol,
                    "status": "skipped",
                    "reason": "no_price",
                })
                continue
            
            # Load inference pipeline
            risk_config = RiskManagerConfig(paper_trade=True)
            pipeline = InferencePipeline(model_dir, risk_config=risk_config)
            pipeline.load()
            
            if not pipeline.models:
                if verbose:
                    print(f"[SKIP] {data_symbol}: No trained models found")
                results["symbols_skipped"] += 1
                results["details"].append({
                    "symbol": data_symbol,
                    "status": "skipped",
                    "reason": "no_models",
                })
                continue
            
            # Estimate volatility (simple: use recent price movement if available)
            volatility = 0.01  # Default 1% daily volatility
            # TODO: Could compute from recent candles if needed
            
            # Run prediction
            try:
                prediction_result = pipeline.predict(
                    feature_row,
                    current_price=current_price,
                    volatility=volatility,
                )
            except Exception as pred_exc:
                error_msg = f"{data_symbol}: Prediction failed: {pred_exc}"
                print(f"[ERROR] {error_msg}")
                if verbose:
                    import traceback
                    print(f"[ERROR] Prediction traceback:\n{traceback.format_exc()}")
                results["errors"].append(error_msg)
                results["symbols_skipped"] += 1
                results["details"].append({
                    "symbol": data_symbol,
                    "status": "error",
                    "error": str(pred_exc),
                    "stage": "prediction",
                })
                continue
            
            consensus = prediction_result.get("consensus", {})
            if not consensus:
                print(f"[SKIP] {data_symbol}: No consensus from models")
                if verbose:
                    print(f"  Prediction result keys: {list(prediction_result.keys())}")
                results["symbols_skipped"] += 1
                results["details"].append({
                    "symbol": data_symbol,
                    "status": "skipped",
                    "reason": "no_consensus",
                })
                continue
            
            # Display predicted action before execution
            action = consensus.get("consensus_action", "hold")
            confidence = consensus.get("consensus_confidence", 0.0)
            expected_move = consensus.get("consensus_expected_move", 0.0)
            print(f"[PREDICTION] {data_symbol}: {action.upper()} (confidence: {confidence*100:.1f}%, expected move: {expected_move*100:+.2f}%)")
            
            # Execute trade with horizon-specific risk parameters
            try:
                execution_result = execution_engine.execute_from_consensus(
                    asset=asset,
                    consensus=consensus,
                    current_price=current_price,
                    dry_run=dry_run,
                    horizon_profile=horizon,  # Pass horizon so engine uses horizon-specific risk config
                )
            except Exception as exec_exc:
                error_msg = f"{data_symbol}: Execution failed: {exec_exc}"
                action = consensus.get("consensus_action", "hold")
                print(f"[ERROR] {error_msg} (predicted action: {action.upper()})")
                if verbose:
                    import traceback
                    print(f"[ERROR] Execution traceback:\n{traceback.format_exc()}")
                results["errors"].append(error_msg)
                results["symbols_skipped"] += 1
                results["details"].append({
                    "symbol": data_symbol,
                    "status": "error",
                    "error": str(exec_exc),
                    "stage": "execution",
                    "consensus_action": consensus.get("consensus_action"),
                })
                continue
            
            if execution_result:
                results["symbols_traded"] += 1
                results["details"].append({
                    "symbol": data_symbol,
                    "status": "traded",
                    "decision": execution_result.get("decision"),
                    "model_action": consensus.get("consensus_action"),
                    "confidence": consensus.get("consensus_confidence"),
                })
                action = consensus.get("consensus_action", "hold")
                decision = execution_result.get("decision", "unknown")
                print(f"[TRADE] {data_symbol}: {action.upper()} -> {decision} (confidence: {consensus.get('consensus_confidence', 0)*100:.1f}%)")
            else:
                # Execution returned None - this means no trade was placed
                action = consensus.get("consensus_action", "hold")
                confidence = consensus.get("consensus_confidence", 0.0)
                print(f"[SKIP] {data_symbol}: No trade executed (action={action}, confidence={confidence*100:.1f}%)")
                if verbose:
                    print(f"  Consensus details: {consensus}")
                results["symbols_skipped"] += 1
                results["details"].append({
                    "symbol": data_symbol,
                    "status": "skipped",
                    "reason": "execution_returned_none",
                    "consensus_action": action,
                    "consensus_confidence": confidence,
                })
            
            results["symbols_processed"] += 1
            
        except Exception as exc:
            import traceback
            error_msg = f"{data_symbol}: {exc}"
            error_traceback = traceback.format_exc()
            results["errors"].append(error_msg)
            results["details"].append({
                "symbol": data_symbol,
                "status": "error",
                "error": str(exc),
                "traceback": error_traceback,
            })
            # Always print errors (not just when verbose) - critical for debugging
            print(f"[ERROR] {error_msg}")
            if verbose:
                print(f"[ERROR] Full traceback:\n{error_traceback}")
            else:
                # Even in non-verbose mode, show a shortened error
                print(f"[ERROR] {type(exc).__name__}: {str(exc)}")
            results["symbols_skipped"] += 1
    
    results["cycle_end"] = datetime.utcnow().isoformat() + "Z"
    results["cycle_duration_seconds"] = (datetime.utcnow() - cycle_start).total_seconds()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Live trading loop: run model predictions and execute trades on Alpaca paper account."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between trading cycles (default: 300 = 5 minutes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log intended trades without sending real orders to Alpaca",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle and exit (useful for testing)",
    )
    parser.add_argument(
        "--timeframe",
        default="1d",
        help="Timeframe for models (default: 1d)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Maximum number of cycles to run (default: unlimited)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress (default: True)",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("LIVE TRADING LOOP - CRYPTO ONLY")
    print("=" * 80)
    print(f"Mode: {'DRY RUN (no real orders)' if args.dry_run else 'LIVE TRADING'}")
    print(f"Interval: {args.interval} seconds")
    print(f"Timeframe: {args.timeframe}")
    print("=" * 80)
    print()
    
    # Show available horizons and their trading behavior
    print_horizon_summary()
    
    # Discover tradable symbols
    print("[DISCOVER] Finding symbols with trained models...")
    tradable = discover_tradable_symbols(asset_type="crypto", timeframe=args.timeframe)
    
    if not tradable:
        print("[ERROR] No tradable symbols found. Train models first:")
        print("  python train_models.py --symbols BTC-USDT ETH-USDT SOL-USDT --timeframe 1d")
        return
    
    print(f"[DISCOVER] Found {len(tradable)} tradable symbol(s):")
    for info in tradable:
        print(f"  - {info['asset'].data_symbol} ({info['asset'].trading_symbol}) - horizon: {info['horizon']}")
    print()
    
    # Initialize execution engine
    try:
        execution_engine = ExecutionEngine()
        print("[INIT] Execution engine ready")
    except Exception as exc:
        print(f"[ERROR] Failed to initialize execution engine: {exc}")
        print("Make sure ALPACA_API_KEY and ALPACA_SECRET_KEY are set in environment")
        return
    
    # Run trading loop
    cycle_count = 0
    try:
        while True:
            cycle_count += 1
            print(f"\n[CYCLE {cycle_count}] Starting trading cycle at {datetime.utcnow().isoformat()}")
            
            cycle_results = run_trading_cycle(
                execution_engine=execution_engine,
                tradable_symbols=tradable,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )
            
            print(f"[CYCLE {cycle_count}] Complete:")
            print(f"  Processed: {cycle_results['symbols_processed']}")
            print(f"  Traded: {cycle_results['symbols_traded']}")
            print(f"  Skipped: {cycle_results['symbols_skipped']}")
            if cycle_results["errors"]:
                print(f"  Errors: {len(cycle_results['errors'])}")
            
            # Log cycle summary
            cycle_log_path = Path("logs/trading/cycles.jsonl")
            cycle_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cycle_log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(cycle_results) + "\n")
            
            if args.once:
                break
            
            if args.max_cycles and cycle_count >= args.max_cycles:
                print(f"[STOP] Reached max cycles ({args.max_cycles})")
                break
            
            print(f"[WAIT] Sleeping {args.interval} seconds until next cycle...")
            time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user")
    except Exception as exc:
        print(f"\n[ERROR] Fatal error in trading loop: {exc}")
        raise


if __name__ == "__main__":
    main()

