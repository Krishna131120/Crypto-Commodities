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
from ml.horizons import DEFAULT_HORIZON_PROFILE, normalize_profile, print_horizon_summary, get_horizon_risk_config
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


def get_current_price_from_features(asset_type: str, symbol: str, timeframe: str, force_live: bool = False) -> Optional[float]:
    """
    Resolve the latest tradable price for a symbol.

    Priority for CRYPTO:
    1. Binance REST API (primary live source - works reliably for crypto)
    2. Alpaca position-based price (if we have an open position)
    3. Local Binance/Yahoo data.json (fallback, only if force_live=False)

    Priority for COMMODITIES:
    1. Alpaca last trade (if available)
    2. Local Yahoo data.json (fallback)

    Args:
        asset_type: Type of asset ("crypto" or "commodities")
        symbol: Data symbol (e.g., "BTC-USDT")
        timeframe: Timeframe (e.g., "1d")
        force_live: If True, will retry more aggressively and NOT fall back to data.json
                    (use this when monitoring active positions)
    
    Returns:
        Current price as float, or None if unavailable
    """
    # For crypto, use Binance as primary source (Alpaca doesn't provide crypto market data)
    if asset_type == "crypto":
        # 1) Try Binance REST API first (most reliable for crypto)
        try:
            from fetchers import get_binance_current_price
            binance_price = get_binance_current_price(symbol)
            if binance_price and binance_price > 0:
                return float(binance_price)
        except Exception:
            pass
        
        # 2) Try Alpaca position-based price (if we have an open position)
        try:
            from trading.alpaca_client import AlpacaClient
            from trading.symbol_universe import find_by_data_symbol as _find

            asset_mapping = _find(symbol)
            if asset_mapping:
                client = AlpacaClient()
                position = client.get_position(asset_mapping.trading_symbol)
                if position:
                    market_value = float(position.get("market_value", 0) or 0)
                    qty = float(position.get("qty", 0) or 0)
                    if qty != 0 and market_value != 0:
                        price = abs(market_value / qty)
                        if price > 0:
                            return float(price)
        except Exception:
            pass
        
        # 3) Fallback: local data.json (only if force_live=False)
        if force_live:
            return None
        
        data_path = Path("data/json/raw") / asset_type / "binance" / symbol / timeframe / "data.json"
        if not data_path.exists():
            data_path = Path("data/json/raw") / asset_type / "alpaca" / symbol / timeframe / "data.json"
        
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
    
    # For commodities, use Alpaca first (if available)
    else:
        # 1) Try Alpaca first
        try:
            from trading.alpaca_client import AlpacaClient
            from trading.symbol_universe import find_by_data_symbol as _find

            asset_mapping = _find(symbol)
            if asset_mapping:
                client = AlpacaClient()
                max_retries = 8 if force_live else 5
                retry_delay = 0.5 if force_live else 1.0
                last_trade = client.get_last_trade(
                    asset_mapping.trading_symbol,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    force_retry=force_live
                )
                if last_trade:
                    price = last_trade.get("price") or last_trade.get("p")
                    if price:
                        return float(price)
        except Exception:
            if force_live:
                return None
            pass
        
        # 2) Fallback: local data.json (only if force_live=False)
        if force_live:
            return None
        
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
        # CRITICAL: Only use the preferred horizon - don't fallback to wrong horizon
        # This ensures each horizon uses its own trained models and produces different predictions
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
                # REMOVED: Fallback to first available horizon
                # This was causing intraday and long to use short model, producing identical predictions
                # elif model_dir is None:
                #     model_dir = horizon_path
                #     used_horizon = horizon_name
        
        # If preferred horizon model doesn't exist, skip this symbol
        # This prevents using wrong horizon models which causes identical predictions
        if model_dir is None or used_horizon != preferred_horizon:
            continue
        
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
    profit_target_pct: Optional[float] = None,
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
        profit_target_pct: Optional profit target percentage (e.g., 0.15 for 0.15%%). If None, uses horizon default.
    
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
    # PRIMARY: Get live price from Binance (for crypto) or Alpaca (for commodities) and update last candle's close price
    # This allows intraday trading with fresh prices even when daily candle isn't complete
    if update_data:
        if verbose:
            print("[UPDATE] Fetching latest live prices from Binance/Alpaca...")
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
                    # For crypto, use Binance as primary source (Alpaca doesn't provide crypto market data)
                    # For commodities, try Alpaca first
                    live_price = None
                    
                    # Try Binance first for crypto
                    try:
                        from fetchers import get_binance_current_price
                        binance_price = get_binance_current_price(symbol)
                        if binance_price and binance_price > 0:
                            live_price = float(binance_price)
                            if verbose:
                                print(f"  [OK] {symbol}: Got live price ${live_price:.2f} from Binance")
                    except Exception as binance_exc:
                        if verbose:
                            print(f"  [WARN] {symbol}: Binance price fetch failed: {binance_exc}")
                    
                    # If Binance failed, try Alpaca position-based price (for crypto) or last trade (for commodities)
                    if live_price is None or live_price <= 0:
                        asset_mapping = find_by_data_symbol(symbol)
                        if asset_mapping:
                            # Try position-based price (works if we have an open position)
                            try:
                                position = client.get_position(asset_mapping.trading_symbol)
                                if position:
                                    market_value = float(position.get("market_value", 0) or 0)
                                    qty = float(position.get("qty", 0) or 0)
                                    if qty != 0 and market_value != 0:
                                        live_price = abs(market_value / qty)
                                        if verbose:
                                            print(f"  [OK] {symbol}: Got live price ${live_price:.2f} from Alpaca position")
                            except Exception:
                                pass
                            
                            # For commodities, try last trade endpoint
                            if (live_price is None or live_price <= 0) and asset_mapping.asset_type != "crypto":
                                try:
                                    last_trade = client.get_last_trade(asset_mapping.trading_symbol, max_retries=3, retry_delay=1.0, force_retry=False)
                                    if last_trade:
                                        price = last_trade.get("price") or last_trade.get("p")
                                        if price:
                                            live_price = float(price)
                                            if verbose:
                                                print(f"  [OK] {symbol}: Got live price ${live_price:.2f} from Alpaca")
                                except Exception:
                                    pass
                    
                    if live_price is None or live_price <= 0:
                        if verbose:
                            print(f"  [WARN] {symbol}: No live price available - will use existing data.json price")
                        continue
                    
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
                        # Determine source for message
                        source = "Binance" if asset_mapping and asset_mapping.asset_type == "crypto" else "Alpaca"
                        print(f"  [OK] {symbol}: Updated last candle close to ${live_price:.2f} ({source} live)")
                        
                except Exception as sym_exc:
                    if verbose:
                        print(f"  [WARN] {symbol}: Failed to update with live price ({sym_exc})")
            
            if verbose:
                print(f"[UPDATE] Live prices updated for {updated_count}/{len(unique_symbols)} symbol(s) (Binance for crypto, Alpaca for commodities)")
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
            updated_count = regenerate_features("crypto", set(unique_symbols), "1d")
            if verbose:
                if updated_count > 0:
                    print(f"[FEATURES] Features regenerated for {updated_count}/{len(unique_symbols)} symbol(s)")
                else:
                    print(f"[WARN] Failed to regenerate features for any symbol(s)")
        except Exception as exc:
            if verbose:
                print(f"[WARN] Failed to regenerate features: {exc}")
            # Continue anyway - we'll use existing features
    
    # Step 3: Check existing positions for stop-loss triggers BEFORE executing new trades
    # This ensures we exit losing positions immediately, protecting capital
    if verbose:
        print("[STOP-LOSS] Checking existing positions for stop-loss triggers...")
    
    try:
        from trading.alpaca_client import AlpacaClient
        from trading.symbol_universe import find_by_data_symbol
        from ml.horizons import get_horizon_risk_config, normalize_profile
        
        client = AlpacaClient()
        all_positions = client.list_positions()
        stop_loss_triggered_count = 0
        
        for position in all_positions:
            try:
                trading_symbol = position.get("symbol", "")
                position_qty = float(position.get("qty", 0) or 0)
                
                if position_qty == 0:
                    continue
                
                # Get data symbol from trading symbol
                asset_mapping = None
                for symbol_info in tradable_symbols:
                    if symbol_info["asset"].trading_symbol.upper() == trading_symbol.upper():
                        asset_mapping = symbol_info["asset"]
                        break
                
                if not asset_mapping:
                    continue
                
                # Get current price - use force_live=True when monitoring positions
                current_price = get_current_price_from_features("crypto", asset_mapping.data_symbol, "1d", force_live=True)
                if current_price is None or current_price <= 0:
                    continue
                
                # Get entry price from position
                avg_entry_price = float(position.get("avg_entry_price", 0) or 0)
                if avg_entry_price <= 0:
                    continue
                
                # Get horizon for this symbol to determine stop-loss percentage
                horizon = normalize_profile(getattr(asset_mapping, "horizon_profile", None) or "short")
                horizon_risk = get_horizon_risk_config(horizon)
                stop_loss_pct = horizon_risk.get("default_stop_loss_pct", 0.02)
                
                # Calculate stop-loss price
                is_long = position_qty > 0
                if is_long:
                    stop_loss_price = avg_entry_price * (1.0 - stop_loss_pct)
                    price_change_pct = ((current_price - avg_entry_price) / avg_entry_price) * 100
                    # Add small buffer (0.1%) to account for slippage - trigger slightly before exact stop-loss
                    stop_loss_triggered = current_price <= (stop_loss_price * 1.001)
                else:
                    stop_loss_price = avg_entry_price * (1.0 + stop_loss_pct)
                    price_change_pct = ((avg_entry_price - current_price) / avg_entry_price) * 100
                    # Add small buffer (0.1%) to account for slippage - trigger slightly before exact stop-loss
                    stop_loss_triggered = current_price >= (stop_loss_price * 0.999)
                
                # Calculate accurate position metrics (ALWAYS recalculate, don't trust Alpaca's values)
                market_value = float(position.get("market_value", 0) or 0)
                
                # ALWAYS recalculate P/L ourselves for accuracy
                if avg_entry_price > 0 and current_price > 0 and abs(position_qty) > 0:
                    if is_long:
                        # Long position: profit when current price > entry price
                        unrealized_pl = (current_price - avg_entry_price) * abs(position_qty)
                        unrealized_pl_pct = ((current_price - avg_entry_price) / avg_entry_price) * 100
                    else:
                        # Short position: profit when current price < entry price
                        unrealized_pl = (avg_entry_price - current_price) * abs(position_qty)
                        unrealized_pl_pct = ((avg_entry_price - current_price) / avg_entry_price) * 100
                else:
                    unrealized_pl = 0
                    unrealized_pl_pct = 0
                
                # Calculate accurate market value if not provided
                if market_value == 0 and current_price > 0 and abs(position_qty) > 0:
                    market_value = abs(position_qty) * current_price
                
                print(f"\n[POSITION] {asset_mapping.data_symbol} ({trading_symbol}):")
                print(f"  Current Price: ${current_price:.2f}")
                print(f"  Entry Price: ${avg_entry_price:.2f}")
                print(f"  Quantity: {abs(position_qty):.8f} {'LONG' if is_long else 'SHORT'}")
                print(f"  Market Value: ${market_value:.2f}")
                print(f"  Unrealized P/L: ${unrealized_pl:.2f} ({unrealized_pl_pct:+.2f}%)")
                if unrealized_pl > 0:
                    print(f"  ✅ IN PROFIT")
                elif unrealized_pl < 0:
                    print(f"  ⚠️  IN LOSS")
                else:
                    print(f"  ➖ Break even")
                print(f"  Stop-Loss Price: ${stop_loss_price:.2f} ({stop_loss_pct*100:.2f}% from entry)")
                print(f"  Price Change: {price_change_pct:+.2f}%")
                
                # Check if stop-loss is triggered OR if prediction changed to SHORT (force exit)
                # First, check if this symbol has a SHORT prediction (we'll check in the trading cycle)
                # For now, just check stop-loss trigger
                should_force_exit = False
                force_exit_reason = None
                
                # Check if prediction is SHORT for this symbol (will be checked in main trading cycle)
                # This is a safety check - if position exists and prediction is SHORT, we should exit
                # The main trading cycle will handle this, but we add a safety check here too
                
                if stop_loss_triggered:
                    should_force_exit = True
                    force_exit_reason = "stop_loss_triggered"
                    print(f"  ⚠️  STOP-LOSS TRIGGERED! Current price ${current_price:.2f} {'<=' if is_long else '>='} stop-loss ${stop_loss_price:.2f}")
                else:
                    # Position is still active, show distance to stop-loss
                    distance_to_stop = abs(current_price - stop_loss_price) / current_price * 100
                    print(f"  ✓ Position active (Distance to stop-loss: {distance_to_stop:.2f}%)")
                
                if should_force_exit and not dry_run:
                    try:
                        # Execute stop-loss: close the position
                        close_qty = abs(position_qty)
                        close_side = "sell" if is_long else "buy"
                        
                        close_resp = client.submit_order(
                            symbol=trading_symbol,
                            qty=close_qty,
                            side=close_side,
                            order_type="market",
                            time_in_force="gtc",
                        )
                        
                        print(f"  ✅ STOP-LOSS EXECUTED: Closed {close_qty:.8f} @ ${current_price:.2f}")
                        print(f"  💰 Realized {'Loss' if unrealized_pl < 0 else 'P/L'}: ${unrealized_pl:.2f} ({unrealized_pl_pct:+.2f}%)")
                        
                        stop_loss_triggered_count += 1
                        results["details"].append({
                            "symbol": asset_mapping.data_symbol,
                            "status": "stop_loss_executed",
                            "entry_price": avg_entry_price,
                            "exit_price": current_price,
                            "quantity": close_qty,
                            "realized_pl": unrealized_pl,
                            "realized_pl_pct": unrealized_pl_pct,
                            "exit_reason": force_exit_reason,
                        })
                    except Exception as stop_exc:
                        print(f"  ❌ ERROR executing stop-loss: {stop_exc}")
                        results["errors"].append(f"{asset_mapping.data_symbol}: Stop-loss execution failed: {stop_exc}")
                    
            except Exception as pos_exc:
                if verbose:
                    print(f"  [WARN] Error checking position: {pos_exc}")
        
        if stop_loss_triggered_count > 0:
            print(f"[STOP-LOSS] Executed {stop_loss_triggered_count} stop-loss order(s)")
        elif all_positions:
            print(f"[STOP-LOSS] Checked {len(all_positions)} position(s), all within stop-loss limits")
        else:
            print("[STOP-LOSS] No open positions to check")
            
    except Exception as exc:
        if verbose:
            print(f"[WARN] Failed to check stop-losses: {exc}")
        # Continue with trading cycle even if stop-loss check fails
    
    # Step 4: Run predictions and execute trades for each symbol
    for symbol_info in tradable_symbols:
        asset = symbol_info["asset"]
        model_dir = symbol_info["model_dir"]
        horizon = symbol_info["horizon"]
        data_symbol = asset.data_symbol
        
        try:
            # Load latest features (now freshly regenerated)
            # Load latest features (with retry in case of timing issues)
            feature_row = None
            for attempt in range(2):  # Try twice in case of file write timing
                feature_row = load_feature_row("crypto", data_symbol, "1d")
                if feature_row is not None and not feature_row.empty:
                    break
                if attempt == 0:  # First attempt failed, wait a bit and try again
                    import time
                    time.sleep(0.5)
            
            if feature_row is None or feature_row.empty:
                if verbose:
                    from pathlib import Path
                    feature_path = Path("data/features") / "crypto" / data_symbol / "1d" / "features.json"
                    if feature_path.exists():
                        print(f"[SKIP] {data_symbol}: Features file exists but couldn't be loaded (may be corrupted)")
                    else:
                        print(f"[SKIP] {data_symbol}: No features available (file not found)")
                results["symbols_skipped"] += 1
                results["details"].append({
                    "symbol": data_symbol,
                    "status": "skipped",
                    "reason": "no_features",
                })
                continue
            
            # Get current price (from Alpaca - always fresh)
            # Get current price - use force_live=False for predictions (allows fallback)
            # But we update data.json with live prices in the update step above
            current_price = get_current_price_from_features("crypto", data_symbol, "1d", force_live=False)
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
            
            # CRITICAL SAFETY CHECK: If we have a LONG position but prediction is SHORT, force exit immediately
            # This ensures we always exit when prediction changes, even if execution engine fails
            action = consensus.get("consensus_action", "hold")
            try:
                from trading.alpaca_client import AlpacaClient
                client_safety = AlpacaClient()
                existing_pos = client_safety.get_position(asset.trading_symbol)
                if existing_pos:
                    existing_pos_qty = float(existing_pos.get("qty", 0) or 0)
                    if existing_pos_qty > 0 and action == "short":
                        # LONG position but prediction is SHORT - force exit immediately (safety check)
                        print(f"  ⚠️  SAFETY CHECK: LONG position detected but prediction is SHORT - forcing exit")
                        try:
                            close_resp = client_safety.submit_order(
                                symbol=asset.trading_symbol,
                                qty=abs(existing_pos_qty),
                                side="sell",
                                order_type="market",
                                time_in_force="gtc",
                            )
                            avg_entry = float(existing_pos.get("avg_entry_price", 0) or 0)
                            realized_pl = (current_price - avg_entry) * abs(existing_pos_qty)
                            realized_pl_pct = ((current_price - avg_entry) / avg_entry) * 100 if avg_entry > 0 else 0
                            print(f"  ✅ FORCED EXIT: Closed LONG position @ ${current_price:.2f}")
                            print(f"  💰 Realized P/L: ${realized_pl:.2f} ({realized_pl_pct:+.2f}%)")
                            print(f"  Reason: Prediction changed to SHORT")
                            # Continue to execution - it will handle the SHORT entry if desired
                        except Exception as force_exc:
                            print(f"  ❌ ERROR in forced exit: {force_exc}")
                            # Continue anyway - execution engine will try again
            except Exception:
                pass  # If we can't check, continue normally - execution engine will handle it
            
            # Display predicted action before execution
            confidence = consensus.get("consensus_confidence", 0.0)
            # Confidence should be 0.0-1.0 (decimal), but check if it's already a percentage
            if confidence > 1.0:
                # Already a percentage, don't multiply
                confidence_pct = confidence
            else:
                # Decimal form, convert to percentage
                confidence_pct = confidence * 100
            # Use predicted_return (consensus_return) for expected move
            expected_move = consensus.get("consensus_return", consensus.get("predicted_return", 0.0))
            print(f"[PREDICTION] {data_symbol}: {action.upper()} (confidence: {confidence_pct:.1f}%, expected move: {expected_move*100:+.2f}%)")
            
            # Execute trade with horizon-specific risk parameters
            try:
                execution_result = execution_engine.execute_from_consensus(
                    asset=asset,
                    consensus=consensus,
                    current_price=current_price,
                    dry_run=dry_run,
                    horizon_profile=horizon,  # Pass horizon so engine uses horizon-specific risk config
                    profit_target_pct=profit_target_pct,  # Pass profit target if specified
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
                
                # Display comprehensive trading information
                confidence_raw = consensus.get('consensus_confidence', 0)
                # Confidence should be 0.0-1.0 (decimal), but check if it's already a percentage
                if confidence_raw > 1.0:
                    # Already a percentage, don't multiply
                    confidence_display = confidence_raw
                else:
                    # Decimal form, convert to percentage
                    confidence_display = confidence_raw * 100
                print(f"\n[TRADE] {data_symbol}: {action.upper()} -> {decision} (confidence: {confidence_display:.1f}%)")
                
                # Get current position details from Alpaca (accurate, not predefined)
                try:
                    from trading.alpaca_client import AlpacaClient
                    client = AlpacaClient()
                    position = client.get_position(asset.trading_symbol)
                except Exception:
                    position = None
                
                # Show detailed trade information if a position was entered or added to
                if decision in ["enter_long", "enter_short", "flip_to_long", "flip_to_short", "add_to_position"]:
                    # Get accurate entry price and quantity from Alpaca position (if exists)
                    if position:
                        avg_entry_price = float(position.get("avg_entry_price", 0) or 0)
                        position_qty = float(position.get("qty", 0) or 0)
                        market_value = float(position.get("market_value", 0) or 0)
                        # ALWAYS recalculate P/L ourselves for accuracy (don't trust Alpaca's value)
                        is_long = position_qty > 0
                        if is_long:
                            # LONG position: profit when current price > entry price
                            unrealized_pl = (current_price - avg_entry_price) * abs(position_qty)
                            unrealized_pl_pct = ((current_price - avg_entry_price) / avg_entry_price) * 100 if avg_entry_price > 0 else 0
                        else:
                            # SHORT position: profit when current price < entry price
                            unrealized_pl = (avg_entry_price - current_price) * abs(position_qty)
                            unrealized_pl_pct = ((avg_entry_price - current_price) / avg_entry_price) * 100 if avg_entry_price > 0 else 0
                    else:
                        # Fallback to execution result if position not yet available
                        avg_entry_price = execution_result.get("entry_price", current_price)
                        position_qty = execution_result.get("entry_qty", execution_result.get("final_qty", 0))
                        market_value = abs(position_qty) * current_price
                        unrealized_pl = 0  # Just entered, no P/L yet
                        unrealized_pl_pct = 0
                    
                    entry_qty = execution_result.get("entry_qty", execution_result.get("additional_qty", 0))
                    entry_notional = execution_result.get("entry_notional", execution_result.get("additional_notional", 0))
                    stop_loss_price = execution_result.get("stop_loss_price", 0)
                    stop_loss_pct = execution_result.get("stop_loss_pct", 0)
                    take_profit_price = execution_result.get("take_profit_price", 0)
                    add_reason = execution_result.get("add_reason", "")
                    is_in_loss = execution_result.get("is_in_loss", False)
                    
                    # Calculate stop-loss if not provided (always show it)
                    if stop_loss_price == 0 and avg_entry_price > 0:
                        from ml.horizons import get_horizon_risk_config
                        horizon_risk = get_horizon_risk_config(horizon)
                        stop_loss_pct = horizon_risk.get("default_stop_loss_pct", 0.02)
                        stop_loss_price = avg_entry_price * (1.0 - stop_loss_pct)
                    
                    print(f"  📊 TRADE DETAILS:")
                    print(f"     Current Price: ${current_price:.2f}")
                    print(f"     Entry Price: ${avg_entry_price:.2f}")
                    if decision == "add_to_position":
                        print(f"     Quantity Added: {entry_qty:.8f} (Reason: {add_reason.replace('_', ' ').title()})")
                        print(f"     Total Quantity: {abs(position_qty):.8f}")
                        if is_in_loss:
                            print(f"     ⚠️  Position is IN LOSS - adding more to recover")
                    else:
                        print(f"     Quantity: {abs(position_qty):.8f}")
                    print(f"     Notional: ${entry_notional:.2f}")
                    print(f"     Market Value: ${market_value:.2f}")
                    if position and abs(position_qty) > 0:
                        print(f"     Unrealized P/L: ${unrealized_pl:.2f} ({unrealized_pl_pct:+.2f}%)")
                        if unrealized_pl > 0:
                            print(f"     ✅ IN PROFIT")
                        elif unrealized_pl < 0:
                            print(f"     ⚠️  IN LOSS")
                    # Always show stop-loss (it's always active)
                    if stop_loss_price > 0:
                        print(f"     Stop-Loss: ${stop_loss_price:.2f} ({stop_loss_pct*100:.2f}% from entry)")
                        print(f"     ⚠️  Stop-loss is ACTIVE and will trigger if price reaches ${stop_loss_price:.2f}")
                    if take_profit_price:
                        print(f"     Take-Profit: ${take_profit_price:.2f}")
                
                # Show position exit information if a position was closed
                elif decision in ["exit_position", "flip_to_flat", "flip_to_long", "flip_to_short"]:
                    close_order = execution_result.get("close_order")
                    exit_reason = execution_result.get("exit_reason", "unknown")
                    entry_price = execution_result.get("entry_price", current_price)
                    exit_price = execution_result.get("exit_price", current_price)
                    realized_pl = execution_result.get("realized_pl", 0)
                    realized_pl_pct = execution_result.get("realized_pl_pct", 0)
                    trade_qty = execution_result.get("trade_qty", 0)
                    trade_side = execution_result.get("trade_side", "")
                    
                    # Calculate accurate P/L if not provided (fallback calculation)
                    if realized_pl == 0 and entry_price > 0 and exit_price > 0 and trade_qty > 0:
                        if trade_side.upper() == "SELL":
                            # Long position closed
                            realized_pl = (exit_price - entry_price) * trade_qty
                            realized_pl_pct = ((exit_price - entry_price) / entry_price) * 100
                        elif trade_side.upper() == "BUY":
                            # Short position closed
                            realized_pl = (entry_price - exit_price) * trade_qty
                            realized_pl_pct = ((entry_price - exit_price) / entry_price) * 100
                    
                    if close_order:
                        # Position was closed - show profit/loss with accurate calculations
                        print(f"  📊 POSITION CLOSED:")
                        print(f"     Reason: {exit_reason.replace('_', ' ').title()}")
                        print(f"     Entry Price: ${entry_price:.2f}")
                        print(f"     Exit Price: ${exit_price:.2f}")
                        print(f"     Quantity Closed: {abs(trade_qty):.8f}")
                        print(f"     Side: {trade_side.upper()}")
                        print(f"     💰 Realized P/L: ${realized_pl:.2f} ({realized_pl_pct:+.2f}%)")
                        
                        if realized_pl > 0:
                            print(f"     ✅ PROFIT LOCKED IN!")
                        elif realized_pl < 0:
                            print(f"     ⚠️  Loss realized")
                        else:
                            print(f"     ➖ Break even")
                        
                        # Check if position still exists (might be partial fill)
                        try:
                            from trading.alpaca_client import AlpacaClient
                            client = AlpacaClient()
                            remaining_position = client.get_position(asset.trading_symbol)
                            
                            if remaining_position:
                                # Position still exists (partial fill)
                                remaining_qty = float(remaining_position.get("qty", 0) or 0)
                                avg_entry = float(remaining_position.get("avg_entry_price", 0) or 0)
                                market_value = float(remaining_position.get("market_value", 0) or 0)
                                
                                # ALWAYS recalculate P/L ourselves for accuracy
                                is_remaining_long = remaining_qty > 0
                                if avg_entry > 0 and current_price > 0 and abs(remaining_qty) > 0:
                                    if is_remaining_long:
                                        unrealized_pl = (current_price - avg_entry) * abs(remaining_qty)
                                        unrealized_pl_pct = ((current_price - avg_entry) / avg_entry) * 100
                                    else:
                                        unrealized_pl = (avg_entry - current_price) * abs(remaining_qty)
                                        unrealized_pl_pct = ((avg_entry - current_price) / avg_entry) * 100
                                else:
                                    unrealized_pl = 0
                                    unrealized_pl_pct = 0
                                
                                print(f"  📊 REMAINING POSITION:")
                                print(f"     Current Price: ${current_price:.2f}")
                                print(f"     Entry Price: ${avg_entry:.2f}")
                                print(f"     Quantity: {abs(remaining_qty):.8f}")
                                print(f"     Market Value: ${market_value:.2f}")
                                print(f"     Unrealized P/L: ${unrealized_pl:.2f} ({unrealized_pl_pct:+.2f}%)")
                                if unrealized_pl > 0:
                                    print(f"     ✅ IN PROFIT")
                                elif unrealized_pl < 0:
                                    print(f"     ⚠️  IN LOSS")
                        except Exception:
                            pass
                
                # Show no-change information if position is maintained
                elif decision == "no_change":
                    try:
                        from trading.alpaca_client import AlpacaClient
                        client = AlpacaClient()
                        position = client.get_position(asset.trading_symbol)
                        
                        if position:
                            # Get accurate position data from Alpaca (not predefined)
                            avg_entry = float(position.get("avg_entry_price", 0) or 0)
                            position_qty = float(position.get("qty", 0) or 0)
                            market_value = float(position.get("market_value", 0) or 0)
                            
                            # ALWAYS recalculate P/L ourselves for accuracy (don't trust Alpaca's value)
                            is_long = position_qty > 0
                            if avg_entry > 0 and current_price > 0 and abs(position_qty) > 0:
                                if is_long:
                                    # Long position: profit when current price > entry price
                                    unrealized_pl = (current_price - avg_entry) * abs(position_qty)
                                    unrealized_pl_pct = ((current_price - avg_entry) / avg_entry) * 100
                                else:
                                    # Short position: profit when current price < entry price
                                    unrealized_pl = (avg_entry - current_price) * abs(position_qty)
                                    unrealized_pl_pct = ((avg_entry - current_price) / avg_entry) * 100
                            else:
                                unrealized_pl = 0
                                unrealized_pl_pct = 0
                            
                            # Get stop-loss from horizon
                            horizon_risk = get_horizon_risk_config(horizon)
                            stop_loss_pct = horizon_risk.get("default_stop_loss_pct", 0.02)
                            stop_loss_price = avg_entry * (1.0 - stop_loss_pct) if is_long else avg_entry * (1.0 + stop_loss_pct)
                            
                            print(f"  📊 POSITION MAINTAINED:")
                            print(f"     Current Price: ${current_price:.2f}")
                            print(f"     Entry Price: ${avg_entry:.2f}")
                            print(f"     Quantity: {abs(position_qty):.8f} {'LONG' if is_long else 'SHORT'}")
                            print(f"     Market Value: ${market_value:.2f}")
                            print(f"     Unrealized P/L: ${unrealized_pl:.2f} ({unrealized_pl_pct:+.2f}%)")
                            if unrealized_pl > 0:
                                print(f"     ✅ IN PROFIT")
                            elif unrealized_pl < 0:
                                print(f"     ⚠️  IN LOSS")
                            else:
                                print(f"     ➖ Break even")
                            print(f"     Stop-Loss: ${stop_loss_price:.2f} ({stop_loss_pct*100:.2f}% from entry)")
                            
                            # Show distance to stop-loss
                            if is_long:
                                distance_to_stop = ((current_price - stop_loss_price) / current_price) * 100
                            else:
                                distance_to_stop = ((stop_loss_price - current_price) / current_price) * 100
                            
                            if distance_to_stop > 0:
                                print(f"     Distance to Stop-Loss: {distance_to_stop:.2f}%")
                            else:
                                print(f"     ⚠️  WARNING: Price is at or below stop-loss! Should trigger on next cycle.")
                    except Exception as e:
                        if verbose:
                            print(f"  [WARN] Could not fetch position details: {e}")
            else:
                # Execution returned None - this means no trade was placed
                action = consensus.get("consensus_action", "hold")
                confidence_raw = consensus.get("consensus_confidence", 0.0)
                # Confidence should be 0.0-1.0 (decimal), but check if it's already a percentage
                if confidence_raw > 1.0:
                    # Already a percentage, don't multiply
                    confidence_display = confidence_raw
                else:
                    # Decimal form, convert to percentage
                    confidence_display = confidence_raw * 100
                print(f"[SKIP] {data_symbol}: No trade executed (action={action}, confidence={confidence_display:.1f}%)")
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
        default=60,
        help="Seconds between trading cycles. Runs forever. Minimum: 30 seconds (to avoid rate limiting). Default: 60 seconds.",
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
    
    # Validate interval (minimum 30 seconds to avoid API rate limiting)
    if args.interval < 30:
        print(f"⚠️  WARNING: Interval {args.interval} seconds is too short. Minimum is 30 seconds to avoid rate limiting.")
        print(f"   Setting interval to 30 seconds.")
        args.interval = 30
    
    print("=" * 80)
    print("LIVE TRADING LOOP - CRYPTO ONLY")
    print("=" * 80)
    print(f"Mode: {'DRY RUN (no real orders)' if args.dry_run else 'LIVE TRADING'}")
    print(f"Interval: {args.interval} seconds (runs forever)")
    print(f"Timeframe: {args.timeframe}")
    if args.max_cycles:
        print(f"Max Cycles: {args.max_cycles}")
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

