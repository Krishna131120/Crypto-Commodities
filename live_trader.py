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


def get_current_price_from_features(asset_type: str, symbol: str, timeframe: str, force_live: bool = False, verbose: bool = False) -> Optional[float]:
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
                        price_val = float(price)
                        if verbose:
                            print(f"  [PRICE] {symbol}: ${price_val:.2f} from Alpaca API (last trade)")
                        return price_val
        except Exception as alpaca_exc:
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
                    price_val = float(latest.get("close", 0))
                    if price_val > 0:
                        timestamp = latest.get("timestamp", "unknown")
                        if verbose:
                            print(f"  [PRICE] {symbol}: ${price_val:.2f} from local Yahoo data.json (latest close, timestamp: {timestamp})")
                        return price_val
                elif isinstance(payload, dict) and "close" in payload:
                    price_val = float(payload["close"])
                    if price_val > 0:
                        if verbose:
                            print(f"  [PRICE] {symbol}: ${price_val:.2f} from local Yahoo data.json (latest close)")
                        return price_val
            except Exception:
                pass
        
        return None


def discover_tradable_symbols(asset_type: str = "crypto", timeframe: str = "1d", override_horizon: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Discover which symbols have trained models and are in our trading universe.
    
    Args:
        asset_type: Type of assets to discover ("crypto" or "commodities")
        timeframe: Timeframe to check (e.g., "1d")
        override_horizon: If provided, use this horizon instead of asset's default horizon
                         This allows command-line horizon to override symbol universe defaults
    
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
        
        # Use override_horizon if provided (from command line), otherwise use asset's horizon_profile
        # This allows --horizon intraday to override the symbol universe default of "short"
        preferred_horizon = override_horizon or asset.horizon_profile or DEFAULT_HORIZON_PROFILE
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
    user_stop_loss_pct: Optional[float] = None,
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
    # PRIMARY: Get live price from Binance (for crypto) or Angel One/Alpaca (for commodities) and update last candle's close price
    # This allows intraday trading with fresh prices even when daily candle isn't complete
    if update_data:
        if verbose:
            print(f"\n[STEP 1/4] UPDATING LIVE DATA")
            print(f"[UPDATE] Fetching latest live prices from Binance/Dhan/Alpaca...")
        try:
            from trading.alpaca_client import AlpacaClient
            from trading.symbol_universe import find_by_data_symbol
            from fetchers import load_json_file, save_json_file, get_data_path
            from pathlib import Path
            import json
            
            # Get unique symbols
            unique_symbols = list(set(info["asset"].data_symbol for info in tradable_symbols))
            updated_count = 0
            
            # COMMODITIES-ONLY: Use DhanClient exclusively (no crypto/Alpaca fallbacks)
            if hasattr(execution_engine, 'client') and execution_engine.client.broker_name == "angelone":
                client = execution_engine.client  # Use AngelOneClient for commodities
                if verbose:
                    print("  [INFO] Using AngelOneClient for commodities data updates (MCX exchange)")
            else:
                # If not AngelOneClient, this is an error for commodities trading
                raise RuntimeError("Commodities trading requires AngelOneClient. AlpacaClient is for crypto only.")
            
            for symbol in unique_symbols:
                try:
                    # COMMODITIES-ONLY: Get asset mapping to verify it's a commodity
                    asset_mapping = find_by_data_symbol(symbol)
                    if not asset_mapping:
                        if verbose:
                            print(f"  [SKIP] {symbol}: Not found in symbol universe")
                        continue
                    
                    # STRICT CHECK: Only process commodities
                    if asset_mapping.asset_type != "commodities":
                        if verbose:
                            print(f"  [SKIP] {symbol}: Not a commodity (asset_type={asset_mapping.asset_type})")
                        continue
                    
                    live_price = None
                    
                    # COMMODITIES-ONLY: Use AngelOneClient to get last trade price (MCX)
                    # NO Binance, NO Alpaca - only Angel One for commodities
                    try:
                        # Get MCX symbol for commodities (required for Dhan)
                        if hasattr(asset_mapping, 'get_mcx_symbol'):
                            # Get horizon from tradable_symbols info
                            symbol_info = next((info for info in tradable_symbols if info["asset"].data_symbol == symbol), None)
                            horizon = symbol_info.get("horizon", "short") if symbol_info else "short"
                            mcx_symbol = asset_mapping.get_mcx_symbol(horizon)
                            trading_symbol_for_price = mcx_symbol
                        else:
                            trading_symbol_for_price = asset_mapping.trading_symbol
                        
                        # Use AngelOneClient to get MCX last trade price
                        last_trade = client.get_last_trade(trading_symbol_for_price, max_retries=3, retry_delay=1.0, force_retry=False)
                        if last_trade:
                            price = last_trade.get("price") or last_trade.get("p") or last_trade.get("ltp")
                            if price:
                                live_price = float(price)
                                if verbose:
                                    print(f"  [OK] {symbol}: Got live price ${live_price:.2f} from Angel One (MCX: {trading_symbol_for_price})")
                    except Exception as trade_exc:
                        if verbose:
                            print(f"  [WARN] {symbol}: Angel One MCX price fetch failed: {trade_exc}")
                        pass
                    
                    if live_price is None or live_price <= 0:
                        if verbose:
                            print(f"  [WARN] {symbol}: No live price available - will use existing data.json price")
                        continue
                    
                    # COMMODITIES-ONLY: Load existing data.json from commodities paths only
                    asset_mapping = find_by_data_symbol(symbol)
                    if not asset_mapping or asset_mapping.asset_type != "commodities":
                        if verbose:
                            print(f"  [SKIP] {symbol}: Not a commodity symbol")
                        continue
                    
                    # Only commodities data paths (no crypto paths)
                    data_paths = [
                        get_data_path("commodities", symbol, "1d", None, "yahoo_chart").parent / "data.json",
                        get_data_path("commodities", symbol, "1d", None, "stooq").parent / "data.json",
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
                        source = "Dhan (MCX)"  # Commodities-only: always Dhan
                        print(f"  [OK] {symbol}: Updated last candle close to ${live_price:.2f} ({source} live)")
                        
                except Exception as sym_exc:
                    if verbose:
                        print(f"  [WARN] {symbol}: Failed to update with live price ({sym_exc})")
            
            if verbose:
                broker_name = execution_engine.client.broker_name if hasattr(execution_engine, 'client') else "Alpaca"
                print(f"[UPDATE] Live prices updated for {updated_count}/{len(unique_symbols)} commodity symbol(s) via Angel One (MCX exchange)")
        except Exception as exc:
            if verbose:
                print(f"[WARN] Failed to update live prices: {exc}")
            # Continue anyway - we'll use existing data
    
    # Step 2: Regenerate features for all symbols (if enabled)
    if regenerate_features_flag:
        if verbose:
            print(f"\n[STEP 2/4] REGENERATING FEATURES")
            print("[FEATURES] Regenerating features with latest data...")
        try:
            from pipeline_runner import regenerate_features
            unique_symbols = list(set(info["asset"].data_symbol for info in tradable_symbols))
            # Determine asset type from first symbol (all should be same type in a cycle)
            # COMMODITIES-ONLY: Verify all symbols are commodities
            asset_type = None
            if tradable_symbols:
                asset_type = tradable_symbols[0]["asset"].asset_type
                # Verify all are commodities
                for info in tradable_symbols:
                    if info["asset"].asset_type != "commodities":
                        raise RuntimeError(f"Non-commodity symbol found: {info['asset'].data_symbol} (type: {info['asset'].asset_type}). Commodities trading only.")
            else:
                asset_type = "commodities"  # Default to commodities for this script
            updated_count = regenerate_features(asset_type, set(unique_symbols), "1d")
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
    # UNLESS manual_stop_loss mode is enabled - then user manages stop-losses manually
    if verbose:
        manual_mode = getattr(execution_engine.risk, 'manual_stop_loss', False)
        if manual_mode:
            print(f"\n[STEP 3/4] CHECKING STOP-LOSSES (MANUAL MODE)")
            print("[STOP-LOSS] MANUAL MODE: Skipping automatic stop-loss checking (you manage stop-losses manually)")
        else:
            print(f"\n[STEP 3/4] CHECKING STOP-LOSSES")
            print("[STOP-LOSS] Checking existing positions for stop-loss triggers...")
    
    # Skip automatic stop-loss execution if manual mode is enabled
    manual_stop_loss_mode = getattr(execution_engine.risk, 'manual_stop_loss', False)
    
    if not manual_stop_loss_mode:
        try:
            from trading.symbol_universe import find_by_data_symbol
            from ml.horizons import get_horizon_risk_config, normalize_profile
            
            # COMMODITIES-ONLY: Use AngelOneClient exclusively
            if hasattr(execution_engine, 'client') and execution_engine.client.broker_name == "angelone":
                client = execution_engine.client  # AngelOneClient for commodities (MCX)
            else:
                raise RuntimeError("Commodities trading requires AngelOneClient. AlpacaClient is for crypto only.")
            all_positions = client.list_positions()
            stop_loss_triggered_count = 0
            
            # COMMODITIES-ONLY: Filter to MCX positions only (protect non-MCX positions)
            mcx_positions = []
            for pos in all_positions:
                exchange_segment = pos.get("exchange_segment", pos.get("_raw_exchange", "")).upper()
                if exchange_segment == "MCX":
                    mcx_positions.append(pos)
            
            # Only process MCX positions for stop-loss checking
            for position in mcx_positions:
                try:
                    trading_symbol = position.get("symbol", "")
                    position_qty = float(position.get("qty", 0) or 0)
                    
                    if position_qty == 0:
                        continue
                    
                    # Get data symbol from trading symbol (MCX commodities only)
                    asset_mapping = None
                    for symbol_info in tradable_symbols:
                        # For commodities, check MCX symbol match
                        if symbol_info["asset"].asset_type == "commodities":
                            if hasattr(symbol_info["asset"], 'get_mcx_symbol'):
                                horizon = symbol_info.get("horizon", "short")
                                mcx_symbol = symbol_info["asset"].get_mcx_symbol(horizon).upper()
                                if mcx_symbol == trading_symbol.upper():
                                    asset_mapping = symbol_info["asset"]
                                    break
                        elif symbol_info["asset"].trading_symbol.upper() == trading_symbol.upper():
                            asset_mapping = symbol_info["asset"]
                            break
                    
                    if not asset_mapping:
                        continue
                    
                    # STRICT CHECK: Only process commodities
                    if asset_mapping.asset_type != "commodities":
                        continue
                    
                    # Get current price - use force_live=True when monitoring positions
                    current_price = get_current_price_from_features(asset_mapping.asset_type, asset_mapping.data_symbol, "1d", force_live=True)
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
                print(f"[STOP-LOSS] Executed {stop_loss_triggered_count} MCX commodity stop-loss order(s)")
            elif mcx_positions:
                print(f"[STOP-LOSS] Checked {len(mcx_positions)} MCX commodity position(s), all within stop-loss limits")
                if len(all_positions) > len(mcx_positions):
                    print(f"[PROTECTED] {len(all_positions) - len(mcx_positions)} non-MCX position(s) exist but are NOT checked")
            else:
                print("[STOP-LOSS] No open positions to check")
                
        except Exception as exc:
            if verbose:
                print(f"[WARN] Failed to check stop-losses: {exc}")
            # Continue with trading cycle even if stop-loss check fails
    else:
        # Manual mode - just show positions without executing stop-losses
        if verbose:
            try:
                # COMMODITIES-ONLY: Use execution engine's AngelOneClient
                if hasattr(execution_engine, 'client') and execution_engine.client.broker_name == "angelone":
                    client = execution_engine.client
                else:
                    raise RuntimeError("Commodities trading requires AngelOneClient.")
                all_positions = client.list_positions()
                # COMMODITIES-ONLY: Filter to MCX positions only
                mcx_positions = [pos for pos in all_positions 
                               if pos.get("exchange_segment", pos.get("_raw_exchange", "")).upper() == "MCX"]
                if mcx_positions:
                    print(f"[STOP-LOSS] Manual mode: {len(mcx_positions)} MCX commodity position(s) active - you manage stop-losses manually")
                    if len(all_positions) > len(mcx_positions):
                        print(f"[PROTECTED] {len(all_positions) - len(mcx_positions)} non-MCX position(s) exist but are NOT managed")
                else:
                    print("[STOP-LOSS] Manual mode: No MCX commodity positions")
            except Exception:
                pass
    
    # Step 4: Run predictions and execute trades for each symbol
    if verbose:
        print(f"\n[STEP 4/4] RUNNING PREDICTIONS & EXECUTING TRADES")
        print(f"[TRADING] Processing {len(tradable_symbols)} symbol(s)...")
    
    for symbol_info in tradable_symbols:
        asset = symbol_info["asset"]
        model_dir = symbol_info["model_dir"]
        horizon = symbol_info["horizon"]
        data_symbol = asset.data_symbol
        
        try:
            # Load latest features (now freshly regenerated)
            # Load latest features (with retry in case of timing issues)
            asset_type = asset.asset_type
            feature_row = None
            for attempt in range(2):  # Try twice in case of file write timing
                feature_row = load_feature_row(asset_type, data_symbol, "1d")
                if feature_row is not None and not feature_row.empty:
                    break
                if attempt == 0:  # First attempt failed, wait a bit and try again
                    import time
                    time.sleep(0.5)
            
            if feature_row is None or feature_row.empty:
                if verbose:
                    from pathlib import Path
                    feature_path = Path("data/features") / asset_type / data_symbol / "1d" / "features.json"
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
            asset_type = asset.asset_type
            current_price = get_current_price_from_features(asset_type, data_symbol, "1d", force_live=False)
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
                # COMMODITIES-ONLY: Use AngelOneClient exclusively
                if hasattr(execution_engine, 'client') and execution_engine.client.broker_name == "angelone":
                    client_safety = execution_engine.client  # AngelOneClient for commodities (MCX)
                else:
                    raise RuntimeError("Commodities trading requires AngelOneClient. AlpacaClient is for crypto only.")
                
                # COMMODITIES-ONLY: Always use MCX symbol
                if hasattr(asset, 'get_mcx_symbol'):
                    trading_symbol_safety = asset.get_mcx_symbol(horizon).upper()
                else:
                    trading_symbol_safety = asset.trading_symbol.upper()
                
                existing_pos = client_safety.get_position(trading_symbol_safety)
                if existing_pos:
                    existing_pos_qty = float(existing_pos.get("qty", 0) or 0)
                    if existing_pos_qty > 0 and action == "short":
                        # LONG position but prediction is SHORT - force exit immediately (safety check)
                        print(f"  ⚠️  SAFETY CHECK: LONG position detected but prediction is SHORT - forcing exit")
                        try:
                            close_resp = client_safety.submit_order(
                                symbol=trading_symbol_safety,
                                qty=abs(existing_pos_qty),  # Commodities use lot-based qty (MCX)
                                notional=None,  # Commodities don't use notional
                                side="sell",
                                order_type="market",
                                time_in_force="gtc",
                            )
                            avg_entry = float(existing_pos.get("avg_entry_price", existing_pos.get("avg_entry_price", 0)) or 0)
                            realized_pl = (current_price - avg_entry) * abs(existing_pos_qty)
                            realized_pl_pct = ((current_price - avg_entry) / avg_entry) * 100 if avg_entry > 0 else 0
                            print(f"  ✅ FORCED EXIT: Closed LONG position @ ${current_price:.2f}")
                            print(f"  💰 Realized P/L: ${realized_pl:.2f} ({realized_pl_pct:+.2f}%)")
                            print(f"  Reason: Prediction changed to SHORT")
                            # Continue to execution - it will handle the SHORT entry if desired
                        except Exception as force_exc:
                            print(f"  ❌ ERROR in forced exit: {force_exc}")
                            # Continue anyway - execution engine will try again
                    elif existing_pos_qty < 0 and action == "long":
                        # SHORT position but prediction is LONG - force exit immediately (safety check)
                        print(f"  ⚠️  SAFETY CHECK: SHORT position detected but prediction is LONG - forcing exit")
                        try:
                            close_resp = client_safety.submit_order(
                                symbol=trading_symbol_safety,
                                qty=abs(existing_pos_qty),  # Commodities use lot-based qty (MCX)
                                notional=None,  # Commodities don't use notional
                                side="buy",
                                order_type="market",
                                time_in_force="gtc",
                            )
                            avg_entry = float(existing_pos.get("avg_entry_price", existing_pos.get("avg_entry_price", 0)) or 0)
                            realized_pl = (avg_entry - current_price) * abs(existing_pos_qty)
                            realized_pl_pct = ((avg_entry - current_price) / avg_entry) * 100 if avg_entry > 0 else 0
                            print(f"  ✅ FORCED EXIT: Closed SHORT position @ ${current_price:.2f}")
                            print(f"  💰 Realized P/L: ${realized_pl:.2f} ({realized_pl_pct:+.2f}%)")
                            print(f"  Reason: Prediction changed to LONG")
                            # Continue to execution - it will handle the LONG entry if desired
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
            # Calculate predicted price from current price and expected move
            predicted_price = current_price * (1.0 + expected_move)
            
            # Show model agreement info
            model_agreement = consensus.get("model_agreement_ratio", None)
            total_models = consensus.get("total_models", None)
            agreement_count = consensus.get("agreement_count", None)
            
            print(f"\n[PREDICTION] {data_symbol} ({asset.asset_type.upper()}):")
            print(f"  Action:           {action.upper()}")
            print(f"  Confidence:       {confidence_pct:.1f}%")
            print(f"  Current Price:    ${current_price:,.2f}")
            print(f"  Predicted Price:  ${predicted_price:,.2f}")
            print(f"  Expected Move:    {expected_move*100:+.2f}%")
            if model_agreement is not None and total_models is not None and total_models > 1:
                print(f"  Model Agreement:  {model_agreement*100:.1f}% ({agreement_count}/{total_models} models agree)")
            elif total_models == 1:
                print(f"  Model Agreement:  Single model (no consensus)")
            
            # Execute trade with horizon-specific risk parameters
            try:
                execution_result = execution_engine.execute_from_consensus(
                    asset=asset,
                    consensus=consensus,
                    current_price=current_price,
                    dry_run=dry_run,
                    horizon_profile=horizon,  # Pass horizon so engine uses horizon-specific risk config
                    profit_target_pct=profit_target_pct,  # Pass profit target if specified
                    user_stop_loss_pct=user_stop_loss_pct,  # Pass user stop-loss if specified
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
            else:
                # Execution returned None - show why trade was skipped
                action = consensus.get("consensus_action", "hold")
                confidence_raw = consensus.get('consensus_confidence', 0)
                if confidence_raw > 1.0:
                    confidence_display = confidence_raw
                else:
                    confidence_display = confidence_raw * 100
                
                # Check why trade was skipped
                skip_reason = "Unknown reason"
                
                # Check if it's a hold/flat action
                if action == "hold" or action == "flat":
                    skip_reason = f"Model prediction is {action.upper()} (no trade signal)"
                # Check confidence threshold
                elif asset.asset_type == "commodities":
                    min_confidence = 0.15  # 15% minimum for commodities
                    if confidence_display < min_confidence * 100:
                        skip_reason = f"Confidence {confidence_display:.1f}% below required {min_confidence*100:.0f}% for commodities (real money)"
                    else:
                        skip_reason = f"Position already aligned or other filter (action={action}, confidence={confidence_display:.1f}%)"
                else:
                    skip_reason = f"Position already aligned or other filter (action={action}, confidence={confidence_display:.1f}%)"
                
                print(f"\n[SKIP] {data_symbol}: No trade executed")
                print(f"  Reason: {skip_reason}")
                print(f"  Action: {action.upper()}, Confidence: {confidence_display:.1f}%")
                
                results["symbols_skipped"] += 1
                results["details"].append({
                    "symbol": data_symbol,
                    "status": "skipped",
                    "reason": skip_reason,
                    "model_action": action,
                    "confidence": confidence_display,
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
    
    if verbose:
        print(f"\n[COMPLETE] Trading cycle finished in {results['cycle_duration_seconds']:.1f} seconds")
    
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
    parser.add_argument(
        "--manual-stop-loss",
        action="store_true",
        help="Enable manual stop-loss management. System will NOT submit or execute stop-loss orders automatically. You manage stop-losses yourself.",
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
    if args.manual_stop_loss:
        print(f"Stop-Loss: MANUAL MODE (you manage stop-losses)")
    else:
        print(f"Stop-Loss: AUTOMATIC (system manages stop-losses)")
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
    
    # Initialize execution engine with risk config
    try:
        risk_config = TradingRiskConfig(manual_stop_loss=args.manual_stop_loss)
        execution_engine = ExecutionEngine(risk_config=risk_config)
        if args.manual_stop_loss:
            print("[INIT] Execution engine ready (MANUAL STOP-LOSS MODE enabled)")
            print("  ⚠️  You are responsible for managing stop-losses manually")
            print("  📝 System will calculate stop-loss levels but will NOT submit or execute them")
        else:
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

