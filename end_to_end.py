"""
End-to-end trading pipeline for crypto and commodities:

Runs the full flow for selected symbols:
1) Historical ingestion  (raw candles)
2) Feature generation    (features.json)
3) Model training        (models/crypto/... or models/commodities/...)
4) Live trading cycles on Alpaca paper account with profit target monitoring

This script is a single entry point so you can:
- Choose symbols yourself via CLI (crypto and/or commodities)
- Let it do all steps in sequence
- Set profit target percentage (MANDATORY)
- Monitor positions for profit targets and stop-loss
- See only a concise summary of what actually happened

IMPORTANT:
- Profit target percentage is MANDATORY (--profit-target)
- Existing open positions in Alpaca are monitored for profit targets
- Positions are automatically closed when profit target is reached or stop-loss is hit
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from pipeline_runner import run_ingestion, regenerate_features
from train_models import train_symbols
from trading.alpaca_client import AlpacaClient
from trading.execution_engine import ExecutionEngine
from trading.symbol_universe import all_enabled, find_by_data_symbol, find_by_trading_symbol
from trading.position_manager import PositionManager
from trading.trade_logger import TradeLogger
from core.model_paths import horizon_dir, list_horizon_dirs
from ml.horizons import (
    DEFAULT_HORIZON_PROFILE,
    normalize_profile,
    print_horizon_summary,
    get_horizon_risk_config,
)
from ml.inference import InferencePipeline
from ml.risk import RiskManagerConfig
from fetchers import load_json_file, save_json_file, get_data_path


def load_feature_row(asset_type: str, symbol: str, timeframe: str) -> Optional[pd.Series]:
    """Load the latest feature row from features.json."""
    feature_path = Path("data/features") / asset_type / symbol / timeframe / "features.json"
    if not feature_path.exists():
        return None
    
    try:
        payload = json.loads(feature_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "features" in payload:
            features_dict = payload["features"]
            clean_features = {}
            for name, value_data in features_dict.items():
                if isinstance(value_data, dict):
                    clean_features[name] = value_data.get("value")
                else:
                    clean_features[name] = value_data
            return pd.Series(clean_features)
        elif isinstance(payload, list):
            if payload:
                return pd.Series(payload[-1])
        return None
    except Exception as exc:
        print(f"[ERROR] Failed to load features for {symbol}: {exc}")
        return None


def sync_existing_alpaca_positions(
    position_manager: PositionManager,
    tradable_symbols: List[Dict[str, Any]],
    profit_target_pct: float,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Check for existing positions in Alpaca and sync them with position manager.
    
    If a position exists in Alpaca but not in position manager, create a Position
    object with the current profit target settings.
    
    Returns:
        Dict with sync results: {
            "positions_found": int,
            "positions_synced": int,
            "positions_details": List[Dict]
        }
    """
    from trading.alpaca_client import AlpacaClient
    from trading.symbol_universe import find_by_data_symbol
    from ml.horizons import get_horizon_risk_config, normalize_profile
    
    results = {
        "positions_found": 0,
        "positions_synced": 0,
        "positions_details": [],
    }
    
    try:
        client = AlpacaClient()
        alpaca_positions = client.list_positions()
        
        if not alpaca_positions:
            if verbose:
                print("[SYNC] No existing positions found in Alpaca")
            return results
        
        results["positions_found"] = len(alpaca_positions)
        
        # Build lookup for tradable symbols
        symbol_lookup = {}
        for info in tradable_symbols:
            asset = info["asset"]
            symbol_lookup[asset.trading_symbol.upper()] = {
                "asset": asset,
                "horizon": info.get("horizon", "short"),
            }
        
        for alpaca_pos in alpaca_positions:
            try:
                trading_symbol = alpaca_pos.get("symbol", "").upper()
                if not trading_symbol:
                    continue
                
                # Check if this symbol is in our tradable list
                symbol_info = symbol_lookup.get(trading_symbol)
                if not symbol_info:
                    if verbose:
                        print(f"  [SKIP] {trading_symbol}: Not in tradable symbols list")
                    continue
                
                asset = symbol_info["asset"]
                qty = float(alpaca_pos.get("qty", 0) or 0)
                
                if qty == 0:
                    continue
                
                # Check if position already exists in position manager
                existing_tracked = position_manager.get_position(trading_symbol)
                if existing_tracked and existing_tracked.status == "open":
                    # Position exists - check if profit target needs updating
                    if abs(existing_tracked.profit_target_pct - profit_target_pct) > 0.01:  # Only update if different
                        # Calculate risk/reward ratios
                        old_rr_ratio = existing_tracked.profit_target_pct / (existing_tracked.stop_loss_pct * 100) if existing_tracked.stop_loss_pct > 0 else 0
                        new_rr_ratio = profit_target_pct / (stop_loss_pct * 100) if stop_loss_pct > 0 else 0
                        
                        # Determine if this is a "tightening" (reducing target) or "widening" (increasing target)
                        is_tightening = profit_target_pct < existing_tracked.profit_target_pct
                        
                        if verbose:
                            print(f"\n{'='*80}")
                            print(f"⚠️  PROFIT TARGET CHANGE DETECTED: {trading_symbol}")
                            print(f"{'='*80}")
                            print(f"  Current Target:  {existing_tracked.profit_target_pct:.2f}%")
                            print(f"  Requested Target: {profit_target_pct:.2f}%")
                            print(f"  Change Type:     {'TIGHTENING' if is_tightening else 'WIDENING'} (Target {'reduced' if is_tightening else 'increased'})")
                            print(f"\n  Risk/Reward Impact:")
                            print(f"    Old R/R Ratio:   {old_rr_ratio:.2f}:1")
                            print(f"    New R/R Ratio:   {new_rr_ratio:.2f}:1")
                        
                        # OPTION B: Allow tightening, block widening
                        if is_tightening:
                            # TIGHTENING - ALLOWED (more conservative)
                            if verbose:
                                print(f"\n  ✅ TIGHTENING ALLOWED (Conservative Change)")
                                print(f"     • You'll exit sooner (reduces risk)")
                                print(f"     • This is a conservative adjustment")
                                print(f"     • Updating profit target to {profit_target_pct:.2f}%")
                                print(f"{'='*80}\n")
                            
                            # Update the profit target
                            position_manager.update_profit_target(
                                trading_symbol,
                                profit_target_pct,
                                stop_loss_pct=stop_loss_pct,
                            )
                            existing_tracked = position_manager.get_position(trading_symbol)  # Refresh
                        else:
                            # WIDENING - BLOCKED (prevents greed-driven changes)
                            if verbose:
                                print(f"\n  ❌ WIDENING BLOCKED (Prevents Greed-Driven Changes)")
                                print(f"     • Cannot increase profit target mid-trade")
                                print(f"     • This prevents moving goalposts based on greed")
                                print(f"     • Original trade plan: {existing_tracked.profit_target_pct:.2f}% target")
                                print(f"\n  💡 To use {profit_target_pct:.2f}% target:")
                                print(f"     1. Close this position manually")
                                print(f"     2. Restart the script with --profit-target {profit_target_pct:.2f}")
                                print(f"     3. System will open a new position with the new target")
                                print(f"\n  ℹ️  Keeping original target: {existing_tracked.profit_target_pct:.2f}%")
                                print(f"{'='*80}\n")
                            
                            # Don't update - keep original target
                            # existing_tracked remains unchanged
                    else:
                        if verbose:
                            print(f"  [INFO] {trading_symbol}: Already tracked with correct profit target ({profit_target_pct:.2f}%)")
                    
                    # Still show details of the existing position
                    current_price = float(alpaca_pos.get("current_price", 0) or 0)
                    if current_price <= 0:
                        # Try to get current price
                        current_price = get_current_price_from_features(
                            asset.asset_type,
                            asset.data_symbol,
                            "1d",
                            verbose=False
                        ) or avg_entry_price
                    
                    # Calculate current P/L
                    if existing_tracked.side == "long":
                        unrealized_pl = (current_price - existing_tracked.entry_price) * existing_tracked.quantity
                        unrealized_pl_pct = ((current_price - existing_tracked.entry_price) / existing_tracked.entry_price) * 100
                    else:
                        unrealized_pl = (existing_tracked.entry_price - current_price) * existing_tracked.quantity
                        unrealized_pl_pct = ((existing_tracked.entry_price - current_price) / existing_tracked.entry_price) * 100
                    
                    initial_investment = existing_tracked.entry_price * abs(existing_tracked.quantity)
                    current_value = current_price * abs(existing_tracked.quantity)
                    
                    # Calculate expected profit/loss
                    if existing_tracked.side == "long":
                        expected_profit_at_target = (existing_tracked.profit_target_price - existing_tracked.entry_price) * abs(existing_tracked.quantity)
                        expected_loss_at_stop = (existing_tracked.stop_loss_price - existing_tracked.entry_price) * abs(existing_tracked.quantity)
                    else:
                        expected_profit_at_target = (existing_tracked.entry_price - existing_tracked.profit_target_price) * abs(existing_tracked.quantity)
                        expected_loss_at_stop = (existing_tracked.stop_loss_price - existing_tracked.entry_price) * abs(existing_tracked.quantity)
                    
                    if verbose:
                        print(f"\n{'='*80}")
                        print(f"EXISTING POSITION (Updated): {trading_symbol}")
                        print(f"{'='*80}")
                        print(f"\n💰 EXISTING INVESTMENT:")
                        print(f"  Initial Investment: ${initial_investment:.2f}")
                        print(f"    └─ Entry Price:   ${existing_tracked.entry_price:.2f}")
                        print(f"    └─ Quantity:      {abs(existing_tracked.quantity):.6f}")
                        print(f"    └─ Side:          {existing_tracked.side.upper()}")
                        print(f"    └─ Asset:         {asset.data_symbol} ({asset.asset_type})")
                        
                        print(f"\n📊 CURRENT STATUS:")
                        print(f"  Current Price:     ${current_price:.2f}")
                        print(f"  Current Value:     ${current_value:.2f}")
                        print(f"  Current P/L:       ${unrealized_pl:+.2f} ({unrealized_pl_pct:+.2f}%)")
                        
                        print(f"\n🎯 TARGET SCENARIOS (Profit Target: {existing_tracked.profit_target_pct:.2f}%):")
                        print(f"  Profit Target:")
                        print(f"    └─ Target Price:  ${existing_tracked.profit_target_price:.2f} ({existing_tracked.profit_target_pct:+.2f}%)")
                        print(f"    └─ Expected Profit: ${expected_profit_at_target:+.2f}")
                        print(f"    └─ Total Value at Target: ${initial_investment + expected_profit_at_target:.2f}")
                        print(f"  Stop-Loss:")
                        print(f"    └─ Stop Price:    ${existing_tracked.stop_loss_price:.2f} ({existing_tracked.stop_loss_pct*100:.2f}%)")
                        print(f"    └─ Expected Loss: ${expected_loss_at_stop:.2f}")
                        print(f"    └─ Total Value at Stop: ${initial_investment + expected_loss_at_stop:.2f}")
                        print(f"{'='*80}\n")
                    
                    results["positions_synced"] += 1
                    continue
                
                # Get position details from Alpaca
                avg_entry_price = float(alpaca_pos.get("avg_entry_price", 0) or 0)
                current_price = float(alpaca_pos.get("current_price", 0) or 0)
                market_value = float(alpaca_pos.get("market_value", 0) or 0)
                unrealized_pl = float(alpaca_pos.get("unrealized_pl", 0) or 0)
                
                if avg_entry_price <= 0 or current_price <= 0:
                    if verbose:
                        print(f"  [WARN] {trading_symbol}: Invalid price data from Alpaca")
                    continue
                
                # Determine side
                side = "long" if qty > 0 else "short"
                
                # Get horizon-specific stop-loss
                horizon = normalize_profile(symbol_info.get("horizon", "short"))
                horizon_risk = get_horizon_risk_config(horizon)
                stop_loss_pct = horizon_risk.get("default_stop_loss_pct", 0.02)
                
                # Calculate profit target and stop-loss prices
                if side == "long":
                    profit_target_price = avg_entry_price * (1.0 + profit_target_pct / 100.0)
                    stop_loss_price = avg_entry_price * (1.0 - stop_loss_pct)
                else:  # short
                    profit_target_price = avg_entry_price * (1.0 - profit_target_pct / 100.0)
                    stop_loss_price = avg_entry_price * (1.0 + stop_loss_pct)
                
                # Create position in position manager
                position = position_manager.save_position(
                    symbol=trading_symbol,
                    data_symbol=asset.data_symbol,
                    asset_type=asset.asset_type,
                    side=side,
                    entry_price=avg_entry_price,
                    quantity=abs(qty),
                    profit_target_pct=profit_target_pct,
                    stop_loss_pct=stop_loss_pct,
                )
                
                # Calculate current P/L
                if side == "long":
                    unrealized_pl_pct = ((current_price - avg_entry_price) / avg_entry_price) * 100
                else:
                    unrealized_pl_pct = ((avg_entry_price - current_price) / avg_entry_price) * 100
                
                initial_investment = avg_entry_price * abs(qty)
                current_value = current_price * abs(qty)
                
                # Calculate expected profit/loss
                if side == "long":
                    expected_profit_at_target = (profit_target_price - avg_entry_price) * abs(qty)
                    expected_loss_at_stop = (stop_loss_price - avg_entry_price) * abs(qty)
                else:
                    expected_profit_at_target = (avg_entry_price - profit_target_price) * abs(qty)
                    expected_loss_at_stop = (stop_loss_price - avg_entry_price) * abs(qty)
                
                results["positions_synced"] += 1
                results["positions_details"].append({
                    "symbol": trading_symbol,
                    "data_symbol": asset.data_symbol,
                    "side": side,
                    "quantity": abs(qty),
                    "entry_price": avg_entry_price,
                    "current_price": current_price,
                })
                
                if verbose:
                    print(f"\n{'='*80}")
                    print(f"EXISTING POSITION SYNCED: {trading_symbol}")
                    print(f"{'='*80}")
                    print(f"\n💰 EXISTING INVESTMENT:")
                    print(f"  Initial Investment: ${initial_investment:.2f}")
                    print(f"    └─ Entry Price:   ${avg_entry_price:.2f}")
                    print(f"    └─ Quantity:      {abs(qty):.6f}")
                    print(f"    └─ Side:          {side.upper()}")
                    print(f"    └─ Asset:         {asset.data_symbol} ({asset.asset_type})")
                    
                    print(f"\n📊 CURRENT STATUS:")
                    print(f"  Current Price:     ${current_price:.2f}")
                    print(f"  Current Value:     ${current_value:.2f}")
                    print(f"  Current P/L:       ${unrealized_pl:+.2f} ({unrealized_pl_pct:+.2f}%)")
                    
                    print(f"\n🎯 TARGET SCENARIOS (New Profit Target: {profit_target_pct}%):")
                    print(f"  Profit Target:")
                    print(f"    └─ Target Price:  ${profit_target_price:.2f} ({profit_target_pct:+.2f}%)")
                    print(f"    └─ Expected Profit: ${expected_profit_at_target:+.2f}")
                    print(f"    └─ Total Value at Target: ${initial_investment + expected_profit_at_target:.2f}")
                    print(f"  Stop-Loss:")
                    print(f"    └─ Stop Price:    ${stop_loss_price:.2f} ({stop_loss_pct*100:.2f}%)")
                    print(f"    └─ Expected Loss: ${expected_loss_at_stop:.2f}")
                    print(f"    └─ Total Value at Stop: ${initial_investment + expected_loss_at_stop:.2f}")
                    print(f"{'='*80}\n")
                    
            except Exception as pos_exc:
                if verbose:
                    print(f"  [ERROR] Failed to sync position {alpaca_pos.get('symbol', '?')}: {pos_exc}")
        
        if verbose and results["positions_synced"] > 0:
            print(f"[SYNC] Synced {results['positions_synced']} existing position(s) from Alpaca")
        
    except Exception as exc:
        if verbose:
            print(f"[WARN] Failed to sync existing positions: {exc}")
    
    return results


def get_current_price_from_features(asset_type: str, symbol: str, timeframe: str, verbose: bool = False, force_live: bool = False) -> Optional[float]:
    """
    Resolve the latest tradable price for a symbol.
    
    Priority for CRYPTO:
    1. Binance REST API (primary live source - works reliably for crypto)
    2. Alpaca position-based price (if we have an open position)
    3. Local Binance/Yahoo data.json (fallback only if force_live=False)
    
    Priority for COMMODITIES:
    1. Alpaca last trade (if available)
    2. Local Yahoo data.json (fallback)
    
    Args:
        asset_type: Type of asset ("crypto" or "commodities")
        symbol: Data symbol (e.g., "BTC-USDT")
        timeframe: Timeframe (e.g., "1d")
        verbose: Whether to print verbose messages
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
                if verbose:
                    print(f"  [PRICE] {symbol}: ${binance_price:.2f} (Binance live)")
                return float(binance_price)
        except Exception as binance_exc:
            if verbose:
                print(f"  [WARN] {symbol}: Binance price fetch failed: {binance_exc}")
        
        # 2) Try Alpaca position-based price (if we have an open position)
        try:
            asset_mapping = find_by_data_symbol(symbol)
            if asset_mapping:
                client = AlpacaClient()
                position = client.get_position(asset_mapping.trading_symbol)
                if position:
                    market_value = float(position.get("market_value", 0) or 0)
                    qty = float(position.get("qty", 0) or 0)
                    if qty != 0 and market_value != 0:
                        price = abs(market_value / qty)
                        if price > 0:
                            if verbose:
                                # Show source - Binance for crypto, Alpaca for commodities
                                source = "Binance" if asset_type == "crypto" else "Alpaca position"
                                print(f"  [PRICE] {symbol}: ${price:.2f} ({source})")
                            return float(price)
        except Exception as alpaca_exc:
            if verbose:
                print(f"  [WARN] {symbol}: Alpaca position price fetch failed: {alpaca_exc}")
        
        # 3) Fallback: local data.json (only if force_live=False)
        if force_live:
            if verbose:
                print(f"  [ERROR] {symbol}: Force live mode - no live price available from Binance or Alpaca")
            return None
        
        data_path = Path("data/json/raw") / asset_type / "binance" / symbol / timeframe / "data.json"
        if not data_path.exists():
            data_path = Path("data/json/raw") / asset_type / "alpaca" / symbol / timeframe / "data.json"
        
        if data_path.exists():
            try:
                payload = json.loads(data_path.read_text(encoding="utf-8"))
                if isinstance(payload, list) and payload:
                    latest = payload[-1]
                    fallback_price = float(latest.get("close", 0))
                    if fallback_price > 0:
                        if verbose:
                            print(f"  [WARN] {symbol}: Using fallback price ${fallback_price:.2f} from data.json (live sources unavailable)")
                        return fallback_price
                elif isinstance(payload, dict) and "close" in payload:
                    fallback_price = float(payload["close"])
                    if fallback_price > 0:
                        if verbose:
                            print(f"  [WARN] {symbol}: Using fallback price ${fallback_price:.2f} from data.json (live sources unavailable)")
                        return fallback_price
            except Exception as fallback_exc:
                if verbose:
                    print(f"  [ERROR] {symbol}: Fallback price fetch also failed: {fallback_exc}")
        
        if verbose:
            print(f"  [ERROR] {symbol}: Could not get price from any source")
        return None
    
    # For commodities, use Alpaca first (if available)
    else:
        # 1) Try Alpaca first
        try:
            asset_mapping = find_by_data_symbol(symbol)
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
                        live_price = float(price)
                        if verbose:
                            print(f"  [PRICE] {symbol}: ${live_price:.2f} (Alpaca live)")
                        return live_price
                else:
                    if verbose:
                        print(f"  [WARN] {symbol}: Alpaca returned no trade data after {max_retries} retries")
            else:
                if verbose:
                    print(f"  [WARN] {symbol}: No asset mapping found")
        except Exception as alpaca_exc:
            if verbose:
                print(f"  [WARN] {symbol}: Alpaca price fetch failed: {alpaca_exc}")
            if force_live:
                if verbose:
                    print(f"  [ERROR] {symbol}: Force live mode - refusing to use fallback price")
                return None
        
        # 2) Fallback: local data.json (only if force_live=False)
        if force_live:
            if verbose:
                print(f"  [ERROR] {symbol}: Force live mode - no fallback available")
            return None
        
        data_path = Path("data/json/raw") / asset_type / "yahoo_chart" / symbol / timeframe / "data.json"
        if data_path.exists():
            try:
                payload = json.loads(data_path.read_text(encoding="utf-8"))
                if isinstance(payload, list) and payload:
                    latest = payload[-1]
                    fallback_price = float(latest.get("close", 0))
                    if fallback_price > 0:
                        if verbose:
                            print(f"  [WARN] {symbol}: Using fallback price ${fallback_price:.2f} from data.json (Alpaca unavailable)")
                        return fallback_price
                elif isinstance(payload, dict) and "close" in payload:
                    fallback_price = float(payload["close"])
                    if fallback_price > 0:
                        if verbose:
                            print(f"  [WARN] {symbol}: Using fallback price ${fallback_price:.2f} from data.json (Alpaca unavailable)")
                        return fallback_price
            except Exception as fallback_exc:
                if verbose:
                    print(f"  [ERROR] {symbol}: Fallback price fetch also failed: {fallback_exc}")
        
        if verbose:
            print(f"  [ERROR] {symbol}: Could not get price from any source")
        return None


def discover_tradable_symbols(asset_type: str = "crypto", timeframe: str = "1d", verbose: bool = False, override_horizon: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Discover which symbols have trained models and are in our trading universe.
    
    Args:
        asset_type: Type of assets to discover ("crypto" or "commodities")
        timeframe: Timeframe to check (e.g., "1d")
        verbose: Whether to print skip messages
        override_horizon: If provided, use this horizon instead of asset's default horizon
                         This allows command-line horizon to override symbol universe defaults
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
        # This allows --crypto-horizon long to override the symbol universe default of "short"
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
            if verbose:
                print(f"[SKIP] {asset.data_symbol}: No model found for preferred horizon '{preferred_horizon}'. "
                      f"Available horizons: {[d.name for d in horizon_dirs if (d / 'summary.json').exists()]}")
            continue
        
        if model_dir and (model_dir / "summary.json").exists():
            # Check if model is marked as tradable
            try:
                with open(model_dir / "summary.json", "r") as f:
                    summary = json.load(f)
                is_tradable = summary.get("tradable", True)
                if not is_tradable:
                    continue
            except Exception:
                continue
            
            tradable.append({
                "asset": asset,
                "model_dir": model_dir,
                "horizon": used_horizon,
            })
    
    return tradable


def display_active_positions(
    position_manager: PositionManager,
    client: AlpacaClient,
    verbose: bool = True,
) -> None:
    """Display all active positions with current status."""
    active_positions = position_manager.get_all_positions()
    
    if not active_positions:
        if verbose:
            print("[ACTIVE POSITIONS] No active positions")
        return
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"ACTIVE POSITIONS ({len(active_positions)} position(s))")
        print("=" * 80)
        
        for pos in active_positions:
            try:
                # Try to get current price from Alpaca
                asset_mapping = find_by_data_symbol(pos.data_symbol)
                current_price = None
                if asset_mapping:
                    alpaca_pos = client.get_position(asset_mapping.trading_symbol)
                    if alpaca_pos:
                        market_value = float(alpaca_pos.get("market_value", 0) or 0)
                        qty = float(alpaca_pos.get("qty", 0) or 0)
                        if qty != 0:
                            current_price = abs(market_value / qty)
                
                if not current_price:
                    current_price = get_current_price_from_features(
                        pos.asset_type,
                        pos.data_symbol,
                        "1d",
                        verbose=False,
                        force_live=True
                    )
                
                if current_price:
                    # Calculate P/L
                    if pos.side == "long":
                        unrealized_pl = (current_price - pos.entry_price) * pos.quantity
                        unrealized_pl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                    else:
                        unrealized_pl = (pos.entry_price - current_price) * pos.quantity
                        unrealized_pl_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100
                    
                    current_value = pos.quantity * current_price
                    
                    print(f"\n{pos.symbol} ({pos.data_symbol}):")
                    print(f"  Side: {pos.side.upper()}")
                    print(f"  Quantity: {pos.quantity:.6f}")
                    print(f"  Entry Price: ${pos.entry_price:.6f}")
                    print(f"  Current Price: ${current_price:.6f}")
                    print(f"  Cost Basis: ${pos.total_cost_basis:.2f}")
                    print(f"  Current Value: ${current_value:.2f}")
                    print(f"  Unrealized P/L: ${unrealized_pl:+.2f} ({unrealized_pl_pct:+.2f}%)")
                    print(f"  Profit Target: ${pos.profit_target_price:.6f} ({pos.profit_target_pct}%)")
                    print(f"  Stop Loss: ${pos.stop_loss_price:.6f} ({pos.stop_loss_pct}%)")
                    print(f"  Entry Time: {pos.entry_time}")
                else:
                    print(f"\n{pos.symbol} ({pos.data_symbol}):")
                    print(f"  Side: {pos.side.upper()}")
                    print(f"  Quantity: {pos.quantity:.6f}")
                    print(f"  Entry Price: ${pos.entry_price:.6f}")
                    print(f"  [WARN] Could not get current price")
            except Exception as e:
                print(f"\n{pos.symbol}: Error displaying position: {e}")
        
        print("=" * 80 + "\n")


def monitor_positions(
    position_manager: PositionManager,
    execution_engine: ExecutionEngine,
    tradable_symbols: List[Dict[str, Any]],
    trade_logger: Optional[TradeLogger] = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Monitor all active positions for profit targets, stop-loss triggers, and prediction changes.
    
    NEW LOGIC: Exit positions when:
    1. Profit target is hit (MANDATORY - must hit target)
    2. Stop-loss is hit
    3. Prediction changes to SHORT or FLAT (for long positions) or LONG/FLAT (for short positions)
    
    Returns:
        Dict with monitoring results
    """
    results = {
        "positions_checked": 0,
        "profit_targets_hit": 0,
        "stop_losses_hit": 0,
        "prediction_changes": 0,
        "positions_active": 0,
        "exits_executed": [],
        "errors": [],
    }
    
    if verbose:
        print("[MONITOR] Checking active positions for profit targets, stop-loss, and prediction changes...")
    
    # Get all active positions from position manager
    active_positions = position_manager.get_all_positions()
    
    if not active_positions:
        if verbose:
            print("[MONITOR] No active positions to monitor")
        return results
    
    client = AlpacaClient()
    
    # CRITICAL: Sync with Alpaca to verify positions are still open
    # If position was closed in Alpaca but position manager still has it, close it properly
    try:
        alpaca_positions = client.list_positions()
        alpaca_symbols = {pos.get("symbol", "").upper() for pos in (alpaca_positions or []) if float(pos.get("qty", 0) or 0) != 0}
        
        # Close positions from position manager that no longer exist in Alpaca
        positions_to_close = []
        for position in active_positions:
            if position.symbol.upper() not in alpaca_symbols:
                # Position closed in Alpaca but still tracked - close it properly
                positions_to_close.append(position)
                if verbose:
                    print(f"  [SYNC] {position.symbol}: Position closed in Alpaca, closing in tracker")
        
        for position in positions_to_close:
            # Try to get exit price - use current market price if available
            exit_price = None
            try:
                exit_price = get_current_price_from_features(
                    position.asset_type,
                    position.data_symbol,
                    "1d",
                    verbose=False,
                    force_live=True
                )
            except:
                pass
            
            # If we can't get current price, use entry price as fallback
            if not exit_price or exit_price <= 0:
                exit_price = position.entry_price
                exit_reason = "closed_externally_unknown_price"
            else:
                exit_reason = "closed_externally_alpaca_sync"
            
            # Close the position properly (preserves history)
            position_manager.close_position(
                position.symbol,
                exit_price,
                exit_reason,
            )
        
        # Update active positions list after cleanup
        active_positions = position_manager.get_all_positions()
    except Exception as sync_exc:
        if verbose:
            print(f"  [WARN] Failed to sync with Alpaca positions: {sync_exc}")
    
    if not active_positions:
        if verbose:
            print("[MONITOR] No active positions after sync")
        return results
    
    results["positions_checked"] = len(active_positions)
    
    # Build a lookup for symbol info to run predictions
    symbol_info_lookup = {}
    for info in tradable_symbols:
        asset = info["asset"]
        symbol_info_lookup[asset.data_symbol] = info
    
    for position in active_positions:
        try:
            # Get current price - ALWAYS fetch fresh from Alpaca
            # Try multiple methods to ensure we get the most accurate live price
            current_price = None
            
            # Method 1: Try to get price from Alpaca position (most accurate)
            try:
                asset_mapping = find_by_data_symbol(position.data_symbol)
                if asset_mapping:
                    alpaca_position = client.get_position(asset_mapping.trading_symbol)
                    if alpaca_position:
                        # Get current market value and quantity to calculate price
                        market_value = float(alpaca_position.get("market_value", 0) or 0)
                        qty = float(alpaca_position.get("qty", 0) or 0)
                        if qty != 0:
                            current_price = abs(market_value / qty)
                            if verbose:
                                # Show source - Binance for crypto, Alpaca for commodities
                                source = "Binance" if position.asset_type == "crypto" else "Alpaca position"
                                print(f"  [PRICE] {position.symbol}: ${current_price:.2f} (from {source})")
            except Exception as pos_exc:
                if verbose:
                    print(f"  [WARN] {position.symbol}: Could not get price from Alpaca position: {pos_exc}")
            
            # Method 2: If position method failed, try last trade API with force_live=True
            # This ensures we get real-time prices when monitoring active positions
            if current_price is None or current_price <= 0:
                current_price = get_current_price_from_features(
                    position.asset_type,
                    position.data_symbol,
                    "1d",
                    verbose=verbose,
                    force_live=True  # Force live prices when monitoring positions
                )
            
            if current_price is None or current_price <= 0:
                if verbose:
                    print(f"  [SKIP] {position.symbol}: Could not get current price from any source")
                continue
            
            # Update position and check if target/stop-loss hit
            exit_info = position_manager.update_position(
                position.symbol,
                current_price,
            )
            
            should_exit = False
            exit_reason = None
            exit_price = current_price
            pos = position
            
            if exit_info and exit_info.get("should_exit"):
                # Profit target or stop-loss hit - execute exit
                should_exit = True
                exit_reason = exit_info["exit_reason"]
                exit_price = exit_info["exit_price"]
                pos = exit_info["position"]
                
                if exit_reason == "profit_target_hit":
                    results["profit_targets_hit"] += 1
                elif exit_reason == "stop_loss_hit":
                    results["stop_losses_hit"] += 1
                
                # Log position closure
                if trade_logger:
                    trade_logger.log_position_closed(
                        symbol=pos.symbol,
                        data_symbol=pos.data_symbol,
                        asset_type=pos.asset_type,
                        side=pos.side,
                        quantity=pos.quantity,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        realized_pl=pos.realized_pl or 0.0,
                        realized_pl_pct=pos.realized_pl_pct or 0.0,
                        cost_basis=pos.total_cost_basis or 0.0,
                        profit_target_pct=pos.profit_target_pct,
                        stop_loss_pct=pos.stop_loss_pct,
                        order_id=None,  # Will be filled if available
                        dry_run=dry_run,
                    )
            else:
                # Position still active - check prediction for pyramiding or exit
                # NEW STRATEGY: Keep buying more as long as prediction stays LONG until profit target is hit
                # Only exit when profit target is hit AND prediction changes to SHORT/FLAT
                symbol_info = symbol_info_lookup.get(position.data_symbol)
                if symbol_info:
                    try:
                        asset = symbol_info["asset"]
                        model_dir = symbol_info["model_dir"]
                        
                        # Load features
                        feature_row = load_feature_row(position.asset_type, position.data_symbol, "1d")
                        if feature_row is not None and not feature_row.empty:
                            # Load inference pipeline
                            from ml.risk import RiskManagerConfig
                            from ml.inference import InferencePipeline
                            
                            risk_config = RiskManagerConfig(paper_trade=True)
                            pipeline = InferencePipeline(model_dir, risk_config=risk_config)
                            pipeline.load()
                            
                            if pipeline.models:
                                # Run prediction
                                volatility = 0.01  # Default
                                prediction_result = pipeline.predict(
                                    feature_row,
                                    current_price=current_price,
                                    volatility=volatility,
                                )
                                
                                consensus = prediction_result.get("consensus", {})
                                current_action = consensus.get("consensus_action", "hold")
                                
                                # Calculate current P/L percentage and detailed metrics
                                if position.side == "long":
                                    unrealized_pl = (current_price - position.entry_price) * position.quantity
                                    unrealized_pl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                                    profit_target_hit = current_price >= position.profit_target_price
                                    progress_to_target = (unrealized_pl_pct / position.profit_target_pct) * 100 if position.profit_target_pct > 0 else 0
                                    remaining_to_target_pct = position.profit_target_pct - unrealized_pl_pct
                                    remaining_to_target_price = position.profit_target_price - current_price
                                else:  # short
                                    unrealized_pl = (position.entry_price - current_price) * position.quantity
                                    unrealized_pl_pct = ((position.entry_price - current_price) / position.entry_price) * 100
                                    profit_target_hit = current_price <= position.profit_target_price
                                    progress_to_target = (unrealized_pl_pct / position.profit_target_pct) * 100 if position.profit_target_pct > 0 else 0
                                    remaining_to_target_pct = position.profit_target_pct - unrealized_pl_pct
                                    remaining_to_target_price = current_price - position.profit_target_price
                                
                                # Calculate investment details
                                initial_investment = position.entry_price * position.quantity
                                current_value = current_price * position.quantity
                                
                                # Calculate expected profit at target
                                if position.side == "long":
                                    expected_profit_at_target = (position.profit_target_price - position.entry_price) * position.quantity
                                    expected_loss_at_stop = (position.stop_loss_price - position.entry_price) * position.quantity
                                else:  # short
                                    expected_profit_at_target = (position.entry_price - position.profit_target_price) * position.quantity
                                    expected_loss_at_stop = (position.stop_loss_price - position.entry_price) * position.quantity
                                
                                # Display comprehensive position information
                                if verbose:
                                    print(f"\n{'='*80}")
                                    print(f"ACTIVE POSITION: {position.symbol}")
                                    print(f"{'='*80}")
                                    print(f"\n💰 INVESTMENT DETAILS:")
                                    print(f"  Initial Investment: ${initial_investment:.2f}")
                                    print(f"    └─ Entry Price:   ${position.entry_price:.2f}")
                                    print(f"    └─ Quantity:      {position.quantity:.6f}")
                                    print(f"    └─ Entry Time:    {position.entry_time}")
                                    print(f"    └─ Side:          {position.side.upper()}")
                                    
                                    print(f"\n📊 CURRENT STATUS:")
                                    print(f"  Current Price:     ${current_price:.2f}")
                                    print(f"  Current Value:     ${current_value:.2f}")
                                    print(f"  Current P/L:       ${unrealized_pl:+.2f} ({unrealized_pl_pct:+.2f}%)")
                                    print(f"  Progress to Target: {progress_to_target:.1f}%")
                                    
                                    print(f"\n🎯 TARGET SCENARIOS (User Target: {position.profit_target_pct:.2f}%):")
                                    print(f"  Profit Target:")
                                    print(f"    └─ Target Price:  ${position.profit_target_price:.2f} ({position.profit_target_pct:+.2f}%)")
                                    print(f"    └─ Expected Profit: ${expected_profit_at_target:+.2f}")
                                    print(f"    └─ Total Value at Target: ${initial_investment + expected_profit_at_target:.2f}")
                                    if not profit_target_hit:
                                        print(f"    └─ Remaining to Target: ${abs(remaining_to_target_price):.2f} ({remaining_to_target_pct:+.2f}% more needed)")
                                    print(f"  Stop-Loss:")
                                    print(f"    └─ Stop Price:    ${position.stop_loss_price:.2f} ({position.stop_loss_pct*100:.2f}%)")
                                    print(f"    └─ Expected Loss: ${expected_loss_at_stop:.2f}")
                                    print(f"    └─ Total Value at Stop: ${initial_investment + expected_loss_at_stop:.2f}")
                                    
                                    print(f"\n📈 PREDICTION STATUS:")
                                    print(f"  Model Prediction:  {current_action.upper()}")
                                    print(f"  Position Status:   {'✅ TARGET HIT' if profit_target_hit else ('⚠️  STOP-LOSS HIT' if exit_info and exit_info.get('should_exit') and exit_reason == 'stop_loss_hit' else '⏳ IN PROGRESS')}")
                                    
                                    if not profit_target_hit:
                                        print(f"\n💡 TO HIT YOUR {position.profit_target_pct:.2f}% TARGET:")
                                        if position.side == "long":
                                            print(f"    • Price needs to rise ${abs(remaining_to_target_price):.2f} more (to ${position.profit_target_price:.2f})")
                                            print(f"    • That's {remaining_to_target_pct:+.2f}% more from current price")
                                        else:  # short
                                            print(f"    • Price needs to fall ${abs(remaining_to_target_price):.2f} more (to ${position.profit_target_price:.2f})")
                                            print(f"    • That's {remaining_to_target_pct:+.2f}% more from current price")
                                    
                                    print(f"{'='*80}\n")
                                
                                # NO PYRAMIDING - Just hold position until target is hit
                                # 1. If profit target is hit -> EXIT IMMEDIATELY (regardless of prediction)
                                # 2. If stop-loss hit -> EXIT (already handled above)
                                # 3. Otherwise -> HOLD position (no additional purchases)
                                
                                if position.side == "long":
                                    if profit_target_hit:
                                        # Profit target hit -> EXIT IMMEDIATELY (regardless of prediction)
                                        should_exit = True
                                        exit_reason = "profit_target_hit"
                                        if verbose:
                                            print(f"  ✅ {position.symbol}: PROFIT TARGET HIT ({unrealized_pl_pct:+.2f}%) -> EXITING IMMEDIATELY")
                                    elif current_action == "long" and not profit_target_hit:
                                        # Prediction still LONG and profit target not hit -> HOLD (no pyramiding)
                                        if verbose:
                                            print(f"  📈 {position.symbol}: Prediction still LONG, holding position ({unrealized_pl_pct:+.2f}% / {position.profit_target_pct:.2f}%)")
                                            print(f"     Waiting for profit target to be reached (NO additional purchases)")
                                    elif current_action in {"short", "hold"} and not profit_target_hit:
                                        # Prediction changed but profit target NOT hit -> HOLD (don't exit yet)
                                        if verbose:
                                            print(f"  ⏸️  {position.symbol}: Prediction changed to {current_action.upper()} but profit target not hit ({unrealized_pl_pct:+.2f}% / {position.profit_target_pct:.2f}%)")
                                            print(f"     HOLDING until profit target is reached")
                                elif position.side == "short":
                                    if profit_target_hit:
                                        # Profit target hit -> EXIT IMMEDIATELY (regardless of prediction)
                                        should_exit = True
                                        exit_reason = "profit_target_hit"
                                        if verbose:
                                            print(f"  ✅ {position.symbol}: PROFIT TARGET HIT ({unrealized_pl_pct:+.2f}%) -> EXITING IMMEDIATELY")
                                    elif current_action == "short" and not profit_target_hit:
                                        # Prediction still SHORT and profit target not hit -> HOLD (no pyramiding)
                                        if verbose:
                                            print(f"  📉 {position.symbol}: Prediction still SHORT, holding position ({unrealized_pl_pct:+.2f}% / {position.profit_target_pct:.2f}%)")
                                            print(f"     Waiting for profit target to be reached (NO additional purchases)")
                                    elif current_action in {"long", "hold"} and not profit_target_hit:
                                        # Prediction changed but profit target NOT hit -> HOLD
                                        if verbose:
                                            print(f"  ⏸️  {position.symbol}: Prediction changed to {current_action.upper()} but profit target not hit ({unrealized_pl_pct:+.2f}% / {position.profit_target_pct:.2f}%)")
                                            print(f"     HOLDING until profit target is reached")
                    except Exception as pred_exc:
                        # If prediction fails, don't exit - just log
                        if verbose:
                            print(f"  [WARN] {position.symbol}: Could not check prediction: {pred_exc}")
            
            if should_exit:
                # Execute exit order
                if exit_reason == "profit_target_hit":
                    if verbose:
                        print(f"  ✅ {pos.symbol}: PROFIT TARGET HIT! (${exit_price:.2f})")
                        print(f"     Entry: ${pos.entry_price:.2f} → Exit: ${exit_price:.2f}")
                        if exit_info:
                            print(f"     Realized P/L: ${exit_info.get('realized_pl', 0):.2f} ({exit_info.get('realized_pl_pct', 0):+.2f}%)")
                elif exit_reason == "stop_loss_hit":
                    if verbose:
                        print(f"  ⚠️  {pos.symbol}: STOP-LOSS HIT! (${exit_price:.2f})")
                        print(f"     Entry: ${pos.entry_price:.2f} → Exit: ${exit_price:.2f}")
                        if exit_info:
                            print(f"     Realized P/L: ${exit_info.get('realized_pl', 0):.2f} ({exit_info.get('realized_pl_pct', 0):+.2f}%)")
                elif exit_reason == "prediction_changed":
                    results["prediction_changes"] += 1
                    if verbose:
                        # Calculate P/L at exit
                        if pos.side == "long":
                            realized_pl = (exit_price - pos.entry_price) * pos.quantity
                            realized_pl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100
                        else:  # short
                            realized_pl = (pos.entry_price - exit_price) * pos.quantity
                            realized_pl_pct = ((pos.entry_price - exit_price) / pos.entry_price) * 100
                        print(f"  🔄 {pos.symbol}: PREDICTION CHANGED - EXITING (${exit_price:.2f})")
                        print(f"     Entry: ${pos.entry_price:.2f} → Exit: ${exit_price:.2f}")
                        print(f"     Realized P/L: ${realized_pl:.2f} ({realized_pl_pct:+.2f}%)")
                
                # Execute exit order
                if not dry_run:
                    try:
                        # Determine exit side
                        exit_side = "sell" if pos.side == "long" else "buy"
                        
                        # Get actual position from Alpaca to get EXACT quantity
                        # This is critical - we must use the exact quantity from Alpaca, not our tracked quantity
                        alpaca_position = client.get_position(pos.symbol)
                        if not alpaca_position:
                            if verbose:
                                print(f"  [WARN] {pos.symbol}: Position not found in Alpaca - may have been closed manually")
                            # Position already closed - mark as closed in position manager
                            position_manager.close_position(
                                pos.symbol,
                                exit_price,
                                exit_reason or "position_not_found",
                                None,
                                None,
                            )
                            continue
                        
                        # Get EXACT quantity from Alpaca position
                        alpaca_qty = float(alpaca_position.get("qty", 0) or 0)
                        exit_qty = abs(alpaca_qty)
                        
                        if exit_qty <= 0:
                            if verbose:
                                print(f"  [WARN] {pos.symbol}: Position quantity is zero in Alpaca - may have been closed")
                            # Position already closed - mark as closed in position manager
                            position_manager.close_position(
                                pos.symbol,
                                exit_price,
                                exit_reason or "position_closed",
                                None,
                                None,
                            )
                            continue
                        
                        # For crypto, use qty directly (Alpaca handles fractional quantities)
                        # For commodities, also use qty
                        try:
                            if pos.asset_type == "crypto":
                                # For crypto, use qty directly - Alpaca supports fractional crypto
                                exit_resp = client.submit_order(
                                    symbol=pos.symbol,
                                    qty=exit_qty,  # Use exact quantity from Alpaca
                                    side=exit_side,
                                    order_type="market",
                                    time_in_force="gtc",
                                )
                            else:
                                # For commodities, use qty
                                exit_resp = client.submit_order(
                                    symbol=pos.symbol,
                                    qty=exit_qty,
                                    side=exit_side,
                                    order_type="market",
                                    time_in_force="gtc",
                                )
                            
                            # Calculate P/L if not already calculated
                            if exit_reason == "prediction_changed":
                                if pos.side == "long":
                                    realized_pl = (exit_price - pos.entry_price) * exit_qty
                                    realized_pl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100
                                else:  # short
                                    realized_pl = (pos.entry_price - exit_price) * exit_qty
                                    realized_pl_pct = ((pos.entry_price - exit_price) / pos.entry_price) * 100
                            else:
                                realized_pl = exit_info.get("realized_pl") if exit_info else None
                                realized_pl_pct = exit_info.get("realized_pl_pct") if exit_info else None
                            
                            # If P/L not calculated, calculate it now using exact exit quantity
                            if realized_pl is None:
                                if pos.side == "long":
                                    realized_pl = (exit_price - pos.entry_price) * exit_qty
                                    realized_pl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100
                                else:  # short
                                    realized_pl = (pos.entry_price - exit_price) * exit_qty
                                    realized_pl_pct = ((pos.entry_price - exit_price) / pos.entry_price) * 100
                            
                            # Mark position as closed in position manager
                            position_manager.close_position(
                                pos.symbol,
                                exit_price,
                                exit_reason,
                                realized_pl,
                                realized_pl_pct,
                            )
                            
                            results["exits_executed"].append({
                                "symbol": pos.symbol,
                                "exit_reason": exit_reason,
                                "exit_price": exit_price,
                                "exit_qty": exit_qty,
                                "realized_pl": realized_pl,
                                "realized_pl_pct": realized_pl_pct,
                            })
                            
                            if verbose:
                                print(f"  ✅ Exit order executed for {pos.symbol}")
                                print(f"     Closed {exit_qty:.8f} {pos.symbol} at ${exit_price:.2f}")
                                if realized_pl is not None:
                                    print(f"     Realized P/L: ${realized_pl:.2f} ({realized_pl_pct:+.2f}%)")
                        
                        except Exception as order_exc:
                            error_msg = str(order_exc)
                            
                            # Handle specific error cases
                            if "insufficient balance" in error_msg.lower() or "403" in error_msg:
                                # Position might have been partially closed or quantity mismatch
                                if verbose:
                                    print(f"  ⚠️  {pos.symbol}: Insufficient balance error - checking actual position...")
                                
                                # Re-check position to see actual quantity
                                recheck_position = client.get_position(pos.symbol)
                                if recheck_position:
                                    actual_qty = abs(float(recheck_position.get("qty", 0) or 0))
                                    if actual_qty > 0 and actual_qty != exit_qty:
                                        if verbose:
                                            print(f"     Position quantity changed: {exit_qty:.8f} → {actual_qty:.8f}")
                                        # Try again with actual quantity
                                        try:
                                            exit_resp = client.submit_order(
                                                symbol=pos.symbol,
                                                qty=actual_qty,
                                                side=exit_side,
                                                order_type="market",
                                                time_in_force="gtc",
                                            )
                                            # Calculate P/L with actual quantity
                                            if pos.side == "long":
                                                realized_pl = (exit_price - pos.entry_price) * actual_qty
                                                realized_pl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100
                                            else:
                                                realized_pl = (pos.entry_price - exit_price) * actual_qty
                                                realized_pl_pct = ((pos.entry_price - exit_price) / pos.entry_price) * 100
                                            
                                            position_manager.close_position(
                                                pos.symbol,
                                                exit_price,
                                                exit_reason,
                                                realized_pl,
                                                realized_pl_pct,
                                            )
                                            
                                            if verbose:
                                                print(f"  ✅ Exit order executed with corrected quantity: {actual_qty:.8f}")
                                            continue
                                        except Exception as retry_exc:
                                            error_msg = f"{pos.symbol}: Failed to execute exit order after retry: {retry_exc}"
                                    elif actual_qty == 0:
                                        if verbose:
                                            print(f"  [INFO] {pos.symbol}: Position already closed in Alpaca")
                                        position_manager.close_position(
                                            pos.symbol,
                                            exit_price,
                                            exit_reason or "already_closed",
                                            None,
                                            None,
                                        )
                                        continue
                            
                            # If we get here, the order failed
                            error_msg = f"{pos.symbol}: Failed to execute exit order: {order_exc}"
                            if verbose:
                                print(f"  ❌ {error_msg}")
                                print(f"     Attempted to close {exit_qty:.8f} {pos.symbol}")
                                print(f"     You may need to close this position manually in Alpaca")
                            results["errors"].append(error_msg)
                            
                    except Exception as exit_exc:
                        error_msg = f"{pos.symbol}: Failed to execute exit order: {exit_exc}"
                        if verbose:
                            print(f"  ❌ {error_msg}")
                            import traceback
                            if verbose:
                                print(f"     Traceback: {traceback.format_exc()}")
                        results["errors"].append(error_msg)
                else:
                    # Dry run - just log
                    if verbose:
                        print(f"  [DRY-RUN] Would execute exit order for {pos.symbol}")
            else:
                # Position still active - provide detailed profit tracking
                results["positions_active"] += 1
                
                # Check if profit target or stop-loss is hit
                if pos.side == "long":
                    profit_target_hit = current_price >= pos.profit_target_price
                    stop_loss_hit = current_price <= pos.stop_loss_price
                else:  # short
                    profit_target_hit = current_price <= pos.profit_target_price
                    stop_loss_hit = current_price >= pos.stop_loss_price
                
                # Calculate detailed P/L metrics
                # Initial investment amount
                initial_investment = pos.entry_price * pos.quantity
                
                # Current value of position
                current_value = current_price * pos.quantity
                
                if pos.side == "long":
                    unrealized_pl = (current_price - pos.entry_price) * pos.quantity
                    unrealized_pl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                    price_change_needed = pos.profit_target_price - current_price
                    pct_change_needed = (price_change_needed / current_price) * 100
                    
                    # Expected profit if target is hit
                    expected_profit_at_target = (pos.profit_target_price - pos.entry_price) * pos.quantity
                    expected_profit_pct_at_target = ((pos.profit_target_price - pos.entry_price) / pos.entry_price) * 100
                    
                    # Expected loss if stop-loss is hit
                    expected_loss_at_stop = (pos.stop_loss_price - pos.entry_price) * pos.quantity
                    expected_loss_pct_at_stop = ((pos.stop_loss_price - pos.entry_price) / pos.entry_price) * 100
                else:  # short
                    unrealized_pl = (pos.entry_price - current_price) * pos.quantity
                    unrealized_pl_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100
                    price_change_needed = current_price - pos.profit_target_price
                    pct_change_needed = (price_change_needed / current_price) * 100
                    
                    # Expected profit if target is hit (for short, profit when price goes down)
                    expected_profit_at_target = (pos.entry_price - pos.profit_target_price) * pos.quantity
                    expected_profit_pct_at_target = ((pos.entry_price - pos.profit_target_price) / pos.entry_price) * 100
                    
                    # Expected loss if stop-loss is hit (for short, loss when price goes up)
                    expected_loss_at_stop = (pos.stop_loss_price - pos.entry_price) * pos.quantity
                    expected_loss_pct_at_stop = ((pos.stop_loss_price - pos.entry_price) / pos.entry_price) * 100
                
                progress_pct = (unrealized_pl_pct / pos.profit_target_pct) * 100 if pos.profit_target_pct > 0 else 0
                distance_to_target = abs(pos.profit_target_price - current_price)
                distance_to_stop = abs(current_price - pos.stop_loss_price)
                
                # Determine why target not hit
                why_not_hit = []
                if not profit_target_hit:
                    if pos.side == "long":
                        if current_price < pos.entry_price:
                            why_not_hit.append(f"Price is DOWN from entry (${current_price:.2f} vs ${pos.entry_price:.2f})")
                        else:
                            why_not_hit.append(f"Price needs to rise ${price_change_needed:.2f} more ({pct_change_needed:+.2f}%) to reach target")
                    else:  # short
                        if current_price > pos.entry_price:
                            why_not_hit.append(f"Price is UP from entry (${current_price:.2f} vs ${pos.entry_price:.2f})")
                        else:
                            why_not_hit.append(f"Price needs to fall ${abs(price_change_needed):.2f} more ({abs(pct_change_needed):+.2f}%) to reach target")
                    
                    if distance_to_stop < distance_to_target:
                        why_not_hit.append(f"Stop-loss is closer than target (${distance_to_stop:.2f} vs ${distance_to_target:.2f} away)")
                
                if verbose:
                    print(f"\n{'='*80}")
                    print(f"\n{'='*80}")
                    print(f"ACTIVE POSITION: {pos.symbol}")
                    print(f"{'='*80}")
                    print(f"\n💰 INVESTMENT DETAILS:")
                    print(f"  Initial Investment: ${initial_investment:.2f}")
                    print(f"    └─ Entry Price:   ${pos.entry_price:.2f}")
                    print(f"    └─ Quantity:      {pos.quantity:.6f}")
                    print(f"    └─ Entry Time:    {pos.entry_time}")
                    print(f"    └─ Side:          {pos.side.upper()}")
                    
                    print(f"\n📊 CURRENT STATUS:")
                    print(f"  Current Price:     ${current_price:.2f}")
                    print(f"  Current Value:     ${current_value:.2f}")
                    print(f"  Current P/L:       ${unrealized_pl:+.2f} ({unrealized_pl_pct:+.2f}%)")
                    print(f"  Progress to Target: {progress_pct:.1f}%")
                    
                    print(f"\n🎯 TARGET SCENARIOS:")
                    print(f"  Profit Target:")
                    print(f"    └─ Target Price:  ${pos.profit_target_price:.2f} ({pos.profit_target_pct:+.2f}%)")
                    print(f"    └─ Expected Profit: ${expected_profit_at_target:+.2f} ({expected_profit_pct_at_target:+.2f}%)")
                    print(f"    └─ Total Value at Target: ${initial_investment + expected_profit_at_target:.2f}")
                    print(f"  Stop-Loss:")
                    print(f"    └─ Stop Price:    ${pos.stop_loss_price:.2f} ({pos.stop_loss_pct:.2f}%)")
                    print(f"    └─ Expected Loss: ${expected_loss_at_stop:.2f} ({expected_loss_pct_at_stop:.2f}%)")
                    print(f"    └─ Total Value at Stop: ${initial_investment + expected_loss_at_stop:.2f}")
                    print(f"\nStatus:           {'✅ TARGET HIT' if profit_target_hit else ('⚠️  STOP-LOSS HIT' if stop_loss_hit else '⏳ IN PROGRESS')}")
                    if not profit_target_hit and not stop_loss_hit:
                        print(f"\nWhy Target Not Hit:")
                        for reason in why_not_hit:
                            print(f"  • {reason}")
                    print(f"{'='*80}\n")
                    
                    # Log position monitoring
                    if trade_logger:
                        trade_logger.log_position_update(
                            symbol=position.symbol,
                            data_symbol=position.data_symbol,
                            asset_type=position.asset_type,
                            side=position.side,
                            quantity=position.quantity,
                            entry_price=position.entry_price,
                            current_price=current_price,
                            position_status=position.status,
                            cost_basis=position.total_cost_basis or 0.0,
                            profit_target_pct=position.profit_target_pct,
                            profit_target_price=position.profit_target_price,
                            stop_loss_pct=position.stop_loss_pct,
                            stop_loss_price=position.stop_loss_price,
                            consensus_action=consensus.get("consensus_action") if consensus else None,
                            consensus_confidence=consensus.get("consensus_confidence") if consensus else None,
                            dry_run=dry_run,
                        )
        
        except Exception as pos_exc:
            error_msg = f"{position.symbol}: Error monitoring position: {pos_exc}"
            if verbose:
                print(f"  ❌ {error_msg}")
            results["errors"].append(error_msg)
    
    if verbose:
        print(f"[MONITOR] Checked {results['positions_checked']} positions:")
        print(f"  Active: {results['positions_active']}")
        print(f"  Profit targets hit: {results['profit_targets_hit']}")
        print(f"  Stop-losses hit: {results['stop_losses_hit']}")
        print(f"  Prediction changes: {results['prediction_changes']}")
    
    return results


def run_trading_cycle(
    execution_engine: ExecutionEngine,
    position_manager: PositionManager,
    tradable_symbols: List[Dict[str, Any]],
    profit_target_pct: float,
    trade_logger: Optional[TradeLogger] = None,
    dry_run: bool = False,
    verbose: bool = True,
    update_data: bool = True,
    regenerate_features_flag: bool = True,
) -> Dict[str, Any]:
    """
    Run one complete trading cycle: monitor positions -> fetch data -> regenerate features -> predict + execute.
    
    Args:
        execution_engine: Execution engine for placing trades
        position_manager: Position manager for tracking positions
        tradable_symbols: List of symbols with trained models
        profit_target_pct: User's desired profit percentage (e.g., 10.0)
        dry_run: If True, don't send real orders
        verbose: Print detailed progress
        update_data: If True, fetch latest live data from Alpaca before each cycle
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
        "monitoring": {},
    }
    
    # Step 0: Display active positions
    if verbose:
        client = AlpacaClient()
        display_active_positions(position_manager, client, verbose=verbose)
    
    # Step 1: FIRST - Monitor existing positions for profit targets and stop-loss
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 1: MONITORING ACTIVE POSITIONS")
        print("=" * 80)
    
    monitoring_results = monitor_positions(
        position_manager,
        execution_engine,
        tradable_symbols,
        trade_logger=trade_logger,
        dry_run=dry_run,
        verbose=verbose,
    )
    results["monitoring"] = monitoring_results
    
    # Step 2: Update live data for all symbols (if enabled)
    if update_data:
        if verbose:
            print("\n" + "=" * 80)
            print("STEP 2: UPDATING LIVE DATA")
            print("=" * 80)
            print("[UPDATE] Fetching latest live prices from Alpaca...")
        
        try:
            unique_symbols = {}
            for info in tradable_symbols:
                asset = info["asset"]
                if asset.asset_type not in unique_symbols:
                    unique_symbols[asset.asset_type] = set()
                unique_symbols[asset.asset_type].add(asset.data_symbol)
            
            updated_count = 0
            client = AlpacaClient()
            
            for asset_type, symbols in unique_symbols.items():
                for symbol in symbols:
                    try:
                        asset_mapping = find_by_data_symbol(symbol)
                        if not asset_mapping:
                            continue
                        
                        # Get live price from Alpaca
                        last_trade = client.get_last_trade(asset_mapping.trading_symbol)
                        if not last_trade:
                            continue
                        
                        live_price = last_trade.get("price") or last_trade.get("p")
                        if not live_price:
                            continue
                        
                        live_price = float(live_price)
                        
                        # Load existing data.json
                        if asset_type == "crypto":
                            data_paths = [
                                get_data_path("crypto", symbol, "1d", None, "alpaca").parent / "data.json",
                                get_data_path("crypto", symbol, "1d", None, "binance").parent / "data.json",
                            ]
                        else:  # commodities
                            data_paths = [
                                get_data_path("commodities", symbol, "1d", None, "yahoo_chart").parent / "data.json",
                            ]
                        
                        data_file = None
                        for path in data_paths:
                            if path.exists():
                                data_file = path
                                break
                        
                        if not data_file or not data_file.exists():
                            continue
                        
                        # Load existing candles
                        existing_candles = load_json_file(data_file)
                        if not existing_candles:
                            continue
                        
                        # Update the last candle's close price with live price
                        last_candle = existing_candles[-1].copy()
                        last_candle["close"] = live_price
                        if live_price > last_candle.get("high", 0):
                            last_candle["high"] = live_price
                        if live_price < last_candle.get("low", float("inf")) or last_candle.get("low", 0) == 0:
                            last_candle["low"] = live_price
                        last_candle["source"] = last_candle.get("source", "alpaca")
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
                print(f"[UPDATE] Live prices updated for {updated_count} symbol(s) from Alpaca")
        except Exception as exc:
            if verbose:
                print(f"[WARN] Failed to update live prices from Alpaca: {exc}")
    
    # Step 3: Regenerate features for all symbols (if enabled)
    if regenerate_features_flag:
        if verbose:
            print("\n" + "=" * 80)
            print("STEP 3: REGENERATING FEATURES")
            print("=" * 80)
            print("[FEATURES] Regenerating features with latest data...")
        
        try:
            unique_symbols_by_type = {}
            for info in tradable_symbols:
                asset = info["asset"]
                if asset.asset_type not in unique_symbols_by_type:
                    unique_symbols_by_type[asset.asset_type] = set()
                unique_symbols_by_type[asset.asset_type].add(asset.data_symbol)
            
            for asset_type, symbols in unique_symbols_by_type.items():
                updated_count = regenerate_features(asset_type, symbols, "1d")
                if verbose:
                    if updated_count > 0:
                        print(f"[FEATURES] Features regenerated for {updated_count}/{len(symbols)} {asset_type} symbol(s)")
                    else:
                        print(f"[WARN] Failed to regenerate features for any {asset_type} symbol(s)")
        except Exception as exc:
            if verbose:
                print(f"[WARN] Failed to regenerate features: {exc}")
    
    # Step 4: Run predictions and rank all symbols (including ones with active positions)
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 4: RUNNING PREDICTIONS AND RANKING SYMBOLS")
        print("=" * 80)
    
    # Collect predictions for all symbols to enable ranking
    symbol_predictions = []
    active_symbols = {pos.symbol.upper() for pos in position_manager.get_all_positions()}
    
    for symbol_info in tradable_symbols:
        asset = symbol_info["asset"]
        model_dir = symbol_info["model_dir"]
        horizon = symbol_info["horizon"]
        data_symbol = asset.data_symbol
        asset_type = asset.asset_type
        
        try:
            # Load latest features
            feature_row = None
            for attempt in range(2):
                feature_row = load_feature_row(asset_type, data_symbol, "1d")
                if feature_row is not None and not feature_row.empty:
                    break
                if attempt == 0:
                    import time
                    time.sleep(0.5)
            
            if feature_row is None or feature_row.empty:
                continue
            
            # Get current price
            current_price = get_current_price_from_features(asset_type, data_symbol, "1d", verbose=False, force_live=False)
            if current_price is None or current_price <= 0:
                continue
            
            # Load inference pipeline and get prediction
            risk_config = RiskManagerConfig(paper_trade=True)
            pipeline = InferencePipeline(model_dir, risk_config=risk_config)
            pipeline.load()
            
            if not pipeline.models:
                continue
            
            volatility = 0.01  # Default
            prediction_result = pipeline.predict(
                feature_row,
                current_price=current_price,
                volatility=volatility,
            )
            
            consensus = prediction_result.get("consensus", {})
            action = consensus.get("consensus_action", "hold")
            confidence = consensus.get("consensus_confidence", 0.0)
            predicted_return = consensus.get("consensus_return", 0.0)
            
            # Check if we have an active position
            has_active_position = asset.trading_symbol.upper() in active_symbols
            
            symbol_predictions.append({
                "symbol_info": symbol_info,
                "asset": asset,
                "data_symbol": data_symbol,
                "current_price": current_price,
                "prediction": consensus,
                "action": action,
                "confidence": confidence,
                "predicted_return": predicted_return,
                "has_active_position": has_active_position,
                "pipeline": pipeline,  # Keep pipeline for later use
                "feature_row": feature_row,
            })
        
        except Exception as e:
            if verbose:
                print(f"[WARN] {data_symbol}: Failed to get prediction for ranking: {e}")
            continue
    
    # Rank symbols by confidence and predicted return
    # Prioritize: high confidence + high predicted return, but also consider active positions
    def rank_symbol(pred):
        action = pred["action"]
        confidence = pred["confidence"]
        predicted_return = pred["predicted_return"]
        has_active_position = pred["has_active_position"]
        
        # Base score from prediction quality
        if action == "hold" or action == "flat":
            base_score = 0
        elif action in ("long", "short"):
            # Score = confidence * abs(predicted_return)
            base_score = confidence * abs(predicted_return) * 100
        else:
            base_score = 0
        
        # Boost score slightly for symbols with active positions (monitor them closely)
        if has_active_position:
            base_score += 10
        
        return base_score
    
    symbol_predictions.sort(key=rank_symbol, reverse=True)
    
    # Display rankings
    if verbose and symbol_predictions:
        print(f"\n[RANKING] Ranked {len(symbol_predictions)} symbol(s) by prediction quality:")
        print("-" * 80)
        for idx, pred in enumerate(symbol_predictions[:20], 1):  # Show top 20
            asset = pred["asset"]
            action = pred["action"]
            confidence = pred["confidence"]
            predicted_return = pred["predicted_return"]
            has_pos = "⭐ ACTIVE" if pred["has_active_position"] else ""
            score = rank_symbol(pred)
            
            print(f"{idx:2d}. {asset.data_symbol:12s} | {action:5s} | Confidence: {confidence*100:5.1f}% | "
                  f"Pred Return: {predicted_return*100:6.2f}% | Score: {score:6.2f} {has_pos}")
        print("-" * 80)
    
    # Step 5: Execute trades based on rankings (process all, not just top ones)
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 5: EXECUTING TRADES FOR ALL SYMBOLS")
        print("=" * 80)
    
        for pred in symbol_predictions:
            symbol_info = pred["symbol_info"]
            asset = pred["asset"]
            model_dir = symbol_info["model_dir"]
            data_symbol = pred["data_symbol"]
            asset_type = asset.asset_type
            current_price = pred["current_price"]
            consensus = pred["prediction"]
            action = pred["action"]
            confidence = pred["confidence"]
            
            # Use cached prediction from ranking step
            results["symbols_processed"] += 1
            
            if verbose:
                confidence_pct = confidence * 100 if confidence <= 1.0 else confidence
                predicted_return = pred["predicted_return"]
                expected_move_pct = predicted_return * 100
                has_pos = "⭐ [ACTIVE]" if pred["has_active_position"] else ""
                
                # Show if neutral guard was triggered
                neutral_guard = consensus.get("neutral_guard_triggered", False)
                if neutral_guard:
                    raw_return = consensus.get("raw_consensus_return", predicted_return)
                    print(f"[PREDICTION] {data_symbol}: {action.upper()} {has_pos} (confidence: {confidence_pct:.1f}%, expected move: {expected_move_pct:+.2f}% [neutral guard engaged, raw: {raw_return*100:+.2f}%])")
                else:
                    print(f"[PREDICTION] {data_symbol}: {action.upper()} {has_pos} (confidence: {confidence_pct:.1f}%, expected move: {expected_move_pct:+.2f}%)")
                
                # Show model bias warnings
                if consensus.get("unanimous_direction"):
                    direction = consensus.get("unanimous_direction")
                    count = consensus.get("unanimous_direction_count", 0)
                    if count >= 3:
                        print(f"  ⚠️  WARNING: All {count} models predict {direction.upper()} - possible model bias/overfitting")
            
            # Execute trade with profit target (using cached prediction)
            try:
                horizon = symbol_info["horizon"]
                execution_result = execution_engine.execute_from_consensus(
                    asset=asset,
                    consensus=consensus,
                    current_price=current_price,
                    dry_run=dry_run,
                    horizon_profile=horizon,
                    profit_target_pct=profit_target_pct,  # Pass profit target
                )
            except Exception as exec_exc:
                error_msg = str(exec_exc)
                # Provide clearer error message for crypto shorting issues
                if "insufficient balance" in error_msg.lower() and "crypto" in asset_type.lower():
                    # Extract asset name from error if possible
                    asset_name = data_symbol.split("-")[0] if "-" in data_symbol else data_symbol
                    enhanced_msg = f"{data_symbol}: Execution failed - {error_msg}\n"
                    enhanced_msg += f"  NOTE: For crypto SHORT positions, you need the actual {asset_name} asset available to borrow, not just USD buying power.\n"
                    enhanced_msg += f"  Alpaca paper trading has limited crypto inventory for shorting. This is expected behavior."
                    print(f"[ERROR] {enhanced_msg}")
                    results["errors"].append(enhanced_msg)
                else:
                    error_msg_full = f"{data_symbol}: Execution failed: {exec_exc}"
                    print(f"[ERROR] {error_msg_full}")
                    results["errors"].append(error_msg_full)
                results["symbols_skipped"] += 1
                continue
            
            if execution_result:
                results["symbols_traded"] += 1
                decision = execution_result.get("decision", "unknown")
                
                # Log trade execution
                if trade_logger:
                    orders = execution_result.get("orders", {})
                    order_details = execution_result.get("order_details", {})
                    
                    # Determine action and side
                    trade_action = "buy" if decision in ("enter", "pyramid", "increase") else "sell" if decision in ("exit", "close", "reduce") else "hold"
                    trade_side = consensus.get("consensus_action", "hold")
                    if trade_side not in ("long", "short"):
                        # Get side from orders or existing position
                        existing_side = orders.get("existing_side", "flat")
                        target_side = orders.get("target_side", "flat")
                        trade_side = target_side if target_side != "flat" else existing_side
                    
                    # Get quantity and price
                    quantity = orders.get("final_qty", 0) or order_details.get("quantity", 0) or 0
                    price = current_price
                    if order_details:
                        price = order_details.get("price") or order_details.get("filled_avg_price") or current_price
                    
                    trade_logger.log_order_execution(
                        symbol=asset.trading_symbol,
                        data_symbol=data_symbol,
                        asset_type=asset.asset_type,
                        action=trade_action,
                        side=trade_side,
                        quantity=abs(quantity) if quantity else 0,
                        price=price,
                        decision=decision,
                        reason=execution_result.get("reason", orders.get("reason", "")),
                        consensus_action=consensus.get("consensus_action"),
                        consensus_confidence=consensus.get("consensus_confidence"),
                        consensus_return=consensus.get("consensus_return"),
                        predicted_price=consensus.get("consensus_price"),
                        order_id=order_details.get("id") if order_details else None,
                        dry_run=dry_run,
                        metadata={
                            "equity": orders.get("equity"),
                            "buying_power": orders.get("buying_power"),
                            "model_action": consensus.get("consensus_action"),
                            "predicted_return": consensus.get("consensus_return"),
                        },
                    )
                
                if verbose:
                    print(f"[TRADE] {data_symbol}: {action.upper()} -> {decision}")
            
            results["symbols_processed"] += 1
        
        except Exception as exc:
            error_msg = f"{data_symbol}: {exc}"
            print(f"[ERROR] {error_msg}")
            results["errors"].append(error_msg)
            results["symbols_skipped"] += 1
    
    results["cycle_end"] = datetime.utcnow().isoformat() + "Z"
    results["cycle_duration_seconds"] = (datetime.utcnow() - cycle_start).total_seconds()
    
    return results


def get_protected_symbols(asset_type: str) -> Set[str]:
    """Read current Alpaca positions and return trading symbols that should not be modified."""
    universe = {asset.trading_symbol.upper() for asset in all_enabled() if asset.asset_type == asset_type}
    
    try:
        client = AlpacaClient()
        positions = client.list_positions()
    except Exception as e:
        import warnings
        warnings.warn(f"Could not fetch Alpaca positions: {e}. Continuing without position protection.", UserWarning)
        return set()
    
    protected: Set[str] = set()
    for pos in positions or []:
        symbol = str(pos.get("symbol", "")).upper()
        if symbol not in universe:
            continue
        qty_str = pos.get("qty", "0")
        try:
            qty = float(qty_str)
        except (TypeError, ValueError):
            qty = 0.0
        if qty != 0.0:
            protected.add(symbol)
    return protected


def normalize_symbol(symbol: str, asset_type: str) -> str:
    """Normalize symbol to canonical data symbol format."""
    # Try exact data_symbol match
    asset = find_by_data_symbol(symbol)
    if asset and asset.asset_type == asset_type:
        return asset.data_symbol.upper()
    
    # Try trading_symbol match
    asset = find_by_trading_symbol(symbol)
    if asset and asset.asset_type == asset_type:
        return asset.data_symbol.upper()
    
    # For crypto: try BTC/USD -> BTC-USDT conversion
    if asset_type == "crypto" and "/" in symbol:
        base, quote = symbol.split("/", 1)
        base = base.strip().upper()
        quote = quote.strip().upper()
        if base and quote == "USD":
            return f"{base}-USDT"
    
    # Return as-is, uppercased
    return symbol.upper()


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end trading pipeline: ingest -> features -> train -> paper-trade with profit targets."
    )
    
    # Symbol arguments
    parser.add_argument(
        "--crypto-symbols",
        nargs="+",
        default=[],
        help="Crypto symbols (e.g. BTC-USDT ETH-USDT). Can be empty if only trading commodities.",
    )
    parser.add_argument(
        "--commodities-symbols",
        nargs="+",
        default=[],
        help="Commodities symbols (e.g. GC=F CL=F). Can be empty if only trading crypto.",
    )
    
    # MANDATORY profit target
    parser.add_argument(
        "--profit-target",
        type=float,
        required=True,
        help="MANDATORY: Desired profit percentage (e.g., 10.0 for 10% profit).",
    )
    
    # Other arguments
    parser.add_argument(
        "--timeframe",
        default="1d",
        help="Timeframe to use for ingestion and models (default: 1d).",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=10.0,
        help="Approximate years of historical data to request (default: 10).",
    )
    parser.add_argument(
        "--crypto-horizon",
        default="short",
        help="Horizon profile for crypto training (intraday/short/long).",
    )
    parser.add_argument(
        "--commodities-horizon",
        default="short",
        help="Horizon profile for commodities training (intraday/short/long).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run trading logic but do not send real orders to Alpaca.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Seconds between trading cycles. Runs forever. Minimum: 30 seconds. Default: 30 seconds.",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.crypto_symbols and not args.commodities_symbols:
        print("[ERROR] Must provide at least one symbol (--crypto-symbols or --commodities-symbols)")
        return
    
    if args.profit_target <= 0:
        print("[ERROR] Profit target must be positive (e.g., 10.0 for 10%)")
        return
    
    if args.interval < 30:
        print(f"⚠️  WARNING: Interval {args.interval} seconds is too short. Minimum is 30 seconds.")
        print(f"   Setting interval to 30 seconds.")
        args.interval = 30
    
    # Normalize symbols
    crypto_symbols = [normalize_symbol(s, "crypto") for s in args.crypto_symbols] if args.crypto_symbols else []
    commodities_symbols = [normalize_symbol(s, "commodities") for s in args.commodities_symbols] if args.commodities_symbols else []
    
    print("=" * 80)
    print("END-TO-END TRADING PIPELINE")
    print("=" * 80)
    if crypto_symbols:
        print(f"Crypto Symbols:   {', '.join(crypto_symbols)}")
    if commodities_symbols:
        print(f"Commodities Symbols: {', '.join(commodities_symbols)}")
    print(f"Timeframe:        {args.timeframe}")
    print(f"Profit Target:    {args.profit_target}%")
    print(f"Crypto Horizon:   {args.crypto_horizon}")
    print(f"Commodities Horizon: {args.commodities_horizon}")
    print(f"Mode:             {'DRY RUN (no real orders)' if args.dry_run else 'LIVE PAPER TRADING'}")
    print("=" * 80)
    print()
    print("[STRATEGY] Buy Low, Sell High, Minimize Losses:")
    print("  ✅ Momentum filters: Block buying during upswings (buy low)")
    print("  ✅ RSI filters: Only buy when oversold (RSI < 30), only short when overbought (RSI > 70)")
    print("  ✅ Mean reversion: Flips SHORT to LONG when oversold (buy low)")
    print("  ✅ Trailing stop: Sell when price drops 2.5% from peak (sell at peak)")
    print("  ✅ Stop-loss: Automatic protection (minimize losses)")
    print("  ✅ Negative prediction filter: Block entries when predicted return < 0 (minimize losses)")
    print("  ✅ Minimum return filter: Require 1.5% minimum predicted return (minimize losses)")
    print("  ✅ Volatility-based sizing: Reduce position size in high volatility (minimize losses)")
    print()
    
    # Show available horizons
    print_horizon_summary()
    
    # Stage 1: Historical ingestion
    print("\n[1/4] Ingesting historical data...")
    run_ingestion(
        mode="historical",
        crypto_symbols=crypto_symbols if crypto_symbols else None,
        commodities_symbols=commodities_symbols if commodities_symbols else None,
        timeframe=args.timeframe,
        years=args.years,
    )
    print("    ✓ Historical data ingestion complete.")
    
    # Stage 2: Feature generation
    print("\n[2/4] Regenerating features...")
    if crypto_symbols:
        regenerate_features("crypto", set(crypto_symbols), args.timeframe)
    if commodities_symbols:
        regenerate_features("commodities", set(commodities_symbols), args.timeframe)
    print("    ✓ Feature generation complete.")
    
    # Stage 3: Model training
    print("\n[3/4] Training models...")
    horizon_map = {}
    if crypto_symbols:
        horizon_map["crypto"] = args.crypto_horizon
    if commodities_symbols:
        horizon_map["commodities"] = args.commodities_horizon
    
    train_symbols(
        crypto_symbols=crypto_symbols if crypto_symbols else [],
        commodities_symbols=commodities_symbols if commodities_symbols else [],
        timeframe=args.timeframe,
        output_dir="models",
        horizon_profiles=horizon_map,
    )
    print("    ✓ Model training complete.")
    
    # Stage 4: Live trading
    print("\n[4/4] Preparing live trading...")
    
    # Discover tradable symbols
    # CRITICAL: Pass the command-line horizon to override symbol universe defaults
    # This ensures that if you train with --crypto-horizon long, it will look for "long" models
    all_tradable = []
    if crypto_symbols:
        crypto_tradable = discover_tradable_symbols(
            asset_type="crypto", 
            timeframe=args.timeframe, 
            verbose=True,
            override_horizon=args.crypto_horizon  # Use command-line horizon instead of symbol universe default
        )
        requested_crypto = {s.upper() for s in crypto_symbols}
        crypto_tradable = [
            info for info in crypto_tradable
            if info["asset"].data_symbol.upper() in requested_crypto
        ]
        all_tradable.extend(crypto_tradable)
    
    if commodities_symbols:
        commodities_tradable = discover_tradable_symbols(
            asset_type="commodities", 
            timeframe=args.timeframe, 
            verbose=True,
            override_horizon=args.commodities_horizon  # Use command-line horizon instead of symbol universe default
        )
        requested_commodities = {s.upper() for s in commodities_symbols}
        commodities_tradable = [
            info for info in commodities_tradable
            if info["asset"].data_symbol.upper() in requested_commodities
        ]
        all_tradable.extend(commodities_tradable)
    
    if not all_tradable:
        print("    ✗ No tradable symbols found after training. Exiting.")
        return
    
    print(f"    ✓ Found {len(all_tradable)} tradable symbol(s):")
    for info in all_tradable:
        asset = info["asset"]
        print(f"      - {asset.data_symbol} ({asset.trading_symbol}) - {asset.asset_type} - horizon: {info['horizon']}")
    
    # Initialize execution engine and position manager
    try:
        position_manager = PositionManager()
        execution_engine = ExecutionEngine(position_manager=position_manager)
        trade_logger = TradeLogger()  # Initialize trade logger
    except Exception as exc:
        print(f"    ✗ Failed to initialize: {exc}")
        print("      Make sure ALPACA_API_KEY and ALPACA_SECRET_KEY are set.")
        return
    
    # Sync existing Alpaca positions with position manager
    print("\n[SYNC] Checking for existing positions in Alpaca...")
    sync_results = sync_existing_alpaca_positions(
        position_manager=position_manager,
        tradable_symbols=all_tradable,
        profit_target_pct=args.profit_target,
        verbose=True,
    )
    
    if sync_results["positions_synced"] > 0:
        print(f"    ✓ Synced {sync_results['positions_synced']} existing position(s)")
        print("    ℹ These positions will be monitored with the new profit target")
    else:
        print("    ✓ No existing positions found (or all already tracked)")
    
    print()
    print("Starting live trading with profit target monitoring...")
    print("=" * 80)
    print(f"Profit Target: {args.profit_target}%")
    print(f"Positions will be automatically closed when:")
    print(f"  - Profit target is reached ({args.profit_target}% gain)")
    print(f"  - Stop-loss is triggered (horizon-specific %)")
    print(f"Trade Logs: data/logs/trades.jsonl")
    print("=" * 80)
    
    cycle_index = 0
    try:
        while True:
            cycle_index += 1
            now = datetime.utcnow().isoformat() + "Z"
            print(f"\n{'='*80}")
            print(f"[CYCLE {cycle_index}] {now}")
            print(f"{'='*80}")
            
            cycle_results = run_trading_cycle(
                execution_engine=execution_engine,
                position_manager=position_manager,
                tradable_symbols=all_tradable,
                profit_target_pct=args.profit_target,
                trade_logger=trade_logger,
                dry_run=args.dry_run,
                verbose=True,
                update_data=True,
                regenerate_features_flag=True,
            )
            
            # Summary
            print(f"\n[CYCLE {cycle_index} SUMMARY]")
            print(f"  Processed: {cycle_results['symbols_processed']}")
            print(f"  Traded: {cycle_results['symbols_traded']}")
            print(f"  Skipped: {cycle_results['symbols_skipped']}")
            if cycle_results['errors']:
                print(f"  Errors: {len(cycle_results['errors'])}")
            
            monitoring = cycle_results.get("monitoring", {})
            if monitoring:
                print(f"  Positions Active: {monitoring.get('positions_active', 0)}")
                print(f"  Profit Targets Hit: {monitoring.get('profit_targets_hit', 0)}")
                print(f"  Stop-Losses Hit: {monitoring.get('stop_losses_hit', 0)}")
            
            # Wait and repeat
            print(f"\n  Waiting {args.interval} seconds before next cycle...")
            print(f"  Press Ctrl+C to stop.")
            time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping due to keyboard interrupt (Ctrl+C).")
        print("   Finalizing any open positions...")
    except Exception as exc:
        print(f"\n[ERROR] Fatal error: {exc}")
        raise


if __name__ == "__main__":
    main()

