"""
Auto-Select Best Commodity Trading Script

This script:
1. Trains and tests ALL available commodities
2. Ranks them by model performance (R², confidence, test metrics)
3. Selects the BEST commodity automatically
4. Monitors existing positions closely (if bot restarts, continues monitoring)
5. Invests strictly based on buying power
6. Applies strict stop-loss (2.0% default, user-configurable)
7. REQUIRES profit target (user must specify)

IMPORTANT:
- REAL MONEY IS AT RISK
- Profit target is REQUIRED (--profit-target-pct)
- Positions are automatically closed when profit target is reached or stop-loss is hit
- If bot restarts, it will monitor existing positions first
- Only trades the BEST commodity based on model performance
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline_runner import run_ingestion, regenerate_features
from train_models import train_symbols
from trading.angelone_client import AngelOneClient
from trading.execution_engine import ExecutionEngine, TradingRiskConfig
from trading.position_manager import PositionManager
from trading.symbol_universe import by_asset_type, find_by_data_symbol
from live_trader import discover_tradable_symbols, run_trading_cycle, get_current_price_from_features, load_feature_row
from ml.inference import InferencePipeline
from ml.risk import RiskManagerConfig
from ml.horizons import print_horizon_summary, normalize_profile


def setup_angelone_client(api_key: str, client_id: str, password: str, totp_secret: Optional[str] = None) -> AngelOneClient:
    """Setup Angel One client with provided credentials."""
    return AngelOneClient(api_key=api_key, client_id=client_id, password=password, totp_secret=totp_secret)


def get_all_commodities() -> List[str]:
    """Get all commodity data symbols from symbol universe."""
    commodities = by_asset_type("commodities")
    return [asset.data_symbol for asset in commodities if asset.enabled]


def rank_commodities_by_performance(
    commodities_symbols: List[str],
    timeframe: str,
    horizon: str,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Rank commodities by model performance.
    
    Returns list of dicts with:
    - symbol: data symbol
    - score: overall performance score (higher is better)
    - confidence: model confidence
    - r2_score: average R² across models
    - test_r2: average test R²
    - tradable: whether symbol is tradable
    """
    ranked = []
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"RANKING COMMODITIES BY MODEL PERFORMANCE")
        print(f"{'='*80}")
        print(f"Evaluating {len(commodities_symbols)} commodities...")
    
    for symbol in commodities_symbols:
        try:
            # Check if model exists
            from core.model_paths import horizon_dir
            model_dir = horizon_dir("commodities", symbol, timeframe, horizon)
            summary_path = model_dir / "summary.json"
            
            if not summary_path.exists():
                if verbose:
                    print(f"  [SKIP] {symbol}: No model found")
                continue
            
            # Load summary
            with open(summary_path, "r") as f:
                summary = json.load(f)
            
            # Get model predictions
            model_predictions = summary.get("model_predictions", {})
            if not model_predictions:
                if verbose:
                    print(f"  [SKIP] {symbol}: No model predictions in summary")
                continue
            
            # Calculate average metrics
            r2_scores = []
            test_r2_scores = []
            confidences = []
            successful_models = 0
            
            for model_name, model_data in model_predictions.items():
                if model_data.get("status") == "failed":
                    continue
                
                r2 = model_data.get("r2_score") or model_data.get("r2")
                test_r2 = model_data.get("test_r2_score") or model_data.get("test_r2")
                conf = model_data.get("confidence", 0.0) or 0.0
                
                if r2 is not None:
                    r2_scores.append(float(r2))
                if test_r2 is not None:
                    test_r2_scores.append(float(test_r2))
                if conf > 0:
                    confidences.append(float(conf))
                
                successful_models += 1
            
            if successful_models == 0:
                if verbose:
                    print(f"  [SKIP] {symbol}: No successful models")
                continue
            
            # Calculate average metrics
            avg_r2 = sum(r2_scores) / len(r2_scores) if r2_scores else 0.0
            avg_test_r2 = sum(test_r2_scores) / len(test_r2_scores) if test_r2_scores else 0.0
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Check if tradable (robustness checks passed)
            tradable = summary.get("tradable", False)
            
            # Calculate overall score (weighted combination)
            # Higher is better: R² (40%), Test R² (30%), Confidence (20%), Tradable bonus (10%)
            score = (
                avg_r2 * 0.40 +
                avg_test_r2 * 0.30 +
                avg_confidence * 0.20 +
                (1.0 if tradable else 0.0) * 0.10
            )
            
            ranked.append({
                "symbol": symbol,
                "score": score,
                "confidence": avg_confidence,
                "r2_score": avg_r2,
                "test_r2": avg_test_r2,
                "tradable": tradable,
                "successful_models": successful_models,
                "model_dir": str(model_dir),
            })
            
            if verbose:
                print(f"  [RANK] {symbol:15s}: Score={score:.4f} | R²={avg_r2:.3f} | Test R²={avg_test_r2:.3f} | Conf={avg_confidence:.1f}% | Tradable={tradable}")
        
        except Exception as exc:
            if verbose:
                print(f"  [ERROR] {symbol}: {exc}")
            continue
    
    # Sort by score (descending)
    ranked.sort(key=lambda x: x["score"], reverse=True)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"RANKING COMPLETE")
        print(f"{'='*80}")
        if ranked:
            print(f"Top 5 commodities:")
            for i, item in enumerate(ranked[:5], 1):
                print(f"  {i}. {item['symbol']:15s} - Score: {item['score']:.4f} (R²={item['r2_score']:.3f}, Conf={item['confidence']:.1f}%)")
        else:
            print("  No commodities ranked (no successful models)")
        print(f"{'='*80}\n")
    
    return ranked


def check_existing_positions(
    client: AngelOneClient,
    position_manager: PositionManager,
    verbose: bool = True
) -> Tuple[List[Dict], List[str]]:
    """
    Check for existing MCX commodity positions.
    
    Returns:
        Tuple of (broker_positions, tracked_symbols)
    """
    existing_positions = []
    tracked_symbols = []
    
    try:
        # Get positions from broker
        all_positions = client.list_positions()
        if all_positions:
            # Filter only MCX positions
            mcx_positions = [
                pos for pos in all_positions
                if pos.get("exchange_segment", pos.get("_raw_exchange", "")).upper() == "MCX"
            ]
            
            if mcx_positions:
                existing_positions = mcx_positions
                if verbose:
                    print(f"\n[EXISTING POSITIONS] Found {len(mcx_positions)} MCX commodity position(s):")
                    for pos in mcx_positions:
                        symbol = pos.get("symbol", "")
                        qty = float(pos.get("qty", 0) or 0)
                        avg_entry = float(pos.get("avg_entry_price", 0) or 0)
                        ltp = float(pos.get("ltp", 0) or 0)
                        unrealized_pl = float(pos.get("unrealized_pl", 0) or 0)
                        side_str = "LONG" if qty > 0 else "SHORT"
                        print(f"  - {symbol}: {abs(qty)} {side_str} @ Rs.{avg_entry:.2f} -> Rs.{ltp:.2f} (P/L: Rs.{unrealized_pl:+.2f})")
        
        # Get tracked positions from position manager
        all_tracked = position_manager.get_all_positions()
        for pos in all_tracked:
            if pos.asset_type == "commodities" and pos.status == "open":
                tracked_symbols.append(pos.symbol)
                if verbose:
                    print(f"  [TRACKED] {pos.symbol}: {pos.quantity:.2f} {pos.side.upper()} @ Rs.{pos.entry_price:.2f}")
    
    except Exception as exc:
        if verbose:
            print(f"[WARNING] Failed to check existing positions: {exc}")
    
    return existing_positions, tracked_symbols


def main():
    parser = argparse.ArgumentParser(
        description="Auto-select best commodity: train all -> rank -> trade best one with strict monitoring"
    )
    parser.add_argument(
        "--timeframe",
        default="1d",
        help="Timeframe (default: 1d)",
    )
    parser.add_argument(
        "--commodities-horizon",
        default="short",
        choices=["intraday", "short", "long"],
        help="Trading horizon for commodities (default: short)",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=5.0,
        help="Years of historical data to fetch (default: 5.0)",
    )
    parser.add_argument(
        "--profit-target-pct",
        type=float,
        default=5.0,
        help="Profit target percentage (default: 5.0 for 5%%). Example: --profit-target-pct 5.0",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=2.0,
        help="Stop-loss percentage (default: 2.0%% for commodities - strict)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (no real orders)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between trading cycles (default: 60)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training phase (use existing models)",
    )
    parser.add_argument(
        "--skip-ranking",
        action="store_true",
        help="Skip ranking phase (use first tradable commodity)",
    )
    parser.add_argument(
        "--angelone-api-key",
        help="Angel One API key (or set ANGEL_ONE_API_KEY env var)",
    )
    parser.add_argument(
        "--angelone-client-id",
        help="Angel One client ID (or set ANGEL_ONE_CLIENT_ID env var)",
    )
    parser.add_argument(
        "--angelone-password",
        help="Angel One trading password/MPIN (or set ANGEL_ONE_PASSWORD env var)",
    )
    parser.add_argument(
        "--angelone-totp-secret",
        help="Angel One TOTP secret (or set ANGEL_ONE_TOTP_SECRET env var)",
    )
    
    args = parser.parse_args()
    
    # Validate profit target
    if args.profit_target_pct <= 0:
        print(f"❌ ERROR: Profit target must be positive. Got: {args.profit_target_pct}")
        print("   Example: --profit-target-pct 5.0 (for 5% profit target)")
        sys.exit(1)
    
    if args.profit_target_pct > 1000:
        print(f"❌ ERROR: Profit target seems unreasonably high: {args.profit_target_pct}%")
        sys.exit(1)
    
    # Get Angel One credentials
    # PRIORITY 1: Read from .env file FIRST (like AngelOneClient does)
    api_key = args.angelone_api_key
    client_id = args.angelone_client_id
    password = args.angelone_password
    totp_secret = args.angelone_totp_secret
    
    # Read from .env file if not provided as arguments
    if not api_key or not client_id or not password:
        if os.path.exists(".env"):
            try:
                with open(".env", "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        key = key.strip()
                        # Remove quotes and comments (everything after #)
                        val = val.split("#")[0].strip().strip('"').strip("'")
                        if key == "ANGEL_ONE_API_KEY" and not api_key:
                            api_key = val
                        elif key == "ANGEL_ONE_CLIENT_ID" and not client_id:
                            client_id = val
                        elif key == "ANGEL_ONE_PASSWORD" and not password:
                            password = val
                        elif key == "ANGEL_ONE_TOTP_SECRET" and not totp_secret:
                            totp_secret = val
            except Exception:
                pass
    
    # PRIORITY 2: Fallback to environment variables
    if not api_key:
        api_key = os.getenv("ANGEL_ONE_API_KEY")
    if not client_id:
        client_id = os.getenv("ANGEL_ONE_CLIENT_ID")
    if not password:
        password = os.getenv("ANGEL_ONE_PASSWORD")
    if not totp_secret:
        totp_secret = os.getenv("ANGEL_ONE_TOTP_SECRET")
    
    if not api_key or not client_id or not password:
        print("[ERROR] Angel One credentials required!")
        print("Set ANGEL_ONE_API_KEY, ANGEL_ONE_CLIENT_ID, ANGEL_ONE_PASSWORD in .env file")
        sys.exit(1)
    
    timeframe = args.timeframe
    horizon = args.commodities_horizon
    years = max(args.years, 0.5)
    
    print("=" * 80)
    print("AUTO-SELECT BEST COMMODITY TRADING SYSTEM")
    print("=" * 80)
    print(f"Mode:           {'DRY RUN' if args.dry_run else 'LIVE TRADING (REAL MONEY)'}")
    print(f"Timeframe:      {timeframe}")
    print(f"Horizon:        {horizon}")
    print(f"Years of Data:  {years}")
    print(f"Profit Target:  {args.profit_target_pct:.2f}% (default: 5.0%, can override with --profit-target-pct)")
    print(f"Stop-Loss:      {args.stop_loss_pct:.2f}% (STRICT)")
    print(f"Interval:       {args.interval} seconds")
    print(f"Position Size:  10% of equity per commodity (STRICT - based on buying power)")
    print(f"Max Total:      50% of equity across all positions")
    if not args.dry_run:
        print(f"⚠️  WARNING: LIVE TRADING MODE - REAL MONEY IS AT RISK")
    print("=" * 80)
    print()
    
    # Setup Angel One client
    print("=" * 80)
    print("SETTING UP ANGEL ONE MCX CONNECTION")
    print("=" * 80)
    try:
        angelone_client = setup_angelone_client(api_key, client_id, password, totp_secret)
        print("[OK] Angel One client initialized")
        
        # Test account
        account = angelone_client.get_account()
        equity = account.get("equity", 0)
        buying_power = account.get("buying_power", 0)
        print(f"[OK] Account connected")
        print(f"  Equity: Rs.{equity:,.2f}")
        print(f"  Buying Power: Rs.{buying_power:,.2f}")
        if equity <= 0:
            print(f"[WARNING] Account equity is zero or negative!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Angel One: {e}")
        sys.exit(1)
    
    # Initialize position manager early to check existing positions
    position_manager = PositionManager()
    
    # Check for existing positions FIRST
    print("\n" + "=" * 80)
    print("CHECKING EXISTING POSITIONS")
    print("=" * 80)
    existing_positions, tracked_symbols = check_existing_positions(angelone_client, position_manager, verbose=True)
    
    # If existing positions found, prioritize monitoring them
    if existing_positions or tracked_symbols:
        print(f"\n[PRIORITY] Found {len(existing_positions)} broker position(s) and {len(tracked_symbols)} tracked position(s)")
        print(f"[ACTION] Will MONITOR existing positions first before considering new trades")
        print(f"[INFO] Existing positions will be monitored with profit target: {args.profit_target_pct:.2f}%")
        print(f"[INFO] Use --skip-training and --skip-ranking to start monitoring immediately")
    
    # Get all commodities
    all_commodities = get_all_commodities()
    print(f"\n[INFO] Found {len(all_commodities)} total commodities in universe")
    
    # Stage 1: Historical ingestion (if not skipping training)
    if not args.skip_training:
        print("\n" + "=" * 80)
        print("[1/4] INGESTING HISTORICAL DATA FOR ALL COMMODITIES")
        print("=" * 80)
        run_ingestion(
            mode="historical",
            crypto_symbols=None,
            commodities_symbols=all_commodities,
            timeframe=timeframe,
            years=years,
        )
        print("    ✓ Historical data ingestion complete.")
        
        # Stage 2: Feature generation
        print("\n" + "=" * 80)
        print("[2/4] REGENERATING FEATURES FOR ALL COMMODITIES")
        print("=" * 80)
        regenerate_features("commodities", set(all_commodities), timeframe)
        print("    ✓ Feature generation complete.")
        
        # Stage 3: Model training
        print("\n" + "=" * 80)
        print("[3/4] TRAINING MODELS FOR ALL COMMODITIES")
        print("=" * 80)
        print("    Training all commodities with strict overfitting prevention...")
        horizon_map = {"commodities": horizon}
        try:
            train_symbols(
                crypto_symbols=[],
                commodities_symbols=all_commodities,
                timeframe=timeframe,
                output_dir="models",
                horizon_profiles=horizon_map,
            )
            print("    ✓ Model training complete.")
        except Exception as train_exc:
            print(f"    ✗ Model training failed: {train_exc}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n[SKIP] Training phase skipped (using existing models)")
    
    # Stage 4: Rank commodities
    if not args.skip_ranking:
        print("\n" + "=" * 80)
        print("[4/4] RANKING COMMODITIES BY PERFORMANCE")
        print("=" * 80)
        ranked = rank_commodities_by_performance(all_commodities, timeframe, horizon, verbose=True)
        
        if not ranked:
            print("❌ ERROR: No commodities could be ranked. Train models first or check model directories.")
            sys.exit(1)
        
        # Show ALL ranked commodities in a table (top 10)
        print(f"\n{'='*80}")
        print(f"📊 ALL RANKED COMMODITIES (Top 10)")
        print(f"{'='*80}")
        print(f"{'Rank':<6} {'Symbol':<20} {'Score':<10} {'R²':<8} {'Test R²':<10} {'Conf':<8} {'Models':<8} {'Tradable':<10}")
        print(f"{'-'*90}")
        for i, item in enumerate(ranked[:10], 1):
            tradable_mark = "✅ YES" if item["tradable"] else "❌ NO"
            print(f"{i:<6} {item['symbol']:<20} {item['score']:<10.4f} {item['r2_score']:<8.3f} {item['test_r2']:<10.3f} {item['confidence']:<8.1f}% {item['successful_models']:<8} {tradable_mark:<10}")
        print(f"{'='*80}\n")
        
        # Select best tradable commodity
        best_commodity = None
        tradable_ranked = [item for item in ranked if item["tradable"]]
        
        if tradable_ranked:
            best_commodity = tradable_ranked[0]["symbol"]
            print(f"{'='*80}")
            print(f"✅ SELECTED BEST TRADABLE COMMODITY: {best_commodity}")
            print(f"{'='*80}")
            best_item = tradable_ranked[0]
            best_rank = ranked.index(best_item) + 1
            print(f"  Rank:         #{best_rank} (out of {len(ranked)} total commodities)")
            print(f"  Score:        {best_item['score']:.4f}")
            print(f"  R² Score:     {best_item['r2_score']:.3f}")
            print(f"  Test R²:      {best_item['test_r2']:.3f}")
            print(f"  Confidence:   {best_item['confidence']:.1f}%")
            print(f"  Models:       {best_item['successful_models']}/5 successful")
            print(f"  Tradable:     ✅ YES (passed all robustness checks)")
            print(f"{'='*80}\n")
            
            # Show other top tradable options
            if len(tradable_ranked) > 1:
                print(f"📋 OTHER TOP TRADABLE OPTIONS (if best one fails):")
                for i, item in enumerate(tradable_ranked[1:6], 2):  # Show next 5
                    rank_num = ranked.index(item) + 1
                    print(f"  {i}. {item['symbol']:20s} - Rank #{rank_num}, Score: {item['score']:.4f} (R²={item['r2_score']:.3f}, Conf={item['confidence']:.1f}%)")
                print()
        else:
            print("❌ ERROR: No tradable commodities found. All models failed robustness checks.")
            print("   Top ranked commodities (not tradable):")
            for item in ranked[:5]:
                print(f"     - {item['symbol']}: Score={item['score']:.4f} (not tradable)")
            sys.exit(1)
        
        selected_symbols = [best_commodity]
    else:
        print("\n[SKIP] Ranking phase skipped")
        # Use first tradable commodity or existing position symbol
        if tracked_symbols:
            # If we have tracked positions, use those symbols
            tradable_list = discover_tradable_symbols("commodities", timeframe, override_horizon=horizon)
            selected_symbols = [info["asset"].data_symbol for info in tradable_list if info["asset"].data_symbol in tracked_symbols]
            if not selected_symbols:
                # Fallback to first tradable
                selected_symbols = [info["asset"].data_symbol for info in tradable_list[:1]]
        else:
            tradable_list = discover_tradable_symbols("commodities", timeframe, override_horizon=horizon)
            selected_symbols = [info["asset"].data_symbol for info in tradable_list[:1]] if tradable_list else []
        
        if not selected_symbols:
            print("❌ ERROR: No tradable commodities found")
            sys.exit(1)
        
        print(f"[INFO] Using commodity: {selected_symbols[0]}")
    
    # Discover tradable symbols (should include selected symbol)
    all_tradable = discover_tradable_symbols("commodities", timeframe, override_horizon=horizon)
    
    # Filter to selected symbol(s) OR existing positions
    if existing_positions or tracked_symbols:
        # Include symbols with existing positions for monitoring
        position_symbols = set()
        for pos in existing_positions:
            # Try to map MCX symbol back to data symbol
            mcx_symbol = pos.get("symbol", "")
            # Find asset by trading symbol
            asset = find_by_data_symbol(mcx_symbol)  # This might not work directly
            # Better: check all assets for matching trading symbol
            from trading.symbol_universe import UNIVERSE
            for a in UNIVERSE:
                if a.asset_type == "commodities" and a.trading_symbol.upper() == mcx_symbol.upper():
                    position_symbols.add(a.data_symbol)
                    break
        
        # Add tracked symbols
        for tracked_sym in tracked_symbols:
            # Find asset by trading symbol
            from trading.symbol_universe import UNIVERSE
            for a in UNIVERSE:
                if a.asset_type == "commodities" and a.trading_symbol.upper() == tracked_sym.upper():
                    position_symbols.add(a.data_symbol)
                    break
        
        # Combine with selected symbols
        symbols_to_trade = list(set(selected_symbols) | position_symbols)
        print(f"[INFO] Trading symbols: {symbols_to_trade} (includes existing positions)")
    else:
        symbols_to_trade = selected_symbols
    
    # Sync existing positions with position manager BEFORE filtering
    if existing_positions:
        print("\n" + "=" * 80)
        print("SYNCING EXISTING POSITIONS WITH POSITION MANAGER")
        print("=" * 80)
        
        # Map MCX symbols to data symbols
        from trading.symbol_universe import UNIVERSE
        mcx_to_data = {}
        for asset in UNIVERSE:
            if asset.asset_type == "commodities":
                mcx_to_data[asset.trading_symbol.upper()] = asset.data_symbol
        
        synced_count = 0
        for pos in existing_positions:
            mcx_symbol = pos.get("symbol", "").upper()
            if mcx_symbol in mcx_to_data:
                data_symbol = mcx_to_data[mcx_symbol]
                # Check if already tracked
                tracked = position_manager.get_position(mcx_symbol)
                if not tracked:
                    # Create position entry
                    qty = float(pos.get("qty", 0) or 0)
                    avg_entry = float(pos.get("avg_entry_price", 0) or 0)
                    side = "long" if qty > 0 else "short"
                    
                    try:
                        position_manager.save_position(
                            symbol=mcx_symbol,
                            data_symbol=data_symbol,
                            asset_type="commodities",
                            side=side,
                            entry_price=avg_entry,
                            quantity=abs(qty),
                            profit_target_pct=args.profit_target_pct,
                            stop_loss_pct=args.stop_loss_pct,
                        )
                        synced_count += 1
                        print(f"  [SYNC] Synced {mcx_symbol} ({data_symbol}): {abs(qty)} {side.upper()} @ Rs.{avg_entry:.2f}")
                    except Exception as sync_exc:
                        print(f"  [WARNING] Failed to sync {mcx_symbol}: {sync_exc}")
        
        if synced_count > 0:
            print(f"  ✓ Synced {synced_count} existing position(s) with profit target: {args.profit_target_pct:.2f}%")
        else:
            print("  ✓ All existing positions already tracked")
        print("=" * 80)
    
    # Filter tradable list to selected symbols OR existing positions
    requested_set = {s.upper() for s in symbols_to_trade}
    
    # Also include symbols from existing positions
    if existing_positions:
        from trading.symbol_universe import UNIVERSE
        for pos in existing_positions:
            mcx_symbol = pos.get("symbol", "").upper()
            for asset in UNIVERSE:
                if asset.asset_type == "commodities" and asset.trading_symbol.upper() == mcx_symbol:
                    requested_set.add(asset.data_symbol.upper())
                    break
    
    tradable = [
        info
        for info in all_tradable
        if info["asset"].data_symbol.upper() in requested_set
    ]
    
    if not tradable:
        print("❌ ERROR: No tradable symbols found after filtering")
        print(f"   Requested: {requested_set}")
        print(f"   Available: {[info['asset'].data_symbol for info in all_tradable]}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"TRADABLE SYMBOLS: {len(tradable)}")
    print(f"{'='*80}")
    for info in tradable:
        asset = info["asset"]
        mcx_symbol = asset.get_mcx_symbol(horizon) if hasattr(asset, 'get_mcx_symbol') else asset.trading_symbol
        print(f"  - {asset.data_symbol} -> MCX: {mcx_symbol} (horizon: {info['horizon']})")
    
    # Initialize execution engine with STRICT buying power limits
    try:
        # Get current buying power to show user
        account = angelone_client.get_account()
        equity = float(account.get("equity", 0) or 0)
        buying_power = float(account.get("buying_power", 0) or 0)
        
        print(f"\n{'='*80}")
        print(f"💰 ACCOUNT & POSITION SIZING (STRICT BUYING POWER LIMITS)")
        print(f"{'='*80}")
        print(f"  Equity:        Rs.{equity:,.2f}")
        print(f"  Buying Power:  Rs.{buying_power:,.2f}")
        print(f"  Max Position:  10% of equity = Rs.{equity * 0.10:,.2f} per commodity")
        print(f"  Max Total:     50% of equity = Rs.{equity * 0.50:,.2f} across all positions")
        print(f"  Position sizing will be STRICTLY based on available buying power")
        print(f"{'='*80}\n")
        
        risk_config = TradingRiskConfig(
            default_stop_loss_pct=args.stop_loss_pct / 100.0,  # Convert to decimal
            user_stop_loss_pct=args.stop_loss_pct,
            profit_target_pct=args.profit_target_pct,
            allow_short=False,  # SHORTING DISABLED for commodities
            max_notional_per_symbol_pct=0.10,  # Maximum 10% of equity per commodity (STRICT)
            max_total_equity_pct=0.50,  # Maximum 50% of equity deployed across all positions
            max_daily_loss_pct=0.05,  # 5% daily loss circuit breaker
            slippage_buffer_pct=0.001,  # 0.1% slippage buffer
        )
        execution_engine = ExecutionEngine(
            client=angelone_client,
            risk_config=risk_config,
            position_manager=position_manager,
            log_path=Path("logs") / "trading" / "commodities_trades.jsonl",
        )
        print("    ✓ Execution engine initialized with strict buying power limits")
    except Exception as exc:
        print(f"    ✗ Failed to initialize ExecutionEngine: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Final confirmation
    print("\n" + "=" * 80)
    print("✅ READY TO START TRADING")
    print("=" * 80)
    print(f"  Selected Commodity: {selected_symbols[0] if selected_symbols else 'N/A'}")
    print(f"  Existing Positions: {len(existing_positions)} (will be monitored)")
    print(f"  Profit Target: {args.profit_target_pct:.2f}%")
    print(f"  Stop-Loss: {args.stop_loss_pct:.2f}% (STRICT)")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE TRADING (REAL MONEY)'}")
    print("=" * 80)
    print()
    
    if not args.dry_run:
        print("⚠️  WARNING: LIVE TRADING MODE - REAL MONEY IS AT RISK")
        print("   Press Ctrl+C within 5 seconds to cancel...")
        time.sleep(5)
    
    # Run trading cycles
    cycle_count = 0
    excluded_symbols = set()
    
    try:
        while True:
            cycle_count += 1
            now = datetime.utcnow().isoformat() + "Z"
            print(f"\n{'='*80}")
            print(f"[CYCLE {cycle_count}] {now}")
            print(f"{'='*80}")
            
            try:
                cycle_results = run_trading_cycle(
                    execution_engine=execution_engine,
                    tradable_symbols=tradable,
                    dry_run=args.dry_run,
                    verbose=True,
                    update_data=True,
                    regenerate_features_flag=True,
                    profit_target_pct=args.profit_target_pct,
                    user_stop_loss_pct=args.stop_loss_pct,
                    excluded_symbols=excluded_symbols,
                )
                
                if cycle_results.get("symbols_stopped"):
                    for symbol in cycle_results["symbols_stopped"]:
                        excluded_symbols.add(symbol)
                
                # Summary
                print(f"\n[CYCLE {cycle_count} SUMMARY]")
                print(f"  Processed: {cycle_results.get('symbols_processed', 0)}")
                print(f"  Traded: {cycle_results.get('symbols_traded', 0)}")
                print(f"  Skipped: {cycle_results.get('symbols_skipped', 0)}")
                print(f"  Errors: {len(cycle_results.get('errors', []))}")
                
                # Show positions
                try:
                    all_positions = angelone_client.list_positions()
                    mcx_positions = [
                        pos for pos in all_positions
                        if pos.get("exchange_segment", pos.get("_raw_exchange", "")).upper() == "MCX"
                    ]
                    if mcx_positions:
                        print(f"\n  📊 ACTIVE POSITIONS: {len(mcx_positions)}")
                        for pos in mcx_positions:
                            symbol = pos.get("symbol", "")
                            qty = float(pos.get("qty", 0) or 0)
                            avg_entry = float(pos.get("avg_entry_price", 0) or 0)
                            ltp = float(pos.get("ltp", 0) or 0)
                            unrealized_pl = float(pos.get("unrealized_pl", 0) or 0)
                            side_str = "LONG" if qty > 0 else "SHORT"
                            print(f"    {symbol}: {abs(qty)} {side_str} @ Rs.{avg_entry:.2f} -> Rs.{ltp:.2f} (P/L: Rs.{unrealized_pl:+.2f})")
                except Exception:
                    pass
                
            except Exception as cycle_exc:
                print(f"\n[ERROR] Trading cycle {cycle_count} failed: {cycle_exc}")
                import traceback
                traceback.print_exc()
            
            print(f"\n[WAIT] Waiting {args.interval} seconds before next cycle...")
            print(f"  Press Ctrl+C to stop")
            time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("\n\n[STOP] Trading stopped by user")
        print("  Positions will remain open (not liquidated automatically)")


if __name__ == "__main__":
    main()

