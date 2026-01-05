"""
End-to-end Commodities Trading Pipeline with Angel One MCX

This script runs the complete flow for selected commodity symbols:
1) Historical ingestion  (raw candles)
2) Feature generation    (features.json)
3) Model training        (models/commodities/...)
4) Live trading cycles on Angel One MCX with profit target monitoring

Features:
- Automatic data fetching, feature calculation, and model training
- Real money trading with MCX futures contracts
- Strengthened stop-loss (2.0% default, user-configurable)
- MCX lot size handling
- Horizon-based contract selection
- Position monitoring for profit targets and stop-loss
- Comprehensive logging to commodities_trades.jsonl

IMPORTANT:
- REAL MONEY IS AT RISK
- Profit target is REQUIRED (--profit-target-pct)
- Positions are automatically closed when profit target is reached or stop-loss is hit
- Test thoroughly in dry-run mode first
- Start with small position sizes
- Monitor positions closely
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline_runner import run_ingestion, regenerate_features
from train_models import train_symbols
from trading.angelone_client import AngelOneClient
from trading.execution_engine import ExecutionEngine, TradingRiskConfig
from trading.position_manager import PositionManager
from trading.symbol_universe import find_by_data_symbol, by_asset_type
from live_trader import discover_tradable_symbols, run_trading_cycle
from ml.horizons import print_horizon_summary


def setup_angelone_client(api_key: str, client_id: str, password: str, totp_secret: Optional[str] = None) -> AngelOneClient:
    """
    Setup Angel One client with provided credentials.
    
    Args:
        api_key: Angel One API key
        client_id: Angel One client ID
        password: Trading password/MPIN
        totp_secret: Optional TOTP secret (if using pyotp library)
        
    Returns:
        Configured AngelOneClient instance
    """
    return AngelOneClient(api_key=api_key, client_id=client_id, password=password, totp_secret=totp_secret)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end commodities trading pipeline: ingest -> features -> train -> trade on MCX using Angel One API"
    )
    parser.add_argument(
        "--commodities-symbols",
        nargs="+",
        help="Commodity symbols (e.g., GC=F CL=F SI=F). If not provided, will auto-discover all enabled commodities from symbol universe.",
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
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        help="User-defined stop-loss percentage (e.g., 1.5 for 1.5%%)",
    )
    parser.add_argument(
        "--profit-target-pct",
        type=float,
        required=True,
        help="REQUIRED: Profit target percentage (e.g., 5.0 for 5%%). You must specify this before trading.",
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
    
    args = parser.parse_args()
    
    # CRITICAL VALIDATION: Profit target is REQUIRED and must be valid
    if args.profit_target_pct is None:
        print("❌ ERROR: --profit-target-pct is REQUIRED and cannot be omitted.")
        print("   You must specify a profit target before trading (e.g., --profit-target-pct 5.0)")
        sys.exit(1)
    
    if args.profit_target_pct <= 0:
        print(f"❌ ERROR: Profit target must be positive. Got: {args.profit_target_pct}")
        print("   Example: --profit-target-pct 5.0 (for 5% profit target)")
        sys.exit(1)
    
    if args.profit_target_pct > 1000:
        print(f"❌ ERROR: Profit target seems unreasonably high: {args.profit_target_pct}%")
        print("   Please verify your profit target. Maximum allowed: 1000%")
        print("   Example: --profit-target-pct 5.0 (for 5% profit target)")
        sys.exit(1)
    
    # Get Angel One credentials
    # Priority: Command line args > .env file > Environment variables
    # User preference: Use .env file only (as requested)
    api_key = args.angelone_api_key
    client_id = args.angelone_client_id
    password = args.angelone_password
    totp_secret = args.angelone_totp_secret
    
    # Read from .env file FIRST (user preference)
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
                    
                    # Remove inline comments (anything after #)
                    if "#" in val:
                        val = val.split("#", 1)[0]
                        
                    key = key.strip()
                    val = val.strip()
                    
                    # Remove quotes if present
                    if val.startswith('"') and val.endswith('"'):
                        val = val[1:-1]
                    elif val.startswith("'") and val.endswith("'"):
                        val = val[1:-1]
                    
                    # Always read from .env if not provided via command line
                    if key == "ANGEL_ONE_API_KEY" and not api_key:
                        api_key = val
                    elif key == "ANGEL_ONE_CLIENT_ID" and not client_id:
                        client_id = val
                    elif key == "ANGEL_ONE_PASSWORD" and not password:
                        password = val
                    elif key == "ANGEL_ONE_TOTP_SECRET" and not totp_secret:
                        totp_secret = val
        except Exception as e:
            print(f"[WARNING] Failed to read .env file: {e}")
            pass  # If .env parsing fails, continue
    
    # Fallback to environment variables only if .env didn't provide values
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
        print()
        print("SET YOUR ANGEL ONE CREDENTIALS IN .env FILE:")
        print("=" * 80)
        print("Create/edit .env file in project root:")
        print('    ANGEL_ONE_API_KEY="your_api_key_here"')
        print('    ANGEL_ONE_CLIENT_ID="your_client_id"')
        print('    ANGEL_ONE_PASSWORD="your_trading_password_or_mpin"')
        print('    ANGEL_ONE_TOTP_SECRET="your_totp_secret"  # Optional if using pyotp')
        print()
        print("NOTE: TOTP must be enabled in your Angel One account.")
        print("      See ANGEL_ONE_REQUIREMENTS.md for setup instructions.")
        print("=" * 80)
        sys.exit(1)
    
    # Auto-discover commodities symbols if not provided
    if args.commodities_symbols:
        # User provided symbols explicitly
        commodities_symbols = [s.upper() for s in args.commodities_symbols]
        print(f"\n[CONFIG] Using user-specified commodities: {', '.join(commodities_symbols)}")
    else:
        # Auto-discover enabled commodities from symbol_universe.py
        print("\n[AUTO-DISCOVERY] No symbols specified, discovering enabled commodities...")
        enabled_commodities = by_asset_type("commodities")
        if not enabled_commodities:
            print("[ERROR] No enabled commodities found in symbol_universe.py!")
            print("        Please enable at least one commodity in trading/symbol_universe.py")
            print("        Set enabled=True for the commodities you want to trade.")
            sys.exit(1)
        
        commodities_symbols = [asset.data_symbol.upper() for asset in enabled_commodities]
        print(f"[AUTO-DISCOVERY] Found {len(commodities_symbols)} enabled commodity(ies):")
        for asset in enabled_commodities:
            mcx_symbol = asset.get_mcx_symbol(args.commodities_horizon) if hasattr(asset, 'get_mcx_symbol') else asset.trading_symbol
            print(f"  ✓ {asset.logical_name}: {asset.data_symbol} → MCX: {mcx_symbol}")
        print()
    timeframe = args.timeframe
    horizon = args.commodities_horizon
    years = max(args.years, 0.5)
    
    print("=" * 80)
    print("END-TO-END COMMODITIES PIPELINE (ANGEL ONE MCX)")
    print("=" * 80)
    print(f"Symbols:        {', '.join(commodities_symbols)}")
    print(f"Timeframe:      {timeframe}")
    print(f"Horizon:        {horizon}")
    print(f"Years of Data:  {years}")
    print(f"Profit Target:  {args.profit_target_pct:.2f}% (REQUIRED - user specified)")
    if args.stop_loss_pct:
        print(f"Stop-Loss:      {args.stop_loss_pct:.2f}% (user specified)")
    else:
        print(f"Stop-Loss:      2.0% (default for commodities - real money)")
    print(f"Mode:           {'DRY RUN (no real orders)' if args.dry_run else 'LIVE TRADING (REAL MONEY)'}")
    if not args.dry_run:
        print(f"⚠️  WARNING: LIVE TRADING MODE - REAL MONEY IS AT RISK")
    print("=" * 80)
    print()
    
    # Show available horizons and their trading behavior
    print_horizon_summary()
    
    # Setup Angel One client
    print("\n" + "=" * 80)
    print("SETTING UP ANGEL ONE MCX CONNECTION")
    print("=" * 80)
    try:
        angelone_client = setup_angelone_client(api_key, client_id, password, totp_secret)
        print("[OK] Angel One client initialized successfully")
        
        # Show token info (first 20 chars for security)
        if hasattr(angelone_client, '_access_token') and angelone_client._access_token:
            token_preview = angelone_client._access_token[:20] + "..."
            print(f"[INFO] Access token obtained: {token_preview}")
            if hasattr(angelone_client, '_token_expiry') and angelone_client._token_expiry:
                from datetime import datetime
                expiry = datetime.fromtimestamp(angelone_client._token_expiry)
                print(f"[INFO] Token expires at: {expiry.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Angel One client: {e}")
        sys.exit(1)
    
    # Test account connection
    try:
        account = angelone_client.get_account()
        equity = account.get("equity", 0)
        buying_power = account.get("buying_power", 0)
        print(f"[OK] Account connected")
        print(f"  Equity: ₹{equity:,.2f}")
        print(f"  Buying Power: ₹{buying_power:,.2f}")
        if equity <= 0:
            print(f"[WARNING] Account equity is zero or negative!")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Angel One account: {e}")
        sys.exit(1)
    
    # ------------------------------------------------------------------
    # Stage 1: Historical ingestion (COMMODITIES ONLY)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[1/4] INGESTING HISTORICAL DATA")
    print("=" * 80)
    run_ingestion(
        mode="historical",
        crypto_symbols=None,  # Commodities-only script
        commodities_symbols=commodities_symbols,
        timeframe=timeframe,
        years=years,
    )
    print("    ✓ Historical data ingestion complete.")
    
    # ------------------------------------------------------------------
    # Stage 2: Feature generation (COMMODITIES ONLY)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[2/4] REGENERATING FEATURES")
    print("=" * 80)
    regenerate_features("commodities", set(commodities_symbols), timeframe)
    print("    ✓ Feature generation complete.")
    
    # ------------------------------------------------------------------
    # Stage 3: Model training (COMMODITIES ONLY)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[3/4] TRAINING MODELS")
    print("=" * 80)
    print("    Training models with strict overfitting prevention for real money trading...")
    horizon_map = {"commodities": horizon}
    try:
        train_symbols(
            crypto_symbols=[],  # Commodities-only script
            commodities_symbols=commodities_symbols,
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
    
    # Force flush to ensure training completion message is visible
    sys.stdout.flush()
    
    # ------------------------------------------------------------------
    # Stage 4: Live trading cycle(s)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[4/4] PREPARING LIVE TRADING")
    print("=" * 80)
    sys.stdout.flush()
    
    # Discover which of the requested symbols actually have trained models.
    # CRITICAL: Pass the trained horizon as override_horizon so discover_tradable_symbols
    # finds the correct models (e.g., short models instead of default)
    print(f"    [DEBUG] Discovering tradable symbols for commodities with horizon: {horizon}")
    try:
        all_tradable = discover_tradable_symbols(
            asset_type="commodities",
            timeframe=timeframe,
            override_horizon=horizon  # Override asset's default horizon_profile with the trained horizon
        )
        print(f"    [DEBUG] Discovered {len(all_tradable)} total tradable commodity symbols")
        if all_tradable:
            print(f"    [DEBUG] Tradable symbols found:")
            for info in all_tradable:
                asset = info["asset"]
                print(f"      - {asset.data_symbol} (horizon: {info['horizon']}, model_dir: {info['model_dir']})")
    except Exception as disc_exc:
        print(f"    ✗ Failed to discover tradable symbols: {disc_exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Restrict to the user-selected symbols only.
    requested_set = {s.upper() for s in commodities_symbols}
    print(f"    [DEBUG] Filtering for requested symbols: {requested_set}")
    tradable = [
        info
        for info in all_tradable
        if info["asset"].data_symbol.upper() in requested_set
    ]
    
    if not tradable:
        print("    ✗ No tradable symbols found after training. Exiting.")
        print("      Make sure models trained successfully for the requested symbols.")
        print(f"      Discovered {len(all_tradable)} total tradable symbols, but none matched requested: {requested_set}")
        if all_tradable:
            print(f"      Available symbols: {[info['asset'].data_symbol for info in all_tradable]}")
        # Check if models exist for requested symbols
        from core.model_paths import horizon_dir
        print(f"\n      [DEBUG] Checking model directories for requested symbols:")
        for symbol in requested_set:
            model_path = horizon_dir("commodities", symbol, timeframe, horizon)
            summary_path = model_path / "summary.json"
            print(f"        {symbol}: model_dir={model_path}, summary exists={summary_path.exists()}")
        sys.exit(1)
    
    print(f"    ✓ Found {len(tradable)} tradable symbol(s):")
    for info in tradable:
        asset = info["asset"]
        mcx_symbol = asset.get_mcx_symbol(horizon) if hasattr(asset, 'get_mcx_symbol') else asset.trading_symbol
        print(f"      - {asset.data_symbol} -> MCX: {mcx_symbol} (horizon: {info['horizon']})")
    sys.stdout.flush()
    
    # Initialize execution engine and position manager
    try:
        risk_config = TradingRiskConfig(
            default_stop_loss_pct=0.020,  # 2.0% for commodities (real money)
            user_stop_loss_pct=args.stop_loss_pct,  # User override if provided
            profit_target_pct=args.profit_target_pct,
            allow_short=False,  # SHORTING DISABLED for commodities (will be enabled later)
            max_total_equity_pct=0.50,  # Maximum 50% of equity deployed across all positions (ENFORCED)
            max_daily_loss_pct=0.05,  # Maximum 5% daily loss before trading halt (circuit breaker)
            slippage_buffer_pct=0.001,  # 0.1% slippage buffer for stop-loss execution
        )
        position_manager = PositionManager()
        # Ensure Path is available (re-import to avoid any shadowing issues)
        from pathlib import Path
        execution_engine = ExecutionEngine(
            client=angelone_client,
            risk_config=risk_config,
            position_manager=position_manager,
            log_path=Path("logs") / "trading" / "commodities_trades.jsonl",
        )
        print("    ✓ Execution engine and position manager initialized")
        sys.stdout.flush()
    except Exception as exc:
        print(f"    ✗ Failed to initialize ExecutionEngine: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # LIQUIDATE ONLY MCX COMMODITIES POSITIONS (if not dry-run)
    # CRITICAL: Do NOT touch any non-MCX positions (stocks, other exchanges, etc.)
    if not args.dry_run:
        print("\n" + "=" * 80)
        print("LIQUIDATING EXISTING MCX COMMODITIES POSITIONS ONLY")
        print("=" * 80)
        print("⚠️  SAFETY: Only MCX (commodities) positions will be closed.")
        print("   All other positions (stocks, other exchanges) will be LEFT UNTOUCHED.")
        print("=" * 80)
        try:
            all_positions = angelone_client.list_positions()
            if all_positions:
                mcx_positions = []
                other_positions = []
                
                # STRICT FILTERING: Only process MCX positions
                for pos in all_positions:
                    exchange_segment = pos.get("exchange_segment", pos.get("_raw_exchange", "")).upper()
                    if exchange_segment == "MCX":
                        mcx_positions.append(pos)
                    else:
                        other_positions.append(pos)
                
                # Report non-MCX positions (but DO NOT touch them)
                if other_positions:
                    print(f"\n[PROTECTED] Found {len(other_positions)} non-MCX position(s) - LEAVING UNTOUCHED:")
                    for pos in other_positions:
                        symbol = pos.get("symbol", "")
                        exchange = pos.get("exchange_segment", pos.get("_raw_exchange", "UNKNOWN"))
                        qty = float(pos.get("qty", 0) or 0)
                        side = "LONG" if qty > 0 else "SHORT"
                        print(f"  🔒 PROTECTED: {symbol} ({exchange}) - {abs(qty)} {side} - NOT TOUCHED")
                    print()
                
                # Only liquidate MCX positions
                if mcx_positions:
                    print(f"[LIQUIDATE] Found {len(mcx_positions)} MCX commodity position(s) - closing...")
                    for pos in mcx_positions:
                        symbol = pos.get("symbol", "")
                        qty = float(pos.get("qty", 0) or 0)
                        if abs(qty) > 0:
                            close_side = "sell" if qty > 0 else "buy"
                            try:
                                close_resp = angelone_client.submit_order(
                                    symbol=symbol,
                                    qty=int(abs(qty)),
                                    side=close_side,
                                    order_type="market",
                                    time_in_force="gtc",
                                )
                                avg_entry = float(pos.get("avg_entry_price", 0) or 0)
                                ltp = float(pos.get("ltp", pos.get("avg_entry_price", 0)) or 0)
                                if qty > 0:
                                    realized_pl = (ltp - avg_entry) * abs(qty)
                                else:
                                    realized_pl = (avg_entry - ltp) * abs(qty)
                                print(f"  ✅ Closed MCX {symbol}: {abs(qty)} @ ₹{ltp:.2f} (P/L: ₹{realized_pl:+.2f})")
                            except Exception as close_exc:
                                print(f"  ❌ Failed to close MCX {symbol}: {close_exc}")
                    print("[LIQUIDATE] All MCX commodity positions liquidated")
                else:
                    print("[LIQUIDATE] No MCX commodity positions to liquidate")
            else:
                print("[LIQUIDATE] No open positions found")
        except Exception as liq_exc:
            print(f"[WARNING] Failed to liquidate MCX positions: {liq_exc}")
        print("=" * 80)
        print()
    
    # Final confirmation before starting cycles
    print()
    print("=" * 80)
    print("✅ ALL SETUP COMPLETE - READY TO START TRADING")
    print("=" * 80)
    print(f"  Tradable Symbols: {len(tradable)}")
    print(f"  Execution Engine: {'DRY RUN' if args.dry_run else 'LIVE TRADING'}")
    print(f"  Profit Target: {args.profit_target_pct:.2f}%")
    print("=" * 80)
    sys.stdout.flush()
    time.sleep(2)  # Brief pause to let user see the confirmation
    
    print()
    print("=" * 80)
    print("🚀 STARTING LIVE TRADING CYCLES 🚀")
    print("=" * 80)
    print(f"Interval: {args.interval} seconds between cycles")
    print(f"Mode: {'DRY RUN (no real orders)' if args.dry_run else 'LIVE TRADING (REAL MONEY)'}")
    print(f"Symbols: {', '.join([info['asset'].data_symbol for info in tradable])}")
    print(f"Press Ctrl+C to stop")
    print("=" * 80)
    print()
    sys.stdout.flush()
    
    # Run trading cycles
    cycle_count = 0
    excluded_symbols = set()  # Track symbols that were manually closed - stop trading them
    try:
        print("⏳ Waiting for first cycle to start...")
        sys.stdout.flush()
        
        while True:
            cycle_count += 1
            now = datetime.utcnow().isoformat() + "Z"
            print(f"\n{'='*80}")
            print(f"🔄 [CYCLE {cycle_count}] {now}")
            print(f"{'='*80}")
            sys.stdout.flush()
            
            # Run trading cycle with user stop-loss parameter
            try:
                cycle_results = run_trading_cycle(
                    execution_engine=execution_engine,
                    tradable_symbols=tradable,
                    dry_run=args.dry_run,
                    verbose=True,  # Always verbose to see what's happening
                    update_data=True,  # Fetch latest live data each cycle
                    regenerate_features_flag=True,  # Regenerate features after updating data
                    profit_target_pct=args.profit_target_pct,
                    user_stop_loss_pct=args.stop_loss_pct,
                    excluded_symbols=excluded_symbols,  # Pass excluded symbols
                )
                
                # Update excluded symbols from cycle results (manually closed positions)
                if cycle_results.get("symbols_stopped"):
                    for symbol in cycle_results["symbols_stopped"]:
                        excluded_symbols.add(symbol)
                        print(f"\n[TRACKING] Added {symbol} to excluded list (manually closed)")
            except Exception as cycle_exc:
                print(f"\n[ERROR] Trading cycle {cycle_count} failed: {cycle_exc}")
                import traceback
                traceback.print_exc()
                print(f"[ERROR] Continuing to next cycle in {args.interval} seconds...")
                sys.stdout.flush()
                time.sleep(args.interval)
                continue
            
            # Show comprehensive cycle summary
            print(f"\n{'='*80}")
            print(f"[CYCLE {cycle_count} SUMMARY]")
            print(f"{'='*80}")
            print(f"  Symbols Processed: {cycle_results.get('symbols_processed', 0)}")
            print(f"  Symbols Traded:    {cycle_results.get('symbols_traded', 0)}")
            print(f"  Symbols Skipped:   {cycle_results.get('symbols_skipped', 0)}")
            print(f"  Errors:           {len(cycle_results.get('errors', []))}")
            
            # Show ONLY MCX active positions (filter out non-commodities)
            try:
                all_positions = angelone_client.list_positions()
                if all_positions:
                    # STRICT FILTERING: Only show MCX positions
                    mcx_positions = [pos for pos in all_positions 
                                   if pos.get("exchange_segment", pos.get("_raw_exchange", "")).upper() == "MCX"]
                    other_positions = [pos for pos in all_positions 
                                     if pos.get("exchange_segment", pos.get("_raw_exchange", "")).upper() != "MCX"]
                    
                    if mcx_positions:
                        print(f"\n  📊 ACTIVE MCX COMMODITIES POSITIONS: {len(mcx_positions)}")
                        total_unrealized_pl = 0.0
                        for pos in mcx_positions:
                            symbol = pos.get("symbol", "")
                            qty = float(pos.get("qty", 0) or 0)
                            avg_entry = float(pos.get("avg_entry_price", 0) or 0)
                            ltp = float(pos.get("ltp", 0) or 0)
                            unrealized_pl = float(pos.get("unrealized_pl", 0) or 0)
                            total_unrealized_pl += unrealized_pl
                            side_str = "LONG" if qty > 0 else "SHORT"
                            print(f"    {symbol} (MCX): {abs(qty)} {side_str} @ ₹{avg_entry:.2f} → ₹{ltp:.2f} (P/L: ₹{unrealized_pl:+.2f})")
                        print(f"  💰 Total Unrealized P/L (MCX): ₹{total_unrealized_pl:+.2f}")
                    else:
                        print(f"\n  📊 ACTIVE MCX COMMODITIES POSITIONS: 0 (flat)")
                    
                    # Report but don't show details of non-MCX positions
                    if other_positions:
                        print(f"\n  🔒 PROTECTED: {len(other_positions)} non-MCX position(s) exist but are NOT managed by this system")
                else:
                    print(f"\n  📊 ACTIVE MCX COMMODITIES POSITIONS: 0 (flat)")
            except Exception:
                pass
            
            # Show trade details
            if cycle_results.get("details"):
                traded_details = [d for d in cycle_results["details"] if d.get("status") == "traded"]
                if traded_details:
                    print(f"\n  📈 TRADES THIS CYCLE:")
                    for detail in traded_details:
                        symbol = detail.get("symbol", "?")
                        decision = detail.get("decision", "?")
                        action = (detail.get("model_action") or "").upper()
                        conf = float(detail.get("confidence") or 0.0)
                        if conf <= 1.0:
                            conf = conf * 100.0
                        print(f"    {symbol}: {action} → {decision} ({conf:.1f}% confidence)")
            
            # Show errors if any
            if cycle_results.get("errors"):
                print(f"\n  ⚠️  ERRORS:")
                for error in cycle_results["errors"][:5]:  # Show first 5 errors
                    print(f"    - {error}")
                if len(cycle_results["errors"]) > 5:
                    print(f"    ... and {len(cycle_results['errors']) - 5} more errors")
            
            print(f"{'='*80}")
            sys.stdout.flush()
            
            # Wait for next cycle
            if cycle_count < 1000:  # Prevent infinite loops in case of issues
                print(f"\n[WAIT] Waiting {args.interval} seconds before next cycle...")
                print(f"  Press Ctrl+C to stop")
                print(f"  Next cycle will be: [CYCLE {cycle_count + 1}]")
                sys.stdout.flush()
                time.sleep(args.interval)
            else:
                print("\n[WARNING] Maximum cycles reached. Stopping.")
                sys.stdout.flush()
                break
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("[STOP] Trading stopped by user")
        print("=" * 80)
        print("  Liquidating all open positions...")
        
        # Liquidate ONLY MCX commodities positions on exit (leave others untouched)
        if not args.dry_run:
            try:
                all_positions = angelone_client.list_positions()
                if all_positions:
                    # STRICT FILTERING: Only process MCX positions
                    mcx_positions = []
                    other_positions = []
                    for pos in all_positions:
                        exchange_segment = pos.get("exchange_segment", pos.get("_raw_exchange", "")).upper()
                        if exchange_segment == "MCX":
                            mcx_positions.append(pos)
                        else:
                            other_positions.append(pos)
                    
                    # Report protected positions
                    if other_positions:
                        print(f"[PROTECTED] Leaving {len(other_positions)} non-MCX position(s) untouched")
                    
                    # Only close MCX positions
                    if mcx_positions:
                        print(f"[LIQUIDATE] Closing {len(mcx_positions)} MCX commodity position(s)...")
                        for pos in mcx_positions:
                            symbol = pos.get("symbol", "")
                            qty = float(pos.get("qty", 0) or 0)
                            if abs(qty) > 0:
                                close_side = "sell" if qty > 0 else "buy"
                                try:
                                    close_resp = angelone_client.submit_order(
                                        symbol=symbol,
                                        qty=int(abs(qty)),
                                        side=close_side,
                                        order_type="market",
                                        time_in_force="gtc",
                                    )
                                    avg_entry = float(pos.get("avg_entry_price", 0) or 0)
                                    ltp = float(pos.get("ltp", pos.get("avg_entry_price", 0)) or 0)
                                    if qty > 0:
                                        realized_pl = (ltp - avg_entry) * abs(qty)
                                    else:
                                        realized_pl = (avg_entry - ltp) * abs(qty)
                                    print(f"  ✅ Closed MCX {symbol}: {abs(qty)} @ ₹{ltp:.2f} (P/L: ₹{realized_pl:+.2f})")
                                except Exception as close_exc:
                                    print(f"  ❌ Failed to close MCX {symbol}: {close_exc}")
                        print("[LIQUIDATE] All MCX commodity positions closed")
                    else:
                        print("[LIQUIDATE] No MCX commodity positions to close")
                else:
                    print("[LIQUIDATE] No open positions found")
            except Exception as liq_exc:
                print(f"[WARNING] Failed to liquidate MCX positions: {liq_exc}")
        else:
            print("[LIQUIDATE] Dry-run mode - no positions to close")
        print("=" * 80)
    except Exception as e:
        print(f"\n[ERROR] Trading cycle failed: {e}")
        import traceback
        traceback.print_exc()
        
        # EMERGENCY EXIT: Try to liquidate ONLY MCX positions on error (leave others untouched)
        # This is a safety mechanism to prevent unlimited losses if the system fails
        if not args.dry_run:
            print("\n" + "=" * 80)
            print("[EMERGENCY EXIT] System error detected - attempting to close MCX positions")
            print("=" * 80)
            print("⚠️  SAFETY: Only MCX (commodities) positions will be closed.")
            print("   All other positions (stocks, other exchanges) will be LEFT UNTOUCHED.")
            print("=" * 80)
            try:
                all_positions = angelone_client.list_positions()
                if all_positions:
                    # STRICT FILTERING: Only process MCX positions
                    mcx_positions = []
                    other_positions = []
                    for pos in all_positions:
                        exchange_seg = pos.get("exchange_segment", pos.get("_raw_exchange", "")).upper()
                        if exchange_seg == "MCX":
                            mcx_positions.append(pos)
                        else:
                            other_positions.append(pos)
                    
                    # Report protected positions
                    if other_positions:
                        print(f"\n[PROTECTED] Leaving {len(other_positions)} non-MCX position(s) untouched")
                    
                    if mcx_positions:
                        print(f"\n[EMERGENCY] Attempting to close {len(mcx_positions)} MCX commodity position(s)...")
                        closed_count = 0
                        failed_count = 0
                        for pos in mcx_positions:
                            symbol = pos.get("symbol", "")
                            qty = float(pos.get("qty", 0) or 0)
                            if abs(qty) > 0:
                                close_side = "sell" if qty > 0 else "buy"
                                try:
                                    angelone_client.submit_order(
                                        symbol=symbol,
                                        qty=int(abs(qty)),
                                        side=close_side,
                                        order_type="market",
                                        time_in_force="gtc",
                                    )
                                    print(f"  ✅ Closed MCX {symbol}: {abs(qty)} {close_side.upper()}")
                                    closed_count += 1
                                except Exception as close_exc:
                                    print(f"  ❌ Failed to close MCX {symbol}: {close_exc}")
                                    failed_count += 1
                        
                        print(f"\n[EMERGENCY] Emergency exit complete: {closed_count} closed, {failed_count} failed")
                        if failed_count > 0:
                            print(f"  ⚠️  WARNING: {failed_count} position(s) could not be closed automatically")
                            print(f"     Please close these manually to prevent further losses")
                    else:
                        print(f"\n[EMERGENCY] No MCX positions to close (other positions left untouched)")
                else:
                    print(f"\n[EMERGENCY] No open positions found")
            except Exception as emergency_exc:
                print(f"\n[EMERGENCY] Failed to execute emergency exit: {emergency_exc}")
                print(f"  ⚠️  CRITICAL: Unable to close positions automatically")
                print(f"     Please check your positions manually immediately")
            print("=" * 80)
        
        sys.exit(1)


if __name__ == "__main__":
    # Fix Windows console encoding
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    main()
