"""
End-to-end crypto pipeline:

Runs the full flow for selected crypto symbols:
1) Historical ingestion  (raw candles)
2) Feature generation    (features.json)
3) Model training        (models/crypto/...)
4) One or more live trading cycles on Alpaca paper account

This script is a single entry point so you can:
- Choose symbols yourself via CLI.
- Let it do all steps in sequence.
- See only a concise summary of what actually happened.

IMPORTANT:
- Existing open positions in Alpaca are *not* touched. Symbols that already
  have a position when this script starts are skipped by the trading engine.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

from pipeline_runner import run_ingestion, regenerate_features
from train_models import train_symbols
from trading.alpaca_client import AlpacaClient
from trading.execution_engine import ExecutionEngine, TradingRiskConfig
from trading.symbol_universe import all_enabled, find_by_data_symbol, find_by_trading_symbol
from live_trader import discover_tradable_symbols, run_trading_cycle
from ml.horizons import print_horizon_summary


def get_protected_symbols() -> Set[str]:
    """
    Read current Alpaca positions and return a set of CRYPTO trading symbols only
    that should not be modified by this script. Non-crypto positions are ignored.
    
    Gracefully handles network errors - returns empty set if Alpaca is unreachable.
    """
    from trading.symbol_universe import all_enabled
    
    # Get only crypto symbols from our universe
    crypto_universe = {asset.trading_symbol.upper() for asset in all_enabled() if asset.asset_type == "crypto"}
    
    try:
        client = AlpacaClient()
        positions = client.list_positions()
    except Exception as e:
        # Network error or API unavailable - log and continue without protection
        # This allows the pipeline to run even when Alpaca is unreachable
        import warnings
        warnings.warn(f"Could not fetch Alpaca positions (network/API error): {e}. Continuing without position protection.", UserWarning)
        return set()
    
    protected: Set[str] = set()
    for pos in positions or []:
        symbol = str(pos.get("symbol", "")).upper()
        # Only consider crypto symbols from our trading universe
        if symbol not in crypto_universe:
            continue
        qty_str = pos.get("qty", "0")
        try:
            qty = float(qty_str)
        except (TypeError, ValueError):
            qty = 0.0
        if qty != 0.0:
            protected.add(symbol)
    return protected


def filter_tradable_symbols(
    tradable: List[Dict[str, Any]],
    protected_trading_symbols: Set[str],
) -> List[Dict[str, Any]]:
    """
    Remove any symbols from the tradable list that already have an open
    position in Alpaca when this script starts.
    """
    result: List[Dict[str, Any]] = []
    for info in tradable:
        asset = info["asset"]
        trading_symbol = asset.trading_symbol.upper()
        if trading_symbol in protected_trading_symbols:
            # Skip this symbol entirely to avoid touching existing positions.
            continue
        result.append(info)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end crypto pipeline: ingest -> features -> train -> paper-trade."
    )
    parser.add_argument(
        "--crypto-symbols",
        nargs="+",
        required=True,
        help="Crypto symbols using your project convention (e.g. BTC-USDT ETH-USDT).",
    )
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
        "--horizon",
        default="short",
        help="Horizon profile for training (intraday/short/long).",
    )
    parser.add_argument(
        "--crypto-horizon",
        default=None,
        help="Alias for --horizon. Horizon profile for training (intraday/short/long).",
    )
    parser.add_argument(
        "--profit-target",
        type=float,
        required=True,
        help="REQUIRED: Profit target percentage (e.g., 10.0 for 10%%). You must specify this before trading.",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=None,
        help="Stop-loss percentage (e.g., 2.0 for 2%%). Default: 3.5%% for crypto, 2.0%% for commodities. If not specified, uses default based on asset type.",
    )
    parser.add_argument(
        "--manual-stop-loss",
        action="store_true",
        help="Enable manual stop-loss management. System will NOT submit or execute stop-loss orders automatically. You manage stop-losses yourself.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run trading logic but do not send real orders to Alpaca.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between trading cycles. Runs forever if set. Minimum: 30 seconds (to avoid rate limiting). Default: 60 seconds.",
    )
    parser.add_argument(
        "--allow-existing-positions",
        action="store_true",
        help="Allow trading even when symbols already have open positions in Alpaca. "
             "By default, symbols with existing positions are SKIPPED to avoid conflicts. "
             "With this flag: trading engine will manage existing positions (may exit them if model prediction changes). "
             "WARNING: This may close your existing positions if the model changes its prediction.",
    )

    args = parser.parse_args()
    
    # Validate interval (minimum 30 seconds to avoid API rate limiting)
    if args.interval is not None and args.interval < 30:
        print(f"⚠️  WARNING: Interval {args.interval} seconds is too short. Minimum is 30 seconds to avoid rate limiting.")
        print(f"   Setting interval to 30 seconds.")
        args.interval = 30

    # Normalize user-provided symbols so both data_symbol (BTC-USDT) and
    # trading_symbol (BTC/USD) styles are accepted on the CLI.
    raw_symbols = [s.strip().upper() for s in args.crypto_symbols if s.strip()]

    def _normalize_crypto_symbol(sym: str) -> str:
        """Return our canonical data symbol for a user-supplied crypto symbol.

        Accepted inputs:
        - Data symbol:   BTC-USDT
        - Trading symbol: BTC/USD
        - Bare symbol styles where we can infer -USDT.
        """
        # 1) Exact data_symbol match
        asset = find_by_data_symbol(sym)
        if asset:
            return asset.data_symbol.upper()

        # 2) Trading symbol match (e.g. BTC/USD)
        asset = find_by_trading_symbol(sym)
        if asset:
            return asset.data_symbol.upper()

        # 3) Simple BASE/QUOTE mapping: BTC/USD -> BTC-USDT
        if "/" in sym:
            base, quote = sym.split("/", 1)
            base = base.strip().upper()
            quote = quote.strip().upper()
            if base and quote == "USD":
                return f"{base}-USDT"

        # 4) Fallback: return as-is, uppercased. Downstream fetchers can still try.
        return sym.upper()

    crypto_symbols = [_normalize_crypto_symbol(s) for s in raw_symbols]
    timeframe = args.timeframe
    years = max(args.years, 0.5)
    # Use --crypto-horizon if provided, otherwise use --horizon
    horizon = args.crypto_horizon if args.crypto_horizon else args.horizon

    print("=" * 80)
    print("END-TO-END CRYPTO PIPELINE")
    print("=" * 80)
    print(f"Symbols:   {', '.join(crypto_symbols)}")
    print(f"Timeframe: {timeframe}")
    print(f"Horizon:   {horizon}")
    print(f"Profit Target: {args.profit_target:.2f}% (REQUIRED - user specified)")
    if args.stop_loss_pct is not None:
        print(f"Stop-Loss: {args.stop_loss_pct:.2f}% (user specified)")
    else:
        print(f"Stop-Loss: 3.5% (default for crypto)")
    if args.manual_stop_loss:
        print(f"Stop-Loss Mode: MANUAL (you manage stop-losses)")
    else:
        print(f"Stop-Loss Mode: AUTOMATIC (system manages stop-losses)")
    print(f"Mode:      {'DRY RUN (no real orders)' if args.dry_run else 'LIVE PAPER TRADING'}")
    print("=" * 80)
    print()
    
    # Show available horizons and their trading behavior
    print_horizon_summary()

    # ------------------------------------------------------------------
    # Stage 1: Historical ingestion (CRYPTO ONLY)
    # ------------------------------------------------------------------
    print("[1/4] Ingesting historical data...")
    run_ingestion(
        mode="historical",
        crypto_symbols=crypto_symbols,
        commodities_symbols=None,  # Crypto-only script - no commodities
        timeframe=timeframe,
        years=years,
    )
    print("    ✓ Historical data ingestion complete.")

    # ------------------------------------------------------------------
    # Stage 2: Feature generation (CRYPTO ONLY)
    # ------------------------------------------------------------------
    print("[2/4] Regenerating features...")
    regenerate_features("crypto", set(crypto_symbols), timeframe)
    print("    ✓ Feature generation complete.")

    # ------------------------------------------------------------------
    # Stage 3: Model training (CRYPTO ONLY)
    # ------------------------------------------------------------------
    print("[3/4] Training models...")
    horizon_map = {"crypto": horizon}  # Crypto-only - no commodities
    train_symbols(
        crypto_symbols=crypto_symbols,
        commodities_symbols=[],  # Crypto-only script - no commodities
        timeframe=timeframe,
        output_dir="models",
        horizon_profiles=horizon_map,
    )
    print("    ✓ Model training complete.")

    # ------------------------------------------------------------------
    # Stage 4: Live trading cycle(s)
    # ------------------------------------------------------------------
    print("[4/4] Preparing live trading...")

    # Discover which of the requested symbols actually have trained models.
    # CRITICAL: Pass the trained horizon as override_horizon so discover_tradable_symbols
    # finds the correct models (e.g., intraday models instead of default short models)
    all_tradable = discover_tradable_symbols(
        asset_type="crypto", 
        timeframe=timeframe,
        override_horizon=horizon  # Override asset's default horizon_profile with the trained horizon
    )
    
    # Restrict to the user-selected symbols only.
    requested_set = {s.upper() for s in crypto_symbols}
    tradable = [
        info
        for info in all_tradable
        if info["asset"].data_symbol.upper() in requested_set
    ]

    if not tradable:
        print("    ✗ No tradable symbols found after training. Exiting.")
        return

    print(f"    ✓ Found {len(tradable)} tradable symbol(s):")
    for info in tradable:
        asset = info["asset"]
        print(
            f"      - {asset.data_symbol} ({asset.trading_symbol}) "
            f"- horizon: {info['horizon']}"
        )

    # Check existing positions for informational purposes only.
    # The execution engine will intelligently manage existing positions:
    # - Add to LONG positions when prediction is LONG (especially on price drops)
    # - Exit positions when prediction changes to SHORT or HOLD
    # - Enter new positions when prediction is LONG and no position exists
    print("    Checking existing Alpaca positions (for informational purposes)...")
    existing_positions = get_protected_symbols()
    if existing_positions:
        print(f"    Found {len(existing_positions)} symbol(s) with open positions:")
        for sym in sorted(existing_positions):
            print(f"      - {sym}")
        print("    ℹ Trading engine will intelligently manage these positions:")
        print("      • LONG positions: Add more if prediction is LONG (especially on price drops)")
        print("      • Exit positions: When prediction changes to SHORT or HOLD")
        print("      • Enter new: When prediction is LONG and no position exists")
    else:
        print("    No existing positions found.")

    # Filter out symbols with existing positions unless --allow-existing-positions is set
    if args.allow_existing_positions:
        tradable_filtered = tradable
        print(f"    ⚠️  WARNING: --allow-existing-positions is enabled.")
        print(f"       The trading engine will manage existing positions (may exit them if model changes).")
    else:
        tradable_filtered = filter_tradable_symbols(tradable, existing_positions)
        if existing_positions and len(tradable_filtered) < len(tradable):
            skipped_count = len(tradable) - len(tradable_filtered)
            print(f"    ⚠️  Skipped {skipped_count} symbol(s) with existing positions (use --allow-existing-positions to trade them).")
    
    print(f"    ✓ {len(tradable_filtered)} symbol(s) ready for trading.")

    print(f"    ✓ {len(tradable_filtered)} symbol(s) eligible for trading after filtering.")

    # Initialize execution engine and position manager (will use env vars for Alpaca).
    try:
        from trading.position_manager import PositionManager
        
        risk_config = TradingRiskConfig(
            manual_stop_loss=args.manual_stop_loss,
            user_stop_loss_pct=args.stop_loss_pct,  # User override if provided
        )
        position_manager = PositionManager()
        engine = ExecutionEngine(risk_config=risk_config, position_manager=position_manager)
        if args.manual_stop_loss:
            print("    ⚠️  MANUAL STOP-LOSS MODE enabled - you are responsible for managing stop-losses")
    except Exception as exc:
        print(f"    ✗ Failed to initialize ExecutionEngine: {exc}")
        print("      Make sure ALPACA_API_KEY and ALPACA_SECRET_KEY are set.")
        return

    # Sync existing Alpaca positions with position manager (if --allow-existing-positions is set)
    if args.allow_existing_positions and existing_positions:
        print("\n[SYNC] Syncing existing Alpaca positions with position manager...")
        from end_to_end import sync_existing_alpaca_positions
        sync_results = sync_existing_alpaca_positions(
            position_manager=position_manager,
            tradable_symbols=tradable_filtered,
            profit_target_pct=args.profit_target,
            verbose=True,
        )
        if sync_results["positions_synced"] > 0:
            print(f"    ✓ Synced {sync_results['positions_synced']} existing position(s)")
            print(f"    ℹ These positions will be monitored with profit target: {args.profit_target:.2f}%")
        else:
            print("    ℹ No positions needed syncing (or all already tracked)")

    print()
    print("Starting live trading...")
    print("=" * 80)

    cycle_index = 0
    while True:
        cycle_index += 1
        now = datetime.utcnow().isoformat() + "Z"
        print(f"\n[CYCLE {cycle_index}] {now}")

        cycle_results = run_trading_cycle(
            execution_engine=engine,
            tradable_symbols=tradable_filtered,
            dry_run=args.dry_run,
            verbose=True,  # Always verbose to see what's happening
            update_data=True,  # Fetch latest live data each cycle
            regenerate_features_flag=True,  # Regenerate features with new data each cycle
            profit_target_pct=args.profit_target,  # REQUIRED - user must specify
            user_stop_loss_pct=args.stop_loss_pct,  # Optional - user override if provided
        )

        # Concise summary
        print(
            f"  Processed: {cycle_results['symbols_processed']}, "
            f"Traded: {cycle_results['symbols_traded']}, "
            f"Skipped: {cycle_results['symbols_skipped']}, "
            f"Errors: {len(cycle_results['errors'])}"
        )

        # Show per-symbol decisions that mattered.
        for detail in cycle_results.get("details", []):
            if detail.get("status") == "traded":
                symbol = detail.get("symbol", "?")
                decision = detail.get("decision", "?")
                action = (detail.get("model_action") or "").upper()
                conf = float(detail.get("confidence") or 0.0) * 100.0
                print(f"    {symbol}: {action} -> {decision} ({conf:.1f}% confidence)")

        # Wait and repeat (runs forever with interval)
        print(f"  Waiting {args.interval} seconds before next cycle...")
        print(f"  Press Ctrl+C to stop.")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping due to keyboard interrupt (Ctrl+C).")
            print("   Finalizing any open positions...")
            break


if __name__ == "__main__":
    main()


