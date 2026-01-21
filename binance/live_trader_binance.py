"""
Live trading loop for Binance: continuously run model predictions and execute trades.

This script is Binance-specific and uses the same concepts as Alpaca but with:
- BinanceClient from binance/ folder
- Logs stored in binance/logs/
- Positions stored in binance/data/positions/
- All Binance-specific files organized in binance/ folder

This script:
1. Discovers crypto symbols with trained models
2. Loads latest features for each symbol
3. Runs InferencePipeline.predict() to get consensus
4. Executes trades via ExecutionEngine with BinanceClient
5. Runs continuously with configurable interval
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress sklearn version compatibility warnings
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Trying to unpickle.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*monotonic_cst.*", category=UserWarning)

import pandas as pd

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.model_paths import horizon_dir, list_horizon_dirs
from ml.horizons import DEFAULT_HORIZON_PROFILE, normalize_profile, print_horizon_summary, get_horizon_risk_config
from ml.inference import InferencePipeline
from ml.risk import RiskManagerConfig
from trading.execution_engine import ExecutionEngine, TradingRiskConfig
from trading.symbol_universe import all_enabled, find_by_data_symbol
from trading.position_manager import PositionManager

# Import BinanceClient from binance package
from binance.binance_client import BinanceClient


# Binance-specific paths
BINANCE_ROOT = Path(__file__).parent
BINANCE_LOGS_DIR = BINANCE_ROOT / "logs"
BINANCE_POSITIONS_FILE = BINANCE_ROOT / "data" / "positions" / "active_positions.json"
BINANCE_TRADES_LOG = BINANCE_LOGS_DIR / "trades.jsonl"
BINANCE_CYCLES_LOG = BINANCE_LOGS_DIR / "cycles.jsonl"


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


def get_current_price_from_features(asset_type: str, symbol: str, timeframe: str, force_live: bool = False, verbose: bool = False) -> Optional[float]:
    """
    Resolve the latest tradable price for a symbol using Binance API.
    
    Priority:
    1. Binance REST API (primary live source)
    2. Local Binance data.json (fallback, only if force_live=False)
    """
    if asset_type == "crypto":
        # 1) Try Binance REST API first
        try:
            from fetchers import get_binance_current_price
            binance_price = get_binance_current_price(symbol)
            if binance_price and binance_price > 0:
                return float(binance_price)
        except Exception:
            pass
        
        # 2) Fallback: local data.json (only if force_live=False)
        if force_live:
            return None
        
        data_path = Path("data/json/raw") / asset_type / "binance" / symbol / timeframe / "data.json"
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
    else:
        return None


def discover_tradable_symbols(asset_type: str = "crypto", timeframe: str = "1d", override_horizon: Optional[str] = None) -> List[Dict[str, Any]]:
    """Discover symbols that have trained models available."""
    tradable = []
    
    # Get all enabled crypto symbols
    enabled_assets = all_enabled()
    crypto_assets = [a for a in enabled_assets if a.asset_type == asset_type]
    
    # Check each symbol for trained models
    for asset in crypto_assets:
        data_symbol = asset.data_symbol
        
        # Check all available horizons
        horizon_dirs = list_horizon_dirs(asset_type, data_symbol, timeframe)
        if not horizon_dirs:
            continue
        
        # Use override_horizon if provided, otherwise use asset's default
        used_horizon = override_horizon or asset.horizon_profile or DEFAULT_HORIZON_PROFILE
        used_horizon = normalize_profile(used_horizon)
        
        # Check if this horizon has models
        model_dir = horizon_dir(asset_type, data_symbol, timeframe, used_horizon)
        if not model_dir.exists():
            continue
        
        # Check if models exist
        model_files = list(model_dir.glob("*.joblib"))
        if not model_files:
            continue
        
        tradable.append({
            "asset": asset,
            "data_symbol": data_symbol,
            "timeframe": timeframe,
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
    excluded_symbols: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Run one complete trading cycle: fetch data -> regenerate features -> predict + execute for all symbols.
    
    Uses the SAME logic as live_trader.py - imports the full run_trading_cycle function
    to ensure identical behavior (mean reversion, momentum filters, position monitoring, etc.)
    """
    # Import the full run_trading_cycle from live_trader.py
    # This ensures we get ALL the logic: mean reversion, momentum filters, position monitoring, etc.
    from live_trader import run_trading_cycle as full_run_trading_cycle
    
    # Call the full implementation - it's broker-agnostic and works with BinanceClient
    return full_run_trading_cycle(
        execution_engine=execution_engine,
        tradable_symbols=tradable_symbols,
        dry_run=dry_run,
        verbose=verbose,
        update_data=update_data,
        regenerate_features_flag=regenerate_features_flag,
        profit_target_pct=profit_target_pct,
        user_stop_loss_pct=user_stop_loss_pct,
        excluded_symbols=excluded_symbols,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Live trading loop for Binance: run model predictions and execute trades on Binance."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Seconds between trading cycles. Runs forever. Minimum: 30 seconds. Default: 30 seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log intended trades without sending real orders to Binance",
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
        help="Enable manual stop-loss management.",
    )
    parser.add_argument(
        "--profit-target-pct",
        type=float,
        required=True,  # MANDATORY: User must specify profit target
        help="REQUIRED: Profit target percentage (e.g., 1.5 for 1.5%%). You must specify this before trading.",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=None,
        help="Stop-loss percentage (e.g., 8.0 for 8%%)",
    )
    args = parser.parse_args()
    
    # CRITICAL: Validate profit target is provided (should be required, but double-check)
    if args.profit_target_pct is None:
        print("ERROR: --profit-target-pct is REQUIRED. You must specify a profit target before trading.")
        print("Example: python binance/live_trader_binance.py --profit-target-pct 1.5 --stop-loss-pct 8.0")
        sys.exit(1)
    
    if args.profit_target_pct <= 0:
        print("ERROR: Profit target must be greater than 0. Example: --profit-target-pct 1.5 for 1.5%")
        sys.exit(1)
    
    # Set default stop loss if not provided (8% for crypto)
    if args.stop_loss_pct is None:
        args.stop_loss_pct = 8.0
        print(f"[INFO] Using default stop-loss: {args.stop_loss_pct}% (crypto default)")
    
    if args.stop_loss_pct <= 0:
        print("ERROR: Stop-loss must be greater than 0. Example: --stop-loss-pct 8.0 for 8%")
        sys.exit(1)
    
    # Validate interval
    if args.interval < 30:
        print(f"WARNING: Interval {args.interval} seconds is too short. Setting to 30 seconds.")
        args.interval = 30
    
    print("=" * 80)
    print("BINANCE LIVE TRADING LOOP - CRYPTO ONLY")
    print("=" * 80)
    print(f"Mode: {'DRY RUN (no real orders)' if args.dry_run else 'LIVE TRADING'}")
    print(f"Broker: BINANCE")
    print(f"Profit Target: {args.profit_target_pct:.2f}% (REQUIRED - user specified)")
    print(f"Stop-Loss: {args.stop_loss_pct:.2f}% ({'user specified' if args.stop_loss_pct != 8.0 else 'default'})")
    print(f"Interval: {args.interval} seconds (runs forever)")
    print(f"Timeframe: {args.timeframe}")
    print(f"Logs: {BINANCE_TRADES_LOG}")
    print(f"Positions: {BINANCE_POSITIONS_FILE}")
    if args.max_cycles:
        print(f"Max Cycles: {args.max_cycles}")
    if args.manual_stop_loss:
        print(f"Stop-Loss: MANUAL MODE")
    else:
        print(f"Stop-Loss: AUTOMATIC")
    print("=" * 80)
    print()
    print("[INFO] System will:")
    print(f"  - Monitor ALL positions for profit target ({args.profit_target_pct:.2f}%)")
    print(f"  - Sell automatically when profit target is hit")
    print(f"  - Log all buy and sell trades to {BINANCE_TRADES_LOG}")
    print(f"  - Keep running continuously (does not stop)")
    print(f"  - Use same profit target ({args.profit_target_pct:.2f}%) throughout")
    print()
    print("[STRATEGY] Buy Low, Sell High, Minimize Losses:")
    print("  ✅ Momentum filters: Block buying during upswings (buy low)")
    print("  ✅ RSI filters: Only buy when oversold, only short when overbought (buy low, sell high)")
    print("  ✅ Trailing stop: Sell when price drops 2.5% from peak (sell at peak)")
    print(f"  ✅ Stop-loss: {args.stop_loss_pct:.1f}% protection (minimize losses)")
    print("  ✅ Negative prediction filter: Block entries when predicted return < 0 (minimize losses)")
    print("  ✅ Minimum return filter: Require 1.5% minimum predicted return (minimize losses)")
    print()
    
    # Show available horizons
    print_horizon_summary()
    
    # Discover tradable symbols (use same function as live_trader.py)
    print("[DISCOVER] Finding symbols with trained models...")
    from live_trader import discover_tradable_symbols as discover_tradable
    tradable = discover_tradable(asset_type="crypto", timeframe=args.timeframe)
    
    if not tradable:
        print("[ERROR] No tradable symbols found. Train models first:")
        print("  python train_models.py --crypto BTC-USDT ETH-USDT SOL-USDT --timeframe 1d")
        return
    
    print(f"[DISCOVER] Found {len(tradable)} tradable symbol(s):")
    for info in tradable:
        print(f"  - {info['asset'].data_symbol} ({info['asset'].trading_symbol}) - horizon: {info['horizon']}")
    print()
    
    # Initialize Binance client and execution engine
    try:
        binance_client = BinanceClient()
        
        # Create Binance-specific position manager
        position_manager = PositionManager(positions_file=BINANCE_POSITIONS_FILE)
        
        risk_config = TradingRiskConfig(
            manual_stop_loss=args.manual_stop_loss,
            profit_target_pct=args.profit_target_pct,
            user_stop_loss_pct=args.stop_loss_pct,
        )
        
        execution_engine = ExecutionEngine(
            client=binance_client,
            risk_config=risk_config,
            log_path=BINANCE_TRADES_LOG,
            position_manager=position_manager,
        )
        
        print("[INIT] Execution engine ready with Binance")
    except Exception as exc:
        print(f"[ERROR] Failed to initialize execution engine: {exc}")
        print("Make sure BINANCE_API_KEY and BINANCE_SECRET_KEY are set in environment or .env file")
        return
    
    # Run trading loop
    cycle_count = 0
    symbol_failures = {}
    MAX_CONSECUTIVE_FAILURES = 3
    
    try:
        while True:
            cycle_count += 1
            print(f"\n[CYCLE {cycle_count}] Starting trading cycle at {datetime.utcnow().isoformat()}")
            
            # Filter out symbols with too many consecutive failures
            symbols_to_skip = {sym for sym, count in symbol_failures.items() if count >= MAX_CONSECUTIVE_FAILURES}
            if symbols_to_skip:
                print(f"[SKIP] Excluding {len(symbols_to_skip)} symbol(s) due to consecutive failures")
            
            cycle_results = run_trading_cycle(
                execution_engine=execution_engine,
                tradable_symbols=tradable,
                dry_run=args.dry_run,
                verbose=args.verbose,
                excluded_symbols=symbols_to_skip,
                profit_target_pct=args.profit_target_pct,
                user_stop_loss_pct=args.stop_loss_pct,
            )
            
            print(f"\n[CYCLE {cycle_count}] Complete:")
            print(f"  Monitored: {cycle_results['symbols_processed']} symbols")
            print(f"  Traded: {cycle_results['symbols_traded']} symbols")
            print(f"  Skipped: {cycle_results['symbols_skipped']} symbols")
            
            # Log cycle results
            BINANCE_CYCLES_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(BINANCE_CYCLES_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(cycle_results) + "\n")
            
            if args.once:
                break
            
            if args.max_cycles and cycle_count >= args.max_cycles:
                print(f"\n[COMPLETE] Reached max cycles ({args.max_cycles})")
                break
            
            # Wait before next cycle
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n[STOPPED] Trading loop interrupted by user")
    except Exception as exc:
        print(f"\n[ERROR] Trading loop failed: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
