"""
End-to-end Binance pipeline:

Runs the full flow for selected crypto symbols:
1) Historical ingestion  (raw candles from Binance)
2) Feature generation    (features.json)
3) Model training        (models/crypto/...)
4) Live trading cycles on Binance with mean reversion (buy low, sell high)

This script is a single entry point so you can:
- Choose symbols yourself via CLI
- Let it do all steps in sequence
- Set profit target percentage (MANDATORY)
- Monitor positions for profit targets and stop-loss
- See only a concise summary of what actually happened

IMPORTANT:
- Uses Binance API for all data and trading
- Mean reversion strategy: buy low, sell at peak
- All logs stored in binance/logs/
- All positions stored in binance/data/positions/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

# Suppress sklearn version compatibility warnings
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Trying to unpickle.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*monotonic_cst.*", category=UserWarning)

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline_runner import run_ingestion, regenerate_features
from train_models import train_symbols
from trading.execution_engine import ExecutionEngine, TradingRiskConfig
from trading.symbol_universe import all_enabled, find_by_data_symbol, find_by_trading_symbol
from trading.position_manager import PositionManager
from ml.horizons import print_horizon_summary

# Import Binance-specific components
from binance.binance_client import BinanceClient

# Binance-specific paths
BINANCE_ROOT = Path(__file__).parent
BINANCE_LOGS_DIR = BINANCE_ROOT / "logs"
BINANCE_POSITIONS_FILE = BINANCE_ROOT / "data" / "positions" / "active_positions.json"
BINANCE_TRADES_LOG = BINANCE_LOGS_DIR / "trades.jsonl"


def get_protected_symbols() -> Set[str]:
    """
    Read current Binance positions and return a set of trading symbols
    that should not be modified by this script.
    
    Gracefully handles network errors - returns empty set if Binance is unreachable.
    """
    from trading.symbol_universe import all_enabled
    
    # Get only crypto symbols from our universe
    crypto_universe = {asset.trading_symbol.upper() for asset in all_enabled() if asset.asset_type == "crypto"}
    
    try:
        client = BinanceClient()
        positions = client.list_positions()
    except Exception as e:
        # Network error or API unavailable - log and continue without protection
        import warnings
        warnings.warn(f"Could not fetch Binance positions (network/API error): {e}. Continuing without position protection.", UserWarning)
        return set()
    
    protected: Set[str] = set()
    for pos in positions or []:
        symbol = str(pos.get("symbol", "")).upper()
        # Only consider crypto symbols from our trading universe
        if symbol not in crypto_universe:
            continue
        qty = float(pos.get("qty", 0) or 0)
        if qty != 0.0:
            protected.add(symbol)
    return protected


def filter_tradable_symbols(
    tradable: List[Dict[str, Any]],
    protected_trading_symbols: Set[str],
) -> List[Dict[str, Any]]:
    """
    Remove any symbols from the tradable list that already have an open
    position in Binance when this script starts.
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


def discover_tradable_symbols(asset_type: str = "crypto", timeframe: str = "1d", override_horizon: str = None) -> List[Dict[str, Any]]:
    """Discover symbols that have trained models available."""
    from core.model_paths import horizon_dir, list_horizon_dirs
    from ml.horizons import DEFAULT_HORIZON_PROFILE, normalize_profile
    
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


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end Binance pipeline: ingest -> features -> train -> paper-trade."
    )
    parser.add_argument(
        "--crypto-symbols",
        nargs="+",
        default=None,
        help="Crypto symbols using your project convention (e.g. BTC-USDT ETH-USDT). If not provided, auto-discovers all enabled crypto symbols.",
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
        help="REQUIRED: Profit target percentage (e.g., 5.0 for 5%%). You must specify this before trading.",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=None,
        help="Stop-loss percentage (e.g., 8.0 for 8%%). Default: 8.0%% for crypto.",
    )
    parser.add_argument(
        "--manual-stop-loss",
        action="store_true",
        help="Enable manual stop-loss management.",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip historical data ingestion (use existing data).",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature generation (use existing features).",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training (use existing models).",
    )
    parser.add_argument(
        "--skip-ranking",
        action="store_true",
        help="Skip symbol ranking (trade all symbols with models).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log intended trades without sending real orders to Binance",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Number of trading cycles to run (default: infinite - runs until stopped).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Seconds between trading cycles. Minimum: 30 seconds. Default: 30 seconds.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress (default: True).",
    )
    args = parser.parse_args()
    
    # Validate interval (minimum 30 seconds to avoid API rate limiting)
    if args.interval < 30:
        print(f"⚠️  WARNING: Interval {args.interval} seconds is too short. Minimum is 30 seconds to avoid rate limiting.")
        print(f"   Setting interval to 30 seconds.")
        args.interval = 30
    
    print("=" * 80)
    print("BINANCE END-TO-END PIPELINE")
    print("=" * 80)
    print()
    
    # Normalize symbols
    if args.crypto_symbols:
        raw_symbols = [s.strip().upper() for s in args.crypto_symbols if s.strip()]
        crypto_symbols = []
        for sym in raw_symbols:
            # Try data symbol first
            asset = find_by_data_symbol(sym)
            if asset and asset.asset_type == "crypto":
                crypto_symbols.append(asset.data_symbol.upper())
                continue
            # Try trading symbol
            asset = find_by_trading_symbol(sym)
            if asset and asset.asset_type == "crypto":
                crypto_symbols.append(asset.data_symbol.upper())
                continue
            # Try adding -USDT if missing
            if "-" not in sym and "USDT" not in sym:
                test_sym = f"{sym}-USDT"
                asset = find_by_data_symbol(test_sym)
                if asset and asset.asset_type == "crypto":
                    crypto_symbols.append(asset.data_symbol.upper())
                    continue
            # If still not found, skip with warning
            print(f"[WARN] Symbol {sym} not found in universe, skipping")
    else:
        # Auto-discover ALL enabled crypto symbols (not just 20-25)
        print("\n[AUTO-DISCOVERY] No symbols specified, discovering ALL enabled crypto symbols...")
        all_crypto_assets = [asset for asset in all_enabled() if asset.asset_type == "crypto"]
        if not all_crypto_assets:
            print("ERROR: No enabled crypto symbols found in symbol_universe.py")
            sys.exit(1)
        crypto_symbols = [asset.data_symbol.upper() for asset in all_crypto_assets]
        print(f"[AUTO-DISCOVERY] Found {len(crypto_symbols)} enabled crypto symbols")
        print(f"[INFO] Processing ALL {len(crypto_symbols)} symbols for predictions and trading")
    
    if not crypto_symbols:
        print("ERROR: No valid crypto symbols found.")
        sys.exit(1)
    
    timeframe = args.timeframe
    years = max(args.years, 0.5)
    horizon = args.crypto_horizon if args.crypto_horizon else args.horizon
    
    # Set default stop loss if not provided
    if args.stop_loss_pct is None:
        args.stop_loss_pct = 8.0
    
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Symbols:   {', '.join(crypto_symbols[:10])}{'...' if len(crypto_symbols) > 10 else ''}")
    print(f"Timeframe: {timeframe}")
    print(f"Horizon:   {horizon}")
    print(f"Profit Target: {args.profit_target:.2f}% (REQUIRED - user specified)")
    print(f"Stop-Loss: {args.stop_loss_pct:.2f}% ({'user specified' if args.stop_loss_pct != 8.0 else 'default'})")
    print(f"Broker: BINANCE")
    print(f"Mode: {'Continuous (runs until stopped)' if args.cycles is None else f'{args.cycles} cycle(s)'}")
    print(f"Interval: {args.interval} seconds between cycles")
    print(f"Logs: {BINANCE_TRADES_LOG}")
    print(f"Positions: {BINANCE_POSITIONS_FILE}")
    print("=" * 80)
    print()
    print("[INFO] System will:")
    print(f"  - Run data ingestion ONCE at startup (smart: skip if data is recent)")
    print(f"  - Generate features ONCE at startup (smart: skip if up-to-date)")
    print(f"  - Train models ONCE at startup (smart: skip if already trained)")
    print(f"  - Then monitor ALL positions continuously every {args.interval} seconds")
    print(f"  - Sell automatically when profit target ({args.profit_target:.2f}%) is hit")
    print(f"  - Stop-loss protection at {args.stop_loss_pct:.2f}%")
    print(f"  - Log all buy and sell trades")
    print(f"  - Keep running until you stop it (Ctrl+C)")
    print()
    print("[STRATEGY] Buy Low, Sell High, Minimize Losses:")
    print("  ✅ Momentum filters: Block buying during upswings (buy low)")
    print("  ✅ RSI filters: Only buy when oversold (RSI < 30), only short when overbought (RSI > 70)")
    print("  ✅ Mean reversion: Flips SHORT to LONG when oversold (buy low)")
    print("  ✅ Trailing stop: Sell when price drops 2.5% from peak (sell at peak)")
    print(f"  ✅ Stop-loss: {args.stop_loss_pct:.1f}% protection (minimize losses)")
    print("  ✅ Negative prediction filter: Block entries when predicted return < 0 (minimize losses)")
    print("  ✅ Minimum return filter: Require 1.5% minimum predicted return (minimize losses)")
    print("  ✅ Volatility-based sizing: Reduce position size in high volatility (minimize losses)")
    print()
    
    # Show horizon summary
    print_horizon_summary()
    print()
    
    # ------------------------------------------------------------------
    # Stage 1: Historical data ingestion (SMART: Check existing data first)
    # ------------------------------------------------------------------
    if not args.skip_ingestion:
        print("[1/4] Checking existing data and ingesting from Binance...")
        
        # Check which symbols have existing data
        from pathlib import Path
        from fetchers import get_last_timestamp_from_existing_data, load_json_file
        
        symbols_needing_full_fetch = []
        symbols_needing_incremental = []
        symbols_with_data = []
        
        for symbol in crypto_symbols:
            data_path = Path("data/json/raw/crypto/binance") / symbol / timeframe / "data.json"
            if data_path.exists():
                # Check if data is recent (within last 24 hours for daily timeframe)
                try:
                    existing_candles = load_json_file(data_path)
                    if existing_candles:
                        last_timestamp = get_last_timestamp_from_existing_data("crypto", symbol, timeframe, "binance")
                        if last_timestamp:
                            from datetime import datetime, timezone, timedelta
                            age_hours = (datetime.now(timezone.utc) - last_timestamp).total_seconds() / 3600
                            if age_hours < 24:
                                symbols_with_data.append(symbol)
                                print(f"  ✓ {symbol}: Data exists and is recent ({age_hours:.1f} hours old) - skipping fetch")
                            else:
                                symbols_needing_incremental.append(symbol)
                                print(f"  ⚠️  {symbol}: Data exists but outdated ({age_hours:.1f} hours old) - will fetch incremental")
                        else:
                            symbols_needing_incremental.append(symbol)
                            print(f"  ⚠️  {symbol}: Data exists but timestamp unclear - will fetch incremental")
                    else:
                        symbols_needing_full_fetch.append(symbol)
                        print(f"  ⚠️  {symbol}: Data file exists but empty - will fetch full historical")
                except Exception as e:
                    symbols_needing_incremental.append(symbol)
                    print(f"  ⚠️  {symbol}: Error checking data ({e}) - will fetch incremental")
            else:
                symbols_needing_full_fetch.append(symbol)
                print(f"  ⚠️  {symbol}: No data found - will fetch full historical")
        
        # Only fetch for symbols that need it
        if symbols_needing_full_fetch or symbols_needing_incremental:
            print(f"\n  Fetching data for {len(symbols_needing_full_fetch)} new + {len(symbols_needing_incremental)} incremental symbol(s)...")
            run_ingestion(
                mode="historical",
                crypto_symbols=crypto_symbols,  # pipeline_runner handles incremental automatically
                commodities_symbols=None,
                timeframe=timeframe,
                years=years,
            )
            print("    ✓ Historical data ingestion complete.")
        else:
            print(f"\n  ✓ All {len(symbols_with_data)} symbols have recent data - skipping fetch")
    else:
        print("[1/4] SKIPPED: Historical data ingestion")
    
    # ------------------------------------------------------------------
    # Stage 2: Feature generation (SMART: Only regenerate if data is new or features missing)
    # ------------------------------------------------------------------
    if not args.skip_features:
        print("[2/4] Checking features and regenerating if needed...")
        
        from pathlib import Path
        
        symbols_needing_features = []
        symbols_with_features = []
        
        for symbol in crypto_symbols:
            feature_path = Path("data/features/crypto") / symbol / timeframe / "features.json"
            data_path = Path("data/json/raw/crypto/binance") / symbol / timeframe / "data.json"
            
            # Check if features exist and are newer than data
            if feature_path.exists() and data_path.exists():
                try:
                    feature_mtime = feature_path.stat().st_mtime
                    data_mtime = data_path.stat().st_mtime
                    # If features are newer than data, skip regeneration
                    if feature_mtime >= data_mtime:
                        symbols_with_features.append(symbol)
                        print(f"  ✓ {symbol}: Features exist and are up-to-date - skipping")
                    else:
                        symbols_needing_features.append(symbol)
                        print(f"  ⚠️  {symbol}: Features outdated (data newer) - will regenerate")
                except Exception:
                    symbols_needing_features.append(symbol)
                    print(f"  ⚠️  {symbol}: Error checking features - will regenerate")
            else:
                symbols_needing_features.append(symbol)
                if not feature_path.exists():
                    print(f"  ⚠️  {symbol}: Features missing - will generate")
                else:
                    print(f"  ⚠️  {symbol}: Data missing - cannot generate features")
        
        # Only regenerate for symbols that need it
        if symbols_needing_features:
            print(f"\n  Regenerating features for {len(symbols_needing_features)} symbol(s)...")
            regenerate_features("crypto", set(symbols_needing_features), timeframe)
            print("    ✓ Feature generation complete.")
        else:
            print(f"\n  ✓ All {len(symbols_with_features)} symbols have up-to-date features - skipping regeneration")
    else:
        print("[2/4] SKIPPED: Feature generation")
    
    # ------------------------------------------------------------------
    # Stage 3: Model training (SMART: Only train if features are new or models missing)
    # ------------------------------------------------------------------
    if not args.skip_training:
        print("[3/4] Checking models and training if needed...")
        
        from pathlib import Path
        from core.model_paths import horizon_dir, summary_path
        from ml.horizons import normalize_profile
        
        # Normalize horizon profile name (e.g., "intraday" -> "intraday")
        normalized_horizon = normalize_profile(horizon)
        
        symbols_needing_training = []
        symbols_with_models = []
        
        for symbol in crypto_symbols:
            # Use the proper model path function that handles normalization
            model_dir = horizon_dir("crypto", symbol, timeframe, normalized_horizon)
            summary_file = summary_path("crypto", symbol, timeframe, normalized_horizon)
            feature_path = Path("data/features/crypto") / symbol / timeframe / "features.json"
            
            # Check if model exists and is newer than features
            if summary_file.exists() and feature_path.exists():
                try:
                    model_mtime = summary_file.stat().st_mtime
                    feature_mtime = feature_path.stat().st_mtime
                    # If model is newer than features, skip training
                    if model_mtime >= feature_mtime:
                        symbols_with_models.append(symbol)
                        print(f"  ✓ {symbol}: Model exists and is up-to-date - skipping training")
                    else:
                        symbols_needing_training.append(symbol)
                        print(f"  ⚠️  {symbol}: Model outdated (features newer) - will retrain")
                except Exception:
                    symbols_needing_training.append(symbol)
                    print(f"  ⚠️  {symbol}: Error checking model - will train")
            else:
                symbols_needing_training.append(symbol)
                if not summary_file.exists():
                    print(f"  ⚠️  {symbol}: Model missing - will train")
                else:
                    print(f"  ⚠️  {symbol}: Features missing - cannot train")
        
        # Only train for symbols that need it
        if symbols_needing_training:
            print(f"\n  Training models for {len(symbols_needing_training)} symbol(s)...")
            print("    NOTE: Each model training starts with 'Trial 0' - this is normal.")
            print("    NOTE: Training logs will be saved to data/logs/training/ (outside binance folder)")
            print()
            
            # Track timing for progress display
            from datetime import datetime, timedelta
            import time
            training_start_time = time.time()
            total_symbols = len(symbols_needing_training)
            
            horizon_map = {symbol: horizon for symbol in symbols_needing_training}
            
            # Override training logger to use data/logs/training/ instead of logs/training/
            # This ensures training logs go to data folder (outside binance) as requested
            import ml.json_logger
            from pathlib import Path
            original_get_training_logger = ml.json_logger.get_training_logger
            
            def custom_get_training_logger(asset_type: str, symbol: str, timeframe: str, base_dir: Path = None):
                """Override to use data/logs/training/ instead of logs/training/"""
                if base_dir is None:
                    base_dir = Path("data/logs")  # Use data/logs/ instead of logs/
                return original_get_training_logger(asset_type, symbol, timeframe, base_dir)
            
            # Temporarily override the logger (train_symbols will use it internally)
            ml.json_logger.get_training_logger = custom_get_training_logger
            
            # Wrap train_symbols to add progress tracking
            import train_models
            from core.model_paths import summary_path
            from ml.horizons import normalize_profile
            
            normalized_horizon = normalize_profile(horizon)
            symbols_trained = []
            
            # Store original train_for_symbol to intercept calls
            original_train_for_symbol = train_models.train_for_symbol
            
            def tracked_train_for_symbol(asset_type, symbol, timeframe, output_dir, **kwargs):
                """Track each symbol training and show progress."""
                horizon_profile = kwargs.get('horizon_profile', 'unknown')
                symbol_start = time.time()
                result = original_train_for_symbol(asset_type, symbol, timeframe, output_dir, **kwargs)
                symbol_end = time.time()
                symbol_duration = symbol_end - symbol_start
                
                symbols_trained.append({
                    'symbol': symbol,
                    'horizon': horizon_profile,
                    'duration': symbol_duration
                })
                
                # Calculate progress based on unique symbols completed
                unique_symbols_completed = len(set(s['symbol'] for s in symbols_trained))
                total_training_ops = len(symbols_trained)
                elapsed = time.time() - training_start_time
                
                # Calculate average time per training operation (accounting for multiple horizons per symbol)
                avg_time_per_op = elapsed / total_training_ops if total_training_ops > 0 else 0
                
                # Estimate remaining: assume each remaining symbol will train same number of horizons as average
                # But simpler: just track by symbol count if we know each symbol trains once
                # For now, show both unique symbols and total operations
                avg_ops_per_symbol = total_training_ops / unique_symbols_completed if unique_symbols_completed > 0 else 1
                remaining_symbols = total_symbols - unique_symbols_completed
                estimated_remaining_ops = remaining_symbols * avg_ops_per_symbol
                estimated_remaining_time = avg_time_per_op * estimated_remaining_ops
                total_estimated_time = avg_time_per_op * (total_symbols * avg_ops_per_symbol)
                
                # Format times
                def format_time(seconds):
                    """Format seconds as readable time string."""
                    if seconds < 60:
                        return f"{int(seconds)}s"
                    elif seconds < 3600:
                        mins, secs = divmod(int(seconds), 60)
                        return f"{mins}m {secs}s"
                    else:
                        hours, remainder = divmod(int(seconds), 3600)
                        mins, secs = divmod(remainder, 60)
                        return f"{hours}h {mins}m {secs}s"
                
                elapsed_str = format_time(elapsed)
                remaining_str = format_time(estimated_remaining_time)
                total_str = format_time(total_estimated_time)
                symbol_time_str = format_time(symbol_duration)
                avg_time_str = format_time(avg_time_per_op)
                
                print()
                print("=" * 80)
                print(f"PROGRESS: {unique_symbols_completed}/{total_symbols} symbols completed ({unique_symbols_completed/total_symbols*100:.1f}%)")
                print(f"  Training operations: {total_training_ops}")
                print(f"  Elapsed time: {elapsed_str}")
                print(f"  Estimated remaining: {remaining_str}")
                print(f"  Estimated total time: {total_str}")
                print(f"  Last completed: {symbol} ({horizon_profile}) - {symbol_time_str}")
                print(f"  Average time per operation: {avg_time_str}")
                print("=" * 80)
                print()
                
                return result
            
            # Temporarily replace train_for_symbol
            train_models.train_for_symbol = tracked_train_for_symbol
            try:
                train_symbols(
                    crypto_symbols=symbols_needing_training,
                    commodities_symbols=None,
                    timeframe=timeframe,
                    output_dir="models",
                    horizon_profiles=horizon_map,
                )
            finally:
                # Restore original train_for_symbol
                train_models.train_for_symbol = original_train_for_symbol
                # Restore original logger
                ml.json_logger.get_training_logger = original_get_training_logger
            
            # Final summary
            total_duration = time.time() - training_start_time
            
            def format_time(seconds):
                """Format seconds as HH:MM:SS."""
                hours, remainder = divmod(int(seconds), 3600)
                minutes, secs = divmod(remainder, 60)
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"
            
            total_duration_str = format_time(total_duration)
            avg_time_str = format_time(total_duration / total_symbols) if total_symbols > 0 else "N/A"
            
            print()
            print("=" * 80)
            print("TRAINING COMPLETE")
            print("=" * 80)
            print(f"  Total symbols trained: {total_symbols}")
            print(f"  Total time: {total_duration_str}")
            print(f"  Average time per symbol: {avg_time_str}")
            print("=" * 80)
            print("    ✓ Model training complete.")
        else:
            print(f"\n  ✓ All {len(symbols_with_models)} symbols have up-to-date models - skipping training")
    else:
        print("[3/4] SKIPPED: Model training")
    
    # ------------------------------------------------------------------
    # Stage 4: Live trading
    # ------------------------------------------------------------------
    print("[4/4] Starting live trading on Binance...")
    print()
    
    # Discover ALL tradable symbols (those with trained models)
    all_tradable = discover_tradable_symbols(
        asset_type="crypto",
        timeframe=timeframe,
        override_horizon=horizon
    )
    
    # Restrict to user-selected symbols (if provided)
    # If no symbols specified, use ALL tradable symbols
    if crypto_symbols:
        requested_set = {s.upper() for s in crypto_symbols}
        tradable = [info for info in all_tradable if info["asset"].data_symbol.upper() in requested_set]
    else:
        # Use ALL tradable symbols (no restriction)
        tradable = all_tradable
    
    if not tradable:
        print("ERROR: No tradable symbols found with trained models.")
        print("   Make sure models were trained successfully.")
        sys.exit(1)
    
    print(f"[TRADING] Found {len(tradable)} tradable symbol(s) with trained models:")
    if len(tradable) <= 20:
        for info in tradable:
            print(f"  - {info['asset'].data_symbol} ({info['asset'].trading_symbol}) - horizon: {info['horizon']}")
    else:
        for info in tradable[:10]:
            print(f"  - {info['asset'].data_symbol} ({info['asset'].trading_symbol}) - horizon: {info['horizon']}")
        print(f"  ... and {len(tradable) - 10} more symbols")
    print()
    print(f"[INFO] ALL {len(tradable)} symbols will be processed for predictions and trading")
    print()
    
    # Get protected symbols (existing positions)
    protected_symbols = get_protected_symbols()
    if protected_symbols:
        print(f"[PROTECT] Protecting {len(protected_symbols)} existing position(s): {', '.join(sorted(protected_symbols))}")
        tradable = filter_tradable_symbols(tradable, protected_symbols)
        print(f"[TRADING] After protection: {len(tradable)} tradable symbol(s)")
        print()
    
    # Initialize Binance client and execution engine
    try:
        binance_client = BinanceClient()
        position_manager = PositionManager(positions_file=BINANCE_POSITIONS_FILE)
        
        risk_config = TradingRiskConfig(
            manual_stop_loss=args.manual_stop_loss,
            profit_target_pct=args.profit_target,
            user_stop_loss_pct=args.stop_loss_pct,
        )
        
        execution_engine = ExecutionEngine(
            client=binance_client,
            risk_config=risk_config,
            log_path=BINANCE_TRADES_LOG,
            position_manager=position_manager,
        )
        
        print("[INIT] Execution engine ready with Binance")
        print(f"[INIT] Mean reversion: ENABLED (buy low, sell at peak)")
        print()
    except Exception as exc:
        print(f"ERROR: Failed to initialize execution engine: {exc}")
        print("Make sure BINANCE_API_KEY and BINANCE_SECRET_KEY are set in .env file")
        sys.exit(1)
    
    # Import run_trading_cycle from live_trader (full implementation)
    from live_trader import run_trading_cycle
    
    # Run trading cycles
    cycle_num = 0
    try:
        while True:
            cycle_num += 1
            
            # Check if we've reached max cycles (if specified)
            if args.cycles is not None and cycle_num > args.cycles:
                print(f"\n{'=' * 80}")
                print(f"REACHED MAX CYCLES ({args.cycles})")
                print(f"{'=' * 80}")
                break
            
            print(f"\n{'=' * 80}")
            print(f"CYCLE {cycle_num}{f'/{args.cycles}' if args.cycles else ''}")
            print(f"{'=' * 80}")
            print()
            
            # Process ALL symbols for predictions (not just tradable ones)
            # This ensures we monitor all symbols and trade any that meet criteria
            cycle_results = run_trading_cycle(
                execution_engine=execution_engine,
                tradable_symbols=tradable,
                dry_run=args.dry_run,
                verbose=args.verbose,
                update_data=True,
                regenerate_features_flag=True,
                profit_target_pct=args.profit_target,
                user_stop_loss_pct=args.stop_loss_pct,
                all_symbols_for_predictions=tradable,  # Process ALL symbols for predictions
            )
            
            print(f"\n[CYCLE {cycle_num}] Summary:")
            print(f"  Monitored: {cycle_results['symbols_processed']} symbols")
            print(f"  Traded: {cycle_results['symbols_traded']} symbols")
            print(f"  Skipped: {cycle_results['symbols_skipped']} symbols")
            if cycle_results.get("errors"):
                print(f"  Errors: {len(cycle_results['errors'])}")
            
            # Log cycle results
            BINANCE_CYCLES_LOG = BINANCE_LOGS_DIR / "cycles.jsonl"
            BINANCE_CYCLES_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(BINANCE_CYCLES_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(cycle_results) + "\n")
            
            # Wait before next cycle (unless this was the last cycle)
            if args.cycles is None or cycle_num < args.cycles:
                print(f"\n[WAIT] Waiting {args.interval} seconds before next cycle...")
                time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print(f"\n\n{'=' * 80}")
        print("TRADING STOPPED BY USER (Ctrl+C)")
        print(f"{'=' * 80}")
        print(f"Completed {cycle_num} cycle(s)")
        print()

    
    print("\n" + "=" * 80)
    print("BINANCE PIPELINE COMPLETE")
    print("=" * 80)
    print()
    print(f"Logs: {BINANCE_TRADES_LOG}")
    print(f"Positions: {BINANCE_POSITIONS_FILE}")
    print()
    print("To view dashboard:")
    print("  python binance/dashboard.py")
    print()


if __name__ == "__main__":
    main()
