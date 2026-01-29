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
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline_runner import run_ingestion, regenerate_features
from train_models import train_symbols
from trading.angelone_client import AngelOneClient
from trading.paper_trading_client import PaperTradingClient, PaperTradingConfig
from trading.execution_engine import ExecutionEngine, TradingRiskConfig
from trading.position_manager import PositionManager
from trading.symbol_universe import by_asset_type, find_by_data_symbol
from live_trader import discover_tradable_symbols, run_trading_cycle, get_current_price_from_features, load_feature_row
from ml.inference import InferencePipeline
from ml.risk import RiskManagerConfig
from ml.horizons import print_horizon_summary, normalize_profile


def setup_paper_trading_client(initial_equity: float = 1000000.0) -> PaperTradingClient:
    """Setup paper trading client (local simulation, no external service needed)."""
    config = PaperTradingConfig(initial_equity=initial_equity, initial_cash=initial_equity)
    return PaperTradingClient(config=config)


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
    client,  # PaperTradingClient or AngelOneClient
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
            # For paper trading, all positions are commodities (no MCX filter needed)
            # For Angel One, filter only MCX positions
            if hasattr(client, 'broker_name') and client.broker_name == "paper_trading":
                # Paper trading - all positions are commodities
                existing_positions = all_positions
            else:
                # Angel One - filter only MCX positions
                mcx_positions = [
                    pos for pos in all_positions
                    if pos.get("exchange_segment", pos.get("_raw_exchange", "")).upper() == "MCX"
                ]
                existing_positions = mcx_positions
            
            if existing_positions:
                if verbose:
                    print(f"\n[EXISTING POSITIONS] Found {len(existing_positions)} commodity position(s):")
                    for pos in existing_positions:
                        symbol = pos.get("symbol", "")
                        qty = float(pos.get("qty", 0) or 0)
                        avg_entry = float(pos.get("avg_entry_price", 0) or 0)
                        ltp = float(pos.get("ltp", avg_entry) or avg_entry)
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
        default=None,  # REQUIRED - no default, user must specify
        help="Profit target percentage (REQUIRED). Example: --profit-target-pct 5.0 for 5%% profit target",
        required=True,  # Make it mandatory
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
        default=30,
        help="Seconds between trading cycles. Minimum: 30 seconds. Default: 30 seconds.",
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
    parser.add_argument(
        "--paper-equity",
        type=float,
        default=1000000.0,
        help="Initial equity for paper trading (default: 10,00,000 Rs.)",
    )
    parser.add_argument(
        "--cycle-delay",
        type=int,
        default=None,
        help="Alias for --interval (seconds between cycles). If set, overrides --interval.",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=5,
        help="Maximum number of concurrent positions (default: 5)",
    )
    parser.add_argument(
        "--skip-data-update",
        action="store_true",
        help="Skip fetching fresh data (use existing data)",
    )
    parser.add_argument(
        "--skip-feature-regen",
        action="store_true",
        help="Skip regenerating features (use existing features)",
    )
    
    args = parser.parse_args()
    
    # Handle cycle-delay alias
    if args.cycle_delay is not None:
        args.interval = args.cycle_delay
    
    # Extract arguments to variables
    timeframe = args.timeframe
    horizon = args.commodities_horizon
    years = args.years
    
    # Validate interval (minimum 30 seconds to avoid API rate limiting)
    if args.interval < 30:
        print(f"WARNING: Interval {args.interval} seconds is too short. Minimum is 30 seconds to avoid rate limiting.")
        print(f"   Setting interval to 30 seconds.")
        args.interval = 30
    
    # Validate profit target (REQUIRED - no default)
    if args.profit_target_pct is None or args.profit_target_pct <= 0:
        print(f"[ERROR] Profit target is REQUIRED and must be positive.")
        print(f"   You specified: {args.profit_target_pct}")
        print("   Example: --profit-target-pct 5.0 (for 5% profit target)")
        print("   Example: --profit-target-pct 1.0 (for 1% profit target)")
        sys.exit(1)
    
    if args.profit_target_pct > 1000:
        print(f"[ERROR] Profit target seems unreasonably high: {args.profit_target_pct}%")
        print("   Maximum recommended: 100.0% (100% profit target)")
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
    
    # For paper trading, Angel One credentials are NOT required
    # The script uses PaperTradingClient which simulates everything locally
    # Angel One credentials are only needed if you want to use AngelOneClient for live trading
    if not api_key or not client_id or not password:
        print("[INFO] Angel One credentials not provided - will use Paper Trading (local simulation)")
        print("[INFO] PaperTradingClient will be used (no external service needed)")
    
    # Show configuration summary
    print("\n" + "=" * 80)
    print("COMMODITIES AUTO-TRADING PIPELINE")
    print("=" * 80)
    print(f"Timeframe:     {timeframe}")
    print(f"Horizon:       {horizon}")
    print(f"Profit Target: {args.profit_target_pct:.2f}% (REQUIRED - user specified)")
    print(f"Stop-Loss:     {args.stop_loss_pct:.2f}% ({'user specified' if args.stop_loss_pct != 2.0 else 'default'})")
    print(f"Broker:        PAPER TRADING (Local Simulation)")
    print(f"Initial Equity: Rs. {args.paper_equity:,.2f}")
    print(f"Mode:          Continuous (runs until stopped)")
    print(f"Interval:      {args.interval} seconds between cycles")
    print("=" * 80)
    print()
    print("[INFO] System will:")
    print(f"  - Check data ONCE at startup (smart: skip if data is recent)")
    print(f"  - Generate features ONCE at startup (smart: skip if up-to-date)")
    print(f"  - Train models ONCE at startup (smart: skip if already trained)")
    print(f"  - Rank commodities by performance")
    print(f"  - Then monitor ALL positions continuously every {args.interval} seconds")
    print(f"  - Sell automatically when profit target ({args.profit_target_pct:.2f}%) is hit")
    print(f"  - Stop-loss protection at {args.stop_loss_pct:.2f}%")
    print(f"  - Log all buy and sell trades")
    print(f"  - Keep running until you stop it (Ctrl+C)")
    print()
    print("[STRATEGY] Buy Low, Sell High, Minimize Losses:")
    print("  ✅ Momentum filters: Block buying during upswings (buy low)")
    print("  ✅ RSI filters: Only buy when oversold (RSI < 30)")
    print("  ✅ Mean reversion: Buy at support, sell at resistance")
    print(f"  ✅ Stop-loss: {args.stop_loss_pct:.1f}% protection (minimize losses)")
    print("  ✅ Negative prediction filter: Block entries when predicted return < 0")
    print("  ✅ Minimum return filter: Require 1.5% minimum predicted return")
    print("  ✅ Volatility-based sizing: Reduce position size in high volatility")
    print()
    
    # Initialize paper trading client with specified equity
    print(f"[SETUP] Initializing Paper Trading Client with Rs. {args.paper_equity:,.2f}")
    try:
        paper_client = setup_paper_trading_client(initial_equity=args.paper_equity)
        print("[OK] Paper trading client initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Paper Trading client: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Initialize position manager early to check existing positions
    position_manager = PositionManager()
    
    # Check for existing positions FIRST
    print("\n" + "=" * 80)
    print("CHECKING EXISTING POSITIONS")
    print("=" * 80)
    existing_positions, tracked_symbols = check_existing_positions(paper_client, position_manager, verbose=True)
    
    # If existing positions found, prioritize monitoring them
    if existing_positions or tracked_symbols:
        print(f"\n[PRIORITY] Found {len(existing_positions)} broker position(s) and {len(tracked_symbols)} tracked position(s)")
        print(f"[ACTION] Will MONITOR existing positions first before considering new trades")
        print(f"[INFO] Existing positions will be monitored with profit target: {args.profit_target_pct:.2f}%")
        print(f"[INFO] Use --skip-training and --skip-ranking to start monitoring immediately")
    
    # Get all commodities
    all_commodities = get_all_commodities()
    print(f"\n[INFO] Found {len(all_commodities)} total commodities in universe")
    
    # Stage 1: Historical ingestion (SMART: Check existing data first)
    if not args.skip_training and not args.skip_data_update:
        print("\n" + "=" * 80)
        print("[1/4] CHECKING EXISTING DATA AND INGESTING IF NEEDED")
        print("=" * 80)
        
        # Check which symbols have existing data
        from fetchers import get_last_timestamp_from_existing_data, load_json_file
        
        symbols_needing_full_fetch = []
        symbols_needing_incremental = []
        symbols_with_data = []
        
        for symbol in all_commodities:
            data_path = Path("data/json/raw/commodities/yahoo") / symbol / timeframe / "data.json"
            if not data_path.exists():
                data_path = Path("data/json/raw/commodities/stooq") / symbol / timeframe / "data.json"
            
            if data_path.exists():
                # Check if data is recent (within last 24 hours for daily timeframe)
                try:
                    existing_candles = load_json_file(data_path)
                    if existing_candles:
                        last_timestamp = get_last_timestamp_from_existing_data("commodities", symbol, timeframe, "yahoo")
                        if last_timestamp:
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
                crypto_symbols=[],  # EMPTY - don't train crypto in commodities bot
                commodities_symbols=all_commodities,  # pipeline_runner handles incremental automatically
                timeframe=timeframe,
                years=years,
            )
            print("    ✓ Historical data ingestion complete.")
        else:
            print(f"\n  ✓ All {len(symbols_with_data)} symbols have recent data - skipping fetch")
    elif args.skip_data_update:
        print("\n[1/4] SKIPPED: Data update (using existing data)")
    else:
        print("\n[1/4] SKIPPED: Data ingestion (training phase skipped)")
    
    # Stage 2: Feature generation (SMART: Only regenerate if data is new or features missing)
    if not args.skip_training and not args.skip_feature_regen:
        print("\n" + "=" * 80)
        print("[2/4] CHECKING FEATURES AND REGENERATING IF NEEDED")
        print("=" * 80)
        
        symbols_needing_features = []
        symbols_with_features = []
        
        for symbol in all_commodities:
            feature_path = Path("data/features/commodities") / symbol / timeframe / "features.json"
            data_path = Path("data/json/raw/commodities/yahoo") / symbol / timeframe / "data.json"
            if not data_path.exists():
                data_path = Path("data/json/raw/commodities/stooq") / symbol / timeframe / "data.json"
            
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
            regenerate_features("commodities", set(symbols_needing_features), timeframe)
            print("    ✓ Feature generation complete.")
        else:
            print(f"\n  ✓ All {len(symbols_with_features)} symbols have up-to-date features - skipping regeneration")
    elif args.skip_feature_regen:
        print("\n[2/4] SKIPPED: Feature regeneration (using existing features)")
    else:
        print("\n[2/4] SKIPPED: Feature generation (training phase skipped)")
    
    # Stage 3: Model training (SMART: Only train if features are new or models missing)
    if not args.skip_training:
        print("\n" + "=" * 80)
        print("[3/4] CHECKING MODELS AND TRAINING IF NEEDED")
        print("=" * 80)
        
        from core.model_paths import horizon_dir, summary_path
        from ml.horizons import normalize_profile
        
        # Normalize horizon profile name
        normalized_horizon = normalize_profile(horizon)
        
        symbols_needing_training = []
        symbols_with_models = []
        
        # Required model files for a complete training
        required_model_files = [
            "random_forest.joblib",
            "lightgbm.joblib",
            "xgboost.joblib",
            "stacked_blend.joblib",
            "dqn.joblib",  # Now that TensorFlow is installed, DQN should train
            "summary.json",
            "feature_scaler.joblib"
        ]
        
        for symbol in all_commodities:
            # Use the proper model path function that handles normalization
            model_dir = horizon_dir("commodities", symbol, timeframe, normalized_horizon)
            summary_file = summary_path("commodities", symbol, timeframe, normalized_horizon)
            feature_path = Path("data/features/commodities") / symbol / timeframe / "features.json"
            
            # Check if ALL required model files exist
            all_models_exist = True
            missing_files = []
            
            if model_dir.exists():
                for required_file in required_model_files:
                    file_path = model_dir / required_file
                    if not file_path.exists():
                        all_models_exist = False
                        missing_files.append(required_file)
            else:
                all_models_exist = False
                missing_files = required_model_files
            
            # If all models exist, check if features are newer (need retrain)
            if all_models_exist and summary_file.exists() and feature_path.exists():
                try:
                    model_mtime = summary_file.stat().st_mtime
                    feature_mtime = feature_path.stat().st_mtime
                    # If model is newer than features, skip training
                    if model_mtime >= feature_mtime:
                        symbols_with_models.append(symbol)
                        print(f"  ✓ {symbol}: All models exist and are up-to-date - skipping training")
                    else:
                        symbols_needing_training.append(symbol)
                        print(f"  ⚠️  {symbol}: Models outdated (features newer) - will retrain")
                except Exception:
                    symbols_needing_training.append(symbol)
                    print(f"  ⚠️  {symbol}: Error checking model timestamps - will train")
            else:
                symbols_needing_training.append(symbol)
                if not all_models_exist:
                    if len(missing_files) <= 3:
                        print(f"  ⚠️  {symbol}: Missing models: {', '.join(missing_files)} - will train")
                    else:
                        print(f"  ⚠️  {symbol}: Missing {len(missing_files)} model files - will train")
                elif not feature_path.exists():
                    print(f"  ⚠️  {symbol}: Features missing - cannot train")
        
        # Only train for symbols that need it
        if symbols_needing_training:
            print(f"\n  Training models for {len(symbols_needing_training)} symbol(s)...")
            print("    NOTE: Each model training starts with 'Trial 0' - this is normal.")
            print()
            
            horizon_map = {"commodities": horizon}
            try:
                train_symbols(
                    crypto_symbols=[],
                    commodities_symbols=symbols_needing_training,
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
            print(f"\n  ✓ All {len(symbols_with_models)} symbols have up-to-date models - skipping training")
    else:
        print("\n[3/4] SKIPPED: Training phase (using existing models)")
    
    # Stage 4: Rank commodities
    if not args.skip_ranking:
        print("\n" + "=" * 80)
        print("[4/4] RANKING COMMODITIES BY PERFORMANCE")
        print("=" * 80)
        ranked = rank_commodities_by_performance(all_commodities, timeframe, horizon, verbose=True)
        
        if not ranked:
            print("[ERROR] No commodities could be ranked. Train models first or check model directories.")
            sys.exit(1)
        
        # Show ALL ranked commodities in a table (top 10)
        print(f"\n{'='*80}")
        print(f"📊 ALL RANKED COMMODITIES (Top 10)")
        print(f"{'='*80}")
        print(f"{'Rank':<6} {'Symbol':<20} {'Score':<10} {'R²':<8} {'Test R²':<10} {'Conf':<8} {'Models':<8} {'Tradable':<10}")
        print(f"{'-'*90}")
        for i, item in enumerate(ranked[:10], 1):
            tradable_mark = "[YES]" if item["tradable"] else "[NO]"
            print(f"{i:<6} {item['symbol']:<20} {item['score']:<10.4f} {item['r2_score']:<8.3f} {item['test_r2']:<10.3f} {item['confidence']:<8.1f}% {item['successful_models']:<8} {tradable_mark:<10}")
        print(f"{'='*80}\n")
        
        # Select best tradable commodity
        best_commodity = None
        tradable_ranked = [item for item in ranked if item["tradable"]]
        
        if tradable_ranked:
            # Trade ALL tradable commodities (not just the best one)
            # Each commodity will be traded based on its model predictions
            selected_symbols = [item["symbol"] for item in tradable_ranked]
            best_commodity = tradable_ranked[0]["symbol"]
            best_item = tradable_ranked[0]
            best_rank = ranked.index(best_item) + 1
            
            print(f"{'='*80}")
            print(f"[OK] TRADING ALL TRADABLE COMMODITIES: {len(selected_symbols)} commodities")
            print(f"{'='*80}")
            print(f"  Total Ranked: {len(ranked)} commodities")
            print(f"  Tradable:     {len(tradable_ranked)} commodities (will be traded)")
            print(f"  Best Ranked:  #{best_rank} - {best_commodity} (Score: {best_item['score']:.4f})")
            print(f"{'='*80}\n")
            
            # Show all tradable commodities that will be traded
            print(f"📋 COMMODITIES TO TRADE (in parallel):")
            for i, item in enumerate(tradable_ranked, 1):
                rank_num = ranked.index(item) + 1
                print(f"  {i}. {item['symbol']:20s} - Rank #{rank_num}, Score: {item['score']:.4f} (R²={item['r2_score']:.3f}, Conf={item['confidence']:.1f}%)")
            print()
            
            print(f"💡 TRADING STRATEGY:")
            print(f"  - Each commodity will be evaluated every cycle")
            print(f"  - Entry: Based on model predictions (buy low signal)")
            print(f"  - Exit: Profit target hit, stop-loss hit, or trailing stop")
            print(f"  - All positions monitored side by side in parallel")
            print(f"  - All trades logged in real-time to data/logs/trades.jsonl")
            print(f"{'='*80}\n")
        else:
            print("[ERROR] No tradable commodities found. All models failed robustness checks.")
            print("   Top ranked commodities (not tradable):")
            for item in ranked[:5]:
                print(f"     - {item['symbol']}: Score={item['score']:.4f} (not tradable)")
            sys.exit(1)
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
            print("[ERROR] No tradable commodities found")
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
            print(f"  [OK] Synced {synced_count} existing position(s) with profit target: {args.profit_target_pct:.2f}%")
        else:
            print("  [OK] All existing positions already tracked")
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
        print("[ERROR] No tradable symbols found after filtering")
        print(f"   Requested: {requested_set}")
        print(f"   Available: {[info['asset'].data_symbol for info in all_tradable]}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"TRADABLE SYMBOLS: {len(tradable)}")
    print(f"{'='*80}")
    for info in tradable:
        asset = info["asset"]
        # Commodities always use paper trading with Yahoo Finance symbols (no MCX mapping needed)
        trading_symbol = asset.data_symbol  # Use Yahoo Finance symbol directly
        print(f"  - {asset.data_symbol} (Paper Trading with live Yahoo Finance prices, horizon: {info['horizon']})")
    
    # Initialize execution engine with STRICT buying power limits
    try:
        # Get current buying power to show user
        account = paper_client.get_account()
        equity = float(account.get("equity", 0) or 0)
        buying_power = float(account.get("buying_power", 0) or 0)
        
        print(f"\n{'='*80}")
        print(f"💰 ACCOUNT & POSITION SIZING (STRICT BUYING POWER LIMITS)")
        print(f"{'='*80}")
        print(f"  Equity:        Rs.{equity:,.2f} (virtual)")
        print(f"  Buying Power:  Rs.{buying_power:,.2f} (virtual)")
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
            client=paper_client,
            risk_config=risk_config,
            position_manager=position_manager,
            log_path=Path("data") / "logs" / "trades.jsonl",
        )
        print("    [OK] Execution engine initialized with strict buying power limits")
    except Exception as exc:
        print(f"    ✗ Failed to initialize ExecutionEngine: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Final confirmation
    print("\n" + "=" * 80)
    print("READY TO START PAPER TRADING")
    print("=" * 80)
    print(f"  Trading Platform: LOCAL PAPER TRADING (no external service)")
    print(f"  Selected Commodity: {selected_symbols[0] if selected_symbols else 'N/A'}")
    print(f"  Existing Positions: {len(existing_positions)} (will be monitored)")
    print(f"  Profit Target: {args.profit_target_pct:.2f}%")
    print(f"  Stop-Loss: {args.stop_loss_pct:.2f}% ({'USER-SPECIFIED' if args.stop_loss_pct != 2.0 else 'DEFAULT'})")
    print(f"  Execution Mode: {'DRY RUN (no orders)' if args.dry_run else 'PAPER TRADING (simulated orders)'}")
    print()
    print("SAFETY CONFIRMATION:")
    print("  - Using LOCAL PAPER TRADING (virtual money, local simulation)")
    print("  - NO real money is at risk")
    print("  - NO external service needed (no TradeTron, no Angel One)")
    print("  - All trades simulated locally on your computer")
    print("  - Positions tracked in: data/positions/active_positions.json")
    print("  - Trade logs saved to: data/logs/trades.jsonl")
    print("  - Safe for testing your strategy")
    print("=" * 80)
    print()
    
    if not args.dry_run:
        print("Starting PAPER TRADING in 3 seconds...")
        print("  (This is local paper trading - virtual money only, no real risk)")
        time.sleep(3)
    
    # Run trading cycles
    cycle_count = 0
    excluded_symbols = set()
    
    try:
        while True:
            cycle_count += 1
            now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
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
                    all_positions = paper_client.list_positions()
                    if all_positions:
                        print(f"\n  📊 ACTIVE POSITIONS: {len(all_positions)}")
                        for pos in all_positions:
                            symbol = pos.get("symbol", "")
                            qty = float(pos.get("qty", 0) or 0)
                            avg_entry = float(pos.get("avg_entry_price", 0) or 0)
                            ltp = float(pos.get("ltp", avg_entry) or avg_entry)
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

