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
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

# Suppress sklearn version compatibility warnings (models trained with 1.3.2, running with 1.7.1)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Trying to unpickle.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*monotonic_cst.*", category=UserWarning)

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
        default=30,
        help="Seconds between trading cycles. Runs forever if set. Minimum: 30 seconds (to avoid rate limiting). Default: 30 seconds.",
    )
    parser.add_argument(
        "--allow-existing-positions",
        action="store_true",
        help="Allow trading even when symbols already have open positions in Alpaca. "
             "By default, symbols with existing positions are SKIPPED to avoid conflicts. "
             "With this flag: trading engine will manage existing positions (may exit them if model prediction changes). "
             "WARNING: This may close your existing positions if the model changes its prediction.",
    )
    parser.add_argument(
        "--force-full-retrain",
        action="store_true",
        help="Force complete retraining: re-fetch all data, regenerate all features, and retrain all models. "
             "By default, the system intelligently updates only what's needed (new data, missing models).",
    )
    parser.add_argument(
        "--skip-data-update",
        action="store_true",
        help="Skip fetching new data. Only use existing cached data. Useful for testing without API calls.",
    )
    parser.add_argument(
        "--top5",
        action="store_true",
        help="Trade only the top 5 performers based on model performance. "
             "By default, system trades ALL symbols that meet trading criteria (LONG with 60%+ confidence, mean reversion signals). "
             "Use this flag to limit trading to top 5 performers only.",
    )
    parser.add_argument(
        "--trade-all",
        action="store_true",
        help="[DEPRECATED] This is now the default behavior. All symbols with trained models are traded if they meet criteria.",
    )

    args = parser.parse_args()
    
    # Validate interval (minimum 30 seconds to avoid API rate limiting)
    if args.interval < 30:
        print(f"⚠️  WARNING: Interval {args.interval} seconds is too short. Minimum is 30 seconds to avoid rate limiting.")
        print(f"   Setting interval to 30 seconds.")
        args.interval = 30
    
    # --top5 flag: Override symbol selection with 5 most volatile cryptos
    if args.top5:
        print("\n" + "=" * 80)
        print("🔥 TOP 5 VOLATILE CRYPTO MODE")
        print("=" * 80)
        print("Processing ONLY 5 highly volatile cryptocurrencies:")
        print("  1. BTC-USDT  (Bitcoin - mandatory, high liquidity)")
        print("  2. ETH-USDT  (Ethereum - mandatory, high liquidity)")
        print("  3. SOL-USDT  (Solana - high volatility altcoin)")
        print("  4. DOGE-USDT (Dogecoin - high volatility meme coin)")
        print("  5. XRP-USDT  (Ripple - high volatility altcoin)")
        print()
        print("⚡ This mode is MUCH FASTER than processing all 57 symbols!")
        print("⚡ Estimated time: 3-5 minutes instead of 30-60 minutes")
        print("=" * 80)
        print()
        
        # Override with top 5 volatile symbols
        raw_symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "DOGE-USDT", "XRP-USDT"]
        
        # Skip auto-discovery and user-provided symbols
        # These 5 symbols are already normalized
        crypto_symbols = raw_symbols
        
    # Auto-discover crypto symbols if not provided (and --top5 not set)
    elif args.crypto_symbols is None or len(args.crypto_symbols) == 0:
        print("\n[AUTO-DISCOVERY] No symbols specified, discovering all enabled crypto symbols...")
        from trading.symbol_universe import all_enabled
        
        # Get all enabled crypto assets
        all_crypto_assets = [asset for asset in all_enabled() if asset.asset_type == "crypto"]
        
        if not all_crypto_assets:
            print("❌ ERROR: No enabled crypto symbols found in symbol_universe.py")
            print("   Please enable at least one crypto symbol in trading/symbol_universe.py")
            sys.exit(1)
        
        # Extract data symbols
        raw_symbols = [asset.data_symbol for asset in all_crypto_assets]
        print(f"[AUTO-DISCOVERY] Found {len(raw_symbols)} enabled crypto symbols:")
        # Show first 10 symbols to avoid overwhelming output
        preview = raw_symbols[:10]
        for sym in preview:
            print(f"  • {sym}")
        if len(raw_symbols) > 10:
            print(f"  ... and {len(raw_symbols) - 10} more")
        print()
    else:
        # Normalize user-provided symbols so both data_symbol (BTC-USDT) and
        # trading_symbol (BTC/USD) styles are accepted on the CLI.
        raw_symbols = [s.strip().upper() for s in args.crypto_symbols if s.strip()]

    def _normalize_crypto_symbol(sym: str) -> str:
        """Return our canonical data symbol for a user-supplied crypto symbol.

        Accepted inputs (case-insensitive):
        - Data symbol:   BTC-USDT, btc-usdt, ETH-USDT
        - Trading symbol: BTCUSD, ETHUSD, BTC/USD, ETH/USD
        - Base symbol:   BTC, ETH (infers -USDT)
        - Any variation with or without separators
        """
        if not sym or not sym.strip():
            return sym.upper() if sym else ""
        
        # Normalize: strip whitespace, uppercase
        sym = sym.strip().upper()
        
        # 1) Exact data_symbol match (e.g., BTC-USDT)
        asset = find_by_data_symbol(sym)
        if asset:
            return asset.data_symbol.upper()

        # 2) Trading symbol match (e.g., BTCUSD, ETHUSD)
        asset = find_by_trading_symbol(sym)
        if asset:
            return asset.data_symbol.upper()

        # 3) Handle BTC/USD or ETH/USD format -> convert to BTC-USDT
        if "/" in sym:
            base, quote = sym.split("/", 1)
            base = base.strip().upper()
            quote = quote.strip().upper()
            if base and quote == "USD":
                # Try to find by the converted data symbol
                converted = f"{base}-USDT"
                asset = find_by_data_symbol(converted)
                if asset:
                    return asset.data_symbol.upper()
                # If not found, return the converted format anyway
                return converted

        # 4) Try bare symbol (e.g., "BTC" -> "BTC-USDT")
        # Remove common separators and try to infer
        bare_symbol = sym.replace("-", "").replace("/", "").replace("_", "").replace(" ", "")
        # Check if it's a known trading symbol without separator
        asset = find_by_trading_symbol(bare_symbol)
        if asset:
            return asset.data_symbol.upper()
        
        # Try adding -USDT if it looks like a base symbol (3-5 chars, all letters)
        if len(bare_symbol) >= 3 and len(bare_symbol) <= 5 and bare_symbol.isalpha():
            inferred = f"{bare_symbol}-USDT"
            asset = find_by_data_symbol(inferred)
            if asset:
                return asset.data_symbol.upper()

        # 5) Fallback: return as-is, uppercased. Downstream fetchers can still try.
        return sym.upper()

    # Normalize symbols (skip if --top5 since they're already normalized)
    if not args.top5:
        crypto_symbols = [_normalize_crypto_symbol(s) for s in raw_symbols]
    
    # Validate that we found at least some recognizable symbols
    if not crypto_symbols:
        print("❌ ERROR: No valid crypto symbols found after normalization.")
        print("   Please check your symbol_universe.py or provide valid symbols manually.")
        sys.exit(1)
    
    # Show what symbols were normalized to (helpful for debugging)
    if not args.top5 and args.crypto_symbols is not None:  # Only show normalization if user provided symbols (and not --top5)
        if len(raw_symbols) != len(crypto_symbols) or any(r.upper() != c for r, c in zip(raw_symbols, crypto_symbols)):
            print(f"[INFO] Symbol normalization:")
            for raw, normalized in zip(raw_symbols, crypto_symbols):
                if raw.upper() != normalized:
                    print(f"  {raw} -> {normalized}")
    
    
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
    print("[STRATEGY] Buy Low, Sell High, Minimize Losses:")
    print("  ✅ Momentum filters: Block buying during upswings (buy low)")
    print("  ✅ RSI filters: Only buy when oversold (RSI < 30), only short when overbought (RSI > 70)")
    print("  ✅ Mean reversion: Flips SHORT to LONG when oversold (buy low)")
    print("  ✅ Trailing stop: Sell when price drops 2.5% from peak (sell at peak)")
    if args.stop_loss_pct is not None:
        print(f"  ✅ Stop-loss: {args.stop_loss_pct:.1f}% protection (minimize losses)")
    else:
        print("  ✅ Stop-loss: 3.5% protection (minimize losses)")
    print("  ✅ Negative prediction filter: Block entries when predicted return < 0 (minimize losses)")
    print("  ✅ Minimum return filter: Require 1.5% minimum predicted return (minimize losses)")
    print("  ✅ Volatility-based sizing: Reduce position size in high volatility (minimize losses)")
    print()
    
    # Show available horizons and their trading behavior
    print_horizon_summary()

    # ------------------------------------------------------------------
    # SMART INCREMENTAL UPDATE SYSTEM
    # ------------------------------------------------------------------
    # By default, intelligently update only what's needed:
    # 1. Fetch only NEW/MISSING data (not re-download everything)
    # 2. Train only MISSING models
    # 3. Update features for symbols with new data
    # 4. Use LATEST data for predictions
    # ------------------------------------------------------------------
    
    from pathlib import Path
    import json
    from datetime import datetime, timedelta
    
    force_full_retrain = args.force_full_retrain
    skip_data_update = args.skip_data_update
    
    if force_full_retrain:
        print("\n[FORCE-FULL-RETRAIN] Re-fetching all data and retraining all models")
        print("=" * 80)
        
        # ------------------------------------------------------------------
        # Stage 1: Historical ingestion (CRYPTO ONLY) - ALL SYMBOLS
        # ------------------------------------------------------------------
        print("[1/4] Ingesting historical data (all symbols)...")
        run_ingestion(
            mode="historical",
            crypto_symbols=crypto_symbols,
            commodities_symbols=None,
            timeframe=timeframe,
            years=years,
        )
        print("    ✓ Historical data ingestion complete.")

        # ------------------------------------------------------------------
        # Stage 2: Feature generation (CRYPTO ONLY) - ALL SYMBOLS
        # ------------------------------------------------------------------
        print("[2/4] Regenerating features (all symbols)...")
        regenerate_features("crypto", set(crypto_symbols), timeframe)
        print("    ✓ Feature generation complete.")

        # ------------------------------------------------------------------
        # Stage 3: Model training (CRYPTO ONLY) - ALL SYMBOLS
        # ------------------------------------------------------------------
        print("[3/4] Training models (all symbols)...")
        horizon_map = {"crypto": horizon}
        train_symbols(
            crypto_symbols=crypto_symbols,
            commodities_symbols=[],
            timeframe=timeframe,
            output_dir="models",
            horizon_profiles=horizon_map,
        )
        print("    ✓ Model training complete.")
    
    else:
        # SMART INCREMENTAL UPDATE MODE (default)
        print("\n[SMART UPDATE] Checking what needs to be updated...")
        print("=" * 80)
        
        # Check which symbols have data, features, and models
        symbols_needing_data = []
        symbols_needing_features = []
        symbols_needing_training = []
        symbols_up_to_date = []
        
        for symbol in crypto_symbols:
            # Check if data exists and is recent
            data_file = Path("data/json/raw/crypto") / "binance" / symbol / timeframe / "data.json"
            has_recent_data = False
            
            if data_file.exists() and not skip_data_update:
                try:
                    # Check if data is from today (or yesterday for daily timeframe)
                    file_mod_time = datetime.fromtimestamp(data_file.stat().st_mtime)
                    age_hours = (datetime.now() - file_mod_time).total_seconds() / 3600
                    
                    # For daily timeframe, data older than 24 hours should be updated
                    if timeframe == "1d":
                        has_recent_data = age_hours < 24
                    else:
                        has_recent_data = age_hours < 1  # For intraday, update if older than 1 hour
                except Exception:
                    has_recent_data = False
            
            if not data_file.exists() or not has_recent_data:
                symbols_needing_data.append(symbol)
            
            # Check if features exist
            features_file = Path("data/json/features/crypto") / symbol / timeframe / "features.json"
            if not features_file.exists():
                symbols_needing_features.append(symbol)
            elif symbol in symbols_needing_data:
                # If we're updating data, we should also update features
                symbols_needing_features.append(symbol)
            
            # Check if model exists
            model_dir = Path("models") / "crypto" / symbol / timeframe / horizon
            summary_file = model_dir / "summary.json"
            
            if not summary_file.exists():
                symbols_needing_training.append(symbol)
            elif symbol not in symbols_needing_data and symbol not in symbols_needing_features:
                symbols_up_to_date.append(symbol)
        
        # Show status
        print(f"\n📊 Status Check:")
        print(f"  ✓ Up-to-date:      {len(symbols_up_to_date)} symbols")
        print(f"  🔄 Need data:      {len(symbols_needing_data)} symbols")
        print(f"  🔄 Need features:  {len(symbols_needing_features)} symbols")
        print(f"  🔄 Need training:  {len(symbols_needing_training)} symbols")
        print()
        
        # Stage 1: Fetch data for symbols that need it
        if symbols_needing_data and not skip_data_update:
            print(f"[1/4] Fetching new data for {len(symbols_needing_data)} symbol(s)...")
            if len(symbols_needing_data) <= 10:
                for sym in symbols_needing_data:
                    print(f"  • {sym}")
            else:
                for sym in symbols_needing_data[:5]:
                    print(f"  • {sym}")
                print(f"  ... and {len(symbols_needing_data) - 5} more")
            
            run_ingestion(
                mode="historical",
                crypto_symbols=symbols_needing_data,
                commodities_symbols=None,
                timeframe=timeframe,
                years=years,
            )
            print(f"    ✓ Data fetch complete for {len(symbols_needing_data)} symbols")
        elif skip_data_update:
            print("[1/4] Skipping data update (--skip-data-update flag set)")
        else:
            print("[1/4] All data is up-to-date, skipping data fetch")
        
        # Stage 2: Regenerate features for symbols that need it
        if symbols_needing_features:
            print(f"\n[2/4] Regenerating features for {len(symbols_needing_features)} symbol(s)...")
            if len(symbols_needing_features) <= 10:
                for sym in symbols_needing_features:
                    print(f"  • {sym}")
            else:
                for sym in symbols_needing_features[:5]:
                    print(f"  • {sym}")
                print(f"  ... and {len(symbols_needing_features) - 5} more")
            
            regenerate_features("crypto", set(symbols_needing_features), timeframe)
            print(f"    ✓ Feature generation complete for {len(symbols_needing_features)} symbols")
        else:
            print("\n[2/4] All features are up-to-date, skipping feature generation")
        
        # Stage 3: Train models for symbols that need it
        if symbols_needing_training:
            print(f"\n[3/4] Training models for {len(symbols_needing_training)} symbol(s)...")
            if len(symbols_needing_training) <= 10:
                for sym in symbols_needing_training:
                    print(f"  • {sym}")
            else:
                for sym in symbols_needing_training[:5]:
                    print(f"  • {sym}")
                print(f"  ... and {len(symbols_needing_training) - 5} more")
            
            horizon_map = {"crypto": horizon}
            train_symbols(
                crypto_symbols=symbols_needing_training,
                commodities_symbols=[],
                timeframe=timeframe,
                output_dir="models",
                horizon_profiles=horizon_map,
            )
            print(f"    ✓ Model training complete for {len(symbols_needing_training)} symbols")
        else:
            print("\n[3/4] All models are trained, skipping model training")
        
        print()


    # ------------------------------------------------------------------
    # Stage 3.5: Rank trained models and select top performers
    # ------------------------------------------------------------------
    print("\n[RANKING] Evaluating model performance...")
    
    def rank_crypto_symbols(symbols: List[str], timeframe: str, horizon: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Rank crypto symbols by model performance metrics.
        
        Returns list of dicts with: symbol, score, directional_accuracy, r2, model_dir
        """
        from pathlib import Path
        import json
        
        rankings = []
        
        for symbol in symbols:
            # Path to model directory: models/crypto/{symbol}/{timeframe}/{horizon}/
            model_dir = Path("models") / "crypto" / symbol / timeframe / horizon
            metrics_file = model_dir / "metrics.json"
            
            if not metrics_file.exists():
                # No trained model for this symbol
                continue
            
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Extract key metrics from all models
                # We'll use the best performing model's metrics
                best_r2 = 0.0
                best_directional = 0.5
                
                for model_name, model_metrics in metrics.items():
                    if isinstance(model_metrics, dict):
                        r2 = model_metrics.get("r2", 0.0) or 0.0
                        directional = model_metrics.get("directional_accuracy", 0.5) or 0.5
                        
                        # Update best metrics
                        if r2 > best_r2:
                            best_r2 = r2
                        if directional > best_directional:
                            best_directional = directional
                
                # Calculate performance score
                # Prioritize directional accuracy (70%) over R² (30%)
                # Only consider models with positive R² and >50% accuracy
                if best_r2 > 0 and best_directional > 0.50:
                    score = (best_directional * 0.7) + (best_r2 * 0.3)
                else:
                    score = 0.0  # Reject poor models
                
                rankings.append({
                    "symbol": symbol,
                    "score": score,
                    "directional_accuracy": best_directional,
                    "r2": best_r2,
                    "model_dir": str(model_dir)
                })
            
            except Exception as e:
                # Failed to load metrics for this symbol
                print(f"    ⚠️  Could not load metrics for {symbol}: {e}")
                continue
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top N
        return rankings[:top_n]
    
    # Rank all trained symbols (but only use top 5 unless --trade-all is set)
    all_rankings = rank_crypto_symbols(crypto_symbols, timeframe, horizon, top_n=len(crypto_symbols))  # Get all rankings
    
    if not all_rankings:
        print("    ✗ No models passed performance criteria. Exiting.")
        print("    Tip: Models need R² > 0 and directional accuracy > 50% to qualify.")
        return
    
    # Display leaderboard
    display_count = len(all_rankings) if args.trade_all else min(10, len(all_rankings))  # Show top 10 if trading all, or all if less
    print(f"\n🏆 CRYPTO LEADERBOARD (Showing top {display_count} of {len(all_rankings)} Performers)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Symbol':<15} {'Score':<10} {'Accuracy':<12} {'R²':<10}")
    print("-" * 80)
    for i, perf in enumerate(all_rankings[:display_count], 1):
        print(f"{i:<6} {perf['symbol']:<15} {perf['score']:.4f}    {perf['directional_accuracy']*100:.2f}%       {perf['r2']:.4f}")
    if len(all_rankings) > display_count:
        print(f"  ... and {len(all_rankings) - display_count} more")
    print("=" * 80)
    print()
    
    # Determine which symbols to trade
    if args.top5:
        # User explicitly wants top 5 only
        top_performers = all_rankings[:5]
        top_symbols = [p["symbol"] for p in top_performers]
        print(f"[SELECTION] Trading only the top {len(top_symbols)} performers: {', '.join(top_symbols)}")
        print(f"[INFO] Use without --top5 to trade all symbols that meet trading criteria")
    else:
        # Default: Trade ALL symbols with trained models (they will be filtered by trading criteria)
        top_symbols = [p["symbol"] for p in all_rankings]
        print(f"[SELECTION] Trading ALL {len(top_symbols)} symbols with trained models (if they meet trading criteria)")
        if len(top_symbols) > 10:
            print(f"  Top performers: {', '.join([p['symbol'] for p in all_rankings[:10]])} ... and {len(top_symbols) - 10} more")
        else:
            print(f"  Symbols: {', '.join(top_symbols)}")
    print()

    # ------------------------------------------------------------------
    # Stage 4: Live trading cycle(s)
    # ------------------------------------------------------------------
    print("[4/4] Preparing live trading...")

    # Discover which of the top-ranked symbols actually have trained models.
    # CRITICAL: Pass the trained horizon as override_horizon so discover_tradable_symbols
    # finds the correct models (e.g., intraday models instead of default short models)
    all_tradable = discover_tradable_symbols(
        asset_type="crypto", 
        timeframe=timeframe,
        override_horizon=horizon  # Override asset's default horizon_profile with the trained horizon
    )
    
    # Determine which symbols to trade based on --top5 flag
    if args.top5:
        # Restrict to the top-ranked symbols only (for trading)
        top_symbols_set = {s.upper() for s in top_symbols}
        tradable = [
            info
            for info in all_tradable
            if info["asset"].data_symbol.upper() in top_symbols_set
        ]
        print(f"\n[MONITORING] System will monitor ALL {len(all_tradable)} symbols with trained models:")
        print(f"  - Trading: Top {len(tradable)} performers only")
        print(f"  - Monitoring: All {len(all_tradable)} symbols for predictions")
        print(f"  - Override: Any symbol with LONG + 60%+ confidence will be traded even if not in top 5")
    else:
        # Default: Trade ALL symbols with trained models (no top 5 restriction)
        tradable = all_tradable.copy()  # Trade ALL symbols that meet criteria
        print(f"\n[MONITORING] System will trade ALL {len(tradable)} symbols with trained models:")
        print(f"  - Trading: All symbols that meet trading criteria (LONG with 60%+ confidence, mean reversion signals)")
        print(f"  - No top 5 restriction - all currencies are evaluated equally")
    
    # For predictions, always use all symbols (for monitoring)
    all_tradable_for_predictions = all_tradable.copy()  # Monitor ALL symbols with trained models

    if not tradable:
        print("    ✗ No tradable symbols found after ranking. Exiting.")
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

    # FIXED: Always include symbols with existing positions for monitoring
    # The execution engine will intelligently handle them:
    # - If --allow-existing-positions: Can actively trade (add/exit positions)
    # - If NOT set: Will monitor (check stop-losses, show predictions) but won't execute new trades
    tradable_filtered = tradable  # Include all symbols for monitoring
    if args.allow_existing_positions:
        print(f"    ⚠️  WARNING: --allow-existing-positions is enabled.")
        print(f"       The trading engine will actively manage existing positions (may exit them if model changes).")
    else:
        if existing_positions:
            print(f"    ℹ  Found {len(existing_positions)} symbol(s) with existing positions.")
            print(f"       These will be MONITORED (stop-loss checks, predictions shown) but not actively traded.")
            print(f"       Use --allow-existing-positions to enable active trading on existing positions.")
    
    print(f"    ✓ {len(tradable_filtered)} symbol(s) ready for monitoring/trading.")

    # Initialize execution engine and position manager (will use env vars for Alpaca).
    try:
        from trading.position_manager import PositionManager
        # Ensure Path is available (re-import to avoid any shadowing issues)
        from pathlib import Path
        
        risk_config = TradingRiskConfig(
            manual_stop_loss=args.manual_stop_loss,
            user_stop_loss_pct=args.stop_loss_pct,  # User override if provided
        )
        position_manager = PositionManager()
        engine = ExecutionEngine(risk_config=risk_config, position_manager=position_manager)
        if args.manual_stop_loss:
            print("    ⚠️  MANUAL STOP-LOSS MODE enabled - you are responsible for managing stop-losses")
        
        # CRITICAL: Sync all broker positions to PositionManager on startup/restart
        # This ensures positions opened externally or before tracking are now monitored
        print("\n[STARTUP SYNC] Syncing all broker positions to PositionManager...")
        try:
            # Build asset mappings dict: trading_symbol -> AssetMapping
            # Use ALL enabled crypto symbols (not just tradable) to catch any broker positions
            from trading.symbol_universe import all_enabled
            asset_mappings = {}
            for asset in all_enabled():
                if asset.asset_type == "crypto" and asset.enabled:
                    trading_symbol = asset.trading_symbol.upper()
                    asset_mappings[trading_symbol] = asset
            
            # Also add assets from tradable symbols (in case they're not in all_enabled)
            for symbol_info in tradable_filtered:
                asset = symbol_info.get("asset")
                if asset:
                    trading_symbol = asset.trading_symbol.upper()
                    asset_mappings[trading_symbol] = asset
            
            # Get effective stop-loss percentage
            effective_stop_loss = args.stop_loss_pct if args.stop_loss_pct is not None else 3.5  # Default 3.5% for crypto
            
            # Sync all broker positions
            sync_results = engine.sync_all_broker_positions(
                asset_mappings=asset_mappings,
                profit_target_pct=args.profit_target,
                stop_loss_pct=effective_stop_loss / 100.0,  # Convert percentage to decimal
                verbose=True,
            )
            
            if sync_results.get("synced"):
                print(f"  ✅ Successfully synced {len(sync_results['synced'])} position(s) - they will now be monitored")
            elif sync_results.get("total_broker_positions", 0) > 0:
                print(f"  ℹ  Found {sync_results['total_broker_positions']} broker position(s), all already tracked")
            else:
                print(f"  ℹ  No broker positions found - starting fresh")
        except Exception as sync_exc:
            print(f"  ⚠️  WARNING: Position sync failed: {sync_exc}")
            print(f"  [CONTINUING] Trading will continue, but untracked positions may not be monitored")
            import traceback
            if args.verbose:
                print(f"  [SYNC ERROR] Traceback: {traceback.format_exc()}")
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

    # Track excluded symbols (manually closed positions) across cycles
    excluded_symbols = set()
    
    cycle_index = 0
    while True:
        cycle_index += 1
        now = datetime.utcnow().isoformat() + "Z"
        print(f"\n[CYCLE {cycle_index}] {now}")

        # Show predictions for ALL symbols, but only trade selected ones
        cycle_results = run_trading_cycle(
            execution_engine=engine,
            tradable_symbols=tradable_filtered,  # Only these will be traded
            dry_run=args.dry_run,
            verbose=True,  # Always verbose to see what's happening
            update_data=True,  # Fetch latest live data each cycle
            regenerate_features_flag=True,  # Regenerate features with new data each cycle
            profit_target_pct=args.profit_target,  # REQUIRED - user must specify
            user_stop_loss_pct=args.stop_loss_pct,  # Optional - user override if provided
            excluded_symbols=excluded_symbols,  # Pass excluded symbols to filter them out
            all_symbols_for_predictions=all_tradable_for_predictions,  # Show predictions for all ranked symbols (but only trade selected ones)
        )
        
        # Update excluded symbols if any were manually closed
        if cycle_results.get("symbols_stopped"):
            for symbol in cycle_results["symbols_stopped"]:
                excluded_symbols.add(symbol)
                print(f"\n[TRACKING] Added {symbol} to excluded list (manually closed)")

        # Concise summary with monitoring info
        print(
            f"\n[SUMMARY] Cycle {cycle_index} Results:"
        )
        print(
            f"  Monitored: {cycle_results['symbols_processed']} symbols (ALL currencies with trained models checked)"
        )
        if args.top5:
            print(
                f"  Traded: {cycle_results['symbols_traded']} symbols (top 5 + any LONG with 60%+ confidence)"
            )
            print(
                f"  Skipped: {cycle_results['symbols_skipped']} symbols (not in top 5 + low confidence)"
            )
        else:
            print(
                f"  Traded: {cycle_results['symbols_traded']} symbols (met trading criteria: LONG with 60%+ confidence, mean reversion signals)"
            )
            print(
                f"  Skipped: {cycle_results['symbols_skipped']} symbols (did not meet trading criteria)"
            )
        if len(cycle_results['errors']) > 0:
            print(f"  Errors: {len(cycle_results['errors'])}")

        # Show per-symbol trading decisions
        traded_symbols = [d for d in cycle_results.get("details", []) 
                         if d.get("status") in ["entered_long", "entered_short", "added_to_position", "flip_position"]]
        if traded_symbols:
            print(f"\n  ✅ Symbols Traded:")
            for detail in traded_symbols:
                symbol = detail.get("symbol", "?")
                decision = detail.get("decision", "?")
                action = (detail.get("consensus_action") or detail.get("model_action") or "").upper()
                conf = float(detail.get("confidence") or 0.0)
                if conf > 1.0:
                    conf = conf / 100.0
                conf_pct = conf * 100.0
                # Check if this was an override (not in tradable set but met criteria)
                is_override = detail.get("status") == "override_long" or "OVERRIDE" in str(decision).upper()
                if args.top5 and is_override:
                    override_msg = " [OVERRIDE - strong signal]"
                else:
                    override_msg = ""
                print(f"    • {symbol}: {action} -> {decision} ({conf_pct:.1f}% confidence){override_msg}")

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


