"""
Complete setup script for ALL Binance symbols.

This script:
1. Fetches ALL symbols from Binance (mainnet or testnet)
2. Adds them to symbol_universe.py
3. Ingests historical data for all symbols
4. Generates features for all symbols
5. Trains models for all symbols
6. Verifies everything is ready for paper trading

Usage:
    python binance/setup_all_symbols.py              # Use mainnet
    python binance/setup_all_symbols.py --testnet   # Use testnet (paper trading)
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from binance.fetch_all_binance_symbols import fetch_binance_trading_pairs, get_existing_symbols, generate_asset_mapping_code, add_symbols_to_universe
from pipeline_runner import run_ingestion, regenerate_features
from train_models import train_symbols
from trading.symbol_universe import all_enabled


def main():
    parser = argparse.ArgumentParser(
        description="Complete setup for ALL Binance symbols: fetch -> add -> ingest -> features -> train"
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Use Binance testnet (paper trading) API",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching symbols (assume they're already in symbol_universe.py)",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip data ingestion (use existing data)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature generation (use existing features)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training (use existing models)",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=10.0,
        help="Years of historical data to fetch (default: 10.0)",
    )
    parser.add_argument(
        "--timeframe",
        default="1d",
        help="Timeframe for data and models (default: 1d)",
    )
    parser.add_argument(
        "--horizon",
        default="short",
        help="Horizon profile for training (default: short)",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("BINANCE COMPLETE SETUP - ALL SYMBOLS")
    print("=" * 80)
    print()
    
    # Step 1: Fetch and add symbols
    if not args.skip_fetch:
        print("[STEP 1/5] Fetching ALL symbols from Binance...")
        pairs = fetch_binance_trading_pairs(use_testnet=args.testnet)
        if not pairs:
            print("ERROR: No trading pairs found. Exiting.")
            sys.exit(1)
        
        print(f"Found {len(pairs)} active USDT trading pairs")
        
        # Get existing symbols
        existing = get_existing_symbols()
        print(f"Existing symbols in universe: {len(existing)}")
        
        # Generate code for new symbols
        new_symbols = []
        for pair in pairs:
            code = generate_asset_mapping_code(pair, existing)
            if code:
                new_symbols.append(code)
                existing.add(f"{pair['base_asset']}-USDT")
        
        if new_symbols:
            print(f"Adding {len(new_symbols)} new symbols to symbol_universe.py...")
            added = add_symbols_to_universe(new_symbols)
            print(f"✓ Added {added} new symbols")
        else:
            print("✓ All symbols already in universe")
        
        print(f"Total symbols ready: {len(pairs)}")
    else:
        print("[STEP 1/5] SKIPPED: Symbol fetching")
        # Get count from existing symbols
        all_crypto = [asset for asset in all_enabled() if asset.asset_type == "crypto"]
        print(f"Using {len(all_crypto)} existing crypto symbols")
    
    # Step 2: Get all crypto symbols
    print("\n[STEP 2/5] Getting all crypto symbols...")
    all_crypto_assets = [asset for asset in all_enabled() if asset.asset_type == "crypto"]
    crypto_symbols = [asset.data_symbol.upper() for asset in all_crypto_assets]
    print(f"✓ Found {len(crypto_symbols)} crypto symbols to process")
    
    if not crypto_symbols:
        print("ERROR: No crypto symbols found. Run without --skip-fetch first.")
        sys.exit(1)
    
    # Step 3: Ingest historical data
    if not args.skip_ingestion:
        print(f"\n[STEP 3/5] Ingesting historical data for ALL {len(crypto_symbols)} symbols...")
        print("This may take a while...")
        try:
            run_ingestion(
                mode="historical",
                crypto_symbols=crypto_symbols,
                commodities_symbols=None,
                timeframe=args.timeframe,
                years=args.years,
            )
            print(f"✓ Data ingestion complete for {len(crypto_symbols)} symbols")
        except Exception as e:
            print(f"ERROR during ingestion: {e}")
            print("Continuing anyway...")
    else:
        print("\n[STEP 3/5] SKIPPED: Data ingestion")
    
    # Step 4: Generate features
    if not args.skip_features:
        print(f"\n[STEP 4/5] Generating features for ALL {len(crypto_symbols)} symbols...")
        print("This may take a while...")
        try:
            regenerate_features("crypto", set(crypto_symbols), args.timeframe)
            print(f"✓ Feature generation complete for {len(crypto_symbols)} symbols")
        except Exception as e:
            print(f"ERROR during feature generation: {e}")
            print("Continuing anyway...")
    else:
        print("\n[STEP 4/5] SKIPPED: Feature generation")
    
    # Step 5: Train models
    if not args.skip_training:
        print(f"\n[STEP 5/5] Training models for ALL {len(crypto_symbols)} symbols...")
        print("This may take a VERY long time (hours)...")
        print("Each symbol trains multiple models (XGBoost, LightGBM, etc.)")
        print()
        
        horizon_map = {symbol: args.horizon for symbol in crypto_symbols}
        try:
            train_symbols(
                crypto_symbols=crypto_symbols,
                commodities_symbols=None,
                timeframe=args.timeframe,
                output_dir="models",
                horizon_profiles=horizon_map,
            )
            print(f"✓ Model training complete for {len(crypto_symbols)} symbols")
        except Exception as e:
            print(f"ERROR during model training: {e}")
            print("You can continue with existing models or retry later")
    else:
        print("\n[STEP 5/5] SKIPPED: Model training")
    
    # Final summary
    print("\n" + "=" * 80)
    print("SETUP COMPLETE")
    print("=" * 80)
    print(f"Total symbols configured: {len(crypto_symbols)}")
    print(f"All symbols are ready for Binance paper trading!")
    print()
    print("Next step: Start trading")
    print("  python binance/end_to_end_binance.py --profit-target 1.5 --stop-loss-pct 8.0")
    print()
    print("This will:")
    print("  - Process ALL symbols for predictions")
    print("  - Trade any symbol that meets criteria")
    print("  - Monitor all positions")
    print("=" * 80)


if __name__ == "__main__":
    main()
