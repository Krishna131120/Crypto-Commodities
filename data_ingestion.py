"""
Unified data ingestion system - Historical first, then live feeds with automatic fallback.
All sources are free and fallback is seamless to ensure uninterrupted data flow.
"""
import time
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import config
from fetchers import (
    FallbackEngine,
    fetch_binance_rest_historical,
    save_historical_data,
    start_binance_live_feed,
    fetch_yahoo_historical,
    save_yahoo_historical,
    poll_yahoo_live,
    fetch_coinbase_historical,
    fetch_kucoin_historical,
    fetch_okx_historical,
    load_from_local_cache,
    fetch_stooq_historical
)


def fetch_crypto_historical_with_fallback(
    symbol: str,
    timeframe: str = "1d",
    years: float = 5,
    fallback_engine: Optional[FallbackEngine] = None
) -> List[Dict[str, Any]]:
    """
    Fetch crypto historical data with automatic fallback.
    Seamlessly switches between free sources when rate limits are hit.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        years: Number of years
        fallback_engine: Optional shared fallback engine
    
    Returns:
        List of canonical candles
    """
    if fallback_engine is None:
        sources = ["binance_rest", "coinbase", "kucoin", "okx", "local_cache"]
        fallback_engine = FallbackEngine(sources, "binance_rest")
    
    all_candles = []
    max_attempts = len(fallback_engine.sources) * 2  # Allow retries
    
    for attempt in range(max_attempts):
        source = fallback_engine.get_current_source()
        
        try:
            print(f"[{symbol}] Fetching historical from {source} (attempt {attempt + 1})...")
            
            if source == "binance_rest":
                candles = fetch_binance_rest_historical(symbol, timeframe, years)
            elif source == "coinbase":
                candles = fetch_coinbase_historical(symbol, timeframe, years)
            elif source == "kucoin":
                candles = fetch_kucoin_historical(symbol, timeframe, years)
            elif source == "okx":
                candles = fetch_okx_historical(symbol, timeframe, years)
            elif source == "local_cache":
                candles = load_from_local_cache(symbol, timeframe, "crypto")
            else:
                candles = []
            
            if candles and len(candles) > 0:
                fallback_engine.mark_success(source)
                all_candles = candles
                print(f"  [OK] Successfully fetched {len(candles)} candles from {source}")
                break
            else:
                fallback_engine.mark_failure(source, "No data returned")
                
        except Exception as e:
            error_msg = str(e)
            # Check for rate limit errors
            status_code = None
            if "429" in error_msg or "rate limit" in error_msg.lower():
                status_code = 429
            elif "503" in error_msg or "502" in error_msg or "500" in error_msg:
                status_code = 500
            
            print(f"  [ERROR] {source}: {error_msg}")
            fallback_engine.mark_failure(source, error_msg, status_code)
        
        # Exponential backoff before trying next source
        if attempt < max_attempts - 1:
            wait_time = min(2 ** (attempt % 3), 10)  # Cap at 10 seconds
            print(f"  Waiting {wait_time}s before trying next source...")
            time.sleep(wait_time)
    
    return all_candles


def fetch_commodities_historical_with_fallback(
    symbol: str,
    timeframe: str = "1d",
    years: float = 5,
    fallback_engine: Optional[FallbackEngine] = None
) -> List[Dict[str, Any]]:
    """
    Fetch commodities historical data with automatic fallback.
    Always uses daily (1d) timeframe for commodities.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe (forced to 1d for commodities)
        years: Number of years requested by the user
        fallback_engine: Optional shared fallback engine
    
    Returns:
        List of canonical candles
    """
    # Force daily timeframe for commodities
    timeframe = "1d"
    if fallback_engine is None:
        sources = ["yahoo", "stooq", "local_cache"]
        fallback_engine = FallbackEngine(sources, "yahoo")
    
    all_candles = []
    max_attempts = len(fallback_engine.sources) * 2
    
    for attempt in range(max_attempts):
        source = fallback_engine.get_current_source()
        
        try:
            print(f"[{symbol}] Fetching historical from {source} (attempt {attempt + 1})...")
            
            if source == "yahoo":
                candles = fetch_yahoo_historical(symbol, timeframe, years)
            elif source == "stooq":
                candles = fetch_stooq_historical(symbol, timeframe, years)
            elif source == "local_cache":
                candles = load_from_local_cache(symbol, timeframe, "commodities")
            else:
                candles = []
            
            if candles and len(candles) > 0:
                fallback_engine.mark_success(source)
                all_candles = candles
                print(f"  [OK] Successfully fetched {len(candles)} candles from {source}")
                break
            else:
                fallback_engine.mark_failure(source, "No data returned")
                
        except Exception as e:
            error_msg = str(e)
            status_code = None
            if "429" in error_msg or "rate limit" in error_msg.lower():
                status_code = 429
            
            print(f"  [ERROR] {source}: {error_msg}")
            fallback_engine.mark_failure(source, error_msg, status_code)
        
        if attempt < max_attempts - 1:
            wait_time = min(2 ** (attempt % 3), 10)
            print(f"  Waiting {wait_time}s before trying next source...")
            time.sleep(wait_time)
    
    return all_candles


def ingest_all_historical(
    crypto_symbols: Optional[List[str]] = None,
    commodities_symbols: Optional[List[str]] = None,
    timeframe: str = "1d",
    years: float = 5
):
    """
    Ingest historical data for all configured symbols with automatic fallback.
    Ensures uninterrupted data flow by switching sources when rate limits hit.
    
    Args:
        crypto_symbols: List of crypto symbols (defaults to config)
        commodities_symbols: List of commodity symbols (defaults to config)
        timeframe: Timeframe
        years: Number of years
    """
    crypto_symbols = crypto_symbols or config.CRYPTO_SYMBOLS
    commodities_symbols = commodities_symbols or config.COMMODITIES_SYMBOLS
    
    print("=" * 80)
    print("PHASE 1: HISTORICAL DATA INGESTION")
    print("=" * 80)
    print(f"Fetching {years} years of DAILY (1d) price data for all symbols")
    print("All sources are FREE. Automatic fallback ensures uninterrupted data flow.")
    print("=" * 80)
    
    # Shared fallback engines for better source management
    crypto_fallback = FallbackEngine(
        ["binance_rest", "coinbase", "kucoin", "okx", "local_cache"],
        "binance_rest"
    )
    commodities_fallback = FallbackEngine(
        ["yahoo", "local_cache"],
        "yahoo"
    )
    
    # Ingest crypto
    print(f"\n[CRYPTO] Ingesting {len(crypto_symbols)} symbols...")
    for idx, symbol in enumerate(crypto_symbols, 1):
        try:
            print(f"\n[{idx}/{len(crypto_symbols)}] {symbol} ({timeframe})")
            candles = fetch_crypto_historical_with_fallback(
                symbol, timeframe, years, crypto_fallback
            )
            if candles:
                save_historical_data(symbol, timeframe, candles)
                print(f"  [SUCCESS] {symbol} complete: {len(candles)} candles saved")
            else:
                print(f"  [WARNING] No data fetched for {symbol}")
        except Exception as e:
            print(f"  [ERROR] Failed to ingest {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # Ingest commodities
    print(f"\n[COMMODITIES] Ingesting {len(commodities_symbols)} symbols...")
    for idx, symbol in enumerate(commodities_symbols, 1):
        try:
            print(f"\n[{idx}/{len(commodities_symbols)}] {symbol} ({timeframe})")
            candles = fetch_commodities_historical_with_fallback(
                symbol, timeframe, years, commodities_fallback
            )
            if candles:
                save_yahoo_historical(symbol, timeframe, candles)
                print(f"  [SUCCESS] {symbol} complete: {len(candles)} candles saved")
            else:
                print(f"  [WARNING] No data fetched for {symbol}")
        except Exception as e:
            print(f"  [ERROR] Failed to ingest {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("HISTORICAL DATA INGESTION COMPLETE")
    print("=" * 80)


def start_live_feeds_with_fallback(
    crypto_symbols: Optional[List[str]] = None,
    commodities_symbols: Optional[List[str]] = None,
    crypto_timeframe: str = "1d",
    commodities_timeframe: str = "1d"
):
    """
    Start live data feeds with automatic fallback.
    Seamlessly switches between free sources when primary source fails.
    Ensures uninterrupted data flow for model training.
    
    Args:
        crypto_symbols: List of crypto symbols
        commodities_symbols: List of commodity symbols
        crypto_timeframe: Timeframe for crypto
        commodities_timeframe: Timeframe for commodities
    """
    crypto_symbols = crypto_symbols or config.CRYPTO_SYMBOLS
    commodities_symbols = commodities_symbols or config.COMMODITIES_SYMBOLS
    
    print("\n" + "=" * 80)
    print("PHASE 2: LIVE DATA FEEDS")
    print("=" * 80)
    print("All sources are FREE. Automatic fallback ensures uninterrupted data flow.")
    print("=" * 80)
    
    # Start crypto WebSocket feeds with fallback
    crypto_clients = []
    crypto_fallbacks = {}
    
    for symbol in crypto_symbols:
        try:
            print(f"\nStarting live feed for {symbol} ({crypto_timeframe})...")
            # Free sources in priority order
            sources = ["binance_ws", "coinbase", "kucoin", "okx"]
            fallback = FallbackEngine(sources, "binance_ws")
            crypto_fallbacks[symbol] = fallback
            
            client = start_binance_live_feed(symbol, crypto_timeframe, fallback)
            crypto_clients.append((symbol, client))
            print(f"  [OK] {symbol} live feed started")
        except Exception as e:
            print(f"  [ERROR] Failed to start live feed for {symbol}: {e}")
    
    # Start commodities polling with fallback
    commodity_threads = []
    
    for symbol in commodities_symbols:
        try:
            print(f"\nStarting live polling for {symbol} ({commodities_timeframe})...")
            sources = ["yahoo"]
            fallback = FallbackEngine(sources, "yahoo")
            
            thread = threading.Thread(
                target=poll_yahoo_live,
                args=(symbol, commodities_timeframe, 60, fallback),
                daemon=True
            )
            thread.start()
            commodity_threads.append((symbol, thread))
            print(f"  [OK] {symbol} live polling started")
        except Exception as e:
            print(f"  [ERROR] Failed to start polling for {symbol}: {e}")
    
    print("\n" + "=" * 80)
    print("LIVE DATA FEEDS RUNNING")
    print("=" * 80)
    print("All feeds are active with automatic fallback protection.")
    print("Data is being saved continuously to ensure uninterrupted model training.")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    
    # Monitor and keep running
    try:
        while True:
            time.sleep(5)
            # Periodically check source status
            for symbol, fallback in crypto_fallbacks.items():
                if fallback.is_fallback_active():
                    print(f"  [INFO] {symbol} using fallback: {fallback.get_fallback_reason()}")
    except KeyboardInterrupt:
        print("\n\nStopping live feeds...")
        for symbol, client in crypto_clients:
            if client:
                try:
                    client.stop()
                    print(f"  [OK] Stopped {symbol}")
                except:
                    pass
        print("Live feeds stopped.")


def run_complete_ingestion(
    crypto_symbols: Optional[List[str]] = None,
    commodities_symbols: Optional[List[str]] = None,
    timeframe: str = "1d",
    years: float = 5
):
    """
    Run complete ingestion: Historical first, then live feeds.
    Ensures uninterrupted data flow with automatic fallback.
    
    Args:
        crypto_symbols: List of crypto symbols
        commodities_symbols: List of commodity symbols
        timeframe: Timeframe for crypto
        years: Number of years of historical data
    """
    print("\n" + "=" * 80)
    print("CRYPTO + COMMODITIES DATA INGESTION SYSTEM")
    print("=" * 80)
    print("All sources: FREE")
    print("Fallback: Automatic and seamless")
    print("Goal: Uninterrupted data flow for model training")
    print("=" * 80)
    
    # Phase 1: Historical data
    ingest_all_historical(crypto_symbols, commodities_symbols, timeframe, years)
    
    # Brief pause before starting live feeds
    print("\n" + "=" * 80)
    print("Transitioning to live feeds in 5 seconds...")
    print("=" * 80)
    time.sleep(5)
    
    # Phase 2: Live feeds
    start_live_feeds_with_fallback(
        crypto_symbols,
        commodities_symbols,
        timeframe,
        "1d"  # Commodities use daily timeframe
    )


def get_user_input():
    """Interactive function to get user input for symbols and options."""
    print("=" * 80)
    print("CRYPTO + COMMODITIES DATA INGESTION SYSTEM")
    print("=" * 80)
    print("Enter the currencies and commodities you want to fetch data for.")
    print("=" * 80)
    
    # Ask what type of data to fetch
    print("\nWhat type of data would you like to fetch?")
    print("  1. Crypto only")
    print("  2. Commodities only")
    print("  3. Both crypto and commodities")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        if choice in ["1", "2", "3"]:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    crypto_symbols = None
    commodities_symbols = None
    
    # Get crypto symbols
    if choice in ["1", "3"]:
        print("\n" + "-" * 80)
        print("CRYPTO SYMBOLS")
        print("-" * 80)
        print("Enter crypto symbols (e.g., BTC-USDT, ETH-USDT, SOL-USDT)")
        print("Format: SYMBOL-USDT (e.g., BTC-USDT)")
        print("Enter multiple symbols separated by commas or spaces")
        print("Example: BTC-USDT ETH-USDT SOL-USDT")
        
        crypto_input = input("\nEnter crypto symbols: ").strip()
        if crypto_input:
            # Split by comma or space
            crypto_symbols = [s.strip().upper() for s in crypto_input.replace(",", " ").split() if s.strip()]
            # Ensure USDT suffix
            crypto_symbols = [s if "-USDT" in s else f"{s}-USDT" for s in crypto_symbols]
            print(f"  Selected crypto symbols: {', '.join(crypto_symbols)}")
        else:
            print("  No crypto symbols entered.")
    
    # Get commodities symbols
    if choice in ["2", "3"]:
        print("\n" + "-" * 80)
        print("COMMODITIES SYMBOLS")
        print("-" * 80)
        print("Enter commodity symbols (e.g., GC=F for Gold, CL=F for Crude Oil)")
        print("Common symbols:")
        print("  GC=F  - Gold")
        print("  SI=F  - Silver")
        print("  CL=F  - Crude Oil")
        print("  NG=F  - Natural Gas")
        print("  ZC=F  - Corn")
        print("  ZS=F  - Soybeans")
        print("  KC=F  - Coffee")
        print("  CT=F  - Cotton")
        print("Enter multiple symbols separated by commas or spaces")
        print("Example: GC=F SI=F CL=F")
        
        commodities_input = input("\nEnter commodity symbols: ").strip()
        if commodities_input:
            # Split by comma or space
            commodities_symbols = [s.strip().upper() for s in commodities_input.replace(",", " ").split() if s.strip()]
            print(f"  Selected commodity symbols: {', '.join(commodities_symbols)}")
        else:
            print("  No commodity symbols entered.")
    
    # Validate that at least one symbol was entered
    if not crypto_symbols and not commodities_symbols:
        print("\n[ERROR] No symbols entered. Please enter at least one crypto or commodity symbol.")
        return None, None, None, None
    
    # Ask for mode
    print("\n" + "-" * 80)
    print("INGESTION MODE")
    print("-" * 80)
    print("  1. Historical data only (fetch past data)")
    print("  2. Live data only (real-time updates)")
    print("  3. Both (historical first, then live)")
    
    while True:
        mode_choice = input("\nEnter mode (1/2/3): ").strip()
        if mode_choice == "1":
            mode = "historical"
            break
        elif mode_choice == "2":
            mode = "live"
            break
        elif mode_choice == "3":
            mode = "both"
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Ask for years (only if historical or both)
    years = 5.0
    if mode in ["historical", "both"]:
        print("\n" + "-" * 80)
        print("HISTORICAL DATA PERIOD")
        print("-" * 80)
        while True:
            years_input = input("Enter number of years of historical data (default 5): ").strip()
            if not years_input:
                years = 5.0
                break
            try:
                years = float(years_input)
                if years <= 0:
                    print("  Please enter a positive number of years.")
                else:
                    break
            except ValueError:
                print("  Invalid input. Please enter a number.")
    
    return crypto_symbols, commodities_symbols, years, mode


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Crypto + Commodities Data Ingestion System (Free Sources Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python data_ingestion.py
  
  # Command line mode
  python data_ingestion.py --crypto-symbols BTC-USDT ETH-USDT --commodities-symbols GC=F SI=F
  
  # Historical only
  python data_ingestion.py --mode historical --crypto-symbols BTC-USDT
  
  # Live only
  python data_ingestion.py --mode live --crypto-symbols BTC-USDT
        """
    )
    parser.add_argument(
        "--mode",
        choices=["historical", "live", "both"],
        help="Ingestion mode (if not provided, interactive mode will ask)"
    )
    parser.add_argument(
        "--crypto-symbols",
        nargs="+",
        help="Crypto symbols to ingest (if not provided, interactive mode will ask)"
    )
    parser.add_argument(
        "--commodities-symbols",
        nargs="+",
        help="Commodity symbols to ingest (if not provided, interactive mode will ask)"
    )
    parser.add_argument(
        "--timeframe",
        default="1d",
        help="Timeframe for historical data (default: 1d for daily prices)"
    )
    parser.add_argument(
        "--years",
        type=float,
        help="Years of historical data to fetch (default: 5)"
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, use interactive mode
    if not args.crypto_symbols and not args.commodities_symbols and not args.mode and args.years is None:
        crypto_symbols, commodities_symbols, years, mode = get_user_input()
        if crypto_symbols is None and commodities_symbols is None:
            print("\nExiting. No symbols provided.")
            return
    else:
        # Command line mode
        crypto_symbols = args.crypto_symbols
        commodities_symbols = args.commodities_symbols
        years = args.years if args.years is not None else 5.0
        mode = args.mode if args.mode else "both"
    
    if years <= 0:
        print("\n[ERROR] Years of historical data must be a positive number.")
        return
    
    # Ensure daily timeframe for historical data
    historical_timeframe = args.timeframe if args.timeframe == "1d" else "1d"
    if args.timeframe != "1d":
        print(f"INFO: Using daily (1d) timeframe for historical data instead of {args.timeframe}")
    
    # Validate symbols
    if not crypto_symbols and not commodities_symbols:
        print("\n[ERROR] No symbols provided. Please specify at least one crypto or commodity symbol.")
        return
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if crypto_symbols:
        print(f"Crypto symbols: {', '.join(crypto_symbols)}")
    if commodities_symbols:
        print(f"Commodity symbols: {', '.join(commodities_symbols)}")
    print(f"Years: {years}")
    print(f"Mode: {mode}")
    print(f"Timeframe: {historical_timeframe}")
    print("=" * 80)
    
    if mode in ["historical", "both"]:
        if mode == "both":
            run_complete_ingestion(
                crypto_symbols,
                commodities_symbols,
                historical_timeframe,
                years
            )
        else:
            ingest_all_historical(
                crypto_symbols,
                commodities_symbols,
                historical_timeframe,
                years
            )
    
    if mode == "live":
        start_live_feeds_with_fallback(
            crypto_symbols,
            commodities_symbols,
            historical_timeframe,
            "1d"
        )


if __name__ == "__main__":
    main()

