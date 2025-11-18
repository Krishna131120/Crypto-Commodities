"""
Main orchestrator for data ingestion with fallback and retry logic.
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
    timeframe: str = "1m",
    years: int = 5
) -> List[Dict[str, Any]]:
    """
    Fetch crypto historical data with automatic fallback.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        years: Number of years
    
    Returns:
        List of canonical candles
    """
    sources = ["binance_rest", "coinbase", "kucoin", "okx", "local_cache"]
    fallback = FallbackEngine(sources, "binance_rest")
    
    all_candles = []
    
    for attempt in range(len(sources)):
        source = fallback.get_current_source()
        
        try:
            print(f"Fetching {symbol} historical from {source}...")
            
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
            
            if candles:
                fallback.mark_success(source)
                all_candles = candles
                print(f"  Successfully fetched {len(candles)} candles from {source}")
                break
            else:
                fallback.mark_failure(source, "No data returned")
                
        except Exception as e:
            print(f"  Error from {source}: {e}")
            fallback.mark_failure(source, str(e))
        
        if attempt < len(sources) - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return all_candles


def fetch_commodities_historical_with_fallback(
    symbol: str,
    timeframe: str = "1d",
    years: int = 5
) -> List[Dict[str, Any]]:
    """
    Fetch commodities historical data with automatic fallback.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        years: Number of years
    
    Returns:
        List of canonical candles
    """
    sources = ["yahoo", "stooq", "local_cache"]
    fallback = FallbackEngine(sources, "yahoo")
    
    all_candles = []
    
    for attempt in range(len(sources)):
        source = fallback.get_current_source()
        
        try:
            print(f"Fetching {symbol} historical from {source}...")
            
            if source == "yahoo":
                candles = fetch_yahoo_historical(symbol, timeframe, years)
            elif source == "stooq":
                candles = fetch_stooq_historical(symbol, timeframe, years)
            elif source == "local_cache":
                candles = load_from_local_cache(symbol, timeframe, "commodities")
            else:
                candles = []
            
            if candles:
                fallback.mark_success(source)
                all_candles = candles
                print(f"  Successfully fetched {len(candles)} candles from {source}")
                break
            else:
                fallback.mark_failure(source, "No data returned")
                
        except Exception as e:
            print(f"  Error from {source}: {e}")
            fallback.mark_failure(source, str(e))
        
        if attempt < len(sources) - 1:
            time.sleep(2 ** attempt)
    
    return all_candles


def ingest_all_historical(
    crypto_symbols: Optional[List[str]] = None,
    commodities_symbols: Optional[List[str]] = None,
    timeframe: str = "1m",
    years: int = 5
):
    """
    Ingest historical data for all configured symbols.
    
    Args:
        crypto_symbols: List of crypto symbols (defaults to config)
        commodities_symbols: List of commodity symbols (defaults to config)
        timeframe: Timeframe
        years: Number of years
    """
    crypto_symbols = crypto_symbols or config.CRYPTO_SYMBOLS
    commodities_symbols = commodities_symbols or config.COMMODITIES_SYMBOLS
    
    print("=" * 80)
    print("STARTING HISTORICAL DATA INGESTION")
    print("=" * 80)
    
    # Ingest crypto
    print(f"\n[CRYPTO] Ingesting {len(crypto_symbols)} symbols...")
    for symbol in crypto_symbols:
        try:
            print(f"\n--- {symbol} ({timeframe}) ---")
            candles = fetch_crypto_historical_with_fallback(symbol, timeframe, years)
            if candles:
                save_historical_data(symbol, timeframe, candles)
            else:
                print(f"  WARNING: No data fetched for {symbol}")
        except Exception as e:
            print(f"  ERROR: Failed to ingest {symbol}: {e}")
    
    # Ingest commodities
    print(f"\n[COMMODITIES] Ingesting {len(commodities_symbols)} symbols...")
    for symbol in commodities_symbols:
        try:
            print(f"\n--- {symbol} ({timeframe}) ---")
            candles = fetch_commodities_historical_with_fallback(symbol, timeframe, years)
            if candles:
                save_yahoo_historical(symbol, timeframe, candles)
            else:
                print(f"  WARNING: No data fetched for {symbol}")
        except Exception as e:
            print(f"  ERROR: Failed to ingest {symbol}: {e}")
    
    print("\n" + "=" * 80)
    print("HISTORICAL DATA INGESTION COMPLETE")
    print("=" * 80)


def start_live_feeds(
    crypto_symbols: Optional[List[str]] = None,
    commodities_symbols: Optional[List[str]] = None,
    crypto_timeframe: str = "1m",
    commodities_timeframe: str = "1d"
):
    """
    Start live data feeds for all configured symbols.
    
    Args:
        crypto_symbols: List of crypto symbols
        commodities_symbols: List of commodity symbols
        crypto_timeframe: Timeframe for crypto
        commodities_timeframe: Timeframe for commodities
    """
    crypto_symbols = crypto_symbols or config.CRYPTO_SYMBOLS
    commodities_symbols = commodities_symbols or config.COMMODITIES_SYMBOLS
    
    print("=" * 80)
    print("STARTING LIVE DATA FEEDS")
    print("=" * 80)
    
    # Start crypto WebSocket feeds
    crypto_clients = []
    for symbol in crypto_symbols:
        try:
            print(f"\nStarting live feed for {symbol} ({crypto_timeframe})...")
            sources = ["binance_ws", "coinbase", "kucoin", "okx"]
            fallback = FallbackEngine(sources, "binance_ws")
            client = start_binance_live_feed(symbol, crypto_timeframe, fallback)
            crypto_clients.append(client)
        except Exception as e:
            print(f"  ERROR: Failed to start live feed for {symbol}: {e}")
    
    # Start commodities polling
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
            commodity_threads.append(thread)
        except Exception as e:
            print(f"  ERROR: Failed to start polling for {symbol}: {e}")
    
    print("\n" + "=" * 80)
    print("LIVE DATA FEEDS RUNNING")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping live feeds...")
        for client in crypto_clients:
            if client:
                client.stop()
        print("Live feeds stopped.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Crypto + Commodities Data Ingestion System")
    parser.add_argument("--mode", choices=["historical", "live", "both"], default="both",
                       help="Ingestion mode")
    parser.add_argument("--crypto-symbols", nargs="+", help="Crypto symbols to ingest")
    parser.add_argument("--commodities-symbols", nargs="+", help="Commodity symbols to ingest")
    parser.add_argument("--timeframe", default="1m", help="Timeframe for crypto")
    parser.add_argument("--years", type=int, default=5, help="Years of historical data")
    
    args = parser.parse_args()
    
    if args.mode in ["historical", "both"]:
        ingest_all_historical(
            args.crypto_symbols,
            args.commodities_symbols,
            args.timeframe,
            args.years
        )
    
    if args.mode in ["live", "both"]:
        start_live_feeds(
            args.crypto_symbols,
            args.commodities_symbols,
            args.timeframe,
            "1d"
        )


if __name__ == "__main__":
    main()

