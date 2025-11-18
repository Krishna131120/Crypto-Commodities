"""
Test script for data fetching functionality.
"""
from fetchers import (
    fetch_binance_rest_historical,
    save_historical_data,
    fetch_yahoo_historical,
    save_yahoo_historical
)
import config

def test_crypto_fetch():
    """Test fetching crypto data."""
    print("=" * 80)
    print("TESTING CRYPTO DATA FETCH")
    print("=" * 80)
    
    # Test with a single symbol, limited timeframe for testing
    symbol = "BTC-USDT"
    timeframe = "1h"  # Use 1h instead of 1m for faster testing
    
    print(f"\nFetching {symbol} ({timeframe})...")
    try:
        # Fetch just 30 days for testing
        candles = fetch_binance_rest_historical(symbol, timeframe, years=0.08)  # ~30 days
        
        if candles:
            print(f"[OK] Successfully fetched {len(candles)} candles")
            save_historical_data(symbol, timeframe, candles)
            print(f"[OK] Data saved and manifest generated")
            
            # Show sample candle
            if candles:
                print(f"\nSample candle:")
                print(f"  Timestamp: {candles[0]['timestamp']}")
                print(f"  Open: {candles[0]['open']}")
                print(f"  High: {candles[0]['high']}")
                print(f"  Low: {candles[0]['low']}")
                print(f"  Close: {candles[0]['close']}")
                print(f"  Volume: {candles[0]['volume']}")
        else:
            print("✗ No candles fetched")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def test_commodities_fetch():
    """Test fetching commodities data."""
    print("\n" + "=" * 80)
    print("TESTING COMMODITIES DATA FETCH")
    print("=" * 80)
    
    symbol = "GC=F"  # Gold
    timeframe = "1d"
    
    print(f"\nFetching {symbol} ({timeframe})...")
    try:
        # Fetch just 30 days for testing
        candles = fetch_yahoo_historical(symbol, timeframe, years=0.08)  # ~30 days
        
        if candles:
            print(f"[OK] Successfully fetched {len(candles)} candles")
            save_yahoo_historical(symbol, timeframe, candles)
            print(f"[OK] Data saved and manifest generated")
            
            # Show sample candle
            if candles:
                print(f"\nSample candle:")
                print(f"  Timestamp: {candles[0]['timestamp']}")
                print(f"  Open: {candles[0]['open']}")
                print(f"  High: {candles[0]['high']}")
                print(f"  Low: {candles[0]['low']}")
                print(f"  Close: {candles[0]['close']}")
                print(f"  Volume: {candles[0]['volume']}")
        else:
            print("✗ No candles fetched")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting data fetch tests...\n")
    
    test_crypto_fetch()
    test_commodities_fetch()
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)

