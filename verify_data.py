"""Quick verification of fetched data"""
import json
from pathlib import Path

# Check BTC-USDT
btc_path = Path("data/json/raw/crypto/binance/BTC-USDT/1d/data.json")
if btc_path.exists():
    with open(btc_path) as f:
        btc_data = json.load(f)
    print("=" * 60)
    print("BTC-USDT Daily Data:")
    print(f"  Total candles: {len(btc_data):,}")
    print(f"  First date: {btc_data[0]['timestamp']}")
    print(f"  Last date: {btc_data[-1]['timestamp']}")
    print(f"  Date range: ~{len(btc_data)/365:.1f} years")
    print(f"  Sample candle: Close=${btc_data[0]['close']:,.2f}")
    print("=" * 60)

# Check GC=F
gold_path = Path("data/json/raw/commodities/yahoo/GC=F/1d/data.json")
if gold_path.exists():
    with open(gold_path) as f:
        gold_data = json.load(f)
    print("\nGC=F (Gold) Daily Data:")
    print(f"  Total candles: {len(gold_data):,}")
    print(f"  First date: {gold_data[0]['timestamp']}")
    print(f"  Last date: {gold_data[-1]['timestamp']}")
    print(f"  Date range: ~{len(gold_data)/365:.1f} years")
    print(f"  Sample candle: Close=${gold_data[0]['close']:,.2f}")
    print("=" * 60)

