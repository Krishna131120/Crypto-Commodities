"""Test Binance API to fetch trading pairs."""

import requests
import json

print("Testing Binance API...")

url = "https://api.binance.com/api/v3/exchangeInfo"
try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    print(f"Total symbols in response: {len(data.get('symbols', []))}")
    
    # Count by status
    status_counts = {}
    for sym in data.get("symbols", []):
        status = sym.get("status", "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"\nSymbol status counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    
    # Count USDT pairs
    usdt_pairs = []
    for sym in data.get("symbols", []):
        quote = sym.get("quoteAsset", "")
        if quote == "USDT" and sym.get("status") == "TRADING":
            permissions = sym.get("permissions", [])
            if "SPOT" in permissions:
                usdt_pairs.append(sym.get("symbol", ""))
    
    print(f"\nActive USDT SPOT pairs: {len(usdt_pairs)}")
    print(f"\nFirst 20 USDT pairs:")
    for pair in sorted(usdt_pairs)[:20]:
        print(f"  {pair}")
    
    # Show sample symbol structure
    if data.get("symbols"):
        sample = data["symbols"][0]
        print(f"\nSample symbol structure:")
        print(json.dumps(sample, indent=2))
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
