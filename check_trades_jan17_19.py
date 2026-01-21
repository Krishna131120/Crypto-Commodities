"""Check for buy/sell logs between January 17-19."""
import json
from datetime import datetime
from pathlib import Path

print("=" * 80)
print("CHECKING TRADES BETWEEN JANUARY 17-19, 2026")
print("=" * 80)

# Date range
start_date = datetime(2026, 1, 17, 0, 0, 0)
end_date = datetime(2026, 1, 19, 23, 59, 59)

print(f"\nDate Range: {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}")

# Check crypto trades
crypto_log = Path("logs/trading/crypto_trades.jsonl")
print(f"\n1. Checking crypto trades: {crypto_log}")
print("-" * 80)

crypto_trades = []
if crypto_log.exists():
    with open(crypto_log, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trade = json.loads(line)
                # Parse timestamp
                timestamp_str = trade.get("timestamp", "")
                if timestamp_str:
                    # Handle ISO format with Z or +00:00
                    if timestamp_str.endswith("Z"):
                        timestamp_str = timestamp_str[:-1] + "+00:00"
                    try:
                        trade_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        # Check if in range
                        if start_date <= trade_time <= end_date:
                            crypto_trades.append((trade_time, trade))
                    except Exception as e:
                        # Try alternative parsing
                        try:
                            trade_time = datetime.strptime(timestamp_str.split(".")[0], "%Y-%m-%dT%H:%M:%S")
                            if start_date <= trade_time <= end_date:
                                crypto_trades.append((trade_time, trade))
                        except:
                            pass
            except json.JSONDecodeError:
                continue

print(f"Found {len(crypto_trades)} crypto trades in date range")

if crypto_trades:
    print("\nCrypto Trades:")
    for trade_time, trade in sorted(crypto_trades, key=lambda x: x[0]):
        decision = trade.get("decision", "unknown")
        final_side = trade.get("final_side", "unknown")
        final_qty = trade.get("final_qty", 0)
        symbol = trade.get("trading_symbol", trade.get("asset", "unknown"))
        timestamp = trade.get("timestamp", "")
        
        action = "BUY" if final_side == "long" and final_qty > 0 else "SELL" if final_side == "short" and final_qty > 0 else "HOLD/FLAT"
        
        print(f"  [{trade_time.strftime('%Y-%m-%d %H:%M:%S')}] {symbol}: {action}")
        print(f"    Decision: {decision}, Side: {final_side}, Qty: {final_qty}")
        if trade.get("alpaca_verified"):
            print(f"    Status: Verified")
        elif trade.get("order_rejected") or trade.get("verification_failed"):
            print(f"    Status: REJECTED/FAILED")
        print()
else:
    print("  No crypto trades found in date range")

# Check for commodity trades (might be in same file or separate)
print("\n" + "=" * 80)
print("2. Checking for commodity trades")
print("-" * 80)

# Check if any trades are commodities
commodity_trades = []
for trade_time, trade in crypto_trades:
    asset_type = trade.get("asset_type", "")
    if asset_type == "commodities":
        commodity_trades.append((trade_time, trade))

if commodity_trades:
    print(f"Found {len(commodity_trades)} commodity trades in date range")
    print("\nCommodity Trades:")
    for trade_time, trade in sorted(commodity_trades, key=lambda x: x[0]):
        decision = trade.get("decision", "unknown")
        final_side = trade.get("final_side", "unknown")
        final_qty = trade.get("final_qty", 0)
        symbol = trade.get("trading_symbol", trade.get("asset", "unknown"))
        
        action = "BUY" if final_side == "long" and final_qty > 0 else "SELL" if final_side == "short" and final_qty > 0 else "HOLD/FLAT"
        
        print(f"  [{trade_time.strftime('%Y-%m-%d %H:%M:%S')}] {symbol}: {action}")
        print(f"    Decision: {decision}, Side: {final_side}, Qty: {final_qty}")
        print()
else:
    print("  No commodity trades found in date range")

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total trades found: {len(crypto_trades)}")
print(f"  - Crypto trades: {len(crypto_trades) - len(commodity_trades)}")
print(f"  - Commodity trades: {len(commodity_trades)}")

# Count buy vs sell
buy_count = 0
sell_count = 0
for trade_time, trade in crypto_trades:
    final_side = trade.get("final_side", "")
    final_qty = trade.get("final_qty", 0)
    if final_side == "long" and final_qty > 0:
        buy_count += 1
    elif final_side == "short" and final_qty > 0:
        sell_count += 1

print(f"\nBuy orders: {buy_count}")
print(f"Sell orders: {sell_count}")
print("=" * 80)
