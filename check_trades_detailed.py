"""Check for buy/sell logs between January 17-19 with detailed analysis."""
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

print("=" * 80)
print("CHECKING TRADES BETWEEN JANUARY 17-19, 2026")
print("=" * 80)

# Date range
start_date = datetime(2026, 1, 17, 0, 0, 0)
end_date = datetime(2026, 1, 19, 23, 59, 59)

print(f"\nDate Range: {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}")

# Check crypto trades
crypto_log = Path("logs/trading/crypto_trades.jsonl")
print(f"\n1. Reading crypto trades from: {crypto_log}")
print("-" * 80)

crypto_trades = []
all_trades = []

if crypto_log.exists():
    with open(crypto_log, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                trade = json.loads(line)
                all_trades.append(trade)
                
                # Parse timestamp
                timestamp_str = trade.get("timestamp", "")
                if not timestamp_str:
                    continue
                    
                # Handle different timestamp formats
                trade_time = None
                try:
                    # Try ISO format with Z (UTC)
                    if timestamp_str.endswith("Z"):
                        # Replace Z with +00:00 for fromisoformat
                        timestamp_str_clean = timestamp_str[:-1] + "+00:00"
                        trade_time = datetime.fromisoformat(timestamp_str_clean)
                        # Convert to naive datetime for comparison (since start_date/end_date are naive)
                        trade_time = trade_time.replace(tzinfo=None)
                    elif "+" in timestamp_str:
                        # Has timezone info
                        trade_time = datetime.fromisoformat(timestamp_str)
                        trade_time = trade_time.replace(tzinfo=None)
                    elif timestamp_str.count("-") >= 3 and "T" in timestamp_str:
                        # ISO format without timezone
                        trade_time = datetime.fromisoformat(timestamp_str)
                    else:
                        # Try simple format
                        trade_time = datetime.strptime(timestamp_str.split(".")[0], "%Y-%m-%dT%H:%M:%S")
                except Exception as e:
                    # Try alternative parsing
                    try:
                        parts = timestamp_str.split("T")
                        if len(parts) >= 1:
                            date_part = parts[0]
                            time_part = parts[1].split(".")[0].split("Z")[0] if len(parts) > 1 else "00:00:00"
                            trade_time = datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S")
                    except:
                        continue
                
                if trade_time:
                    # Check if in range
                    if start_date <= trade_time <= end_date:
                        crypto_trades.append((trade_time, trade))
            except json.JSONDecodeError as e:
                continue
            except Exception as e:
                continue

print(f"Total trades in file: {len(all_trades)}")
print(f"Trades in date range (Jan 17-19): {len(crypto_trades)}")

# Show recent trades to verify date parsing
if all_trades:
    print(f"\nLast 5 trades in file (to verify date format):")
    for trade in all_trades[-5:]:
        ts = trade.get("timestamp", "N/A")
        symbol = trade.get("trading_symbol", trade.get("asset", "unknown"))
        decision = trade.get("decision", "unknown")
        side = trade.get("final_side", "unknown")
        qty = trade.get("final_qty", 0)
        print(f"  {ts[:25]} | {symbol:15s} | {decision:15s} | {side:10s} | Qty: {qty}")

if crypto_trades:
    print("\n" + "=" * 80)
    print("TRADES FOUND IN DATE RANGE (JAN 17-19, 2026)")
    print("=" * 80)
    
    # Group by type
    buy_trades = []
    sell_trades = []
    other_trades = []
    
    for trade_time, trade in sorted(crypto_trades, key=lambda x: x[0]):
        decision = trade.get("decision", "unknown")
        final_side = trade.get("final_side", "unknown")
        final_qty = trade.get("final_qty", 0)
        symbol = trade.get("trading_symbol", trade.get("asset", "unknown"))
        asset_type = trade.get("asset_type", "unknown")
        timestamp = trade.get("timestamp", "")
        
        # Determine action
        # Check decision first (exit_long = SELL, exit_short = BUY to cover)
        decision = trade.get("decision", "")
        trade_side = trade.get("trade_side", "")
        existing_side = trade.get("existing_side", "")
        
        # Check if this is an exit (old format: "exit_position", new format: "exit_long"/"exit_short")
        is_exit = decision in ["exit_position", "exit_long", "exit_short", "would_exit_position"]
        
        if is_exit:
            # This is an exit - determine if it's SELL (closing long) or BUY (closing short)
            if existing_side == "long" or trade_side == "sell" or decision == "exit_long":
                action = "SELL"
                sell_trades.append((trade_time, trade))
            elif existing_side == "short" or trade_side == "buy" or decision == "exit_short":
                action = "BUY"  # BUY to cover short
                buy_trades.append((trade_time, trade))
            else:
                # Fallback: check trade_side
                if trade_side == "sell":
                    action = "SELL"
                    sell_trades.append((trade_time, trade))
                elif trade_side == "buy":
                    action = "BUY"
                    buy_trades.append((trade_time, trade))
                else:
                    action = "EXIT (unknown side)"
                    other_trades.append((trade_time, trade))
        elif decision in ["exit_long", "sell"] or (final_side == "short" and final_qty > 0):
            action = "SELL"
            sell_trades.append((trade_time, trade))
        elif decision in ["exit_short", "buy"] or (final_side == "long" and final_qty > 0):
            action = "BUY"
            buy_trades.append((trade_time, trade))
        elif final_side == "long" and final_qty > 0:
            action = "BUY"
            buy_trades.append((trade_time, trade))
        elif final_side == "short" and final_qty > 0:
            action = "SELL"
            sell_trades.append((trade_time, trade))
        else:
            action = "HOLD/FLAT/SKIP"
            other_trades.append((trade_time, trade))
        
        status = "VERIFIED" if trade.get("alpaca_verified") else "REJECTED" if trade.get("order_rejected") or trade.get("verification_failed") else "UNKNOWN"
        
        print(f"\n[{trade_time.strftime('%Y-%m-%d %H:%M:%S')}] {symbol} ({asset_type})")
        print(f"  Action: {action}")
        print(f"  Decision: {decision}")
        print(f"  Side: {final_side}, Quantity: {final_qty}")
        print(f"  Status: {status}")
        if trade.get("entry_price"):
            print(f"  Entry Price: {trade.get('entry_price')}")
        if trade.get("exit_price"):
            print(f"  Exit Price: {trade.get('exit_price')}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total trades in range: {len(crypto_trades)}")
    print(f"  - BUY orders: {len(buy_trades)}")
    print(f"  - SELL orders: {len(sell_trades)}")
    print(f"  - Other (HOLD/FLAT/SKIP): {len(other_trades)}")
    
    # Count by asset type
    by_type = defaultdict(int)
    for trade_time, trade in crypto_trades:
        asset_type = trade.get("asset_type", "unknown")
        by_type[asset_type] += 1
    
    print(f"\nBy asset type:")
    for asset_type, count in sorted(by_type.items()):
        print(f"  - {asset_type}: {count}")
    
    # Count verified vs rejected
    verified = sum(1 for _, t in crypto_trades if t.get("alpaca_verified"))
    rejected = sum(1 for _, t in crypto_trades if t.get("order_rejected") or t.get("verification_failed"))
    print(f"\nVerification status:")
    print(f"  - Verified: {verified}")
    print(f"  - Rejected/Failed: {rejected}")
    print(f"  - Unknown: {len(crypto_trades) - verified - rejected}")
    
else:
    print("\n" + "=" * 80)
    print("NO TRADES FOUND IN DATE RANGE")
    print("=" * 80)
    print("No buy or sell orders were logged between January 17-19, 2026.")
    print("\nPossible reasons:")
    print("  1. Bot was not running during this period")
    print("  2. No trading signals were generated")
    print("  3. All trades were filtered out (risk limits, cooldowns, etc.)")
    print("  4. Date format in logs might be different")

print("=" * 80)
