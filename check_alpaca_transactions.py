"""Check Alpaca transaction history and compare with logs."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

print("=" * 80)
print("CHECKING ALPACA TRANSACTION HISTORY (JAN 17-19, 2026)")
print("=" * 80)

# Date range
start_date = datetime(2026, 1, 17, 0, 0, 0, tzinfo=timezone.utc)
end_date = datetime(2026, 1, 19, 23, 59, 59, tzinfo=timezone.utc)

print(f"\nDate Range: {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize Alpaca client
try:
    from trading.alpaca_client import AlpacaClient
    client = AlpacaClient()
    print("\n[OK] Alpaca client initialized")
except Exception as e:
    print(f"\n[ERROR] Failed to initialize Alpaca client: {e}")
    exit(1)

# Get orders from Alpaca
print("\n" + "=" * 80)
print("FETCHING ORDERS FROM ALPACA")
print("=" * 80)

try:
    # Alpaca API: GET /v2/orders?status=all&after={start}&until={end}
    # Note: Alpaca uses ISO format dates
    orders_url = f"{client.config.base_url}/orders"
    params = {
        "status": "all",  # all, open, closed, filled, etc.
        "after": start_date.isoformat(),
        "until": end_date.isoformat(),
        "limit": 500,  # Max orders to fetch
        "nested": "true"  # Include nested order details
    }
    
    response = client._session.get(orders_url, params=params, timeout=30)
    response.raise_for_status()
    alpaca_orders = response.json()
    
    print(f"\n[OK] Fetched {len(alpaca_orders)} orders from Alpaca")
    
except Exception as e:
    print(f"\n[ERROR] Failed to fetch orders: {e}")
    alpaca_orders = []

# Get positions from Alpaca
print("\n" + "=" * 80)
print("FETCHING CURRENT POSITIONS FROM ALPACA")
print("=" * 80)

try:
    positions = client.list_positions()
    print(f"\n[OK] Found {len(positions)} current positions")
    for pos in positions:
        symbol = pos.get("symbol", "unknown")
        qty = pos.get("qty", 0)
        side = "long" if float(qty) > 0 else "short"
        print(f"  {symbol}: {qty} ({side})")
except Exception as e:
    print(f"\n[ERROR] Failed to fetch positions: {e}")
    positions = []

# Analyze orders
print("\n" + "=" * 80)
print("ANALYZING ALPACA ORDERS")
print("=" * 80)

buy_orders = []
sell_orders = []

for order in alpaca_orders:
    symbol = order.get("symbol", "unknown")
    side = order.get("side", "").lower()  # buy or sell
    qty = float(order.get("qty", 0) or 0)
    status = order.get("status", "").lower()
    filled_qty = float(order.get("filled_qty", 0) or 0)
    filled_avg_price = order.get("filled_avg_price")
    created_at = order.get("created_at", "")
    filled_at = order.get("filled_at", "")
    
    # Parse timestamp
    try:
        if created_at:
            order_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        elif filled_at:
            order_time = datetime.fromisoformat(filled_at.replace("Z", "+00:00"))
        else:
            continue
    except:
        continue
    
    # Only count filled orders
    if status in ["filled", "partially_filled"] and filled_qty > 0:
        order_info = {
            "symbol": symbol,
            "side": side,
            "qty": filled_qty,
            "price": filled_avg_price,
            "status": status,
            "order_id": order.get("id"),
            "created_at": created_at,
            "filled_at": filled_at,
            "order_time": order_time
        }
        
        if side == "buy":
            buy_orders.append(order_info)
        elif side == "sell":
            sell_orders.append(order_info)

print(f"\nBuy orders (filled): {len(buy_orders)}")
print(f"Sell orders (filled): {len(sell_orders)}")

# Group by symbol
print("\n" + "=" * 80)
print("ORDERS BY SYMBOL")
print("=" * 80)

from collections import defaultdict
orders_by_symbol = defaultdict(lambda: {"buy": [], "sell": []})

for order in buy_orders + sell_orders:
    symbol = order["symbol"]
    side = order["side"]
    orders_by_symbol[symbol][side].append(order)

for symbol in sorted(orders_by_symbol.keys()):
    buys = orders_by_symbol[symbol]["buy"]
    sells = orders_by_symbol[symbol]["sell"]
    print(f"\n{symbol}:")
    print(f"  BUY: {len(buys)} orders")
    for b in buys:
        print(f"    [{b['order_time'].strftime('%Y-%m-%d %H:%M:%S')}] {b['qty']} @ ${b['price']} (ID: {b['order_id'][:20]}...)")
    print(f"  SELL: {len(sells)} orders")
    for s in sells:
        print(f"    [{s['order_time'].strftime('%Y-%m-%d %H:%M:%S')}] {s['qty']} @ ${s['price']} (ID: {s['order_id'][:20]}...)")

# Compare with logs
print("\n" + "=" * 80)
print("COMPARING WITH LOGS")
print("=" * 80)

# Read log file
log_file = Path("logs/trading/crypto_trades.jsonl")
logged_trades = []
if log_file.exists():
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trade = json.loads(line)
                logged_trades.append(trade)
            except:
                continue

print(f"\nTotal trades in log file: {len(logged_trades)}")

# Find missing SELL entries
print("\n" + "=" * 80)
print("MISSING SELL ENTRIES IN LOGS")
print("=" * 80)

missing_sells = []

for symbol in orders_by_symbol.keys():
    alpaca_buys = orders_by_symbol[symbol]["buy"]
    alpaca_sells = orders_by_symbol[symbol]["sell"]
    
    # Count logged buys and sells for this symbol
    logged_buys = [t for t in logged_trades 
                   if (t.get("trading_symbol", t.get("asset", "")) == symbol
                   and t.get("decision") in ["enter_long", "enter_short"]
                   and t.get("final_side") in ["long", "short"]
                   and t.get("final_qty", 0) > 0)]
    
    logged_sells = [t for t in logged_trades 
                    if (t.get("trading_symbol", t.get("asset", "")) == symbol
                    and (t.get("decision") in ["exit_long", "exit_short", "exit_position", "sell"]
                         or (t.get("trade_side") == "sell" and t.get("final_qty", 0) > 0)))]
    
    if len(alpaca_sells) > len(logged_sells):
        missing_count = len(alpaca_sells) - len(logged_sells)
        print(f"\n{symbol}:")
        print(f"  Alpaca SELL orders: {len(alpaca_sells)}")
        print(f"  Logged SELL entries: {len(logged_sells)}")
        print(f"  MISSING: {missing_count} SELL entries")
        
        # Find which sells are missing
        for sell_order in alpaca_sells:
            # Check if this sell is logged
            is_logged = False
            for logged in logged_sells:
                logged_time = logged.get("timestamp", "")
                try:
                    if logged_time:
                        if logged_time.endswith("Z"):
                            logged_time = logged_time[:-1] + "+00:00"
                        logged_dt = datetime.fromisoformat(logged_time)
                        # Check if within 5 minutes (orders might have slight time differences)
                        time_diff = abs((sell_order["order_time"] - logged_dt.replace(tzinfo=timezone.utc)).total_seconds())
                        if time_diff < 300:  # 5 minutes
                            is_logged = True
                            break
                except:
                    pass
            
            if not is_logged:
                missing_sells.append({
                    "symbol": symbol,
                    "order": sell_order,
                    "reason": "Not found in logs"
                })
                print(f"    MISSING: [{sell_order['order_time'].strftime('%Y-%m-%d %H:%M:%S')}] SELL {sell_order['qty']} @ ${sell_order['price']}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Alpaca BUY orders (filled): {len(buy_orders)}")
print(f"Alpaca SELL orders (filled): {len(sell_orders)}")
print(f"Missing SELL entries in logs: {len(missing_sells)}")
print(f"Current open positions: {len(positions)}")

if missing_sells:
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("Some SELL orders from Alpaca are not in the logs.")
    print("These may have been executed externally or the logging failed.")
    print("Consider backfilling these entries to maintain accurate trade history.")

print("=" * 80)
