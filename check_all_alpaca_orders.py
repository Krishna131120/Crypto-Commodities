"""Check ALL Alpaca orders to understand BUY/SELL balance."""
import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict

print("=" * 80)
print("CHECKING ALL ALPACA ORDERS (INCLUDING BEFORE JAN 17)")
print("=" * 80)

# Initialize Alpaca client
try:
    from trading.alpaca_client import AlpacaClient
    client = AlpacaClient()
    print("\n[OK] Alpaca client initialized")
except Exception as e:
    print(f"\n[ERROR] Failed to initialize: {e}")
    exit(1)

# Get orders from a wider date range (last 30 days to catch positions opened before Jan 17)
end_date = datetime(2026, 1, 19, 23, 59, 59, tzinfo=timezone.utc)
start_date = end_date - timedelta(days=30)  # Go back 30 days

print(f"\nDate Range: {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}")

# Get orders from Alpaca
print("\nFetching orders from Alpaca...")
try:
    orders_url = f"{client.config.base_url}/orders"
    params = {
        "status": "all",
        "after": start_date.isoformat(),
        "until": end_date.isoformat(),
        "limit": 500,
        "nested": "true"
    }
    response = client._session.get(orders_url, params=params, timeout=30)
    response.raise_for_status()
    all_orders = response.json()
    print(f"[OK] Fetched {len(all_orders)} total orders from Alpaca")
except Exception as e:
    print(f"[ERROR] Failed to fetch orders: {e}")
    exit(1)

# Filter orders in Jan 17-19 range
jan17 = datetime(2026, 1, 17, 0, 0, 0, tzinfo=timezone.utc)
jan19 = datetime(2026, 1, 19, 23, 59, 59, tzinfo=timezone.utc)

orders_in_range = []
orders_before_range = []

for order in all_orders:
    filled_at = order.get("filled_at") or order.get("created_at", "")
    if not filled_at:
        continue
    try:
        order_time = datetime.fromisoformat(filled_at.replace("Z", "+00:00"))
        if jan17 <= order_time <= jan19:
            orders_in_range.append(order)
        elif order_time < jan17:
            orders_before_range.append(order)
    except:
        continue

print(f"\nOrders in Jan 17-19 range: {len(orders_in_range)}")
print(f"Orders before Jan 17: {len(orders_before_range)}")

# Analyze orders in Jan 17-19 range
print("\n" + "=" * 80)
print("ORDERS IN JAN 17-19 RANGE")
print("=" * 80)

buy_orders_range = []
sell_orders_range = []

for order in orders_in_range:
    side = order.get("side", "").lower()
    status = order.get("status", "").lower()
    filled_qty = float(order.get("filled_qty", 0) or 0)
    
    if status in ["filled", "partially_filled"] and filled_qty > 0:
        if side == "buy":
            buy_orders_range.append(order)
        elif side == "sell":
            sell_orders_range.append(order)

print(f"BUY orders (filled): {len(buy_orders_range)}")
print(f"SELL orders (filled): {len(sell_orders_range)}")

# Analyze orders before Jan 17
print("\n" + "=" * 80)
print("ORDERS BEFORE JAN 17 (that might explain extra SELLs)")
print("=" * 80)

buy_orders_before = []
sell_orders_before = []

for order in orders_before_range:
    side = order.get("side", "").lower()
    status = order.get("status", "").lower()
    filled_qty = float(order.get("filled_qty", 0) or 0)
    
    if status in ["filled", "partially_filled"] and filled_qty > 0:
        if side == "buy":
            buy_orders_before.append(order)
        elif side == "sell":
            sell_orders_before.append(order)

print(f"BUY orders before Jan 17: {len(buy_orders_before)}")
print(f"SELL orders before Jan 17: {len(sell_orders_before)}")

# Group by symbol to understand the flow
print("\n" + "=" * 80)
print("SYMBOL-BY-SYMBOL ANALYSIS")
print("=" * 80)

orders_by_symbol = defaultdict(lambda: {"buy_before": [], "sell_before": [], "buy_range": [], "sell_range": []})

# Process all orders
for order in all_orders:
    symbol = order.get("symbol", "unknown")
    side = order.get("side", "").lower()
    status = order.get("status", "").lower()
    filled_qty = float(order.get("filled_qty", 0) or 0)
    filled_at = order.get("filled_at") or order.get("created_at", "")
    
    if status not in ["filled", "partially_filled"] or filled_qty <= 0 or not filled_at:
        continue
    
    try:
        order_time = datetime.fromisoformat(filled_at.replace("Z", "+00:00"))
        if order_time < jan17:
            if side == "buy":
                orders_by_symbol[symbol]["buy_before"].append(order)
            elif side == "sell":
                orders_by_symbol[symbol]["sell_before"].append(order)
        elif jan17 <= order_time <= jan19:
            if side == "buy":
                orders_by_symbol[symbol]["buy_range"].append(order)
            elif side == "sell":
                orders_by_symbol[symbol]["sell_range"].append(order)
    except:
        continue

# Show symbols with SELLs in range but BUYs before range
print("\nSymbols with SELLs in Jan 17-19 that closed positions opened BEFORE Jan 17:")
print("-" * 80)

for symbol in sorted(orders_by_symbol.keys()):
    buys_before = orders_by_symbol[symbol]["buy_before"]
    sells_before = orders_by_symbol[symbol]["sell_before"]
    buys_range = orders_by_symbol[symbol]["buy_range"]
    sells_range = orders_by_symbol[symbol]["sell_range"]
    
    if len(sells_range) > 0 and len(buys_before) > 0:
        print(f"\n{symbol}:")
        print(f"  BUY orders BEFORE Jan 17: {len(buys_before)}")
        for b in buys_before:
            time_str = (b.get("filled_at") or b.get("created_at", ""))[:19]
            print(f"    [{time_str}] BUY {b.get('filled_qty', 0)} @ ${b.get('filled_avg_price', 'N/A')}")
        
        print(f"  SELL orders in Jan 17-19: {len(sells_range)}")
        for s in sells_range:
            time_str = (s.get("filled_at") or s.get("created_at", ""))[:19]
            print(f"    [{time_str}] SELL {s.get('filled_qty', 0)} @ ${s.get('filled_avg_price', 'N/A')}")
        
        print(f"  BUY orders in Jan 17-19: {len(buys_range)}")
        for b in buys_range:
            time_str = (b.get("filled_at") or b.get("created_at", ""))[:19]
            print(f"    [{time_str}] BUY {b.get('filled_qty', 0)} @ ${b.get('filled_avg_price', 'N/A')}")

# Final summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"BUY orders in Jan 17-19: {len(buy_orders_range)}")
print(f"SELL orders in Jan 17-19: {len(sell_orders_range)}")
print(f"Difference: {len(sell_orders_range) - len(buy_orders_range)} extra SELLs")
print(f"\nBUY orders before Jan 17: {len(buy_orders_before)}")
print(f"SELL orders before Jan 17: {len(sell_orders_before)}")

print("\n[EXPLANATION]")
print(f"The {len(sell_orders_range) - len(buy_orders_range)} extra SELL orders in Jan 17-19")
print("are closing positions that were opened BEFORE Jan 17.")
print("These are legitimate exits and should be logged.")
print("=" * 80)
