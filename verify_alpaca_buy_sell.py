"""Verify BUY and SELL orders from Alpaca between Jan 17-19."""
import json
from datetime import datetime, timezone
from collections import defaultdict

print("=" * 80)
print("VERIFYING ALPACA BUY AND SELL ORDERS (JAN 17-19, 2026)")
print("=" * 80)

# Initialize Alpaca client
try:
    from trading.alpaca_client import AlpacaClient
    client = AlpacaClient()
    print("\n[OK] Alpaca client initialized")
except Exception as e:
    print(f"\n[ERROR] Failed to initialize: {e}")
    exit(1)

# Date range
start_date = datetime(2026, 1, 17, 0, 0, 0, tzinfo=timezone.utc)
end_date = datetime(2026, 1, 19, 23, 59, 59, tzinfo=timezone.utc)

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
    alpaca_orders = response.json()
    print(f"[OK] Fetched {len(alpaca_orders)} total orders from Alpaca")
except Exception as e:
    print(f"[ERROR] Failed to fetch orders: {e}")
    exit(1)

# Analyze all orders
print("\n" + "=" * 80)
print("ANALYZING ALL ORDERS")
print("=" * 80)

all_buy_orders = []
all_sell_orders = []
orders_by_symbol = defaultdict(lambda: {"buy": [], "sell": []})

for order in alpaca_orders:
    symbol = order.get("symbol", "unknown")
    side = order.get("side", "").lower()
    status = order.get("status", "").lower()
    qty = float(order.get("qty", 0) or 0)
    filled_qty = float(order.get("filled_qty", 0) or 0)
    filled_avg_price = order.get("filled_avg_price")
    created_at = order.get("created_at", "")
    filled_at = order.get("filled_at", "")
    order_id = order.get("id", "")
    
    # Parse timestamp
    order_time = None
    try:
        if filled_at:
            order_time = datetime.fromisoformat(filled_at.replace("Z", "+00:00"))
        elif created_at:
            order_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except:
        continue
    
    if not order_time:
        continue
    
    # Only count filled orders
    if status in ["filled", "partially_filled"] and filled_qty > 0:
        order_info = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "filled_qty": filled_qty,
            "price": filled_avg_price,
            "status": status,
            "order_id": order_id,
            "created_at": created_at,
            "filled_at": filled_at,
            "order_time": order_time
        }
        
        if side == "buy":
            all_buy_orders.append(order_info)
            orders_by_symbol[symbol]["buy"].append(order_info)
        elif side == "sell":
            all_sell_orders.append(order_info)
            orders_by_symbol[symbol]["sell"].append(order_info)

print(f"\nTotal BUY orders (filled): {len(all_buy_orders)}")
print(f"Total SELL orders (filled): {len(all_sell_orders)}")

# Detailed breakdown by symbol
print("\n" + "=" * 80)
print("DETAILED BREAKDOWN BY SYMBOL")
print("=" * 80)

for symbol in sorted(orders_by_symbol.keys()):
    buys = orders_by_symbol[symbol]["buy"]
    sells = orders_by_symbol[symbol]["sell"]
    
    print(f"\n{symbol}:")
    print(f"  BUY orders: {len(buys)}")
    for i, b in enumerate(buys, 1):
        print(f"    {i}. [{b['order_time'].strftime('%Y-%m-%d %H:%M:%S')}] BUY {b['filled_qty']} @ ${b['price']}")
        print(f"       Order ID: {b['order_id'][:30]}...")
    
    print(f"  SELL orders: {len(sells)}")
    for i, s in enumerate(sells, 1):
        print(f"    {i}. [{s['order_time'].strftime('%Y-%m-%d %H:%M:%S')}] SELL {s['filled_qty']} @ ${s['price']}")
        print(f"       Order ID: {s['order_id'][:30]}...")
    
    # Check balance
    total_buy_qty = sum(b['filled_qty'] for b in buys)
    total_sell_qty = sum(s['filled_qty'] for s in sells)
    net_qty = total_buy_qty - total_sell_qty
    
    print(f"  Balance: BUY {total_buy_qty:.6f} - SELL {total_sell_qty:.6f} = NET {net_qty:.6f}")
    if net_qty > 0:
        print(f"    -> Still holding {net_qty:.6f} (LONG position)")
    elif net_qty < 0:
        print(f"    -> Short position: {abs(net_qty):.6f}")
    else:
        print(f"    -> Position closed (flat)")

# Check current positions
print("\n" + "=" * 80)
print("CURRENT POSITIONS IN ALPACA")
print("=" * 80)

try:
    positions = client.list_positions()
    print(f"\nCurrent open positions: {len(positions)}")
    for pos in positions:
        symbol = pos.get("symbol", "unknown")
        qty = float(pos.get("qty", 0) or 0)
        side = "long" if qty > 0 else "short"
        avg_entry = pos.get("avg_entry_price", "N/A")
        current_price = pos.get("current_price", "N/A")
        unrealized_pl = pos.get("unrealized_pl", "N/A")
        print(f"  {symbol}: {qty} ({side}) | Entry: ${avg_entry} | Current: ${current_price} | P/L: ${unrealized_pl}")
except Exception as e:
    print(f"[ERROR] Failed to fetch positions: {e}")

# Summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"Alpaca BUY orders (filled): {len(all_buy_orders)}")
print(f"Alpaca SELL orders (filled): {len(all_sell_orders)}")
print(f"Difference: {len(all_sell_orders) - len(all_buy_orders)}")

if len(all_sell_orders) > len(all_buy_orders):
    print(f"\n[NOTE] More SELL orders than BUY orders.")
    print("This could mean:")
    print("  1. Some positions were opened before Jan 17")
    print("  2. Multiple SELL orders for same position (partial exits)")
    print("  3. Some BUY orders were rejected/failed")
elif len(all_buy_orders) > len(all_sell_orders):
    print(f"\n[NOTE] More BUY orders than SELL orders.")
    print("This means some positions are still open.")
else:
    print(f"\n[OK] BUY and SELL orders match - all positions closed.")

print("=" * 80)
