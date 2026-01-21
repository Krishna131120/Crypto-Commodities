"""Final comprehensive P&L report for January 17-19, 2026."""
import json
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

print("=" * 80)
print("FINAL PROFIT & LOSS REPORT")
print("JANUARY 17-19, 2026")
print("=" * 80)

# Date range
start_date = datetime(2026, 1, 17, 0, 0, 0)
end_date = datetime(2026, 1, 19, 23, 59, 59)

print(f"\nReport Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Fetch orders from Alpaca
print("\nFetching transaction data from Alpaca...")
try:
    from trading.alpaca_client import AlpacaClient
    client = AlpacaClient()
    
    orders_url = f"{client.config.base_url}/orders"
    params = {
        "status": "all",
        "after": datetime(2026, 1, 17, 0, 0, 0, tzinfo=timezone.utc).isoformat(),
        "until": datetime(2026, 1, 19, 23, 59, 59, tzinfo=timezone.utc).isoformat(),
        "limit": 500,
        "nested": "true"
    }
    response = client._session.get(orders_url, params=params, timeout=30)
    response.raise_for_status()
    all_orders = response.json()
    
    # Filter filled orders
    filled_orders = []
    for order in all_orders:
        status = order.get("status", "").lower()
        filled_qty = float(order.get("filled_qty", 0) or 0)
        filled_avg_price = order.get("filled_avg_price")
        
        if status in ["filled", "partially_filled"] and filled_qty > 0 and filled_avg_price:
            filled_at = order.get("filled_at", "")
            if filled_at:
                try:
                    order_time = datetime.fromisoformat(filled_at.replace("Z", "+00:00"))
                    order_time = order_time.replace(tzinfo=None)
                    
                    if start_date <= order_time <= end_date:
                        filled_orders.append({
                            "symbol": order.get("symbol", "").replace("/", ""),
                            "side": order.get("side", "").lower(),
                            "time": order_time,
                            "qty": filled_qty,
                            "price": float(filled_avg_price),
                            "order_id": order.get("id", "")
                        })
                except:
                    pass
    
    print(f"[OK] Found {len(filled_orders)} filled orders")
    
    # Get current positions
    positions = client.list_positions()
    print(f"[OK] Found {len(positions)} current positions")
    
except Exception as e:
    print(f"[ERROR] Failed to fetch data: {e}")
    exit(1)

# Group orders by symbol
trades_by_symbol = defaultdict(lambda: {"buys": [], "sells": []})

for order in filled_orders:
    symbol = order["symbol"]
    if order["side"] == "buy":
        trades_by_symbol[symbol]["buys"].append(order)
    elif order["side"] == "sell":
        trades_by_symbol[symbol]["sells"].append(order)

# Sort by time
for symbol in trades_by_symbol:
    trades_by_symbol[symbol]["buys"].sort(key=lambda x: x["time"])
    trades_by_symbol[symbol]["sells"].sort(key=lambda x: x["time"])

# Match BUY and SELL using FIFO
print("\n" + "=" * 80)
print("REALIZED P&L (CLOSED TRADES)")
print("=" * 80)

realized_trades = []

for symbol in sorted(trades_by_symbol.keys()):
    buys = trades_by_symbol[symbol]["buys"]
    sells = trades_by_symbol[symbol]["sells"]
    
    if not buys or not sells:
        continue
    
    # FIFO matching
    buy_queue = buys.copy()
    sell_queue = sells.copy()
    
    while buy_queue and sell_queue:
        buy = buy_queue[0]
        sell = sell_queue[0]
        
        # Only match if sell happens after buy
        if sell["time"] < buy["time"]:
            # This sell closes a position from before our date range
            # Skip it for this report
            sell_queue.pop(0)
            continue
        
        matched_qty = min(buy["qty"], sell["qty"])
        realized_pl = (sell["price"] - buy["price"]) * matched_qty
        realized_pl_pct = ((sell["price"] - buy["price"]) / buy["price"]) * 100 if buy["price"] > 0 else 0
        
        realized_trades.append({
            "symbol": symbol,
            "buy_time": buy["time"],
            "sell_time": sell["time"],
            "buy_price": buy["price"],
            "sell_price": sell["price"],
            "quantity": matched_qty,
            "realized_pl": realized_pl,
            "realized_pl_pct": realized_pl_pct,
            "hold_duration": (sell["time"] - buy["time"]).total_seconds() / 3600,  # hours
        })
        
        # Update quantities
        if buy["qty"] > matched_qty:
            buy["qty"] -= matched_qty
        else:
            buy_queue.pop(0)
        
        if sell["qty"] > matched_qty:
            sell["qty"] -= matched_qty
        else:
            sell_queue.pop(0)

# Print realized trades
if realized_trades:
    total_realized = 0.0
    profitable_count = 0
    losing_count = 0
    
    print("\nClosed Trades (Sorted by Exit Time):")
    print("-" * 80)
    
    for i, trade in enumerate(sorted(realized_trades, key=lambda x: x["sell_time"]), 1):
        status = "PROFIT" if trade['realized_pl'] > 0 else "LOSS"
        if trade['realized_pl'] > 0:
            profitable_count += 1
        else:
            losing_count += 1
        
        print(f"\n{i}. {trade['symbol']} - {status}")
        print(f"   Entry:  [{trade['buy_time'].strftime('%Y-%m-%d %H:%M:%S')}] ${trade['buy_price']:.6f}")
        print(f"   Exit:   [{trade['sell_time'].strftime('%Y-%m-%d %H:%M:%S')}] ${trade['sell_price']:.6f}")
        print(f"   Quantity: {trade['quantity']:.6f}")
        print(f"   Hold Duration: {trade['hold_duration']:.2f} hours")
        print(f"   Realized P/L: ${trade['realized_pl']:+.2f} ({trade['realized_pl_pct']:+.2f}%)")
        
        total_realized += trade['realized_pl']
    
    print(f"\n{'='*80}")
    print(f"TOTAL REALIZED P/L: ${total_realized:+.2f}")
    print(f"Profitable Trades: {profitable_count} | Losing Trades: {losing_count}")
    print(f"{'='*80}")
else:
    print("\nNo closed trades in date range.")
    total_realized = 0.0

# Unrealized P/L
print("\n" + "=" * 80)
print("UNREALIZED P&L (OPEN POSITIONS)")
print("=" * 80)

if positions:
    total_unrealized = 0.0
    
    print("\nOpen Positions:")
    print("-" * 80)
    
    for i, pos in enumerate(sorted(positions, key=lambda x: x.get("symbol", "")), 1):
        symbol = pos.get("symbol", "").replace("/", "")
        qty = float(pos.get("qty", 0) or 0)
        avg_entry = float(pos.get("avg_entry_price", 0) or 0)
        current_price = float(pos.get("current_price", 0) or 0)
        unrealized_pl = float(pos.get("unrealized_pl", 0) or 0)
        unrealized_pl_pct = ((current_price - avg_entry) / avg_entry) * 100 if avg_entry > 0 else 0
        
        status = "PROFIT" if unrealized_pl > 0 else "LOSS"
        
        print(f"\n{i}. {symbol} - {status}")
        print(f"   Entry Price: ${avg_entry:.6f}")
        print(f"   Current Price: ${current_price:.6f}")
        print(f"   Quantity: {qty:.6f}")
        print(f"   Unrealized P/L: ${unrealized_pl:+.2f} ({unrealized_pl_pct:+.2f}%)")
        
        total_unrealized += unrealized_pl
    
    print(f"\n{'='*80}")
    print(f"TOTAL UNREALIZED P/L: ${total_unrealized:+.2f}")
    print(f"{'='*80}")
else:
    print("\nNo open positions.")
    total_unrealized = 0.0

# Final summary
print("\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)

total_pnl = total_realized + total_unrealized

print(f"\nRealized P/L (Closed Trades):     ${total_realized:+,.2f}")
print(f"Unrealized P/L (Open Positions):  ${total_unrealized:+,.2f}")
print(f"{'='*80}")
print(f"TOTAL P&L:                        ${total_pnl:+,.2f}")
print(f"{'='*80}")

# Detailed statistics
if realized_trades:
    profitable = [t for t in realized_trades if t['realized_pl'] > 0]
    losing = [t for t in realized_trades if t['realized_pl'] < 0]
    
    print(f"\nTrade Statistics:")
    print(f"  Total Closed Trades: {len(realized_trades)}")
    print(f"  Profitable: {len(profitable)} trades (${sum(t['realized_pl'] for t in profitable):+,.2f})")
    print(f"  Losing: {len(losing)} trades (${sum(t['realized_pl'] for t in losing):+,.2f})")
    
    if profitable:
        avg_profit = sum(t['realized_pl'] for t in profitable) / len(profitable)
        max_profit = max(t['realized_pl'] for t in profitable)
        print(f"  Average Profit: ${avg_profit:+,.2f}")
        print(f"  Best Trade: ${max_profit:+,.2f}")
    
    if losing:
        avg_loss = sum(t['realized_pl'] for t in losing) / len(losing)
        max_loss = min(t['realized_pl'] for t in losing)
        print(f"  Average Loss: ${avg_loss:+,.2f}")
        print(f"  Worst Trade: ${max_loss:+,.2f}")
    
    if realized_trades:
        win_rate = (len(profitable) / len(realized_trades)) * 100
        print(f"  Win Rate: {win_rate:.1f}%")
        
        # Risk-reward ratio
        if profitable and losing:
            avg_profit = sum(t['realized_pl'] for t in profitable) / len(profitable)
            avg_loss = abs(sum(t['realized_pl'] for t in losing) / len(losing))
            if avg_loss > 0:
                risk_reward = avg_profit / avg_loss
                print(f"  Risk-Reward Ratio: {risk_reward:.2f}")

print("\n" + "=" * 80)
print("END OF REPORT")
print("=" * 80)
