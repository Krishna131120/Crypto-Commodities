"""Calculate accurate P&L by properly matching BUY and SELL orders."""
import json
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

print("=" * 80)
print("PROFIT & LOSS REPORT (JANUARY 17-19, 2026)")
print("=" * 80)

# Date range
start_date = datetime(2026, 1, 17, 0, 0, 0)
end_date = datetime(2026, 1, 19, 23, 59, 59)

print(f"\nDate Range: {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}")

# Read trade logs
log_file = Path("logs/trading/crypto_trades.jsonl")
all_trades = []

if log_file.exists():
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trade = json.loads(line)
                    timestamp_str = trade.get("timestamp", "")
                    if timestamp_str:
                        try:
                            if timestamp_str.endswith("Z"):
                                timestamp_str = timestamp_str[:-1] + "+00:00"
                            trade_time = datetime.fromisoformat(timestamp_str)
                            trade_time = trade_time.replace(tzinfo=None)
                            
                            if start_date <= trade_time <= end_date:
                                all_trades.append((trade_time, trade))
                        except:
                            pass
                except:
                    continue

print(f"\nTotal trades in date range: {len(all_trades)}")

# Get actual orders from Alpaca for accurate matching
print("\nFetching orders from Alpaca for accurate P&L calculation...")
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
    alpaca_orders = response.json()
    print(f"[OK] Fetched {len(alpaca_orders)} orders from Alpaca")
except Exception as e:
    print(f"[WARN] Could not fetch Alpaca orders: {e}")
    alpaca_orders = []

# Process Alpaca orders for accurate P&L
alpaca_trades = []
for order in alpaca_orders:
    side = order.get("side", "").lower()
    status = order.get("status", "").lower()
    filled_qty = float(order.get("filled_qty", 0) or 0)
    filled_avg_price = order.get("filled_avg_price")
    filled_at = order.get("filled_at", "")
    symbol = order.get("symbol", "").replace("/", "")
    
    if status in ["filled", "partially_filled"] and filled_qty > 0 and filled_avg_price:
        try:
            order_time = datetime.fromisoformat(filled_at.replace("Z", "+00:00"))
            order_time = order_time.replace(tzinfo=None)
            
            if start_date <= order_time <= end_date:
                alpaca_trades.append({
                    "symbol": symbol,
                    "side": side,
                    "time": order_time,
                    "qty": filled_qty,
                    "price": float(filled_avg_price),
                    "order_id": order.get("id", "")
                })
        except:
            pass

# Group by symbol
trades_by_symbol = defaultdict(lambda: {"buys": [], "sells": []})

for trade in alpaca_trades:
    symbol = trade["symbol"]
    if trade["side"] == "buy":
        trades_by_symbol[symbol]["buys"].append(trade)
    elif trade["side"] == "sell":
        trades_by_symbol[symbol]["sells"].append(trade)

# Sort by time
for symbol in trades_by_symbol:
    trades_by_symbol[symbol]["buys"].sort(key=lambda x: x["time"])
    trades_by_symbol[symbol]["sells"].sort(key=lambda x: x["time"])

# Calculate P&L using FIFO matching
print("\n" + "=" * 80)
print("CALCULATING REALIZED P&L (FIFO MATCHING)")
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
        
        # Ensure sell happens after buy
        if sell["time"] < buy["time"]:
            # This sell is from a position opened before our date range
            # Skip it or match with earlier buy if available
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

# Get current positions for unrealized P/L
print("\nFetching current positions...")
try:
    positions = client.list_positions()
    current_positions = []
    for pos in positions:
        symbol = pos.get("symbol", "").replace("/", "")
        qty = float(pos.get("qty", 0) or 0)
        avg_entry = float(pos.get("avg_entry_price", 0) or 0)
        current_price = float(pos.get("current_price", 0) or 0)
        unrealized_pl = float(pos.get("unrealized_pl", 0) or 0)
        
        if qty > 0 and avg_entry > 0:
            current_positions.append({
                "symbol": symbol,
                "quantity": qty,
                "entry_price": avg_entry,
                "current_price": current_price,
                "unrealized_pl": unrealized_pl,
                "unrealized_pl_pct": ((current_price - avg_entry) / avg_entry) * 100 if avg_entry > 0 else 0,
            })
    print(f"[OK] Found {len(current_positions)} open positions")
except Exception as e:
    print(f"[WARN] Could not fetch positions: {e}")
    current_positions = []

# Print comprehensive report
print("\n" + "=" * 80)
print("REALIZED P&L (CLOSED TRADES)")
print("=" * 80)

if realized_trades:
    total_realized = 0.0
    
    for i, trade in enumerate(sorted(realized_trades, key=lambda x: x["sell_time"]), 1):
        print(f"\n{i}. {trade['symbol']}")
        print(f"   Entry: [{trade['buy_time'].strftime('%Y-%m-%d %H:%M:%S')}] ${trade['buy_price']:.6f}")
        print(f"   Exit:  [{trade['sell_time'].strftime('%Y-%m-%d %H:%M:%S')}] ${trade['sell_price']:.6f}")
        print(f"   Quantity: {trade['quantity']:.6f}")
        print(f"   Realized P/L: ${trade['realized_pl']:+.2f} ({trade['realized_pl_pct']:+.2f}%)")
        
        total_realized += trade['realized_pl']
    
    print(f"\n{'='*80}")
    print(f"TOTAL REALIZED P/L: ${total_realized:+.2f}")
    print(f"{'='*80}")
else:
    print("\nNo closed trades found.")
    total_realized = 0.0

print("\n" + "=" * 80)
print("UNREALIZED P&L (OPEN POSITIONS)")
print("=" * 80)

if current_positions:
    total_unrealized = 0.0
    
    for i, pos in enumerate(sorted(current_positions, key=lambda x: x["symbol"]), 1):
        print(f"\n{i}. {pos['symbol']}")
        print(f"   Entry Price: ${pos['entry_price']:.6f}")
        print(f"   Current Price: ${pos['current_price']:.6f}")
        print(f"   Quantity: {pos['quantity']:.6f}")
        print(f"   Unrealized P/L: ${pos['unrealized_pl']:+.2f} ({pos['unrealized_pl_pct']:+.2f}%)")
        
        total_unrealized += pos['unrealized_pl']
    
    print(f"\n{'='*80}")
    print(f"TOTAL UNREALIZED P/L: ${total_unrealized:+.2f}")
    print(f"{'='*80}")
else:
    print("\nNo open positions.")
    total_unrealized = 0.0

# Overall summary
print("\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)
print(f"\nRealized P/L (Closed Trades): ${total_realized:+.2f}")
print(f"Unrealized P/L (Open Positions): ${total_unrealized:+.2f}")
print(f"{'='*80}")
print(f"TOTAL P&L: ${total_realized + total_unrealized:+.2f}")
print(f"{'='*80}")

# Statistics
if realized_trades:
    profitable = [t for t in realized_trades if t['realized_pl'] > 0]
    losing = [t for t in realized_trades if t['realized_pl'] < 0]
    
    print(f"\nTrade Statistics:")
    print(f"  Total Closed Trades: {len(realized_trades)}")
    print(f"  Profitable: {len(profitable)} (${sum(t['realized_pl'] for t in profitable):+.2f})")
    print(f"  Losing: {len(losing)} (${sum(t['realized_pl'] for t in losing):+.2f})")
    if realized_trades:
        win_rate = (len(profitable) / len(realized_trades)) * 100
        print(f"  Win Rate: {win_rate:.1f}%")

print("=" * 80)
