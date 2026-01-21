"""Calculate comprehensive P&L for trades between Jan 17-19."""
import json
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

print("=" * 80)
print("PROFIT & LOSS REPORT (JANUARY 17-19, 2026)")
print("=" * 80)

# Date range
start_date = datetime(2026, 1, 17, 0, 0, 0)
end_date = datetime(2026, 1, 19, 23, 59, 59)

print(f"\nDate Range: {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}")

# Read trade logs
log_file = Path("logs/trading/crypto_trades.jsonl")
trades = []

if log_file.exists():
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trade = json.loads(line)
                    # Parse timestamp
                    timestamp_str = trade.get("timestamp", "")
                    if timestamp_str:
                        try:
                            if timestamp_str.endswith("Z"):
                                timestamp_str = timestamp_str[:-1] + "+00:00"
                            trade_time = datetime.fromisoformat(timestamp_str)
                            trade_time = trade_time.replace(tzinfo=None)
                            
                            if start_date <= trade_time <= end_date:
                                trades.append((trade_time, trade))
                        except:
                            pass
                except:
                    continue

print(f"\nTotal trades found in date range: {len(trades)}")

# Separate BUY and SELL trades
buy_trades = []
sell_trades = []

for trade_time, trade in trades:
    decision = trade.get("decision", "")
    final_side = trade.get("final_side", "")
    final_qty = trade.get("final_qty", 0)
    trade_side = trade.get("trade_side", "")
    existing_side = trade.get("existing_side", "")
    
    # Determine if BUY or SELL
    is_buy = False
    is_sell = False
    
    if decision in ["enter_long", "enter_short"]:
        if final_side in ["long", "short"] and final_qty > 0:
            is_buy = True
    elif decision in ["exit_long", "exit_short", "exit_position"]:
        if trade_side == "sell" or (final_side == "short" and final_qty > 0):
            is_sell = True
        elif trade_side == "buy" or (final_side == "long" and final_qty > 0):
            is_buy = True
    
    # Also check by trade_side
    if trade_side == "buy" and final_qty > 0:
        is_buy = True
    elif trade_side == "sell" and final_qty > 0:
        is_sell = True
    
    if is_buy:
        buy_trades.append((trade_time, trade))
    elif is_sell:
        sell_trades.append((trade_time, trade))

print(f"BUY trades: {len(buy_trades)}")
print(f"SELL trades: {len(sell_trades)}")

# Match BUY and SELL trades by symbol
print("\n" + "=" * 80)
print("MATCHING TRADES AND CALCULATING P&L")
print("=" * 80)

# Group by symbol
trades_by_symbol = defaultdict(lambda: {"buys": [], "sells": []})

for trade_time, trade in buy_trades:
    symbol = trade.get("trading_symbol", trade.get("asset", "unknown"))
    trades_by_symbol[symbol]["buys"].append((trade_time, trade))

for trade_time, trade in sell_trades:
    symbol = trade.get("trading_symbol", trade.get("asset", "unknown"))
    trades_by_symbol[symbol]["sells"].append((trade_time, trade))

# Calculate P&L for each symbol
realized_trades = []
unrealized_positions = []

for symbol in sorted(trades_by_symbol.keys()):
    buys = sorted(trades_by_symbol[symbol]["buys"], key=lambda x: x[0])
    sells = sorted(trades_by_symbol[symbol]["sells"], key=lambda x: x[0])
    
    print(f"\n{symbol}:")
    print(f"  BUY orders: {len(buys)}")
    print(f"  SELL orders: {len(sells)}")
    
    # Match buys and sells (FIFO - First In First Out)
    buy_queue = buys.copy()
    sell_queue = sells.copy()
    
    while buy_queue and sell_queue:
        buy_time, buy_trade = buy_queue[0]
        sell_time, sell_trade = sell_queue[0]
        
        # Get quantities and prices
        buy_qty = float(buy_trade.get("final_qty", buy_trade.get("trade_qty", 0)) or 0)
        sell_qty = float(sell_trade.get("final_qty", sell_trade.get("trade_qty", 0)) or 0)
        buy_price = float(buy_trade.get("entry_price", buy_trade.get("current_price", 0)) or 0)
        sell_price = float(sell_trade.get("exit_price", sell_trade.get("current_price", 0)) or 0)
        
        if buy_qty <= 0 or sell_qty <= 0 or buy_price <= 0 or sell_price <= 0:
            # Skip invalid trades
            if buy_qty <= 0 or buy_price <= 0:
                buy_queue.pop(0)
            if sell_qty <= 0 or sell_price <= 0:
                sell_queue.pop(0)
            continue
        
        # Match quantities
        matched_qty = min(buy_qty, sell_qty)
        
        # Calculate P/L
        realized_pl = (sell_price - buy_price) * matched_qty
        realized_pl_pct = ((sell_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
        
        # Store realized trade
        realized_trades.append({
            "symbol": symbol,
            "buy_time": buy_time,
            "sell_time": sell_time,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "quantity": matched_qty,
            "realized_pl": realized_pl,
            "realized_pl_pct": realized_pl_pct,
            "buy_order_id": buy_trade.get("close_order", {}).get("id", "N/A"),
            "sell_order_id": sell_trade.get("close_order", {}).get("id", "N/A"),
        })
        
        print(f"    [{buy_time.strftime('%Y-%m-%d %H:%M')}] BUY {matched_qty:.6f} @ ${buy_price:.6f}")
        print(f"    [{sell_time.strftime('%Y-%m-%d %H:%M')}] SELL {matched_qty:.6f} @ ${sell_price:.6f}")
        print(f"    P/L: ${realized_pl:+.2f} ({realized_pl_pct:+.2f}%)")
        
        # Update quantities
        if buy_qty > matched_qty:
            # Partial fill - update buy trade
            buy_trade["final_qty"] = buy_qty - matched_qty
            buy_queue[0] = (buy_time, buy_trade)
        else:
            buy_queue.pop(0)
        
        if sell_qty > matched_qty:
            # Partial fill - update sell trade
            sell_trade["final_qty"] = sell_qty - matched_qty
            sell_queue[0] = (sell_time, sell_trade)
        else:
            sell_queue.pop(0)
    
    # Remaining buys = unrealized positions
    for buy_time, buy_trade in buy_queue:
        buy_qty = float(buy_trade.get("final_qty", buy_trade.get("trade_qty", 0)) or 0)
        buy_price = float(buy_trade.get("entry_price", buy_trade.get("current_price", 0)) or 0)
        current_price = float(buy_trade.get("current_price", buy_price) or buy_price)
        
        if buy_qty > 0 and buy_price > 0:
            unrealized_pl = (current_price - buy_price) * buy_qty
            unrealized_pl_pct = ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
            
            unrealized_positions.append({
                "symbol": symbol,
                "buy_time": buy_time,
                "buy_price": buy_price,
                "current_price": current_price,
                "quantity": buy_qty,
                "unrealized_pl": unrealized_pl,
                "unrealized_pl_pct": unrealized_pl_pct,
            })

# Get current positions from Alpaca for accurate unrealized P/L
print("\n" + "=" * 80)
print("FETCHING CURRENT POSITIONS FROM ALPACA")
print("=" * 80)

try:
    from trading.alpaca_client import AlpacaClient
    client = AlpacaClient()
    positions = client.list_positions()
    
    # Update unrealized positions with actual current prices
    for pos in positions:
        symbol = pos.get("symbol", "").replace("/", "")
        qty = float(pos.get("qty", 0) or 0)
        avg_entry = float(pos.get("avg_entry_price", 0) or 0)
        current_price = float(pos.get("current_price", 0) or 0)
        unrealized_pl = float(pos.get("unrealized_pl", 0) or 0)
        
        if qty > 0 and avg_entry > 0:
            # Update or add unrealized position
            found = False
            for up in unrealized_positions:
                if up["symbol"] == symbol:
                    up["current_price"] = current_price
                    up["unrealized_pl"] = unrealized_pl
                    up["unrealized_pl_pct"] = ((current_price - avg_entry) / avg_entry) * 100 if avg_entry > 0 else 0
                    found = True
                    break
            
            if not found:
                unrealized_positions.append({
                    "symbol": symbol,
                    "buy_time": None,  # Unknown
                    "buy_price": avg_entry,
                    "current_price": current_price,
                    "quantity": qty,
                    "unrealized_pl": unrealized_pl,
                    "unrealized_pl_pct": ((current_price - avg_entry) / avg_entry) * 100 if avg_entry > 0 else 0,
                })
    
    print(f"[OK] Fetched {len(positions)} current positions")
except Exception as e:
    print(f"[WARN] Could not fetch current positions: {e}")

# Print comprehensive P&L report
print("\n" + "=" * 80)
print("COMPREHENSIVE PROFIT & LOSS REPORT")
print("=" * 80)

print("\n" + "=" * 80)
print("REALIZED P/L (CLOSED TRADES)")
print("=" * 80)

if realized_trades:
    total_realized_pl = 0.0
    
    for i, trade in enumerate(sorted(realized_trades, key=lambda x: x["sell_time"]), 1):
        print(f"\n{i}. {trade['symbol']}")
        print(f"   Entry: [{trade['buy_time'].strftime('%Y-%m-%d %H:%M:%S')}] ${trade['buy_price']:.6f}")
        print(f"   Exit:  [{trade['sell_time'].strftime('%Y-%m-%d %H:%M:%S')}] ${trade['sell_price']:.6f}")
        print(f"   Quantity: {trade['quantity']:.6f}")
        print(f"   Realized P/L: ${trade['realized_pl']:+.2f} ({trade['realized_pl_pct']:+.2f}%)")
        
        total_realized_pl += trade['realized_pl']
    
    print(f"\n{'='*80}")
    print(f"TOTAL REALIZED P/L: ${total_realized_pl:+.2f}")
    print(f"{'='*80}")
else:
    print("\nNo closed trades found in date range.")

print("\n" + "=" * 80)
print("UNREALIZED P/L (OPEN POSITIONS)")
print("=" * 80)

if unrealized_positions:
    total_unrealized_pl = 0.0
    
    for i, pos in enumerate(sorted(unrealized_positions, key=lambda x: x["symbol"]), 1):
        buy_time_str = pos['buy_time'].strftime('%Y-%m-%d %H:%M:%S') if pos['buy_time'] else "Unknown"
        print(f"\n{i}. {pos['symbol']}")
        print(f"   Entry: [{buy_time_str}] ${pos['buy_price']:.6f}")
        print(f"   Current: ${pos['current_price']:.6f}")
        print(f"   Quantity: {pos['quantity']:.6f}")
        print(f"   Unrealized P/L: ${pos['unrealized_pl']:+.2f} ({pos['unrealized_pl_pct']:+.2f}%)")
        
        total_unrealized_pl += pos['unrealized_pl']
    
    print(f"\n{'='*80}")
    print(f"TOTAL UNREALIZED P/L: ${total_unrealized_pl:+.2f}")
    print(f"{'='*80}")
else:
    print("\nNo open positions found.")

# Overall summary
print("\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)

total_realized = sum(t['realized_pl'] for t in realized_trades)
total_unrealized = sum(p['unrealized_pl'] for p in unrealized_positions)
total_pnl = total_realized + total_unrealized

print(f"\nRealized P/L (Closed Trades): ${total_realized:+.2f}")
print(f"Unrealized P/L (Open Positions): ${total_unrealized:+.2f}")
print(f"{'='*80}")
print(f"TOTAL P&L: ${total_pnl:+.2f}")
print(f"{'='*80}")

# Statistics
print("\n" + "=" * 80)
print("TRADE STATISTICS")
print("=" * 80)

if realized_trades:
    profitable_trades = [t for t in realized_trades if t['realized_pl'] > 0]
    losing_trades = [t for t in realized_trades if t['realized_pl'] < 0]
    breakeven_trades = [t for t in realized_trades if t['realized_pl'] == 0]
    
    print(f"\nTotal Closed Trades: {len(realized_trades)}")
    print(f"  Profitable: {len(profitable_trades)} (${sum(t['realized_pl'] for t in profitable_trades):+.2f})")
    print(f"  Losing: {len(losing_trades)} (${sum(t['realized_pl'] for t in losing_trades):+.2f})")
    print(f"  Breakeven: {len(breakeven_trades)}")
    
    if profitable_trades:
        avg_profit = sum(t['realized_pl'] for t in profitable_trades) / len(profitable_trades)
        print(f"\n  Average Profit: ${avg_profit:+.2f}")
    
    if losing_trades:
        avg_loss = sum(t['realized_pl'] for t in losing_trades) / len(losing_trades)
        print(f"  Average Loss: ${avg_loss:+.2f}")
    
    win_rate = (len(profitable_trades) / len(realized_trades)) * 100 if realized_trades else 0
    print(f"  Win Rate: {win_rate:.1f}%")

print(f"\nOpen Positions: {len(unrealized_positions)}")
if unrealized_positions:
    profitable_positions = [p for p in unrealized_positions if p['unrealized_pl'] > 0]
    losing_positions = [p for p in unrealized_positions if p['unrealized_pl'] < 0]
    print(f"  In Profit: {len(profitable_positions)}")
    print(f"  In Loss: {len(losing_positions)}")

print("=" * 80)
