"""
Calculate profit and loss for a specific date from trading logs.
"""
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

def calculate_daily_pnl(target_date: str):
    """
    Calculate profit and loss for a specific date.
    
    Args:
        target_date: Date in format "YYYY-MM-DD" (e.g., "2025-01-10" or "2026-01-10")
    """
    log_path = Path("logs/trading/crypto_trades.jsonl")
    
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        return
    
    # Parse target date
    try:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        target_date_str = target_dt.strftime("%Y-%m-%d")
    except ValueError:
        print(f"Error: Invalid date format. Use YYYY-MM-DD (e.g., 2025-01-10)")
        return
    
    print(f"\n{'='*80}")
    print(f"PROFIT & LOSS CALCULATION FOR {target_date_str}")
    print(f"{'='*80}\n")
    
    # Read all trades
    trades = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                trade = json.loads(line.strip())
                trades.append(trade)
            except json.JSONDecodeError:
                continue
    
    # Filter trades for target date
    date_trades = []
    for trade in trades:
        timestamp = trade.get("timestamp", "")
        if timestamp.startswith(target_date_str):
            date_trades.append(trade)
    
    if not date_trades:
        print(f"No trades found for {target_date_str}")
        print(f"\nAvailable dates in log file:")
        dates = sorted(set(t.get("timestamp", "")[:10] for t in trades if t.get("timestamp")))
        for d in dates[-10:]:
            count = sum(1 for t in trades if t.get("timestamp", "").startswith(d))
            print(f"  {d}: {count} trades")
        return
    
    print(f"Found {len(date_trades)} trade(s) on {target_date_str}\n")
    
    # Calculate realized P/L (closed positions)
    realized_pl = 0.0
    realized_trades = []
    
    # Calculate unrealized P/L (open positions at end of day)
    unrealized_pl = 0.0
    open_positions = {}
    
    # Track entries and exits
    entries = {}  # symbol -> list of entry trades
    exits = {}    # symbol -> list of exit trades
    
    for trade in date_trades:
        symbol = trade.get("trading_symbol", trade.get("symbol", "UNKNOWN"))
        decision = trade.get("decision", "")
        timestamp = trade.get("timestamp", "")
        
        # Track entries
        if decision in ["enter_long", "enter_short"]:
            if symbol not in entries:
                entries[symbol] = []
            entries[symbol].append({
                "timestamp": timestamp,
                "entry_price": trade.get("entry_price"),
                "quantity": trade.get("entry_qty") or trade.get("final_qty", 0),
                "side": "long" if decision == "enter_long" else "short",
                "entry_cost": trade.get("entry_notional") or trade.get("entry_cost", 0),
            })
        
        # Track exits
        if decision == "exit_position":
            if symbol not in exits:
                exits[symbol] = []
            
            exit_price = trade.get("exit_price")
            entry_price = trade.get("entry_price")
            quantity = trade.get("trade_qty") or trade.get("final_qty", 0)
            realized_pl_value = trade.get("realized_pl", 0.0)
            realized_pl_pct = trade.get("realized_pl_pct", 0.0)
            exit_reason = trade.get("exit_reason", "unknown")
            
            exits[symbol].append({
                "timestamp": timestamp,
                "exit_price": exit_price,
                "entry_price": entry_price,
                "quantity": quantity,
                "realized_pl": realized_pl_value,
                "realized_pl_pct": realized_pl_pct,
                "exit_reason": exit_reason,
            })
            
            realized_pl += realized_pl_value
            realized_trades.append({
                "symbol": symbol,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "realized_pl": realized_pl_value,
                "realized_pl_pct": realized_pl_pct,
                "exit_reason": exit_reason,
                "timestamp": timestamp,
            })
        
        # Track open positions (hold_position or recent entries)
        if decision in ["hold_position", "enter_long", "enter_short"]:
            current_price = trade.get("current_price")
            if current_price:
                position_status = trade.get("position_status", {})
                if position_status:
                    unrealized_pl_value = position_status.get("unrealized_pl", 0.0)
                    unrealized_pl_pct = position_status.get("unrealized_pl_pct", 0.0)
                    
                    # Update with latest position data
                    open_positions[symbol] = {
                        "entry_price": position_status.get("entry_price") or trade.get("entry_price"),
                        "current_price": current_price,
                        "quantity": position_status.get("quantity") or trade.get("final_qty", 0),
                        "side": position_status.get("side") or ("long" if decision == "enter_long" else "short"),
                        "unrealized_pl": unrealized_pl_value,
                        "unrealized_pl_pct": unrealized_pl_pct,
                        "timestamp": timestamp,
                    }
    
    # Print realized P/L
    print(f"{'REALIZED PROFIT/LOSS (Closed Positions)':^80}")
    print(f"{'-'*80}")
    
    if realized_trades:
        total_realized = 0.0
        for trade in realized_trades:
            pl_sign = "+" if trade["realized_pl"] >= 0 else ""
            print(f"{trade['symbol']:12} | Entry: ${trade['entry_price']:>10.4f} | Exit: ${trade['exit_price']:>10.4f} | "
                  f"Qty: {trade['quantity']:>12.4f} | P/L: {pl_sign}${trade['realized_pl']:>10.2f} ({pl_sign}{trade['realized_pl_pct']:>6.2f}%) | "
                  f"Reason: {trade['exit_reason']}")
            total_realized += trade["realized_pl"]
        
        print(f"{'-'*80}")
        pl_sign = "+" if total_realized >= 0 else ""
        print(f"{'TOTAL REALIZED P/L':>50} {pl_sign}${total_realized:>10.2f}")
    else:
        print("No positions closed on this date")
        print(f"{'-'*80}")
        print(f"{'TOTAL REALIZED P/L':>50} $0.00")
    
    print()
    
    # Print unrealized P/L (open positions at end of day)
    print(f"{'UNREALIZED PROFIT/LOSS (Open Positions at End of Day)':^80}")
    print(f"{'-'*80}")
    
    if open_positions:
        total_unrealized = 0.0
        for symbol, pos in sorted(open_positions.items()):
            pl_sign = "+" if pos["unrealized_pl"] >= 0 else ""
            print(f"{symbol:12} | Entry: ${pos['entry_price']:>10.4f} | Current: ${pos['current_price']:>10.4f} | "
                  f"Qty: {pos['quantity']:>12.4f} | Side: {pos['side']:>4} | "
                  f"P/L: {pl_sign}${pos['unrealized_pl']:>10.2f} ({pl_sign}{pos['unrealized_pl_pct']:>6.2f}%)")
            total_unrealized += pos["unrealized_pl"]
        
        print(f"{'-'*80}")
        pl_sign = "+" if total_unrealized >= 0 else ""
        print(f"{'TOTAL UNREALIZED P/L':>50} {pl_sign}${total_unrealized:>10.2f}")
    else:
        print("No open positions at end of day")
        print(f"{'-'*80}")
        print(f"{'TOTAL UNREALIZED P/L':>50} $0.00")
    
    print()
    
    # Print summary
    print(f"{'SUMMARY':^80}")
    print(f"{'-'*80}")
    total_realized = sum(t["realized_pl"] for t in realized_trades)
    total_unrealized = sum(p["unrealized_pl"] for p in open_positions.values())
    total_pl = total_realized + total_unrealized
    
    print(f"Realized P/L:     ${total_realized:>12.2f}")
    print(f"Unrealized P/L:  ${total_unrealized:>12.2f}")
    print(f"{'-'*80}")
    pl_sign = "+" if total_pl >= 0 else ""
    print(f"TOTAL P/L:       {pl_sign}${abs(total_pl):>12.2f}")
    print(f"{'='*80}\n")
    
    # Show entries and exits summary
    if entries or exits:
        print(f"{'TRADE SUMMARY':^80}")
        print(f"{'-'*80}")
        print(f"New Positions Opened: {sum(len(v) for v in entries.values())}")
        print(f"Positions Closed:      {sum(len(v) for v in exits.values())}")
        print(f"Open Positions (EOD):  {len(open_positions)}")
        print(f"{'-'*80}\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
    else:
        # Default to today or most recent date in logs
        target_date = "2026-01-10"  # Default fallback
    
    calculate_daily_pnl(target_date)
