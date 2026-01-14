"""
Analyze recently sold trades: Calculate profit/loss, iterations, and totals.
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any

def analyze_recent_trades(log_file: Path, positions_file: Path = None, days: int = 7) -> None:
    """
    Analyze recently sold trades from trade log and positions file.
    
    Args:
        log_file: Path to JSONL trade log file
        positions_file: Path to positions JSON file (optional)
        days: Number of recent days to analyze (default: 7)
    """
    # Read trades from log file
    trades = []
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trade = json.loads(line)
                    trades.append(trade)
                except json.JSONDecodeError:
                    continue
    
    # Read closed positions from positions file
    if positions_file is None:
        positions_file = Path("data/positions/active_positions.json")
    
    if positions_file.exists():
        try:
            with open(positions_file, 'r', encoding='utf-8') as f:
                positions_data = json.load(f)
            
            # Convert closed positions to trade format
            for symbol, pos_data in positions_data.items():
                status = pos_data.get("status", "")
                if status in ["profit_target_hit", "stop_loss_hit", "closed"]:
                    # Convert position to trade format
                    trade = {
                        "data_symbol": pos_data.get("data_symbol", symbol),
                        "trading_symbol": symbol,
                        "decision": "exit_position",
                        "exit_reason": pos_data.get("exit_reason", status),
                        "entry_price": pos_data.get("entry_price", 0.0),
                        "exit_price": pos_data.get("exit_price", 0.0),
                        "realized_pl": pos_data.get("realized_pl", 0.0),
                        "realized_pl_pct": pos_data.get("realized_pl_pct", 0.0),
                        "timestamp": pos_data.get("exit_time") or pos_data.get("entry_time", ""),
                        "entry_time": pos_data.get("entry_time", ""),
                        "exit_time": pos_data.get("exit_time", ""),
                        "quantity": pos_data.get("quantity", 0.0),
                        "side": pos_data.get("side", "long"),
                    }
                    trades.append(trade)
        except Exception as e:
            print(f"[WARN] Could not read positions file: {e}")
    
    if not trades:
        print("No trades found in log file or positions file")
        return
    
    # Filter for closed/sold trades
    closed_trades = []
    entry_trades = {}  # Track entry trades by symbol to calculate holding period
    
    for trade in trades:
        decision = trade.get("decision", "")
        exit_reason = trade.get("exit_reason", "")
        symbol = trade.get("data_symbol") or trade.get("trading_symbol", "UNKNOWN")
        
        # Track entry trades
        if decision in ["enter_long", "enter_short"]:
            entry_time = trade.get("timestamp", "")
            entry_price = trade.get("entry_price", 0.0)
            if entry_time and entry_price > 0:
                if symbol not in entry_trades:
                    entry_trades[symbol] = []
                entry_trades[symbol].append({
                    "timestamp": entry_time,
                    "entry_price": entry_price,
                })
        
        # Check if this is a closed trade
        if decision in ["exit_position", "profit_target_executed", "stop_loss_executed"] or \
           exit_reason in ["profit_target_hit", "stop_loss_hit", "trailing_stop_from_peak", "trailing_stop_from_bottom"]:
            closed_trades.append(trade)
    
    if not closed_trades:
        print("No closed trades found")
        return
    
    # Group by symbol and calculate metrics
    symbol_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "trades": [],
        "total_pl": 0.0,
        "total_pl_pct": 0.0,
        "iterations": 0,  # Number of times this symbol was traded (entry+exit cycles)
        "entry_count": 0,  # Number of times entered
        "exit_count": 0,   # Number of times exited
        "wins": 0,
        "losses": 0,
        "profit_target_hits": 0,
        "stop_loss_hits": 0,
        "total_holding_time_minutes": 0.0,
    })
    
    # Count entry trades
    for symbol, entries in entry_trades.items():
        symbol_stats[symbol]["entry_count"] = len(entries)
    
    for trade in closed_trades:
        symbol = trade.get("data_symbol") or trade.get("trading_symbol", "UNKNOWN")
        realized_pl = trade.get("realized_pl", 0.0)
        realized_pl_pct = trade.get("realized_pl_pct", 0.0)
        exit_reason = trade.get("exit_reason", "unknown")
        entry_price = trade.get("entry_price", 0.0)
        exit_price = trade.get("exit_price", 0.0)
        entry_time = trade.get("entry_time") or trade.get("timestamp", "")
        exit_time = trade.get("exit_time") or trade.get("timestamp", "")
        timestamp = trade.get("timestamp", "")
        
        # Calculate holding period if we have entry and exit times
        holding_minutes = 0.0
        if entry_time and exit_time:
            try:
                entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                exit_dt = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
                holding_minutes = (exit_dt - entry_dt).total_seconds() / 60.0
            except:
                pass
        
        # Calculate P/L if not provided
        if realized_pl == 0.0 and entry_price > 0 and exit_price > 0:
            qty = trade.get("trade_qty") or trade.get("quantity", 0.0)
            if qty > 0:
                side = trade.get("side", "long")
                if side == "long":
                    realized_pl = (exit_price - entry_price) * qty
                    realized_pl_pct = ((exit_price - entry_price) / entry_price) * 100
                else:  # short
                    realized_pl = (entry_price - exit_price) * qty
                    realized_pl_pct = ((entry_price - exit_price) / entry_price) * 100
        
        symbol_stats[symbol]["trades"].append({
            "timestamp": timestamp,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "realized_pl": realized_pl,
            "realized_pl_pct": realized_pl_pct,
            "exit_reason": exit_reason,
            "holding_minutes": holding_minutes,
        })
        
        symbol_stats[symbol]["total_pl"] += realized_pl
        symbol_stats[symbol]["exit_count"] += 1
        symbol_stats[symbol]["iterations"] = max(symbol_stats[symbol]["entry_count"], symbol_stats[symbol]["exit_count"])
        symbol_stats[symbol]["total_holding_time_minutes"] += holding_minutes
        
        if realized_pl > 0:
            symbol_stats[symbol]["wins"] += 1
        elif realized_pl < 0:
            symbol_stats[symbol]["losses"] += 1
        
        if "profit_target" in exit_reason.lower():
            symbol_stats[symbol]["profit_target_hits"] += 1
        elif "stop_loss" in exit_reason.lower():
            symbol_stats[symbol]["stop_loss_hits"] += 1
    
    # Calculate average P/L percentage for each symbol
    for symbol, stats in symbol_stats.items():
        if stats["iterations"] > 0:
            stats["avg_pl_pct"] = stats["total_pl_pct"] / stats["iterations"] if stats["total_pl_pct"] != 0 else 0.0
            # Calculate average from individual trades if total_pl_pct is 0
            if stats["avg_pl_pct"] == 0 and stats["trades"]:
                total_pct = sum(t.get("realized_pl_pct", 0) for t in stats["trades"])
                stats["avg_pl_pct"] = total_pct / len(stats["trades"])
    
    # Sort by total P/L (descending)
    sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]["total_pl"], reverse=True)
    
    # Print summary
    print("=" * 100)
    print("RECENTLY SOLD TRADES ANALYSIS")
    print("=" * 100)
    print(f"Total Closed Trades: {len(closed_trades)}")
    print(f"Unique Symbols: {len(symbol_stats)}")
    print()
    
    # Overall totals
    total_profit = sum(s["total_pl"] for s in symbol_stats.values() if s["total_pl"] > 0)
    total_loss = sum(s["total_pl"] for s in symbol_stats.values() if s["total_pl"] < 0)
    total_pl = sum(s["total_pl"] for s in symbol_stats.values())
    total_wins = sum(s["wins"] for s in symbol_stats.values())
    total_losses = sum(s["losses"] for s in symbol_stats.values())
    total_closed_trades = len(closed_trades)  # Each closed trade is one complete cycle
    
    print("=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)
    print(f"Total Closed Trades: {total_closed_trades}")
    print(f"Winning Trades: {total_wins} ({total_wins/total_closed_trades*100:.1f}%)" if total_closed_trades > 0 else "Winning Trades: 0")
    print(f"Losing Trades: {total_losses} ({total_losses/total_closed_trades*100:.1f}%)" if total_closed_trades > 0 else "Losing Trades: 0")
    print(f"Total Profit: ${total_profit:+.2f}")
    print(f"Total Loss: ${total_loss:.2f}")
    print(f"Net P/L: ${total_pl:+.2f}")
    if total_closed_trades > 0:
        avg_pl_per_trade = total_pl / total_closed_trades
        print(f"Average P/L per Trade: ${avg_pl_per_trade:+.2f}")
    print()
    
    # Per-symbol breakdown
    print("=" * 100)
    print("PER-SYMBOL BREAKDOWN")
    print("=" * 100)
    print(f"{'Symbol':<20} {'Trades':<8} {'Wins':<8} {'Losses':<8} {'Total P/L':<15} {'Avg P/L %':<12} {'Avg Hold':<12} {'Exit Reasons':<20}")
    print("-" * 100)
    
    for symbol, stats in sorted_symbols:
        iterations = stats["exit_count"]  # Number of times this symbol was sold
        wins = stats["wins"]
        losses = stats["losses"]
        total_pl = stats["total_pl"]
        
        # Calculate average P/L percentage from individual trades
        if stats["trades"]:
            total_pct = sum(t.get("realized_pl_pct", 0) for t in stats["trades"])
            avg_pl_pct = total_pct / len(stats["trades"])
        else:
            avg_pl_pct = 0.0
        
        # Calculate average holding time
        avg_hold_minutes = stats["total_holding_time_minutes"] / len(stats["trades"]) if stats["trades"] else 0.0
        if avg_hold_minutes < 60:
            avg_hold_str = f"{avg_hold_minutes:.1f}m"
        elif avg_hold_minutes < 1440:
            avg_hold_str = f"{avg_hold_minutes/60:.1f}h"
        else:
            avg_hold_str = f"{avg_hold_minutes/1440:.1f}d"
        
        # Count exit reasons
        exit_reasons = []
        if stats["profit_target_hits"] > 0:
            exit_reasons.append(f"Target:{stats['profit_target_hits']}")
        if stats["stop_loss_hits"] > 0:
            exit_reasons.append(f"Stop:{stats['stop_loss_hits']}")
        exit_reason_str = ", ".join(exit_reasons) if exit_reasons else "Other"
        
        print(f"{symbol:<20} {iterations:<8} {wins:<8} {losses:<8} ${total_pl:+12.2f} {avg_pl_pct:+10.2f}% {avg_hold_str:<12} {exit_reason_str:<20}")
    
    print()
    
    # Detailed trade list (last 20 trades)
    print("=" * 100)
    print("RECENT TRADES (Last 20)")
    print("=" * 100)
    print(f"{'Symbol':<20} {'Entry':<12} {'Exit':<12} {'P/L':<15} {'P/L %':<12} {'Reason':<20} {'Time':<20}")
    print("-" * 100)
    
    # Sort all trades by timestamp (most recent first)
    all_trades_sorted = sorted(closed_trades, key=lambda t: t.get("timestamp", ""), reverse=True)
    
    for trade in all_trades_sorted[:20]:
        symbol = trade.get("data_symbol") or trade.get("trading_symbol", "UNKNOWN")
        entry_price = trade.get("entry_price", 0.0)
        exit_price = trade.get("exit_price", 0.0)
        realized_pl = trade.get("realized_pl", 0.0)
        realized_pl_pct = trade.get("realized_pl_pct", 0.0)
        exit_reason = trade.get("exit_reason", "unknown")
        timestamp = trade.get("timestamp", "")
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            time_str = dt.strftime("%Y-%m-%d %H:%M")
        except:
            time_str = timestamp[:16] if len(timestamp) > 16 else timestamp
        
        print(f"{symbol:<20} ${entry_price:>10.4f} ${exit_price:>10.4f} ${realized_pl:+12.2f} {realized_pl_pct:+10.2f}% {exit_reason:<20} {time_str:<20}")
    
    print()
    
    # Best and worst performers
    print("=" * 100)
    print("TOP PERFORMERS")
    print("=" * 100)
    winners = [(s, stats) for s, stats in sorted_symbols if stats["total_pl"] > 0]
    if winners:
        print(f"{'Symbol':<20} {'Trades':<8} {'Total P/L':<15} {'Avg P/L %':<12} {'Win Rate':<12}")
        print("-" * 100)
        for symbol, stats in sorted(winners, key=lambda x: x[1]["total_pl"], reverse=True)[:10]:
            win_rate = (stats["wins"] / stats["exit_count"] * 100) if stats["exit_count"] > 0 else 0
            avg_pl_pct = sum(t.get("realized_pl_pct", 0) for t in stats["trades"]) / len(stats["trades"]) if stats["trades"] else 0
            print(f"{symbol:<20} {stats['exit_count']:<8} ${stats['total_pl']:+12.2f} {avg_pl_pct:+10.2f}% {win_rate:>10.1f}%")
    else:
        print("No winning trades found")
    
    print()
    print("=" * 100)
    print("WORST PERFORMERS")
    print("=" * 100)
    losers = [(s, stats) for s, stats in sorted_symbols if stats["total_pl"] < 0]
    if losers:
        print(f"{'Symbol':<20} {'Trades':<8} {'Total P/L':<15} {'Avg P/L %':<12} {'Loss Rate':<12}")
        print("-" * 100)
        for symbol, stats in sorted(losers, key=lambda x: x[1]["total_pl"])[:10]:
            loss_rate = (stats["losses"] / stats["exit_count"] * 100) if stats["exit_count"] > 0 else 0
            avg_pl_pct = sum(t.get("realized_pl_pct", 0) for t in stats["trades"]) / len(stats["trades"]) if stats["trades"] else 0
            print(f"{symbol:<20} {stats['exit_count']:<8} ${stats['total_pl']:+12.2f} {avg_pl_pct:+10.2f}% {loss_rate:>10.1f}%")
    else:
        print("No losing trades found")
    
    print()
    print("=" * 100)

if __name__ == "__main__":
    log_file = Path("logs/trading/crypto_trades.jsonl")
    analyze_recent_trades(log_file, days=30)
