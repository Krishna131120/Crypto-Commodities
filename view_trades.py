"""
View recent trading activity from logs.

Shows:
- Recent trades (entries/exits)
- Current positions
- Cycle summaries
- Errors
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from trading.alpaca_client import AlpacaClient


def load_trade_log(limit: int = 50) -> List[Dict[str, Any]]:
    """Load recent trades from crypto_trades.jsonl."""
    log_path = Path("logs/trading/crypto_trades.jsonl")
    if not log_path.exists():
        return []
    
    trades = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    return trades[-limit:]


def load_cycle_log(limit: int = 10) -> List[Dict[str, Any]]:
    """Load recent cycle summaries from cycles.jsonl."""
    log_path = Path("logs/trading/cycles.jsonl")
    if not log_path.exists():
        return []
    
    cycles = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                cycles.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    return cycles[-limit:]


def format_trade_summary(trade: Dict[str, Any]) -> str:
    """Format a single trade record for display."""
    symbol = trade.get("trading_symbol", "?")
    decision = trade.get("decision", "unknown")
    model_action = trade.get("model_action", "?")
    timestamp = trade.get("timestamp", "?")
    
    lines = [f"  [{timestamp}] {symbol}: {decision.upper()}"]
    
    if decision in ("enter_long", "enter_short"):
        qty = trade.get("entry_qty", 0)
        price = trade.get("current_price", 0)
        stop = trade.get("stop_loss_price", 0)
        lines.append(f"    Model: {model_action.upper()}")
        lines.append(f"    Entry: {qty:.6f} @ ${price:,.2f}")
        lines.append(f"    Stop-loss: ${stop:,.2f}")
    elif decision in ("exit_long", "exit_short"):
        qty = trade.get("trade_qty", 0)
        price = trade.get("current_price", 0)
        lines.append(f"    Exit: {qty:.6f} @ ${price:,.2f}")
    elif decision == "no_change":
        existing = trade.get("existing_side", "?")
        lines.append(f"    Current: {existing} (no change needed)")
    
    return "\n".join(lines)


def show_recent_trades(limit: int = 20):
    """Display recent trades."""
    trades = load_trade_log(limit)
    if not trades:
        print("No trades logged yet.")
        return
    
    print(f"\n{'='*80}")
    print(f"RECENT TRADES (last {len(trades)} entries)")
    print(f"{'='*80}\n")
    
    for trade in trades:
        print(format_trade_summary(trade))
        print()


def show_cycle_summary(limit: int = 5):
    """Display recent cycle summaries."""
    cycles = load_cycle_log(limit)
    if not cycles:
        print("No cycles logged yet.")
        return
    
    print(f"\n{'='*80}")
    print(f"RECENT CYCLES (last {len(cycles)} cycles)")
    print(f"{'='*80}\n")
    
    for cycle in cycles:
        start = cycle.get("cycle_start", "?")
        duration = cycle.get("cycle_duration_seconds", 0)
        processed = cycle.get("symbols_processed", 0)
        traded = cycle.get("symbols_traded", 0)
        skipped = cycle.get("symbols_skipped", 0)
        errors = len(cycle.get("errors", []))
        
        print(f"  [{start}]")
        print(f"    Processed: {processed} | Traded: {traded} | Skipped: {skipped} | Errors: {errors}")
        print(f"    Duration: {duration:.2f}s")
        print()


def show_current_positions():
    """Display current Alpaca positions."""
    try:
        client = AlpacaClient()
        account = client.get_account()
        positions = client.list_positions()
        
        equity = float(account.get("equity", 0))
        buying_power = float(account.get("buying_power", 0))
        
        print(f"\n{'='*80}")
        print("CURRENT ALPACA ACCOUNT STATUS")
        print(f"{'='*80}\n")
        print(f"Equity: ${equity:,.2f}")
        print(f"Buying Power: ${buying_power:,.2f}")
        print()
        
        if not positions:
            print("No open positions.")
            return
        
        print(f"Open Positions ({len(positions)}):")
        print()
        for pos in positions:
            symbol = pos.get("symbol", "?")
            qty = float(pos.get("qty", 0))
            avg_entry = float(pos.get("avg_entry_price", 0))
            current_price = float(pos.get("current_price", 0))
            market_value = float(pos.get("market_value", 0))
            unrealized_pl = float(pos.get("unrealized_pl", 0))
            side = "LONG" if qty > 0 else "SHORT"
            
            print(f"  {symbol} ({side})")
            print(f"    Quantity: {qty:.6f}")
            print(f"    Avg Entry: ${avg_entry:,.2f}")
            print(f"    Current: ${current_price:,.2f}")
            print(f"    Market Value: ${market_value:,.2f}")
            print(f"    Unrealized P/L: ${unrealized_pl:+,.2f}")
            print()
    
    except Exception as exc:
        print(f"[ERROR] Failed to load positions: {exc}")
        print("Make sure ALPACA_API_KEY and ALPACA_SECRET_KEY are set")


def show_statistics():
    """Show trading statistics from logs."""
    trades = load_trade_log(limit=1000)
    if not trades:
        print("No trades to analyze.")
        return
    
    print(f"\n{'='*80}")
    print("TRADING STATISTICS")
    print(f"{'='*80}\n")
    
    decisions = defaultdict(int)
    symbols = defaultdict(int)
    
    for trade in trades:
        decision = trade.get("decision", "unknown")
        symbol = trade.get("trading_symbol", "?")
        decisions[decision] += 1
        symbols[symbol] += 1
    
    print("Decision counts:")
    for decision, count in sorted(decisions.items(), key=lambda x: -x[1]):
        print(f"  {decision}: {count}")
    
    print("\nSymbol activity:")
    for symbol, count in sorted(symbols.items(), key=lambda x: -x[1]):
        print(f"  {symbol}: {count} trades")


def main():
    parser = argparse.ArgumentParser(description="View trading logs and current positions.")
    parser.add_argument(
        "--trades",
        action="store_true",
        help="Show recent trades",
    )
    parser.add_argument(
        "--cycles",
        action="store_true",
        help="Show recent cycle summaries",
    )
    parser.add_argument(
        "--positions",
        action="store_true",
        help="Show current Alpaca positions",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show trading statistics",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show everything",
    )
    args = parser.parse_args()
    
    if args.all or not any([args.trades, args.cycles, args.positions, args.stats]):
        # Default: show everything
        show_recent_trades()
        show_cycle_summary()
        show_current_positions()
        show_statistics()
    else:
        if args.trades:
            show_recent_trades()
        if args.cycles:
            show_cycle_summary()
        if args.positions:
            show_current_positions()
        if args.stats:
            show_statistics()


if __name__ == "__main__":
    main()

