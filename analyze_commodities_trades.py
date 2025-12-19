"""Analyze commodities trade logs to calculate total profit/loss and show live positions."""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Optional

def get_live_positions() -> List[Dict[str, Any]]:
    """Get live positions from Dhan MCX API."""
    try:
        from trading.dhan_client import DhanClient, DhanConfig
        import os
        
        # Get credentials from environment variables
        access_token = os.getenv("DHAN_ACCESS_TOKEN")
        client_id = os.getenv("DHAN_CLIENT_ID")
        
        if not access_token or not client_id:
            print("  ⚠️  Could not load DHAN credentials.")
            print("      Set DHAN_ACCESS_TOKEN and DHAN_CLIENT_ID environment variables.")
            print("      PowerShell: $env:DHAN_ACCESS_TOKEN=\"your_token\"")
            print("      PowerShell: $env:DHAN_CLIENT_ID=\"1107954503\"")
            return []
        
        client = DhanClient(access_token=access_token, client_id=client_id)
        positions = client.list_positions()
        
        live_positions = []
        for pos in positions or []:
            # DhanClient normalizes positions to this format:
            # {"symbol": str, "qty": float, "market_value": float, "avg_entry_price": float, "ltp": float, "unrealized_pl": float}
            qty = float(pos.get("qty", 0) or 0)
            if qty == 0:
                continue
            
            symbol = pos.get("symbol", "")
            avg_entry = float(pos.get("avg_entry_price", 0) or 0)
            current_price = float(pos.get("ltp", pos.get("current_price", 0)) or 0)
            market_value = float(pos.get("market_value", 0) or 0)
            unrealized_pl = float(pos.get("unrealized_pl", 0) or 0)
            
            # Determine side (long/short) - positive qty = long, negative = short
            side = "long" if qty > 0 else "short"
            
            # Calculate unrealized P/L percentage if we have entry price
            if avg_entry > 0:
                if side == "long":
                    unrealized_pl_pct = ((current_price - avg_entry) / avg_entry) * 100
                else:
                    unrealized_pl_pct = ((avg_entry - current_price) / avg_entry) * 100
            else:
                unrealized_pl_pct = 0.0
            
            # If market value not provided, calculate it
            if market_value == 0 and current_price > 0:
                market_value = abs(qty) * current_price
            
            live_positions.append({
                "symbol": symbol,
                "side": side,
                "quantity": abs(qty),
                "entry_price": avg_entry,
                "current_price": current_price,
                "market_value": market_value,
                "unrealized_pl": unrealized_pl,
                "unrealized_pl_pct": unrealized_pl_pct,
            })
        
        return live_positions
    except Exception as exc:
        print(f"  ⚠️  Could not fetch live positions from Dhan MCX: {exc}")
        return []


def parse_trading_logs(log_file: Path, filter_symbol: Optional[str] = None, filter_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Parse trading logs to extract closed trades with P/L."""
    if not log_file.exists():
        return []
    
    closed_trades = []
    current_positions = {}  # symbol -> entry info
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    trade = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Apply filters
                symbol = trade.get('trading_symbol', '')
                if filter_symbol and symbol != filter_symbol:
                    continue
                
                timestamp = trade.get('timestamp', '')
                if filter_date and not timestamp.startswith(filter_date):
                    continue
                
                decision = trade.get('decision', '')
                
                # Track entries
                if decision == 'enter_long' or decision == 'enter_short':
                    entry_price = trade.get('entry_price', 0)
                    entry_qty = trade.get('entry_qty', 0)
                    if entry_price > 0 and entry_qty > 0:
                        current_positions[symbol] = {
                            'entry_price': entry_price,
                            'quantity': entry_qty,
                            'entry_time': timestamp,
                            'side': 'long' if decision == 'enter_long' else 'short',
                        }
                
                # Track exits - look for exit_position decision
                elif decision in ('exit_position', 'would_exit_position'):
                    # Exit has realized P/L directly in the log
                    exit_price = trade.get('exit_price', 0)
                    realized_pl = trade.get('realized_pl', 0)
                    realized_pl_pct = trade.get('realized_pl_pct', 0)
                    exit_reason = trade.get('exit_reason', 'unknown')
                    entry_price = trade.get('entry_price', 0)
                    
                    # If we have entry info from current_positions, use it; otherwise use from log
                    if symbol in current_positions:
                        entry_info = current_positions[symbol]
                        entry_price = entry_info['entry_price']
                        entry_time = entry_info['entry_time']
                        side = entry_info['side']
                        del current_positions[symbol]
                    else:
                        # Try to infer from log
                        entry_time = timestamp  # Fallback
                        side = 'long' if trade.get('existing_side') == 'long' else 'short'
                    
                    # Only add if we have valid P/L data
                    if exit_price > 0 and entry_price > 0:
                        closed_trades.append({
                            'symbol': symbol,
                            'side': side,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'quantity': trade.get('trade_qty', trade.get('entry_qty', 0)),
                            'realized_pl': realized_pl,
                            'realized_pl_pct': realized_pl_pct,
                            'exit_reason': exit_reason,
                            'entry_time': entry_time,
                            'exit_time': timestamp,
                        })
    
    except Exception as exc:
        print(f"  ⚠️  Error reading trading logs: {exc}")
    
    return closed_trades


def get_closed_positions_from_file(positions_file: Path, filter_symbol: Optional[str] = None, filter_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get closed positions from position manager file."""
    if not positions_file.exists():
        return []
    
    closed_trades = []
    
    try:
        with open(positions_file, 'r', encoding='utf-8') as f:
            saved_positions = json.load(f)
        
        for symbol, pos_data in saved_positions.items():
            # Apply filters
            if filter_symbol and symbol != filter_symbol:
                continue
            
            exit_time = pos_data.get('exit_time') or ''
            if filter_date and exit_time and not exit_time.startswith(filter_date):
                continue
            
            # Only include closed positions with P/L data
            status = pos_data.get('status', '')
            if status != 'open' and pos_data.get('realized_pl') is not None:
                # Only include commodities positions
                if pos_data.get('asset_type') == 'commodities':
                    closed_trades.append({
                        'symbol': symbol,
                        'side': pos_data.get('side', 'long'),
                        'entry_price': pos_data.get('entry_price', 0),
                        'exit_price': pos_data.get('exit_price', 0),
                        'quantity': abs(pos_data.get('quantity', 0)),
                        'realized_pl': pos_data.get('realized_pl', 0),
                        'realized_pl_pct': pos_data.get('realized_pl_pct', 0),
                        'exit_reason': pos_data.get('exit_reason', 'unknown'),
                        'entry_time': pos_data.get('entry_time', ''),
                        'exit_time': exit_time,
                    })
    except Exception as exc:
        print(f"  ⚠️  Error reading position file: {exc}")
    
    return closed_trades


def analyze_commodities_trades(filter_symbol: Optional[str] = None, filter_date: Optional[str] = None):
    """
    Analyze commodities trades: show live positions first, then historical P/L from logs.
    
    Args:
        filter_symbol: Only analyze trades for this symbol (e.g., 'GOLD')
        filter_date: Only analyze trades from this date (e.g., '2025-12-19')
    """
    print("=" * 80)
    print("COMMODITIES TRADE ANALYSIS (MCX - Dhan)")
    print("=" * 80)
    
    filter_info = []
    if filter_symbol:
        filter_info.append(f"Symbol: {filter_symbol}")
    if filter_date:
        filter_info.append(f"Date: {filter_date}")
    if filter_info:
        print(f"Filters: {', '.join(filter_info)}")
    print()
    
    # ========================================================================
    # PART 1: LIVE POSITIONS
    # ========================================================================
    print("=" * 80)
    print("LIVE POSITIONS (Current Open Positions on MCX)")
    print("=" * 80)
    
    live_positions = get_live_positions()
    
    if live_positions:
        # Apply filters to live positions
        if filter_symbol:
            live_positions = [p for p in live_positions if p['symbol'] == filter_symbol]
        
        if live_positions:
            total_unrealized_pl = 0.0
            total_market_value = 0.0
            
            print(f"\nFound {len(live_positions)} open position(s):\n")
            
            for i, pos in enumerate(live_positions, 1):
                symbol = pos['symbol']
                side = pos['side'].upper()
                qty = pos['quantity']
                entry = pos['entry_price']
                current = pos['current_price']
                market_val = pos['market_value']
                unrealized_pl = pos['unrealized_pl']
                unrealized_pl_pct = pos['unrealized_pl_pct']
                
                status = "[PROFIT]" if unrealized_pl > 0 else "[LOSS]" if unrealized_pl < 0 else "[FLAT]"
                
                print(f"Position {i}: {symbol} ({side})")
                print(f"  Entry Price:      ₹{entry:,.2f}")
                print(f"  Current Price:     ₹{current:,.2f}")
                print(f"  Quantity:          {qty:.2f} lots")
                print(f"  Market Value:      ₹{market_val:,.2f}")
                print(f"  Unrealized P/L:    ₹{unrealized_pl:+,.2f} ({unrealized_pl_pct:+.2f}%)")
                print(f"  Status:            {status}")
                print()
                
                total_unrealized_pl += unrealized_pl
                total_market_value += market_val
            
            print("-" * 80)
            print("LIVE POSITIONS SUMMARY")
            print("-" * 80)
            print(f"Total Positions:        {len(live_positions)}")
            print(f"Total Market Value:     ₹{total_market_value:,.2f}")
            print(f"Total Unrealized P/L:   ₹{total_unrealized_pl:+,.2f}")
            print()
        else:
            print("No open positions found (after filtering).")
            print()
    else:
        print("No open positions found.")
        print()
    
    # ========================================================================
    # PART 2: HISTORICAL P/L FROM TRADING LOGS
    # ========================================================================
    print("=" * 80)
    print("HISTORICAL TRADES (From Trading Logs)")
    print("=" * 80)
    
    log_file = Path("logs/trading/commodities_trades.jsonl")
    positions_file = Path("data/positions/active_positions.json")
    
    # Parse closed trades from logs
    closed_trades_from_logs = parse_trading_logs(log_file, filter_symbol, filter_date)
    
    # Also get closed trades from position manager file
    closed_trades_from_file = get_closed_positions_from_file(positions_file, filter_symbol, filter_date)
    
    # Merge and deduplicate (prefer log data if available)
    all_closed_trades = []
    seen_trades = set()
    
    # Add trades from logs first
    for trade in closed_trades_from_logs:
        key = (trade['symbol'], trade['entry_time'], trade['entry_price'])
        if key not in seen_trades:
            all_closed_trades.append(trade)
            seen_trades.add(key)
    
    # Add trades from file that aren't in logs
    for trade in closed_trades_from_file:
        key = (trade['symbol'], trade['entry_time'], trade['entry_price'])
        if key not in seen_trades:
            all_closed_trades.append(trade)
            seen_trades.add(key)
    
    # Sort by exit time (most recent first)
    all_closed_trades.sort(key=lambda x: x.get('exit_time', ''), reverse=True)
    
    if all_closed_trades:
        total_profit = 0.0
        total_loss = 0.0
        profitable_trades = 0
        losing_trades = 0
        
        print(f"\nFound {len(all_closed_trades)} closed trade(s):\n")
        
        for i, trade in enumerate(all_closed_trades, 1):
            symbol = trade['symbol']
            side = trade['side'].upper()
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            quantity = trade['quantity']
            realized_pl = trade['realized_pl']
            realized_pl_pct = trade['realized_pl_pct']
            exit_reason = trade['exit_reason']
            
            status = "[PROFIT]" if realized_pl > 0 else "[LOSS]" if realized_pl < 0 else "[FLAT]"
            
            if realized_pl > 0:
                total_profit += realized_pl
                profitable_trades += 1
            elif realized_pl < 0:
                total_loss += abs(realized_pl)
                losing_trades += 1
            
            print(f"Trade {i}: {symbol} ({side})")
            print(f"  Entry:              ₹{entry_price:,.2f}")
            print(f"  Exit:               ₹{exit_price:,.2f}")
            print(f"  Quantity:           {quantity:.2f} lots")
            print(f"  Realized P/L:        ₹{realized_pl:+,.2f} ({realized_pl_pct:+.2f}%)")
            print(f"  Exit Reason:         {exit_reason}")
            print(f"  Status:              {status}")
            if trade.get('entry_time'):
                print(f"  Entry Time:          {trade['entry_time']}")
            if trade.get('exit_time'):
                print(f"  Exit Time:           {trade['exit_time']}")
            print()
        
        net_profit = total_profit - total_loss
        total_trades = len(all_closed_trades)
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        print("-" * 80)
        print("HISTORICAL TRADES SUMMARY")
        print("-" * 80)
        print(f"Total Closed Trades:     {total_trades}")
        print(f"  Profitable:            {profitable_trades}")
        print(f"  Losing:                {losing_trades}")
        print(f"  Win Rate:              {win_rate:.1f}%")
        print()
        print(f"Total Profit:            ₹{total_profit:,.2f}")
        print(f"Total Loss:              ₹{total_loss:,.2f}")
        print(f"Net Profit/Loss:         ₹{net_profit:+,.2f}")
        
        if (total_profit + total_loss) > 0:
            net_return_pct = (net_profit / (total_profit + total_loss)) * 100
            print(f"Net Return:             {net_return_pct:+.2f}%")
        print()
    else:
        if not log_file.exists() and not positions_file.exists():
            print("No trading logs found.")
            print("  • Trading logs: logs/trading/commodities_trades.jsonl")
            print("  • Position file: data/positions/active_positions.json")
        elif not log_file.exists():
            print("Trading log file not found: logs/trading/commodities_trades.jsonl")
            print("Showing closed positions from position manager file only...")
        else:
            print("No closed trades found in logs (after filtering).")
        print()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    if live_positions or all_closed_trades:
        print("=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        
        if live_positions:
            total_unrealized = sum(p['unrealized_pl'] for p in live_positions)
            print(f"Live Positions:          {len(live_positions)}")
            print(f"  Total Unrealized P/L:  ₹{total_unrealized:+,.2f}")
        
        if all_closed_trades:
            net_historical = sum(t['realized_pl'] for t in all_closed_trades)
            print(f"Historical Trades:       {len(all_closed_trades)}")
            print(f"  Net Realized P/L:      ₹{net_historical:+,.2f}")
        
        if live_positions and all_closed_trades:
            total_unrealized = sum(p['unrealized_pl'] for p in live_positions)
            net_historical = sum(t['realized_pl'] for t in all_closed_trades)
            combined_pl = total_unrealized + net_historical
            print(f"\nCombined P/L:            ₹{combined_pl:+,.2f}")
            print(f"  (Unrealized: ₹{total_unrealized:+,.2f} + Realized: ₹{net_historical:+,.2f})")
        print()


if __name__ == "__main__":
    import sys
    
    filter_symbol = None
    filter_date = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            filter_symbol = None
            filter_date = None
        elif sys.argv[1] == "--help":
            print("Usage: python analyze_commodities_trades.py [--all] [--symbol SYMBOL] [--date YYYY-MM-DD]")
            print()
            print("Options:")
            print("  --all              Show all trades (no filters)")
            print("  --symbol SYMBOL     Filter by trading symbol (e.g., GOLD)")
            print("  --date YYYY-MM-DD  Filter by date (e.g., 2025-12-19)")
            print()
            print("Default: Shows all trades without filters")
            sys.exit(0)
        else:
            # Parse arguments
            i = 1
            while i < len(sys.argv):
                if sys.argv[i] == "--symbol" and i + 1 < len(sys.argv):
                    filter_symbol = sys.argv[i + 1]
                    i += 2
                elif sys.argv[i] == "--date" and i + 1 < len(sys.argv):
                    filter_date = sys.argv[i + 1]
                    i += 2
                else:
                    i += 1
    
    analyze_commodities_trades(filter_symbol=filter_symbol, filter_date=filter_date)
