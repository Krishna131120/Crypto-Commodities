"""Analyze trade logs to calculate total profit/loss."""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def analyze_trades(filter_symbol=None, filter_date=None):
    """
    Analyze all trades from the crypto_trades.jsonl file.
    
    Args:
        filter_symbol: Only analyze trades for this symbol (e.g., 'BTCUSD')
        filter_date: Only analyze trades from this date (e.g., '2025-12-15')
    """
    log_file = Path("logs/trading/crypto_trades.jsonl")
    
    if not log_file.exists():
        print(f"Trade log file not found: {log_file}")
        return
    
    trades = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    trade = json.loads(line)
                    # Apply filters
                    if filter_symbol and trade.get('trading_symbol') != filter_symbol:
                        continue
                    if filter_date and not trade.get('timestamp', '').startswith(filter_date):
                        continue
                    trades.append(trade)
                except json.JSONDecodeError:
                    continue
    
    filter_info = []
    if filter_symbol:
        filter_info.append(f"Symbol: {filter_symbol}")
    if filter_date:
        filter_info.append(f"Date: {filter_date}")
    
    print(f"Total trade entries: {len(trades)}" + (f" (filtered: {', '.join(filter_info)})" if filter_info else ""))
    print("=" * 80)
    
    # Track positions: symbol -> list of entry/exit pairs
    position_history = defaultdict(list)  # symbol -> list of {entry, exit} dicts
    current_positions = {}  # symbol -> current entry info
    
    # Process trades chronologically
    for trade in trades:
        decision = trade.get('decision', '')
        symbol = trade.get('trading_symbol', '')
        timestamp = trade.get('timestamp', '')
        existing_side = trade.get('existing_side', 'flat')
        existing_qty = trade.get('existing_qty', 0.0)
        
        # Track entries: when we go from flat to long
        if decision == 'enter_long':
            entry_price = trade.get('entry_price', 0)
            quantity = trade.get('entry_qty', 0)
            entry_notional = trade.get('entry_notional', 0)
            
            current_positions[symbol] = {
                'entry_price': entry_price,
                'quantity': quantity,
                'entry_notional': entry_notional,
                'entry_time': timestamp,
                'entry_order_id': trade.get('entry_order', {}).get('id', ''),
            }
        
        # Track exits: when we go from long to flat (or when position_status shows exit)
        elif decision == 'hold_position':
            position_status = trade.get('position_status', {})
            if position_status:
                profit_target_hit = position_status.get('profit_target_hit', False)
                stop_loss_hit = position_status.get('stop_loss_hit', False)
                current_price = position_status.get('current_price', 0)
                
                # If position was hit, this might be the exit
                if (profit_target_hit or stop_loss_hit) and symbol in current_positions:
                    entry_info = current_positions[symbol]
                    exit_price = current_price
                    exit_qty = position_status.get('quantity', entry_info['quantity'])
                    
                    # Calculate P/L
                    pl = (exit_price - entry_info['entry_price']) * exit_qty
                    pl_pct = ((exit_price - entry_info['entry_price']) / entry_info['entry_price']) * 100
                    
                    exit_reason = 'profit_target_hit' if profit_target_hit else 'stop_loss_hit'
                    
                    position_history[symbol].append({
                        'entry_price': entry_info['entry_price'],
                        'exit_price': exit_price,
                        'quantity': exit_qty,
                        'realized_pl': pl,
                        'realized_pl_pct': pl_pct,
                        'exit_reason': exit_reason,
                        'entry_time': entry_info['entry_time'],
                        'exit_time': timestamp,
                    })
                    
                    # Remove from current positions (position closed)
                    del current_positions[symbol]
        
        # Also check if existing_side changed from long to flat (another exit indicator)
        elif existing_side == 'flat' and symbol in current_positions:
            # Position was closed (went from long to flat)
            # Try to find exit price from position_status if available
            position_status = trade.get('position_status', {})
            if position_status:
                exit_price = position_status.get('current_price', 0)
                if exit_price > 0:
                    entry_info = current_positions[symbol]
                    exit_qty = position_status.get('quantity', entry_info['quantity'])
                    
                    pl = (exit_price - entry_info['entry_price']) * exit_qty
                    pl_pct = ((exit_price - entry_info['entry_price']) / entry_info['entry_price']) * 100
                    
                    position_history[symbol].append({
                        'entry_price': entry_info['entry_price'],
                        'exit_price': exit_price,
                        'quantity': exit_qty,
                        'realized_pl': pl,
                        'realized_pl_pct': pl_pct,
                        'exit_reason': 'unknown',
                        'entry_time': entry_info['entry_time'],
                        'exit_time': timestamp,
                    })
                    
                    del current_positions[symbol]
    
    # Also check position manager file for closed positions (but apply filters)
    positions_file = Path("data/positions/active_positions.json")
    if positions_file.exists():
        with open(positions_file, 'r', encoding='utf-8') as f:
            saved_positions = json.load(f)
        
        for symbol, pos_data in saved_positions.items():
            # Apply filters
            if filter_symbol and symbol != filter_symbol:
                continue
            exit_time = pos_data.get('exit_time') or ''
            if filter_date and exit_time and not exit_time.startswith(filter_date):
                # Only include if exit happened on filter_date
                continue
            
            if pos_data.get('status') != 'open' and pos_data.get('realized_pl') is not None:
                # Check if we already have this trade
                existing = False
                for trade in position_history.get(symbol, []):
                    if abs(trade['entry_price'] - pos_data.get('entry_price', 0)) < 0.01:
                        existing = True
                        break
                
                if not existing:
                    position_history[symbol].append({
                        'symbol': symbol,
                        'entry_price': pos_data.get('entry_price', 0),
                        'exit_price': pos_data.get('exit_price', 0),
                        'quantity': pos_data.get('quantity', 0),
                        'realized_pl': pos_data.get('realized_pl', 0),
                        'realized_pl_pct': pos_data.get('realized_pl_pct', 0),
                        'exit_reason': pos_data.get('exit_reason', ''),
                        'entry_time': pos_data.get('entry_time', ''),
                        'exit_time': pos_data.get('exit_time', ''),
                    })
    
    # Flatten all closed trades
    closed_trades = []
    for symbol, trades_list in position_history.items():
        for trade in trades_list:
            trade['symbol'] = symbol
            closed_trades.append(trade)
    
    # Also check the trade logs for exit information from terminal output patterns
    # Look for entries that show position was closed
    print("\n" + "=" * 80)
    print("CLOSED TRADES ANALYSIS")
    print("=" * 80)
    
    if closed_trades:
        total_profit = 0
        total_loss = 0
        profitable_trades = 0
        losing_trades = 0
        
        print(f"\nFound {len(closed_trades)} closed positions:\n")
        
        for i, trade in enumerate(closed_trades, 1):
            pl = trade['realized_pl']
            pl_pct = trade['realized_pl_pct']
            reason = trade['exit_reason']
            
            if pl > 0:
                total_profit += pl
                profitable_trades += 1
                status = "[PROFIT]"
            else:
                total_loss += abs(pl)
                losing_trades += 1
                status = "[LOSS]"
            
            print(f"Trade {i}: {trade['symbol']}")
            print(f"  Entry: ${trade['entry_price']:.2f} -> Exit: ${trade['exit_price']:.2f}")
            print(f"  Quantity: {trade['quantity']:.6f}")
            print(f"  P/L: ${pl:.2f} ({pl_pct:+.2f}%)")
            print(f"  Reason: {reason}")
            print(f"  Status: {status}")
            print()
        
        net_profit = total_profit - total_loss
        
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Closed Trades: {len(closed_trades)}")
        print(f"  Profitable: {profitable_trades}")
        print(f"  Losing: {losing_trades}")
        print(f"  Win Rate: {(profitable_trades/len(closed_trades)*100):.1f}%")
        print()
        print(f"Total Profit: ${total_profit:.2f}")
        print(f"Total Loss: ${total_loss:.2f}")
        print(f"Net Profit/Loss: ${net_profit:.2f}")
        print(f"Net Return: {(net_profit/(total_profit+total_loss)*100 if (total_profit+total_loss) > 0 else 0):.2f}%")
    else:
        print("No closed positions found in position manager file.")
        print("Checking for active positions...")
        
        if positions_file.exists():
            with open(positions_file, 'r', encoding='utf-8') as f:
                saved_positions = json.load(f)
            
            active = [s for s in saved_positions.values() if s.get('status') == 'open']
            print(f"Active positions: {len(active)}")
            
            for pos in active:
                print(f"  {pos.get('symbol')}: Entry ${pos.get('entry_price', 0):.2f}, "
                      f"Qty: {pos.get('quantity', 0):.6f}")

if __name__ == "__main__":
    import sys
    # Default: filter for BTCUSD trades from today (2025-12-15)
    filter_symbol = "BTCUSD"
    filter_date = "2025-12-15"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            filter_symbol = None
            filter_date = None
        elif sys.argv[1] == "--help":
            print("Usage: python analyze_trades.py [--all]")
            print("  --all: Analyze all trades (not just today's BTC)")
            print("  Default: Analyze only BTCUSD trades from 2025-12-15")
            sys.exit(0)
    
    analyze_trades(filter_symbol=filter_symbol, filter_date=filter_date)

