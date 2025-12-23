"""
View Angel One MCX Commodities Positions and P&L

This script displays:
- Current account status (equity, buying power, margin)
- Open positions (MCX commodities only)
- Profit and Loss for each position
- Total unrealized P&L
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any, Dict, List

from trading.angelone_client import AngelOneClient


def show_angelone_positions(filter_mcx: bool = True):
    """Display current Angel One positions with P&L."""
    try:
        print("=" * 80)
        print("ANGEL ONE MCX COMMODITIES - POSITIONS & P&L")
        print("=" * 80)
        print()
        
        # Initialize client
        print("Connecting to Angel One...")
        client = AngelOneClient()
        print("✓ Connected successfully!")
        print()
        
        # Get account details
        account = client.get_account()
        equity = float(account.get("equity", 0))
        buying_power = float(account.get("buying_power", 0))
        cash = float(account.get("cash", 0))
        margin_used = float(account.get("margin_used", 0))
        margin_available = float(account.get("margin_available", 0))
        
        print("=" * 80)
        print("ACCOUNT STATUS")
        print("=" * 80)
        print(f"Equity:           ₹{equity:,.2f}")
        print(f"Cash:             ₹{cash:,.2f}")
        print(f"Buying Power:     ₹{buying_power:,.2f}")
        print(f"Margin Used:      ₹{margin_used:,.2f}")
        print(f"Margin Available: ₹{margin_available:,.2f}")
        print()
        
        # Get all positions
        all_positions = client.list_positions()
        
        # Filter MCX positions if requested
        if filter_mcx:
            positions = [
                pos for pos in all_positions 
                if pos.get("exchange_segment", "").upper() == "MCX"
            ]
            print("=" * 80)
            print(f"MCX COMMODITIES POSITIONS ({len(positions)} open)")
            print("=" * 80)
        else:
            positions = all_positions
            print("=" * 80)
            print(f"ALL POSITIONS ({len(positions)} open)")
            print("=" * 80)
        
        if not positions:
            print("\nNo open positions.")
            return
        
        print()
        
        # Calculate totals
        total_unrealized_pl = 0.0
        total_market_value = 0.0
        
        # Display each position
        for i, pos in enumerate(positions, 1):
            symbol = pos.get("symbol", "?")
            qty = float(pos.get("qty", 0))
            avg_entry = float(pos.get("avg_entry_price", 0))
            current_price = float(pos.get("ltp", 0))  # Last Traded Price
            market_value = float(pos.get("market_value", 0))
            unrealized_pl = float(pos.get("unrealized_pl", 0))
            exchange = pos.get("exchange_segment", "?")
            
            # Determine side
            side = "LONG" if qty > 0 else "SHORT"
            
            # Calculate P&L percentage
            if avg_entry > 0:
                if side == "LONG":
                    pl_pct = ((current_price - avg_entry) / avg_entry) * 100
                else:  # SHORT
                    pl_pct = ((avg_entry - current_price) / avg_entry) * 100
            else:
                pl_pct = 0.0
            
            # Accumulate totals
            total_unrealized_pl += unrealized_pl
            total_market_value += market_value
            
            # Display position
            print(f"[{i}] {symbol} ({exchange}) - {side}")
            print(f"    Quantity:        {abs(qty):.2f} lots")
            print(f"    Avg Entry Price: ₹{avg_entry:,.2f}")
            print(f"    Current Price:    ₹{current_price:,.2f}")
            print(f"    Market Value:    ₹{market_value:,.2f}")
            print(f"    Unrealized P/L:   ₹{unrealized_pl:+,.2f} ({pl_pct:+.2f}%)")
            print()
        
        # Display summary
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Positions:        {len(positions)}")
        print(f"Total Market Value:     ₹{total_market_value:,.2f}")
        print(f"Total Unrealized P/L:   ₹{total_unrealized_pl:+,.2f}")
        
        if equity > 0:
            total_pl_pct = (total_unrealized_pl / equity) * 100
            print(f"Total P/L % of Equity: {total_pl_pct:+.2f}%")
        
        print()
        print("=" * 80)
        
    except Exception as exc:
        print(f"\n[ERROR] Failed to load positions: {exc}")
        print("\nMake sure your .env file has:")
        print("  - ANGEL_ONE_API_KEY")
        print("  - ANGEL_ONE_CLIENT_ID")
        print("  - ANGEL_ONE_PASSWORD")
        print("  - ANGEL_ONE_TOTP_SECRET")
        import traceback
        traceback.print_exc()


def show_trade_history():
    """Display recent trade history from logs if available."""
    from pathlib import Path
    import json
    
    # Check for commodities trade logs
    log_paths = [
        Path("logs/trading/commodities_trades.jsonl"),
        Path("logs/trading/crypto_trades.jsonl"),
    ]
    
    trades = []
    for log_path in log_paths:
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        trade = json.loads(line)
                        # Filter for commodities/MCX trades
                        if "commodities" in log_path.name or trade.get("asset_type") == "commodities":
                            trades.append(trade)
                    except json.JSONDecodeError:
                        continue
    
    if not trades:
        print("\nNo trade history found in logs.")
        return
    
    # Show last 20 trades
    recent_trades = trades[-20:]
    
    print("\n" + "=" * 80)
    print(f"RECENT TRADE HISTORY (last {len(recent_trades)} trades)")
    print("=" * 80)
    print()
    
    for trade in recent_trades:
        symbol = trade.get("trading_symbol", trade.get("symbol", "?"))
        decision = trade.get("decision", "?")
        timestamp = trade.get("timestamp", "?")
        price = trade.get("current_price", trade.get("price", 0))
        qty = trade.get("entry_qty", trade.get("qty", 0))
        
        print(f"[{timestamp}] {symbol}: {decision.upper()}")
        if price > 0:
            print(f"    Price: ₹{price:,.2f}, Qty: {qty}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="View Angel One MCX commodities positions and P&L"
    )
    parser.add_argument(
        "--all-exchanges",
        action="store_true",
        help="Show positions from all exchanges (not just MCX)",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show recent trade history from logs",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show positions and trade history",
    )
    
    args = parser.parse_args()
    
    # Show positions
    show_angelone_positions(filter_mcx=not args.all_exchanges)
    
    # Show history if requested
    if args.all or args.history:
        show_trade_history()


if __name__ == "__main__":
    main()

