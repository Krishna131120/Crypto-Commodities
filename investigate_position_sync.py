"""
Investigation script to understand why positions closed in Alpaca 
weren't updated in the JSON file.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from trading.alpaca_client import AlpacaClient


def get_alpaca_orders(client: AlpacaClient, symbol: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get orders from Alpaca, optionally filtered by symbol and status."""
    try:
        params = {}
        if status:
            params["status"] = status  # "all", "open", "closed", "filled", "canceled"
        if symbol:
            params["symbols"] = symbol.upper()
        
        # Get all orders (closed/filled)
        orders = client._request("GET", "/orders", params=params if params else None)
        return orders if isinstance(orders, list) else []
    except Exception as e:
        print(f"Error fetching orders from Alpaca: {e}")
        return []


def analyze_closed_position(client: AlpacaClient, symbol: str, position_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze why a position might have been closed."""
    analysis = {
        "symbol": symbol,
        "entry_time": position_data.get("entry_time"),
        "status": position_data.get("status"),
        "findings": [],
        "likely_reason": None,
        "exit_info_found": False,
        "exit_price": None,
        "exit_time": None,
    }
    
    # Try to find closing orders
    orders = get_alpaca_orders(client, symbol=symbol, status="all")
    
    # Filter to filled/closed orders that would close this position
    entry_time = position_data.get("entry_time")
    side = position_data.get("side", "long")
    
    closing_orders = []
    for order in orders:
        order_symbol = order.get("symbol", "").upper()
        if order_symbol != symbol.upper():
            continue
        
        order_status = order.get("status", "").lower()
        order_side = order.get("side", "").lower()
        order_filled_at = order.get("filled_at") or order.get("created_at")
        
        # Check if this is a closing order (opposite side, after entry)
        is_closing = False
        if side == "long" and order_side == "sell":
            is_closing = True
        elif side == "short" and order_side == "buy":
            is_closing = True
        
        if is_closing and order_status in ("filled", "closed"):
            if order_filled_at and entry_time:
                # Check if order was filled after entry
                try:
                    order_time = datetime.fromisoformat(order_filled_at.replace("Z", "+00:00"))
                    entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                    if order_time > entry_dt:
                        closing_orders.append(order)
                except:
                    closing_orders.append(order)
            else:
                closing_orders.append(order)
    
    if closing_orders:
        # Find the most recent closing order
        latest_order = max(closing_orders, key=lambda o: o.get("filled_at") or o.get("created_at") or "")
        analysis["exit_info_found"] = True
        analysis["exit_price"] = float(latest_order.get("filled_avg_price") or latest_order.get("limit_price") or 0)
        analysis["exit_time"] = latest_order.get("filled_at") or latest_order.get("created_at")
        analysis["exit_order_id"] = latest_order.get("id")
        analysis["exit_order_type"] = latest_order.get("order_type")
        
        # Determine likely reason
        order_type = latest_order.get("order_type", "").lower()
        if "stop" in order_type or "stop_loss" in latest_order.get("legs", []):
            analysis["likely_reason"] = "stop_loss_hit"
            analysis["findings"].append("Position closed by stop-loss order")
        elif "limit" in order_type:
            analysis["likely_reason"] = "profit_target_hit"
            analysis["findings"].append("Position closed by take-profit limit order")
        else:
            analysis["likely_reason"] = "manual_or_broker_closed"
            analysis["findings"].append("Position closed by market order or manually")
        
        analysis["findings"].append(f"Exit order found: {latest_order.get('id')}")
        analysis["findings"].append(f"Exit price: ${analysis['exit_price']}")
        analysis["findings"].append(f"Exit time: {analysis['exit_time']}")
    else:
        analysis["findings"].append("No closing orders found in Alpaca order history")
        analysis["findings"].append("Position may have been closed manually via Alpaca dashboard")
        analysis["likely_reason"] = "unknown_external_close"
    
    # Check if monitoring script would have detected this
    analysis["findings"].append(
        "Position exists in JSON but not in Alpaca - sync mechanism should have caught this"
    )
    analysis["findings"].append(
        "The monitor_positions() function has sync logic but may not have run recently"
    )
    
    return analysis


def main():
    """Main investigation function."""
    print("=" * 80)
    print("POSITION SYNC INVESTIGATION")
    print("=" * 80)
    print()
    
    # Load positions from JSON
    json_path = Path("data/positions/active_positions.json")
    print(f"[1] Loading positions from: {json_path}")
    print("-" * 80)
    
    with open(json_path, "r", encoding="utf-8") as f:
        all_positions = json.load(f)
    
    # Find active positions
    active_positions = {
        symbol: data 
        for symbol, data in all_positions.items() 
        if data.get("status") == "open"
    }
    
    print(f"Found {len(active_positions)} positions marked as 'open' in JSON")
    print()
    
    # Check Alpaca
    print("[2] Checking Alpaca for active positions")
    print("-" * 80)
    
    try:
        client = AlpacaClient()
        alpaca_positions = client.list_positions()
        alpaca_symbols = {
            pos.get("symbol", "").upper() 
            for pos in (alpaca_positions or []) 
            if float(pos.get("qty", 0) or 0) != 0
        }
        
        print(f"Found {len(alpaca_symbols)} active positions in Alpaca: {alpaca_symbols}")
        print()
        
        # Find mismatches
        json_symbols = set(active_positions.keys())
        missing_in_alpaca = json_symbols - alpaca_symbols
        
        if not missing_in_alpaca:
            print("[OK] All positions are in sync!")
            return
        
        print(f"[ISSUE] {len(missing_in_alpaca)} position(s) in JSON but not in Alpaca:")
        for symbol in missing_in_alpaca:
            print(f"  - {symbol}")
        print()
        
        # Investigate each missing position
        print("[3] Investigating why positions weren't synced")
        print("-" * 80)
        
        for symbol in missing_in_alpaca:
            print(f"\nInvestigating {symbol}...")
            print("-" * 60)
            
            pos_data = active_positions[symbol]
            analysis = analyze_closed_position(client, symbol, pos_data)
            
            print(f"Entry Time: {analysis['entry_time']}")
            print(f"Current Status in JSON: {analysis['status']}")
            print()
            print("Findings:")
            for finding in analysis["findings"]:
                print(f"  - {finding}")
            
            if analysis["exit_info_found"]:
                print()
                print("Exit Information Found:")
                print(f"  - Exit Price: ${analysis['exit_price']}")
                print(f"  - Exit Time: {analysis['exit_time']}")
                print(f"  - Likely Reason: {analysis['likely_reason']}")
                print(f"  - Order ID: {analysis.get('exit_order_id', 'N/A')}")
            
            print()
            print("Root Cause Analysis:")
            print("  - The monitor_positions() function in end_to_end.py has sync logic")
            print("    (lines 651-667) that should remove positions not found in Alpaca")
            print("  - However, it calls remove_position() which DELETES the position")
            print("  - It should instead call close_position() to preserve exit information")
            print("  - The monitoring script likely hasn't run since these positions closed")
            print()
    
    except Exception as e:
        print(f"Error during investigation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
