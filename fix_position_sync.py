"""
Fix script to properly close positions that were closed in Alpaca 
but still marked as open in JSON.

This script:
1. Finds positions that are open in JSON but closed in Alpaca
2. Attempts to get exit information from Alpaca order history
3. Properly closes them in the JSON file with exit details
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from trading.position_manager import PositionManager
from trading.alpaca_client import AlpacaClient


def get_recent_orders(client: AlpacaClient, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Get recent orders for a symbol."""
    try:
        params = {
            "symbols": symbol.upper(),
            "status": "all",  # Get all orders (filled, canceled, etc.)
            "limit": limit,
            "nested": "true",  # Include nested orders
        }
        orders = client._request("GET", "/orders", params=params)
        return orders if isinstance(orders, list) else []
    except Exception as e:
        print(f"  [WARN] Could not fetch orders for {symbol}: {e}")
        return []


def find_closing_order(orders: List[Dict[str, Any]], symbol: str, entry_time: str, side: str) -> Optional[Dict[str, Any]]:
    """Find the order that closed this position."""
    entry_dt = None
    try:
        entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
    except:
        pass
    
    # Sort orders by time (most recent first)
    sorted_orders = sorted(
        orders,
        key=lambda o: o.get("filled_at") or o.get("created_at") or "",
        reverse=True
    )
    
    for order in sorted_orders:
        order_symbol = order.get("symbol", "").upper()
        if order_symbol != symbol.upper():
            continue
        
        order_status = order.get("status", "").lower()
        order_side = order.get("side", "").lower()
        
        # Check if this is a closing order
        is_closing = False
        if side == "long" and order_side == "sell":
            is_closing = True
        elif side == "short" and order_side == "buy":
            is_closing = True
        
        if not is_closing:
            continue
        
        # Check if order was filled
        if order_status not in ("filled", "closed"):
            continue
        
        # Check if order was after entry
        order_time_str = order.get("filled_at") or order.get("created_at") or ""
        if order_time_str and entry_dt:
            try:
                order_dt = datetime.fromisoformat(order_time_str.replace("Z", "+00:00"))
                if order_dt <= entry_dt:
                    continue  # Order was before entry, skip
            except:
                pass  # Can't parse time, assume it's valid
        
        # This looks like a closing order
        return order
    
    return None


def get_exit_info_from_alpaca(client: AlpacaClient, symbol: str, position_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Try to get exit information from Alpaca order history."""
    orders = get_recent_orders(client, symbol)
    
    if not orders:
        return None
    
    entry_time = position_data.get("entry_time")
    side = position_data.get("side", "long")
    
    closing_order = find_closing_order(orders, symbol, entry_time, side)
    
    if not closing_order:
        return None
    
    # Extract exit information
    exit_price = None
    exit_time = None
    exit_reason = "external_close"
    
    # Try to get filled average price
    exit_price = closing_order.get("filled_avg_price")
    if exit_price:
        exit_price = float(exit_price)
    
    # Try limit price if no filled price
    if not exit_price:
        exit_price = closing_order.get("limit_price")
        if exit_price:
            exit_price = float(exit_price)
    
    # Try stop price for stop orders
    if not exit_price:
        exit_price = closing_order.get("stop_price")
        if exit_price:
            exit_price = float(exit_price)
    
    exit_time = closing_order.get("filled_at") or closing_order.get("created_at")
    
    # Determine reason
    order_type = closing_order.get("order_type", "").lower()
    if "stop" in order_type:
        exit_reason = "stop_loss_hit"
    elif "limit" in order_type:
        # Check if it's a take-profit by comparing to profit target
        profit_target = position_data.get("profit_target_price")
        if profit_target and exit_price:
            if side == "long" and exit_price >= profit_target * 0.99:  # Within 1% of target
                exit_reason = "profit_target_hit"
            elif side == "short" and exit_price <= profit_target * 1.01:
                exit_reason = "profit_target_hit"
    
    if exit_price and exit_time:
        return {
            "exit_price": exit_price,
            "exit_time": exit_time,
            "exit_reason": exit_reason,
            "order_id": closing_order.get("id"),
        }
    
    return None


def fix_positions():
    """Main fix function."""
    print("=" * 80)
    print("POSITION SYNC FIX")
    print("=" * 80)
    print()
    
    # Initialize position manager
    position_manager = PositionManager()
    
    # Get active positions
    active_positions = position_manager.get_all_positions()
    
    if not active_positions:
        print("[OK] No active positions to check")
        return
    
    print(f"[1] Found {len(active_positions)} active position(s) in JSON")
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
        
        print(f"Found {len(alpaca_symbols)} active positions in Alpaca: {alpaca_symbols if alpaca_symbols else 'None'}")
        print()
        
        # Find positions that need fixing
        json_symbols = {pos.symbol.upper() for pos in active_positions}
        missing_in_alpaca = json_symbols - alpaca_symbols
        
        if not missing_in_alpaca:
            print("[OK] All positions are in sync!")
            return
        
        print(f"[3] Found {len(missing_in_alpaca)} position(s) that need to be closed:")
        print()
        
        fixed_count = 0
        skipped_count = 0
        
        for pos in active_positions:
            if pos.symbol.upper() not in missing_in_alpaca:
                continue
            
            print(f"Processing {pos.symbol}...")
            
            # Try to get exit info from Alpaca
            pos_dict = {
                "entry_time": pos.entry_time,
                "side": pos.side,
                "profit_target_price": pos.profit_target_price,
            }
            exit_info = get_exit_info_from_alpaca(client, pos.symbol, pos_dict)
            
            if exit_info:
                # Close with exit information
                print(f"  - Found exit information:")
                print(f"    Exit Price: ${exit_info['exit_price']}")
                print(f"    Exit Time: {exit_info['exit_time']}")
                print(f"    Exit Reason: {exit_info['exit_reason']}")
                
                position_manager.close_position(
                    pos.symbol,
                    exit_info["exit_price"],
                    exit_info["exit_reason"],
                )
                print(f"  [OK] Position closed successfully")
                fixed_count += 1
            else:
                # Close without exit info (use current market price as best estimate)
                print(f"  - No exit order found in Alpaca history")
                print(f"  - Attempting to get current market price...")
                
                try:
                    # Try to get current price using the data_symbol
                    try:
                        # Use the same method as monitor_positions uses
                        from end_to_end import get_current_price_from_features
                        
                        exit_price = get_current_price_from_features(
                            pos.asset_type,
                            pos.data_symbol,
                            "1d",
                            verbose=False,
                            force_live=True
                        )
                        
                        if exit_price and exit_price > 0:
                            print(f"  - Using current market price: ${exit_price}")
                            position_manager.close_position(
                                pos.symbol,
                                exit_price,
                                "closed_externally_no_order_history",
                            )
                            print(f"  [OK] Position closed with current market price")
                            fixed_count += 1
                        else:
                            # Fallback to entry price if we can't get current price
                            print(f"  [WARN] Could not get current price, using entry price as fallback")
                            position_manager.close_position(
                                pos.symbol,
                                pos.entry_price,
                                "closed_externally_unknown_price",
                            )
                            print(f"  [OK] Position closed with entry price as fallback")
                            fixed_count += 1
                    except ImportError:
                        # Fallback if function not available
                        print(f"  [WARN] Could not import price function, using entry price")
                        position_manager.close_position(
                            pos.symbol,
                            pos.entry_price,
                            "closed_externally_unknown_price",
                        )
                        print(f"  [OK] Position closed with entry price as fallback")
                        fixed_count += 1
                except Exception as e:
                    print(f"  [ERROR] Failed to close position: {e}")
                    import traceback
                    traceback.print_exc()
                    skipped_count += 1
            
            print()
        
        print("=" * 80)
        print(f"SUMMARY:")
        print(f"  - Fixed: {fixed_count} position(s)")
        print(f"  - Skipped: {skipped_count} position(s)")
        print("=" * 80)
    
    except Exception as e:
        print(f"[ERROR] Failed to fix positions: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    fix_positions()
