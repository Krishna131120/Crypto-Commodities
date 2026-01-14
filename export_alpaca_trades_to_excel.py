"""
Export accurate trade history from Alpaca activities to Excel.
Only includes executed/filled trades, excludes rejected orders.
Groups by currency and calculates profit/loss.
Date range: January 5-6, 2026
"""

import json
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional
import traceback

try:
    from trading.alpaca_client import AlpacaClient
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("[ERROR] Alpaca client not available!")


def fetch_alpaca_activities(client: AlpacaClient, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Fetch all account activities (filled orders) from Alpaca API."""
    activities = []
    
    try:
        # Alpaca Activities API endpoint: /v2/account/activities/{activity_type}
        # Activity types: FILL (filled orders), ORDER, etc.
        
        params = {
                    "activity_types": "FILL",  # Only get filled orders
            "date": None,  # Will use after/until
            "until": end_date,
            "after": start_date,
            "direction": "desc",  # Most recent first
            "page_size": 100,  # Max allowed by Alpaca
        }
        
        print(f"[INFO] Fetching FILL activities from {start_date} to {end_date}...")
        
        # Primary method: Use orders endpoint (more reliable for pagination)
        all_orders = []
        page_token = None
        max_pages = 50  # Safety limit
        
        try:
            params_orders = {
                "status": "filled",
                "after": start_date,
                "until": end_date,
                "limit": 100,  # Max page size
                "direction": "desc",
            }
            
            for page_num in range(max_pages):
                if page_token:
                    params_orders["page_token"] = page_token
                
                try:
                    response = client._request("GET", "/orders", params=params_orders)
                    
                    if isinstance(response, list):
                        page_orders = response
                        all_orders.extend(page_orders)
                        print(f"  Page {page_num + 1}: {len(page_orders)} orders")
                        if len(page_orders) < params_orders["limit"]:
                            break  # Last page
                        # For list response, try to get next page by using last order's submitted_at
                        if len(page_orders) > 0:
                            last_time = page_orders[-1].get("submitted_at") or page_orders[-1].get("created_at")
                            if last_time:
                                params_orders["until"] = last_time
                        else:
                            break
                    else:
                        # Dict response with pagination
                        page_orders = response.get("orders", [])
                        all_orders.extend(page_orders)
                        print(f"  Page {page_num + 1}: {len(page_orders)} orders")
                        
                        page_token = response.get("next_page_token")
                        if not page_token or len(page_orders) < params_orders["limit"]:
                            break
                
                except Exception as e:
                    print(f"[WARN] Error on page {page_num + 1}: {e}")
                    break
            
            activities = all_orders
            print(f"[OK] Total fetched: {len(activities)} filled orders")
            
        except Exception as e1:
            print(f"[WARN] Orders pagination failed: {e1}")
        
        # Fallback: Try activities endpoint
        if len(activities) == 0:
            try:
                result = client._request("GET", "/account/activities/FILL", params=params)
                if isinstance(result, list):
                    activities.extend(result)
                    print(f"[OK] Fetched {len(result)} FILL activities")
                elif isinstance(result, dict):
                    activities.extend(result.get("activities", []))
                    print(f"[OK] Fetched {len(activities)} FILL activities")
            except Exception as e1:
                print(f"[WARN] Activities endpoint failed: {e1}")
        
        # Also fetch from activities with different approach
        if len(activities) == 0:
            print(f"[INFO] Trying all activity types...")
            for activity_type in ["FILL", "FILLS"]:
                try:
                    result = client._request("GET", f"/account/activities/{activity_type}", params={
                        "after": start_date,
                        "until": end_date,
                        "page_size": 500,
                    })
                    if isinstance(result, list):
                        activities.extend(result)
                    elif isinstance(result, dict):
                        activities.extend(result.get("activities", []))
                except:
                    pass
        
        print(f"[OK] Total activities/orders fetched: {len(activities)}")
        return activities
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch Alpaca activities: {e}")
        traceback.print_exc()
        return []


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol format."""
    if not symbol:
        return ""
    return symbol.replace("/", "").upper()


def parse_alpaca_order(order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse an Alpaca order/activity into a standardized trade format."""
    try:
        # Get symbol
        symbol = order.get("symbol", "")
        if not symbol:
            return None
        
        symbol = normalize_symbol(symbol)
        
        # Get side
        side = order.get("side", "").lower()
        if side not in ["buy", "sell"]:
            return None
        
        # Get filled details - these are the actual executed trades
        filled_qty_str = order.get("filled_qty") or order.get("qty") or "0"
        filled_price_str = order.get("filled_avg_price") or order.get("avg_fill_price") or "0"
        
        # For activities format
        if "transaction_time" in order:
            filled_qty_str = order.get("quantity") or filled_qty_str
            filled_price_str = order.get("price") or filled_price_str
            side = order.get("side", "").lower()
            if "sell" in side.lower():
                side = "sell"
            elif "buy" in side.lower():
                side = "buy"
        
        try:
            filled_qty = float(filled_qty_str) if filled_qty_str else 0.0
            filled_price = float(filled_price_str) if filled_price_str else 0.0
        except:
            return None
        
        # Must have both qty and price
        if filled_qty == 0 or filled_price == 0:
            return None
        
        # Get timestamps
        filled_at = order.get("filled_at") or order.get("transaction_time") or order.get("created_at") or order.get("submitted_at")
        if not filled_at:
            return None
        
        # Calculate notional
        notional = filled_qty * filled_price
        
        return {
            "symbol": symbol,
            "side": side,
            "qty": filled_qty,
            "price": filled_price,
            "notional": notional,
            "filled_at": filled_at,
            "order_id": order.get("id") or order.get("order_id", ""),
            "status": order.get("status", ""),
        }
    except Exception as e:
        print(f"[WARN] Error parsing order: {e}")
        return None


def calculate_fifo_pl(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate P/L using FIFO matching."""
    positions = defaultdict(list)  # symbol -> list of buys
    closed_trades = []
    
    # Sort by time
    sorted_trades = sorted(trades, key=lambda x: x.get("filled_at", ""))
    
    for trade in sorted_trades:
        symbol = trade["symbol"]
        side = trade["side"]
        
        if side == "buy":
            # Add to position queue
            positions[symbol].append({
                "qty": trade["qty"],
                "price": trade["price"],
                "notional": trade["notional"],
                "time": trade["filled_at"],
                "order_id": trade.get("order_id", ""),
            })
        elif side == "sell":
            # Match against buys (FIFO)
            sell_qty = trade["qty"]
            sell_price = trade["price"]
            
            while sell_qty > 0 and positions[symbol]:
                buy = positions[symbol][0]
                matched_qty = min(sell_qty, buy["qty"])
                
                # Calculate P/L
                pl = (sell_price - buy["price"]) * matched_qty
                pl_pct = ((sell_price - buy["price"]) / buy["price"]) * 100 if buy["price"] > 0 else 0
                
                closed_trades.append({
                    "symbol": symbol,
                    "entry_price": buy["price"],
                    "exit_price": sell_price,
                    "qty": matched_qty,
                    "entry_time": buy["time"],
                    "exit_time": trade["filled_at"],
                    "realized_pl": pl,
                    "realized_pl_pct": pl_pct,
                    "entry_order_id": buy.get("order_id", ""),
                    "exit_order_id": trade.get("order_id", ""),
                })
                
                sell_qty -= matched_qty
                buy["qty"] -= matched_qty
                
                if buy["qty"] <= 0:
                    positions[symbol].pop(0)
    
    return {
        "closed_trades": closed_trades,
        "open_positions": {k: v for k, v in positions.items() if v},
    }


def main():
    """Main function to export trades to Excel."""
    print("=" * 120)
    print("EXPORT ALPACA TRADE HISTORY TO EXCEL")
    print("Date Range: January 5-6, 2026")
    print("=" * 120)
    print()
    
    if not ALPACA_AVAILABLE:
        print("[ERROR] Cannot proceed without Alpaca client!")
        return
    
    # Date range: Jan 5-6, 2026
    start_date = datetime(2026, 1, 5, 0, 0, 0, tzinfo=timezone.utc)
    end_date = datetime(2026, 1, 7, 0, 0, 0, tzinfo=timezone.utc)  # Up to end of Jan 6
    
    start_iso = start_date.isoformat().replace("+00:00", "Z")
    end_iso = end_date.isoformat().replace("+00:00", "Z")
    
    print(f"Fetching trades from: {start_iso}")
    print(f"                   to: {end_iso}")
    print()
    
    # Fetch from Alpaca
    client = AlpacaClient()
    activities = fetch_alpaca_activities(client, start_iso, end_iso)
    
    if not activities:
        print("[ERROR] No activities fetched from Alpaca!")
        return
    
    # Parse all orders
    print()
    print("[INFO] Parsing orders...")
    trades = []
    for activity in activities:
        parsed = parse_alpaca_order(activity)
        if parsed:
            trades.append(parsed)
    
    print(f"[OK] Parsed {len(trades)} valid executed trades")
    print()
    
    if not trades:
        print("[ERROR] No valid trades found!")
        return
    
    # Group by symbol
    by_symbol = defaultdict(lambda: {"buys": [], "sells": []})
    for trade in trades:
        by_symbol[trade["symbol"]]["buys" if trade["side"] == "buy" else "sells"].append(trade)
    
    # Calculate P/L
    print("[INFO] Calculating profit/loss using FIFO matching...")
    pl_results = calculate_fifo_pl(trades)
    
    # Prepare Excel data
    print("[INFO] Preparing Excel export...")
    
    # Sheet 1: All Trades (Detailed)
    all_trades_data = []
    for trade in sorted(trades, key=lambda x: x.get("filled_at", "")):
        all_trades_data.append({
            "Date/Time": trade.get("filled_at", ""),
            "Symbol": trade["symbol"],
            "Side": trade["side"].upper(),
            "Quantity": trade["qty"],
            "Price": trade["price"],
            "Notional": trade["notional"],
            "Order ID": trade.get("order_id", ""),
        })
    
    df_all = pd.DataFrame(all_trades_data)
    
    # Sheet 2: Summary by Currency
    summary_data = []
    for symbol in sorted(by_symbol.keys()):
        buys = by_symbol[symbol]["buys"]
        sells = by_symbol[symbol]["sells"]
        
        total_buys = sum(t["notional"] for t in buys)
        total_sells = sum(t["notional"] for t in sells)
        net_cash = total_sells - total_buys
        
        # Get P/L from closed trades
        closed = [t for t in pl_results["closed_trades"] if t["symbol"] == symbol]
        total_pl = sum(t["realized_pl"] for t in closed)
        wins = [t for t in closed if t["realized_pl"] > 0]
        losses = [t for t in closed if t["realized_pl"] < 0]
        
        summary_data.append({
            "Symbol": symbol,
            "Total Buy Orders": len(buys),
            "Total Sell Orders": len(sells),
            "Total Cash Out (Buys)": total_buys,
            "Total Cash In (Sells)": total_sells,
            "Net Cash Flow": net_cash,
            "Realized P/L": total_pl,
            "Closed Positions": len(closed),
            "Wins": len(wins),
            "Losses": len(losses),
            "Total Wins $": sum(t["realized_pl"] for t in wins),
            "Total Losses $": abs(sum(t["realized_pl"] for t in losses)),
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Sheet 3: Closed Positions (P/L Details)
    closed_data = []
    for trade in sorted(pl_results["closed_trades"], key=lambda x: x.get("exit_time", "")):
        closed_data.append({
            "Symbol": trade["symbol"],
            "Entry Time": trade["entry_time"],
            "Exit Time": trade["exit_time"],
            "Quantity": trade["qty"],
            "Entry Price": trade["entry_price"],
            "Exit Price": trade["exit_price"],
            "Realized P/L": trade["realized_pl"],
            "Realized P/L %": trade["realized_pl_pct"],
            "Entry Order ID": trade.get("entry_order_id", ""),
            "Exit Order ID": trade.get("exit_order_id", ""),
        })
    
    df_closed = pd.DataFrame(closed_data)
    
    # Sheet 4: Open Positions
    open_data = []
    for symbol, positions in pl_results["open_positions"].items():
        for pos in positions:
            open_data.append({
                "Symbol": symbol,
                "Entry Time": pos["time"],
                "Quantity": pos["qty"],
                "Entry Price": pos["price"],
                "Cost Basis": pos["notional"],
                "Order ID": pos.get("order_id", ""),
            })
    
    df_open = pd.DataFrame(open_data)
    
    # Export to Excel
    excel_file = Path("Alpaca_Trade_History_Jan5_6_2026.xlsx")
    print(f"[INFO] Writing to Excel: {excel_file}")
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df_all.to_excel(writer, sheet_name='All Trades', index=False)
        df_summary.to_excel(writer, sheet_name='Summary by Currency', index=False)
        df_closed.to_excel(writer, sheet_name='Closed Positions P&L', index=False)
        df_open.to_excel(writer, sheet_name='Open Positions', index=False)
    
    print()
    print("=" * 120)
    print("EXPORT COMPLETE!")
    print("=" * 120)
    print(f"Excel file: {excel_file.absolute()}")
    print()
    print(f"Total Trades: {len(trades)}")
    print(f"Currencies: {len(by_symbol)}")
    print(f"Closed Positions: {len(pl_results['closed_trades'])}")
    print(f"Open Positions: {sum(len(p) for p in pl_results['open_positions'].values())}")
    print()
    
    # Print summary
    total_pl = sum(t["realized_pl"] for t in pl_results["closed_trades"])
    print(f"Total Realized P/L: ${total_pl:,.2f}")
    print()


if __name__ == "__main__":
    main()
