"""Backfill missing SELL entries from Alpaca into trade logs."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

print("=" * 80)
print("BACKFILLING MISSING SELL ENTRIES FROM ALPACA")
print("=" * 80)

# Initialize Alpaca client
try:
    from trading.alpaca_client import AlpacaClient
    from trading.symbol_universe import find_by_trading_symbol
    client = AlpacaClient()
    print("\n[OK] Alpaca client initialized")
except Exception as e:
    print(f"\n[ERROR] Failed to initialize: {e}")
    exit(1)

# Date range
start_date = datetime(2026, 1, 17, 0, 0, 0, tzinfo=timezone.utc)
end_date = datetime(2026, 1, 19, 23, 59, 59, tzinfo=timezone.utc)

# Get orders from Alpaca
print("\nFetching orders from Alpaca...")
try:
    orders_url = f"{client.config.base_url}/orders"
    params = {
        "status": "all",
        "after": start_date.isoformat(),
        "until": end_date.isoformat(),
        "limit": 500,
        "nested": "true"
    }
    response = client._session.get(orders_url, params=params, timeout=30)
    response.raise_for_status()
    alpaca_orders = response.json()
    print(f"[OK] Fetched {len(alpaca_orders)} orders")
except Exception as e:
    print(f"[ERROR] Failed to fetch orders: {e}")
    exit(1)

# Read existing logs
log_file = Path("logs/trading/crypto_trades.jsonl")
logged_trades = []
if log_file.exists():
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    logged_trades.append(json.loads(line))
                except:
                    pass

print(f"[OK] Read {len(logged_trades)} existing log entries")

# Find missing SELL orders
print("\nAnalyzing orders...")
missing_sells = []

for order in alpaca_orders:
    side = order.get("side", "").lower()
    status = order.get("status", "").lower()
    filled_qty = float(order.get("filled_qty", 0) or 0)
    
    if side == "sell" and status in ["filled", "partially_filled"] and filled_qty > 0:
        symbol = order.get("symbol", "").replace("/", "")  # Convert BTC/USD to BTCUSD
        filled_at = order.get("filled_at", "")
        filled_avg_price = order.get("filled_avg_price")
        order_id = order.get("id")
        
        if not filled_at:
            continue
        
        # Parse timestamp
        try:
            order_time = datetime.fromisoformat(filled_at.replace("Z", "+00:00"))
        except:
            continue
        
        # Check if this order is already logged
        is_logged = False
        for logged in logged_trades:
            logged_time_str = logged.get("timestamp", "")
            if not logged_time_str:
                continue
            
            try:
                if logged_time_str.endswith("Z"):
                    logged_time_str = logged_time_str[:-1] + "+00:00"
                logged_time = datetime.fromisoformat(logged_time_str)
                
                # Check if within 5 minutes and same symbol
                logged_symbol = logged.get("trading_symbol", logged.get("asset", "")).replace("/", "")
                if logged_symbol == symbol:
                    time_diff = abs((order_time - logged_time.replace(tzinfo=timezone.utc)).total_seconds())
                    if time_diff < 300:  # 5 minutes
                        # Check if it's a sell
                        decision = logged.get("decision", "")
                        trade_side = logged.get("trade_side", "")
                        if decision in ["exit_long", "exit_short", "exit_position", "sell"] or trade_side == "sell":
                            is_logged = True
                            break
            except:
                continue
        
        if not is_logged:
            missing_sells.append({
                "symbol": symbol,
                "order": order,
                "order_time": order_time,
                "qty": filled_qty,
                "price": filled_avg_price,
                "order_id": order_id
            })

print(f"\n[FOUND] {len(missing_sells)} missing SELL entries")

if not missing_sells:
    print("\n[OK] No missing SELL entries found. All orders are logged.")
    exit(0)

# Get position manager to find entry prices
print("\nFetching position history to find entry prices...")
from trading.position_manager import PositionManager
position_manager = PositionManager()

# Backfill missing entries
print("\n" + "=" * 80)
print("BACKFILLING MISSING SELL ENTRIES")
print("=" * 80)

backfilled_count = 0

for missing in missing_sells:
    symbol = missing["symbol"]
    order = missing["order"]
    order_time = missing["order_time"]
    qty = float(missing["qty"])
    price = float(missing["price"] or 0)
    order_id = missing["order_id"]
    
    # Find asset mapping
    asset_mapping = find_by_trading_symbol(symbol)
    if not asset_mapping:
        print(f"\n[SKIP] {symbol}: No asset mapping found")
        continue
    
    # Try to find entry price from position manager or order history
    entry_price = None
    entry_time = None
    
    # Check if there's a position file with entry info
    try:
        position = position_manager.get_position(symbol)
        if position and position.entry_price:
            entry_price = position.entry_price
            entry_time_str = position.entry_time
            try:
                if entry_time_str.endswith("Z"):
                    entry_time_str = entry_time_str[:-1] + "+00:00"
                entry_time = datetime.fromisoformat(entry_time_str)
            except:
                pass
    except:
        pass
    
    # If no entry price from position, try to find matching BUY order
    if not entry_price:
        # Look for BUY order for this symbol before the SELL
        for buy_order in alpaca_orders:
            if (buy_order.get("side", "").lower() == "buy" 
                and buy_order.get("symbol", "").replace("/", "") == symbol
                and buy_order.get("status", "").lower() in ["filled", "partially_filled"]):
                buy_time_str = buy_order.get("filled_at") or buy_order.get("created_at", "")
                if buy_time_str:
                    try:
                        buy_time = datetime.fromisoformat(buy_time_str.replace("Z", "+00:00"))
                        if buy_time < order_time:
                            entry_price = buy_order.get("filled_avg_price")
                            entry_time = buy_time
                            break
                    except:
                        pass
    
    # Calculate P/L if we have entry price
    realized_pl = None
    realized_pl_pct = None
    if entry_price and price:
        try:
            realized_pl = (price - entry_price) * qty
            realized_pl_pct = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        except:
            pass
    
    # Create log entry
    log_entry = {
        "asset": asset_mapping.logical_name,
        "data_symbol": asset_mapping.data_symbol,
        "trading_symbol": symbol,
        "asset_type": asset_mapping.asset_type,
        "current_price": price,
        "model_action": "flat",
        "target_side": "flat",
        "existing_side": "long",  # Assuming we're closing a long position
        "existing_qty": qty,
        "timestamp": order_time.isoformat().replace("+00:00", "Z"),
        "dry_run": False,
        "decision": "exit_long",  # New format
        "trade_qty": qty,
        "trade_side": "sell",
        "final_side": "short",  # New format to indicate SELL
        "final_qty": qty,  # New format to indicate SELL
        "close_order": {
            "id": order_id,
            "status": "filled",
            "filled_at": order.get("filled_at"),
            "filled_avg_price": price,
            "filled_qty": qty
        },
        "exit_reason": "backfilled_from_alpaca",
        "entry_price": entry_price,
        "exit_price": price,
        "realized_pl": realized_pl,
        "realized_pl_pct": realized_pl_pct,
        "market_value_at_exit": price * qty,
        "alpaca_verified": True,
        "backfilled": True,
        "backfill_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }
    
    if entry_price:
        log_entry["pl_summary"] = {
            "entry_price": entry_price,
            "exit_price": price,
            "quantity": qty,
            "realized_pl": realized_pl,
            "realized_pl_pct": realized_pl_pct,
            "is_profit": realized_pl > 0 if realized_pl is not None else None
        }
    
    # Append to log file
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        print(f"\n[BACKFILLED] {symbol}: SELL {qty} @ ${price}")
        if entry_price:
            print(f"  Entry: ${entry_price}, P/L: ${realized_pl:+.2f} ({realized_pl_pct:+.2f}%)")
        else:
            print(f"  Entry price: Unknown")
        print(f"  Order ID: {order_id[:20]}...")
        backfilled_count += 1
    except Exception as e:
        print(f"\n[ERROR] Failed to backfill {symbol}: {e}")

print("\n" + "=" * 80)
print("BACKFILL SUMMARY")
print("=" * 80)
print(f"Missing SELL entries found: {len(missing_sells)}")
print(f"Successfully backfilled: {backfilled_count}")
print(f"Failed: {len(missing_sells) - backfilled_count}")
print("=" * 80)
