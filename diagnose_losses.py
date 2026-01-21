"""Diagnose why trades are losing - comprehensive analysis."""
import json
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

print("=" * 80)
print("DIAGNOSING WHY TRADES ARE LOSING")
print("=" * 80)

# Read trade logs
log_file = Path("logs/trading/crypto_trades.jsonl")
trades = []

if log_file.exists():
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trade = json.loads(line)
                    timestamp_str = trade.get("timestamp", "")
                    if timestamp_str:
                        try:
                            if timestamp_str.endswith("Z"):
                                timestamp_str = timestamp_str[:-1] + "+00:00"
                            trade_time = datetime.fromisoformat(timestamp_str)
                            trade_time = trade_time.replace(tzinfo=None)
                            
                            # Focus on Jan 17-19
                            if datetime(2026, 1, 17) <= trade_time <= datetime(2026, 1, 19, 23, 59, 59):
                                trades.append((trade_time, trade))
                        except:
                            pass
                except:
                    continue

print(f"\nAnalyzing {len(trades)} trades from Jan 17-19...")

# Get Alpaca orders for accurate analysis
print("\nFetching Alpaca orders...")
try:
    from trading.alpaca_client import AlpacaClient
    client = AlpacaClient()
    
    orders_url = f"{client.config.base_url}/orders"
    params = {
        "status": "all",
        "after": datetime(2026, 1, 17, 0, 0, 0, tzinfo=timezone.utc).isoformat(),
        "until": datetime(2026, 1, 19, 23, 59, 59, tzinfo=timezone.utc).isoformat(),
        "limit": 500,
        "nested": "true"
    }
    response = client._session.get(orders_url, params=params, timeout=30)
    response.raise_for_status()
    alpaca_orders = response.json()
    
    # Match trades with orders
    orders_by_symbol = defaultdict(list)
    for order in alpaca_orders:
        if order.get("status", "").lower() in ["filled", "partially_filled"]:
            symbol = order.get("symbol", "").replace("/", "")
            orders_by_symbol[symbol].append(order)
    
    print(f"[OK] Found {len(alpaca_orders)} orders")
except Exception as e:
    print(f"[ERROR] {e}")
    orders_by_symbol = {}

# Analyze each losing trade
print("\n" + "=" * 80)
print("ANALYZING LOSING TRADES")
print("=" * 80)

issues_found = {
    "stop_loss_too_wide": [],
    "stop_loss_not_enforced": [],
    "entry_at_peak": [],
    "exit_too_early": [],
    "low_confidence_entry": [],
    "momentum_filter_failed": [],
    "model_prediction_wrong": [],
}

# Match BUY and SELL orders
for symbol in sorted(orders_by_symbol.keys()):
    symbol_orders = sorted(orders_by_symbol[symbol], key=lambda x: x.get("filled_at", ""))
    buys = [o for o in symbol_orders if o.get("side", "").lower() == "buy"]
    sells = [o for o in symbol_orders if o.get("side", "").lower() == "sell"]
    
    if not buys or not sells:
        continue
    
    # Match FIFO
    buy_queue = buys.copy()
    sell_queue = sells.copy()
    
    while buy_queue and sell_queue:
        buy = buy_queue[0]
        sell = sell_queue[0]
        
        if sell.get("filled_at", "") < buy.get("filled_at", ""):
            sell_queue.pop(0)
            continue
        
        buy_price = float(buy.get("filled_avg_price", 0) or 0)
        sell_price = float(sell.get("filled_avg_price", 0) or 0)
        buy_time = datetime.fromisoformat(buy.get("filled_at", "").replace("Z", "+00:00"))
        sell_time = datetime.fromisoformat(sell.get("filled_at", "").replace("Z", "+00:00"))
        qty = float(buy.get("filled_qty", 0) or 0)
        
        if buy_price <= 0 or sell_price <= 0 or qty <= 0:
            if buy_price <= 0:
                buy_queue.pop(0)
            if sell_price <= 0:
                sell_queue.pop(0)
            continue
        
        # Calculate P/L
        pl = (sell_price - buy_price) * qty
        pl_pct = ((sell_price - buy_price) / buy_price) * 100
        
        if pl < 0:  # Losing trade
            print(f"\n{symbol}: LOSS ${pl:.2f} ({pl_pct:.2f}%)")
            print(f"  Entry: [{buy_time.strftime('%Y-%m-%d %H:%M')}] ${buy_price:.6f}")
            print(f"  Exit:  [{sell_time.strftime('%Y-%m-%d %H:%M')}] ${sell_price:.6f}")
            print(f"  Hold: {(sell_time - buy_time).total_seconds() / 3600:.2f} hours")
            
            # Check if stop-loss was supposed to protect
            # Default stop-loss is 8% for crypto
            expected_stop_loss = buy_price * 0.92  # 8% down
            actual_loss_pct = abs(pl_pct)
            
            if actual_loss_pct > 8.0:
                print(f"  [ISSUE] Loss ({actual_loss_pct:.2f}%) exceeds stop-loss (8%)")
                print(f"     Expected stop-loss price: ${expected_stop_loss:.6f}")
                print(f"     Actual exit price: ${sell_price:.6f}")
                issues_found["stop_loss_not_enforced"].append({
                    "symbol": symbol,
                    "expected_stop": expected_stop_loss,
                    "actual_exit": sell_price,
                    "loss_pct": actual_loss_pct,
                })
            
            # Check entry conditions from logs
            for trade_time, trade in trades:
                if (trade.get("trading_symbol", "") == symbol and 
                    abs((trade_time - buy_time.replace(tzinfo=None)).total_seconds()) < 300):
                    # Found matching entry log
                    confidence = trade.get("confidence", 0) or 0
                    predicted_return = trade.get("predicted_return", 0) or 0
                    model_action = trade.get("model_action", "")
                    
                    print(f"  Entry Conditions:")
                    print(f"    Confidence: {confidence*100:.1f}%")
                    print(f"    Predicted Return: {predicted_return*100:.2f}%")
                    print(f"    Model Action: {model_action}")
                    
                    if confidence < 0.10:
                        print(f"    [ISSUE] LOW CONFIDENCE: {confidence*100:.1f}% < 10%")
                        issues_found["low_confidence_entry"].append({
                            "symbol": symbol,
                            "confidence": confidence,
                        })
                    
                    if predicted_return < 0:
                        print(f"    [ISSUE] MODEL PREDICTED NEGATIVE RETURN")
                        issues_found["model_prediction_wrong"].append({
                            "symbol": symbol,
                            "predicted_return": predicted_return,
                        })
                    
                    # Check momentum filter
                    momentum_skipped = trade.get("decision") == "skipped_momentum_filter"
                    if momentum_skipped:
                        print(f"    [INFO] MOMENTUM FILTER WAS SKIPPED (entry blocked)")
                    else:
                        # Check if price was at peak
                        price_lag_1 = trade.get("price_lag_1")
                        if price_lag_1:
                            recent_change = ((buy_price - price_lag_1) / price_lag_1) * 100 if price_lag_1 > 0 else 0
                            if recent_change > 1.0:
                                print(f"    [ISSUE] ENTERED AFTER {recent_change:.2f}% UPSWING (possible peak entry)")
                                issues_found["entry_at_peak"].append({
                                    "symbol": symbol,
                                    "upswing_pct": recent_change,
                                })
                    break
        
        # Update queues
        if float(buy.get("filled_qty", 0) or 0) > qty:
            buy["filled_qty"] = float(buy.get("filled_qty", 0) or 0) - qty
        else:
            buy_queue.pop(0)
        
        if float(sell.get("filled_qty", 0) or 0) > qty:
            sell["filled_qty"] = float(sell.get("filled_qty", 0) or 0) - qty
        else:
            sell_queue.pop(0)

# Summary of issues
print("\n" + "=" * 80)
print("ISSUES FOUND")
print("=" * 80)

total_issues = sum(len(v) for v in issues_found.values())
print(f"\nTotal issues identified: {total_issues}")

for issue_type, occurrences in issues_found.items():
    if occurrences:
        print(f"\n{issue_type.replace('_', ' ').title()}: {len(occurrences)}")
        for occ in occurrences[:3]:  # Show first 3
            print(f"  - {occ.get('symbol', 'unknown')}: {occ}")

# Recommendations
print("\n" + "=" * 80)
print("ROOT CAUSE ANALYSIS & RECOMMENDATIONS")
print("=" * 80)

print("\n1. STOP-LOSS ENFORCEMENT:")
if issues_found["stop_loss_not_enforced"]:
    print("   [PROBLEM] Stop-losses are NOT being enforced properly")
    print("   [FIX] Ensure stop-loss orders are submitted and executed")
    print("   [FIX] Add position monitoring that force-exits at stop-loss price")
else:
    print("   [WARN] Stop-losses may be too wide (8%) - consider tightening to 5-6%")

print("\n2. ENTRY TIMING:")
if issues_found["entry_at_peak"]:
    print("   [PROBLEM] Entering positions after price upswings (buying at peaks)")
    print("   [FIX] Strengthen momentum filter (currently 0.5%/0.8% - may need 1.0%/1.5%)")
    print("   [FIX] Add confirmation: wait for pullback after upswing before entering")
else:
    print("   [WARN] Momentum filter may need adjustment")

print("\n3. MODEL PREDICTIONS:")
if issues_found["model_prediction_wrong"]:
    print("   [PROBLEM] Model predicting negative returns but still entering")
    print("   [FIX] Add check to block entries when predicted_return < 0")
    print("   [FIX] Require minimum predicted_return (e.g., 2%) before entering")

print("\n4. CONFIDENCE THRESHOLD:")
if issues_found["low_confidence_entry"]:
    print("   [PROBLEM] Entering on low confidence (<10%)")
    print("   [FIX] Increase min_confidence from 0.05 (5%) to 0.10 (10%) or higher")
    print("   [FIX] Require higher confidence for larger position sizes")

print("\n5. EXIT LOGIC:")
print("   [WARN] Many trades exited on Jan 19 at losses - possible mass stop-loss trigger")
print("   [FIX] Review exit logic - ensure profit targets are realistic (2.5% may be too high)")
print("   [FIX] Consider using trailing stops more aggressively to protect profits")

print("\n" + "=" * 80)
print("IMMEDIATE ACTION ITEMS")
print("=" * 80)

print("""
1. ENFORCE STOP-LOSSES:
   - Add position monitoring that checks price every cycle
   - Force-exit immediately when stop-loss price is hit
   - Don't rely on broker stop-loss orders alone

2. IMPROVE ENTRY FILTERING:
   - Increase momentum filter thresholds (1.0% for 1 bar, 1.5% for 2 bars)
   - Block entries when predicted_return < 0
   - Require minimum confidence of 10% (not 5%)

3. FIX EXIT LOGIC:
   - Lower profit target to 1.5-2.0% (more achievable)
   - Use tighter trailing stops (1.5% instead of 2.5%)
   - Exit on any profit if model flips to opposite signal

4. ADD SAFETY CHECKS:
   - Don't enter if price moved >1% in last bar
   - Don't enter if RSI > 75 (even more lenient than current 80)
   - Don't enter if model confidence < 10%

5. TEST IN DRY-RUN:
   - Run with new settings in dry-run mode first
   - Monitor for 24-48 hours before going live
""")

print("=" * 80)
