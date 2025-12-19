# Stop-Loss Regulation & Shorting Logic

## 📊 STOP-LOSS REGULATION

### 1. **Stop-Loss Calculation**

Stop-loss is calculated differently for **LONG** vs **SHORT** positions:

```python
# Lines 896-912 in execution_engine.py

if target_side == "long":
    # Long: lose if price drops, win if price rises
    stop_loss_price = current_price * (1.0 - stop_pct)  
    # Example: $100 * 0.98 = $98 (2% down = loss)
    take_profit_price = current_price * (1.0 + stop_pct * tp_mult)  
    # Example: $100 * 1.04 = $104 (4% up = profit)
    side = "buy"
else:  # short
    # Short: lose if price rises, win if price drops
    stop_loss_price = current_price * (1.0 + stop_pct)  
    # Example: $100 * 1.02 = $102 (2% up = loss for short)
    take_profit_price = current_price * (1.0 - stop_pct * tp_mult)  
    # Example: $100 * 0.96 = $96 (4% down = profit for short)
    side = "sell"
```

### 2. **Stop-Loss Percentage by Asset Type**

```python
# Lines 59-82 in execution_engine.py

def get_effective_stop_loss_pct(self, asset_type: str = "crypto") -> float:
    """
    Get effective stop-loss percentage based on asset type and user override.
    
    For commodities (real money): 2.0% default (tighter)
    For crypto (paper trading): 3.5% default (wider)
    User override takes precedence.
    """
    if self.user_stop_loss_pct is not None:
        # User override takes precedence (clamped 0.5% to 10%)
        return max(0.005, min(0.10, self.user_stop_loss_pct))
    
    if asset_type == "commodities":
        return 0.020  # 2.0% for commodities (real money - tighter)
    else:
        return 0.035  # 3.5% for crypto (paper trading - wider)
```

### 3. **Stop-Loss Monitoring & Execution**

The system monitors stop-loss in **real-time** during each trading cycle:

```python
# Lines 241-250 in execution_engine.py

# PRIORITY 2: Also exit on stop-loss (safety mechanism)
stop_loss_hit = False
if not effective_risk.manual_stop_loss and tracked_position and tracked_position.status == "open":
    if tracked_position.side == "long":
        # Long position: exit if price drops to/below stop-loss
        stop_loss_hit = current_price <= tracked_position.stop_loss_price
    elif tracked_position.side == "short":
        # Short position: exit if price rises to/above stop-loss
        stop_loss_hit = current_price >= tracked_position.stop_loss_price

if stop_loss_hit:
    must_exit_position = True  # Force immediate exit
```

### 4. **Stop-Loss Placement Methods**

#### **A. Broker-Level (Commodities with DHAN)**
```python
# Lines 989-992 in execution_engine.py

if self.client.broker_name == "dhan":
    # DHAN MCX: stop-loss submitted as separate order
    if not effective_risk.manual_stop_loss:
        order_kwargs["stop_loss_price"] = stop_loss_price
        # This creates a broker-level SL order that executes even if script stops
```

#### **B. Broker-Level (Crypto with Alpaca)**
```python
# Lines 1014-1030 in execution_engine.py

if is_crypto and stop_loss_price and not effective_risk.manual_stop_loss:
    try:
        # Submit stop-loss order for crypto (broker-level protection)
        stop_side = "sell" if target_side == "long" else "buy"
        stop_order_resp = self.client.submit_stop_order(
            symbol=trading_symbol,
            qty=implied_qty,
            stop_price=stop_loss_price,
            side=stop_side,
            time_in_force="gtc",
        )
        stop_loss_order_id = stop_order_resp.get("id")
        print(f"  ✅ Stop-loss order placed at broker level: ${stop_loss_price:.2f}")
    except Exception as stop_exc:
        print(f"  ⚠️  WARNING: Failed to place broker-level stop-loss order")
        print(f"     Stop-loss will only work while monitoring script is running")
```

#### **C. System-Level Monitoring**
- If broker-level placement fails, system monitors in real-time
- Only works while script is running
- Less reliable than broker-level

### 5. **Manual Stop-Loss Mode**

```python
# Line 53 in execution_engine.py

manual_stop_loss: bool = False  
# If True, user manages stop-losses manually
# System won't submit or execute stop-loss orders
```

When enabled:
- System calculates stop-loss price but doesn't place order
- User must manage stop-loss manually
- Useful for advanced traders who want full control

---

## 📉 SHORTING LOGIC (When Prediction is SHORT)

### 1. **Determining Target Side from Prediction**

```python
# Lines 218-224 in execution_engine.py

# Determine target side from model action
if action == "short" and self.risk.allow_short:
    target_side = "short"  # ✅ Shorting enabled and prediction is SHORT
elif action == "long":
    target_side = "long"
else:
    target_side = "flat"  # hold or no action
```

**Key Check**: `self.risk.allow_short` must be `True` (default: `True`)

### 2. **Confidence & Agreement Requirements for Shorting**

```python
# Lines 312-314 in execution_engine.py (Crypto)
# Lines 290-303 in execution_engine.py (Commodities)

# For CRYPTO (paper trading):
if target_side == "short" and action == "short" and confidence < effective_risk.min_confidence:
    # For SHORT positions, require confidence threshold (more risky)
    return None  # Skip trade

# For COMMODITIES (real money):
if is_commodities:
    # Commodities require:
    # 1. Confidence >= 15% minimum
    # 2. Model agreement >= 66% (if multiple models)
    # 3. Action must match target side
    if confidence < min_confidence_for_commodities:  # 15% minimum
        return None  # Skip trade
    if not agreement_met:  # 66% agreement required
        return None  # Skip trade
```

### 3. **Entering SHORT Position from FLAT**

```python
# Lines 863-912 in execution_engine.py

if side_in_market == "flat" and target_side in {"long", "short"} and desired_notional > 0:
    # Calculate stop-loss for SHORT
    if target_side == "short":
        # Short: lose if price rises, win if price drops
        stop_loss_price = current_price * (1.0 + stop_pct)  
        # Example: $100 * 1.02 = $102 (2% up = loss for short)
        take_profit_price = current_price * (1.0 - stop_pct * tp_mult)  
        # Example: $100 * 0.96 = $96 (4% down = profit for short)
        side = "sell"  # ✅ SELL to open short position
```

### 4. **SHORT Position Execution (Crypto vs Commodities)**

#### **A. Crypto Short (Alpaca)**
```python
# Lines 707-775 in execution_engine.py

if is_crypto and target_side == "short":
    # Crypto shorts: use USD notional, rounded to 2 decimals
    notional_rounded = round(desired_notional, 2)
    
    # Safety checks
    if notional_rounded <= 0 or notional_rounded < 0.01:
        return orders  # Skip if too small
    
    # Re-fetch buying power after closing any existing position
    account_after_close = self.client.get_account()
    buying_power_after = float(account_after_close.get("buying_power", 0.0) or 0.0)
    
    # Ensure we don't exceed buying power
    if notional_rounded > buying_power_after:
        notional_rounded = round(buying_power_after, 2)
    
    try:
        entry_resp = self.client.submit_order(
            symbol=trading_symbol,
            notional=notional_rounded,  # ✅ Use notional for crypto
            side="sell",  # ✅ SELL to short
            order_type="market",
            time_in_force="gtc",
        )
    except RuntimeError as exc:
        # Handle crypto short rejections (paper trading limitation)
        if "insufficient balance" in error_msg.lower():
            # Alpaca paper trading has limited crypto inventory for shorting
            # Need actual crypto asset (ETH) available to borrow, not just USD
            orders["decision"] = "flip_to_short_rejected"
            return orders
```

#### **B. Commodities Short (DHAN/MCX)**
```python
# Lines 875-912 in execution_engine.py

if is_commodities and self.client.broker_name == "dhan":
    # MCX requires lot-based trading - round quantity to nearest lot
    raw_qty = max(desired_notional / current_price, 0.0)
    trade_qty = round_to_lot_size(raw_qty, asset.data_symbol)
    trade_notional = trade_qty * current_price
    implied_qty = trade_qty
    
    # Build order
    order_kwargs = {
        "symbol": trading_symbol,
        "qty": implied_qty,  # ✅ Use quantity (lot-based) for commodities
        "side": "sell",  # ✅ SELL to short
        "order_type": "market",
        "time_in_force": "gtc",
    }
    
    # Add stop-loss if not manual mode
    if not effective_risk.manual_stop_loss:
        order_kwargs["stop_loss_price"] = stop_loss_price
    
    entry_resp = self.client.submit_order(**order_kwargs)
```

### 5. **Flipping from LONG to SHORT**

```python
# Lines 642-828 in execution_engine.py

# Flipping positions: long -> short or short -> long
if side_in_market in {"long", "short"} and target_side in {"long", "short"} and side_in_market != target_side:
    # Step 1: Close existing position IMMEDIATELY
    close_resp = self.client.submit_order(
        symbol=trading_symbol,
        qty=abs(existing_qty),
        side="sell" if existing_qty > 0 else "buy",  # Close long or short
        order_type="market",
        time_in_force="gtc",
    )
    
    # Step 2: Open new position in opposite direction
    if desired_notional > 0:
        if target_side == "short":
            stop_loss_price = current_price * (1.0 + effective_risk.default_stop_loss_pct)
            side = "sell"  # ✅ SELL to open short
        
        # Execute new short position (same logic as entering from flat)
        entry_resp = self.client.submit_order(...)
```

### 6. **Short Position Stop-Loss Monitoring**

```python
# Lines 244-247 in execution_engine.py

if tracked_position.side == "short":
    # Short position: exit if price RISES to/above stop-loss
    stop_loss_hit = current_price >= tracked_position.stop_loss_price
    # Example: Entry $100, Stop $102, Current $102.50 → EXIT (loss)
```

### 7. **Short Position Profit Target**

```python
# Lines 234-236 in execution_engine.py

if tracked_position.side == "short":
    # Short position: profit target hit if price DROPS to/below target
    profit_target_hit = current_price <= tracked_position.profit_target_price
    # Example: Entry $100, Target $96, Current $95.50 → EXIT (profit)
```

---

## 🔄 COMPLETE SHORTING FLOW

### Scenario: Prediction is SHORT, Currently FLAT

```
1. Model Prediction: "short" with 20% confidence, 3/4 models agree (75%)
   ↓
2. Check allow_short: True ✅
   ↓
3. Check confidence: 20% >= 15% (commodities) ✅
   ↓
4. Check agreement: 75% >= 66% ✅
   ↓
5. Calculate position size based on equity and risk limits
   ↓
6. Calculate stop-loss: current_price * (1.0 + 0.02) = $102 (if entry $100)
   ↓
7. Calculate take-profit: current_price * (1.0 - 0.04) = $96 (if entry $100)
   ↓
8. Submit SELL order to open short position
   ↓
9. Place broker-level stop-loss order at $102
   ↓
10. Monitor position:
    - If price rises to $102 → Stop-loss triggers → BUY to close (loss)
    - If price drops to $96 → Profit target → BUY to close (profit)
    - If prediction changes to "long" → Close short, open long
```

### Scenario: Prediction is SHORT, Currently LONG

```
1. Model Prediction: "short" (opposite of current position)
   ↓
2. IMMEDIATELY close LONG position (sell)
   ↓
3. Then open SHORT position (sell) with new stop-loss
   ↓
4. This ensures we exit immediately when prediction changes
```

---

## ⚠️ IMPORTANT NOTES

1. **Shorting Requires**:
   - `allow_short = True` in risk config
   - Sufficient margin/buying power
   - For crypto: Asset available to borrow (Alpaca limitation)
   - For commodities: MCX supports shorting natively

2. **Stop-Loss for Shorts**:
   - Placed ABOVE entry price (opposite of long)
   - Triggers when price RISES (not drops)
   - Protects against unlimited losses

3. **Risk Management**:
   - Commodities: Stricter (15% confidence, 66% agreement)
   - Crypto: More lenient (10% confidence, 50% agreement)
   - Stop-loss: 2% for commodities, 3.5% for crypto

4. **Broker-Level vs System-Level**:
   - Broker-level: Executes even if script crashes (SAFER)
   - System-level: Only works while script runs (LESS SAFE)
