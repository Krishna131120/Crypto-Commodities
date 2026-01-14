# Tradetron Paper Trading Setup Guide for Commodities

## Overview: How Tradetron Works vs Angel One

**Angel One (Current Setup):**
- You directly place orders via API calls
- Your code sends buy/sell orders directly to the broker
- Full control over order execution

**Tradetron (New Platform):**
- You create a "strategy" on Tradetron's platform
- Your code sends **signals** (not direct orders) to Tradetron
- Tradetron executes trades on your behalf based on your signals
- Better for automation and backtesting

**Is Tradetron Right for Paper Trading?**
✅ **YES!** Tradetron is excellent for paper trading because:
- Free paper trading environment
- No real money risk
- Same API works for both paper and live trading
- Easy to test strategies before going live

---

## Step-by-Step Setup Process

### STEP 1: Create Your Tradetron Account ✅ (You've Done This!)

You mentioned you already created an account. Great!

---

### STEP 2: Get Your API Credentials

1. **Log in to Tradetron** → Go to your dashboard
2. **Navigate to API Settings:**
   - Look for "API" or "Developer" section in your account settings
   - Or go to: Settings → API Keys / Developer Tools
3. **Generate API Credentials:**
   - You'll need:
     - **API Key** (like a username)
     - **API Secret** (like a password)
     - **Strategy ID** (you'll get this after creating a strategy)
4. **Save these credentials** - you'll need them for your code

---

### STEP 3: Create a Strategy on Tradetron Platform

**What is a Strategy?**
- A strategy is like a "recipe" that tells Tradetron what to do when it receives signals
- For paper trading, you'll create a simple strategy that:
  - Accepts BUY signals → Places buy orders
  - Accepts SELL signals → Places sell orders
  - Manages stop-loss and take-profit automatically

**How to Create:**

1. **Go to "Strategies" section** in Tradetron dashboard
2. **Click "Create New Strategy"**
3. **Choose Strategy Type:**
   - Select "API Signal Based" or "External Signal"
   - This allows your Python code to send signals
4. **Configure Basic Settings:**
   - **Name:** "Commodities ML Strategy" (or any name)
   - **Exchange:** MCX (Multi Commodity Exchange)
   - **Product Type:** INTRADAY or DELIVERY (choose based on your needs)
5. **Set Up Signal Reception:**
   - Enable "Accept External Signals via API"
   - This allows your code to send trading signals
6. **Configure Risk Management:**
   - Set default stop-loss percentage (e.g., 2%)
   - Set default take-profit percentage (e.g., 5%)
   - These can be overridden by your signals
7. **Save the Strategy**
8. **Note the Strategy ID** - you'll need this in your code

---

### STEP 4: Deploy Strategy in Paper Trading Mode

**This is the KEY step for paper trading:**

1. **After creating your strategy, click "Deploy"**
2. **In Deployment Settings:**
   - **Execution Type:** Select "Paper Trading" (NOT "Live Trading")
   - **Broker:** Select "TT Paper Trading" (Tradetron's paper trading broker)
   - **Capital:** Set virtual capital (e.g., ₹1,00,000 for testing)
3. **Confirm Deployment**
4. **Your strategy is now running in paper mode!** 🎉

---

### STEP 5: Understand How Your Code Will Send Signals

**How It Works (Line by Line in Simple Words):**

```
1. Your Python code runs the ML model (same as before)
   → Model predicts: "GOLD should go UP" with 75% confidence

2. Your code decides: "This is a BUY signal"
   → Calculates: Buy 1 lot of GOLD at current market price
   → Sets stop-loss at 2% below entry
   → Sets take-profit at 5% above entry

3. Your code sends a SIGNAL to Tradetron (not a direct order):
   → "Hey Tradetron, I want to BUY 1 lot of GOLD"
   → "Stop-loss: ₹60,000"
   → "Take-profit: ₹63,000"

4. Tradetron receives the signal:
   → Checks: Is this strategy deployed? ✅
   → Checks: Is it in paper trading mode? ✅
   → Executes: Places a BUY order in the paper trading account
   → Sets stop-loss and take-profit orders automatically

5. Tradetron sends back confirmation:
   → "Order placed successfully"
   → "Order ID: 12345"
   → "Position opened: 1 lot GOLD @ ₹61,000"

6. Your code logs this and continues monitoring
```

**Key Difference from Angel One:**
- **Angel One:** Your code → Direct API call → Order placed immediately
- **Tradetron:** Your code → Signal sent → Tradetron processes → Order placed

---

### STEP 6: Adapt Your Current Code

**What Needs to Change:**

1. **Create a TradetronClient** (similar to AngelOneClient)
   - This will send signals to Tradetron API
   - Instead of placing orders, it sends signal requests

2. **Modify ExecutionEngine** to use TradetronClient
   - When it wants to buy/sell, it calls TradetronClient.send_signal()
   - Instead of calling broker.submit_order()

3. **Keep Everything Else the Same:**
   - ✅ Data ingestion (same)
   - ✅ Feature generation (same)
   - ✅ Model training (same)
   - ✅ Prediction logic (same)
   - ✅ Risk management (same)
   - ❌ Only the order execution changes

---

### STEP 7: How Signals Work (Technical Details)

**Tradetron API Signal Format:**

```python
# Example: Send a BUY signal
signal = {
    "strategy_id": "your-strategy-id-123",
    "symbol": "GOLDDEC24",  # MCX symbol
    "action": "BUY",  # or "SELL" or "EXIT"
    "quantity": 1,  # Number of lots
    "order_type": "MARKET",  # or "LIMIT"
    "price": 61000,  # For limit orders (optional for market)
    "stop_loss": 60000,  # Stop-loss price
    "target": 63000,  # Take-profit price
    "product_type": "INTRADAY",  # or "DELIVERY"
    "exchange": "MCX"
}

# Send to Tradetron API
response = tradetron_client.send_signal(signal)
```

**What Happens Next:**
1. Tradetron validates the signal
2. Checks if strategy is active and in paper trading mode
3. Places the order in paper trading account
4. Returns order confirmation

---

### STEP 8: Testing Your Setup

**Before Running Full Strategy:**

1. **Test API Connection:**
   ```python
   # Simple test script
   client = TradetronClient(api_key="...", api_secret="...")
   account = client.get_account()  # Should return paper trading account
   print(f"Paper Trading Balance: ₹{account['balance']}")
   ```

2. **Send a Test Signal:**
   ```python
   # Send a small test signal
   test_signal = {
       "strategy_id": "your-strategy-id",
       "symbol": "GOLDDEC24",
       "action": "BUY",
       "quantity": 1,
       "order_type": "MARKET"
   }
   result = client.send_signal(test_signal)
   print(f"Test order placed: {result}")
   ```

3. **Check Tradetron Dashboard:**
   - Go to "Orders" or "Positions" section
   - You should see your test order in paper trading account
   - Verify it executed correctly

4. **If Test Works:**
   - ✅ You're ready to run your full strategy!
   - ✅ All trades will be in paper mode (no real money)

---

### STEP 9: Running Your Full Strategy

**Once Everything is Set Up:**

1. **Your code runs exactly as before:**
   ```bash
   python end_to_end_commodities.py \
       --commodities-symbols GOLDDEC24 SILVERDEC24 \
       --profit-target 5.0 \
       --stop-loss-pct 2.0 \
       --dry-run false  # Set to false for paper trading
   ```

2. **But now it uses Tradetron instead of Angel One:**
   - Same ML models ✅
   - Same predictions ✅
   - Same risk management ✅
   - Different execution (Tradetron signals instead of direct orders) ✅

3. **Monitor in Tradetron Dashboard:**
   - See all orders in real-time
   - Track P&L
   - View positions
   - All in paper trading mode (safe!)

---

## Important Notes

### ✅ Advantages of Tradetron Paper Trading:

1. **Free Testing:** No real money at risk
2. **Realistic Environment:** Paper trading uses real market prices
3. **Easy Monitoring:** Dashboard shows everything clearly
4. **Strategy Management:** Can pause/resume strategies easily
5. **Backtesting:** Can test on historical data before live trading

### ⚠️ Things to Remember:

1. **Paper Trading ≠ Live Trading:**
   - Paper trading may have slight delays
   - Slippage might be different
   - Always test thoroughly before going live

2. **API Rate Limits:**
   - Tradetron may have rate limits on API calls
   - Don't send too many signals too quickly
   - Your current code already has delays (60 seconds between cycles)

3. **Symbol Format:**
   - Make sure MCX symbols match Tradetron's format
   - Example: "GOLDDEC24" vs "GOLD DEC24" - check exact format

4. **Strategy Must Be Active:**
   - Strategy must be deployed and active
   - If you pause the strategy, signals won't execute

---

## Next Steps

1. ✅ **Get API credentials from Tradetron**
2. ✅ **Create a strategy on Tradetron platform**
3. ✅ **Deploy strategy in paper trading mode**
4. ⏳ **I'll create TradetronClient for you** (similar to AngelOneClient)
5. ⏳ **Modify ExecutionEngine to use TradetronClient**
6. ⏳ **Test with a small signal**
7. ⏳ **Run full strategy in paper trading mode**

---

## Questions?

**Q: Can I use the same strategy for both paper and live trading?**
A: Yes! Just change the deployment from "Paper Trading" to "Live Trading" when ready.

**Q: What if I want to test with different parameters?**
A: You can create multiple strategies with different settings, or modify the strategy settings anytime.

**Q: How do I know if my signals are working?**
A: Check the Tradetron dashboard - you'll see all orders and positions in real-time.

**Q: Can I still use Angel One for live trading later?**
A: Yes! You can keep both implementations. Use Tradetron for paper trading, Angel One for live trading.

---

## Summary

**Tradetron is PERFECT for paper trading commodities!** 

The process is:
1. Create strategy on Tradetron platform
2. Deploy in paper trading mode
3. Your code sends signals (not direct orders)
4. Tradetron executes trades in paper account
5. Monitor everything in dashboard
6. When ready, switch to live trading

**Your current strategy logic stays the same** - only the execution method changes from direct orders (Angel One) to signals (Tradetron).

---

## Quick Start: Using TradetronClient in Your Code

### Step 1: Add API Token to .env File

Create or edit `.env` file in your project root:

```bash
# Tradetron API Configuration
TRADETRON_API_TOKEN=your-api-oauth-token-here
TRADETRON_API_URL=https://api.tradetron.tech/api
```

**How to get your token:**
1. Log in to Tradetron
2. Go to "My Strategies"
3. Click three dots (⋯) next to your strategy
4. Select "API OAUTH Token"
5. Click "Link" → "Proceed"
6. Copy the generated token (looks like: `ab4ee4c7-4413-4110-993e-cf9b9b927d4a`)

### Step 2: Use TradetronClient Instead of AngelOneClient

**Before (Angel One):**
```python
from trading.angelone_client import AngelOneClient

client = AngelOneClient()
order = client.submit_order(
    symbol="GOLDDEC24",
    qty=1,
    side="buy",
    order_type="market"
)
```

**After (Tradetron):**
```python
from trading.tradetron_client import TradetronClient

client = TradetronClient()  # Automatically loads from .env
order = client.submit_order(
    symbol="GOLDDEC24",
    qty=1,
    side="buy",
    order_type="market"
)
```

**That's it!** The interface is the same - just swap the client.

### Step 3: Update ExecutionEngine to Use TradetronClient

In your trading script, change:

```python
# OLD: Using Angel One
from trading.angelone_client import AngelOneClient
client = AngelOneClient()
engine = ExecutionEngine(client=client, ...)

# NEW: Using Tradetron
from trading.tradetron_client import TradetronClient
client = TradetronClient()
engine = ExecutionEngine(client=client, ...)
```

### Step 4: Test with a Simple Script

Create `test_tradetron.py`:

```python
"""Simple test script for Tradetron paper trading."""

from trading.tradetron_client import TradetronClient

def test_tradetron():
    print("Testing Tradetron connection...")
    
    # Initialize client (loads from .env)
    client = TradetronClient()
    print(f"✅ Connected to Tradetron (broker: {client.broker_name})")
    
    # Test account info
    account = client.get_account()
    print(f"📊 Account info: {account}")
    
    # Test signal (small quantity for testing)
    print("\n📤 Sending test BUY signal for GOLDDEC24...")
    result = client.submit_order(
        symbol="GOLDDEC24",
        qty=1,  # 1 lot
        side="buy",
        order_type="market"
    )
    
    print(f"✅ Signal sent: {result}")
    print("\n💡 Check Tradetron dashboard to see if order was placed in paper trading account!")
    
    return result

if __name__ == "__main__":
    test_tradetron()
```

Run it:
```bash
python test_tradetron.py
```

### Step 5: Run Your Full Strategy

Once testing works, run your full commodities strategy:

```python
# Your existing code - just change the client
from trading.tradetron_client import TradetronClient
from trading.execution_engine import ExecutionEngine
from trading.position_manager import PositionManager

# Initialize with Tradetron
client = TradetronClient()
position_manager = PositionManager()
engine = ExecutionEngine(
    client=client,  # Use Tradetron instead of Angel One
    position_manager=position_manager,
    ...
)

# Rest of your code stays the same!
```

---

## Important Notes About Tradetron Signals

### Signal Format

Tradetron uses key-value pairs for signals. The `TradetronClient` automatically converts your orders to this format:

**Your code sends:**
```python
client.submit_order(symbol="GOLDDEC24", qty=1, side="buy")
```

**TradetronClient converts to:**
```json
{
  "auth-token": "your-token",
  "GOLDDEC24_long": "1",
  "GOLDDEC24_short": "0",
  "GOLDDEC24_long_lots": "1"
}
```

### Symbol Format

Make sure your MCX symbols match what Tradetron expects:
- ✅ Good: `GOLDDEC24`, `SILVERDEC24`
- ❌ Bad: `GOLD DEC24` (spaces), `golddec24` (lowercase)

The client automatically normalizes symbols, but check your strategy configuration on Tradetron platform.

### Strategy Configuration

Your Tradetron strategy must be configured to:
1. ✅ Accept external signals via API
2. ✅ Be deployed and active
3. ✅ Be in "Paper Trading" mode
4. ✅ Have the correct symbol names matching your code

---

## Troubleshooting

### Error: "Tradetron authentication failed (401)"
- **Solution:** Check your API token in `.env` file
- Get fresh token from Tradetron dashboard if needed

### Error: "Signal sent but no order appears"
- **Solution:** 
  1. Check if strategy is deployed and active
  2. Verify strategy is in "Paper Trading" mode
  3. Check symbol names match exactly
  4. Look at Tradetron dashboard for error messages

### Error: "Symbol not found"
- **Solution:**
  1. Verify symbol format (e.g., `GOLDDEC24` not `GOLD DEC24`)
  2. Check if symbol is available in your strategy's exchange (MCX)
  3. Ensure symbol is enabled in strategy settings

### Orders not executing
- **Solution:**
  1. Check Tradetron dashboard → Orders section
  2. Verify strategy is not paused
  3. Check if there are any error messages
  4. Verify paper trading account has sufficient virtual capital

---

## Next Steps After Paper Trading Works

1. ✅ **Test thoroughly** - Run for several days/weeks in paper mode
2. ✅ **Monitor performance** - Check P&L, win rate, drawdowns
3. ✅ **Adjust parameters** - Fine-tune stop-loss, take-profit, position sizing
4. ✅ **When ready for live trading:**
   - Change strategy deployment from "Paper Trading" to "Live Trading"
   - Connect your real broker account (e.g., Maitra Commodities)
   - Start with small position sizes
   - Monitor closely for first few days
