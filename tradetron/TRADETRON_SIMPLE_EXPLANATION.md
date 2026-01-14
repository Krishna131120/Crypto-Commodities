# Tradetron Paper Trading - Simple Explanation

## What You Asked For

You want to:
1. ✅ Test your commodities trading strategy in paper trading mode
2. ✅ Use Tradetron platform for paper trading
3. ✅ Keep the same strategy you have for Angel One
4. ✅ Understand how it works step-by-step

**Answer: YES, Tradetron is perfect for this!** ✅

---

## How It Works (Super Simple)

### Your Current Setup (Angel One):
```
Your Code → Direct Order → Angel One Broker → Trade Executed
```

### New Setup (Tradetron Paper Trading):
```
Your Code → Signal → Tradetron Platform → Paper Trading Account → Trade Executed (with fake money)
```

**Key Difference:** Instead of placing orders directly, you send "signals" to Tradetron, and Tradetron executes the trades for you in a paper trading account (fake money, no risk).

---

## Step-by-Step: What You Need to Do

### STEP 1: Get Your API Token from Tradetron ✅

**IMPORTANT:** You need to create a strategy FIRST before you can get an API token!

1. **First, create a strategy:**
   - Go to "Create" → "Strategy"
   - Create a simple strategy that accepts external signals
   - Save it

2. **Then get the API token:**
   - Go to "Strategies" → "MY STRATEGIES"
   - Find your strategy in the list
   - Look for a menu/actions button (might be three dots, gear icon, or "Actions" dropdown)
   - Click it and find "API OAUTH Token" or "API Link"
   - Click "Link" → "Proceed"
   - Copy the token (looks like: `ab4ee4c7-4413-4110-993e-cf9b9b927d4a`)

**If you can't find the menu:**
- Make sure you've created a strategy first
- Try the strategy edit page → look for API/Settings section
- Contact Tradetron support: support@tradetron.tech

**Save this token** - you'll need it!

---

### STEP 2: Create a Strategy on Tradetron (If You Haven't)

1. Go to "Strategies" → "Create New Strategy"
2. Name it: "Commodities ML Strategy"
3. Choose: "API Signal Based" or "External Signal"
4. Exchange: MCX
5. Enable: "Accept External Signals via API"
6. Save the strategy

---

### STEP 3: Deploy Strategy in Paper Trading Mode

1. Click "Deploy" next to your strategy
2. Select: **"Paper Trading"** (NOT "Live Trading")
3. Select: **"TT Paper Trading"** as broker
   - **NOTE:** This is Tradetron's built-in paper trading broker
   - **You DON'T need to connect Angel One for paper trading!**
   - Angel One is only needed for live trading (later)
4. Set virtual capital: ₹1,00,000 (or any amount)
5. Click "Deploy"

**Now your strategy is running in paper mode!** 🎉

**Important:** For paper trading, you do NOT need to connect any external broker. Tradetron provides its own paper trading environment.

---

### STEP 4: Add Token to Your Code

Create or edit `.env` file in your project folder:

```bash
TRADETRON_API_TOKEN=your-token-here
```

Replace `your-token-here` with the token you copied in Step 1.

---

### STEP 5: Change One Line in Your Code

**Before (using Angel One):**
```python
from trading.angelone_client import AngelOneClient
client = AngelOneClient()
```

**After (using Tradetron):**
```python
from trading.tradetron_client import TradetronClient
client = TradetronClient()
```

**That's it!** Everything else stays the same.

---

### STEP 6: Test It

Run the test script:
```bash
python test_tradetron.py
```

This will:
1. Connect to Tradetron
2. Send a test signal (buy 1 lot of GOLD)
3. Show you if it worked

**Then check Tradetron dashboard** - you should see the order in your paper trading account!

---

### STEP 7: Run Your Full Strategy

Once testing works, run your full strategy exactly as before:

```bash
python end_to_end_commodities.py \
    --commodities-symbols GOLDDEC24 SILVERDEC24 \
    --profit-target 5.0 \
    --stop-loss-pct 2.0
```

**The only difference:** It now uses Tradetron instead of Angel One, and all trades are in paper mode (fake money).

---

## How Signals Work (Line by Line)

Let's say your ML model predicts: **"GOLD should go UP"**

### Line 1: Your Code Makes Prediction
```python
# Your existing code (unchanged)
prediction = model.predict(features)
# Result: "BUY GOLD with 75% confidence"
```

### Line 2: Your Code Decides to Trade
```python
# Your existing code (unchanged)
if prediction == "BUY" and confidence > 0.6:
    # Calculate: Buy 1 lot at current price
    # Set stop-loss: 2% below entry
    # Set take-profit: 5% above entry
```

### Line 3: Your Code Sends Signal to Tradetron
```python
# NEW: Instead of placing order directly, send signal
client = TradetronClient()  # Loads token from .env
client.submit_order(
    symbol="GOLDDEC24",
    qty=1,
    side="buy"
)
```

### Line 4: Tradetron Receives Signal
```
Tradetron checks:
- ✅ Is strategy active? YES
- ✅ Is it in paper trading mode? YES
- ✅ Signal format correct? YES
→ Places order in paper trading account
```

### Line 5: Tradetron Executes Trade
```
Paper Trading Account:
- Buy 1 lot GOLD @ ₹61,000
- Stop-loss: ₹60,000 (2% below)
- Take-profit: ₹63,000 (5% above)
```

### Line 6: Your Code Gets Confirmation
```python
# Tradetron sends back:
{
    "status": "accepted",
    "order_id": "12345",
    "message": "Signal received"
}
```

### Line 7: Monitor in Dashboard
```
Go to Tradetron dashboard:
- See order in "Orders" section
- See position in "Positions" section
- Track P&L (profit/loss)
- All in paper mode (safe!)
```

---

## What Stays the Same ✅

- ✅ Data fetching (same)
- ✅ Feature generation (same)
- ✅ Model training (same)
- ✅ Prediction logic (same)
- ✅ Risk management (same)
- ✅ Profit targets (same)
- ✅ Stop-losses (same)

**Only thing that changes:** How orders are placed (signals vs direct orders)

---

## What's Different ⚠️

### Angel One (Direct Orders):
- Your code → Direct API call → Order placed immediately
- You control everything
- Real money (when live)

### Tradetron (Signals):
- Your code → Signal sent → Tradetron processes → Order placed
- Tradetron manages execution
- Paper trading (fake money) or live (real money)

---

## Advantages of Tradetron for Paper Trading

1. **Free Testing** - No real money at risk
2. **Easy Monitoring** - Dashboard shows everything
3. **Strategy Management** - Can pause/resume easily
4. **Same API** - Works for both paper and live trading
5. **No Broker Account Needed** - For paper trading, Tradetron provides virtual account

---

## Important Notes

### ✅ Paper Trading is Safe
- All trades use fake money
- No real money at risk
- Perfect for testing

### ⚠️ Before Going Live
- Test thoroughly in paper mode first
- Check performance, win rate, drawdowns
- When ready, change deployment to "Live Trading"
- Connect your real broker account

### 📊 Monitoring
- Check Tradetron dashboard regularly
- See all orders and positions
- Track P&L in real-time

---

## Quick Checklist

- [ ] Created Tradetron account ✅ (You said you did this)
- [ ] Created strategy on Tradetron platform
- [ ] Deployed strategy in "Paper Trading" mode
- [ ] Got API token from strategy settings
- [ ] Added token to `.env` file
- [ ] Tested with `test_tradetron.py`
- [ ] Verified order appears in Tradetron dashboard
- [ ] Ready to run full strategy!

---

## Summary

**Tradetron is PERFECT for paper trading commodities!**

1. ✅ Create strategy on Tradetron
2. ✅ Deploy in paper trading mode
3. ✅ Get API token
4. ✅ Change one line in your code (use TradetronClient instead of AngelOneClient)
5. ✅ Run your strategy - everything else stays the same!
6. ✅ Monitor in Tradetron dashboard
7. ✅ When ready, switch to live trading

**Your strategy logic doesn't change** - only the execution method changes from direct orders to signals.

---

## Questions?

**Q: Do I need to connect Angel One to Tradetron for paper trading?**
A: **NO!** For paper trading, Tradetron has its own built-in paper trading broker ("TT Paper Trading"). You only need to connect Angel One when you're ready for live trading with real money.

**Q: How does Tradetron work vs direct broker API?**
A: 
- **Direct API (Angel One):** Your code → Direct order → Broker → Trade executed
- **Tradetron:** Your code → Signal → Tradetron → Broker (via Tradetron) → Trade executed
- Tradetron handles position management, stop-losses, etc. automatically

**Q: Is Tradetron the right platform?**
A: Yes! It's perfect for paper trading. Free, easy to use, and works great for testing strategies.

**Q: Can I use both Angel One and Tradetron?**
A: Yes! Use Tradetron for paper trading (testing). For live trading, you can either:
- Use Tradetron with Angel One connected (Tradetron executes via Angel One)
- Or use Angel One directly (your current setup)

**Q: What if I can't find the API token?**
A: 
1. Make sure you've created a strategy first
2. Look in "MY STRATEGIES" → find your strategy → look for menu/actions button
3. Contact Tradetron support: support@tradetron.tech

**Q: How do I know if it's working?**
A: Check Tradetron dashboard - you'll see all orders and positions in real-time.

---

## Files Created for You

1. **`TRADETRON_PAPER_TRADING_GUIDE.md`** - Detailed guide with all steps
2. **`TRADETRON_SIMPLE_EXPLANATION.md`** - This file (simple explanation)
3. **`trading/tradetron_client.py`** - The Tradetron client code
4. **`test_tradetron.py`** - Test script to verify setup

**You're all set!** 🚀
