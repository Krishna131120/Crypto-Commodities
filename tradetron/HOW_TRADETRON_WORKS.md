# How Tradetron Actually Works - Clarified

## Understanding Tradetron's Role

**Tradetron is NOT a broker** - it's a **strategy automation platform** that:
1. Receives trading signals from your code (via webhook/API)
2. Executes trades on your behalf through a connected broker
3. Manages positions, stop-losses, and take-profits automatically

## The Complete Flow

```
Your ML Code → Generates Signal → Sends to Tradetron API → Tradetron → Executes via Broker → Trade Done
```

### Step-by-Step:

1. **Your Code** (Python with ML models)
   - Analyzes market data
   - Generates BUY/SELL signals
   - Sends signal to Tradetron via webhook

2. **Tradetron Platform**
   - Receives your signal
   - Checks if strategy is active
   - Executes trade through connected broker

3. **Broker** (Where actual trade happens)
   - **For Paper Trading:** Tradetron's built-in paper trading broker (TT Paper Trading)
   - **For Live Trading:** Your real broker (Angel One, Maitra Commodities, etc.)

## Do You Need to Connect Angel One?

### For Paper Trading: **NO** ✅
- Tradetron has its own paper trading broker
- You just need to:
  1. Create a strategy on Tradetron
  2. Deploy it in "Paper Trading" mode
  3. Use "TT Paper Trading" as the broker
  4. Send signals from your code

### For Live Trading: **YES** (Later)
- When you're ready for real trading:
  1. Connect your Angel One account to Tradetron
  2. Change deployment from "Paper Trading" to "Live Trading"
  3. Select "Angel One" as the broker
  4. Same signals, but now with real money

## How to Get Your API Token

Based on Tradetron's documentation, here's where to find it:

### Option 1: Through Strategy Settings
1. Go to **"Strategies"** → **"MY STRATEGIES"**
2. Find your strategy in the list
3. Look for a menu icon (three dots, settings icon, or "Actions" button) next to your strategy
4. Click it and look for **"API OAUTH Token"** or **"API Link"**
5. Click **"Link"** → **"Proceed"** to generate the token
6. Copy the token (looks like: `ab4ee4c7-4413-4110-993e-cf9b9b927d4a`)

### Option 2: If You Can't Find It
1. Make sure you've **created a strategy first**
2. The API token is **strategy-specific** - you need a strategy to get a token
3. Try looking in:
   - Strategy edit page → Settings/Configuration
   - Strategy deployment page
   - "Broker & Exchanges" section (though this is for connecting brokers, not API tokens)

### Option 3: Contact Support
- Email: support@tradetron.tech
- Ask: "How do I get the API OAuth token for my strategy to send signals?"

## What Tradetron's API Documentation Means

When Tradetron says:
> "Tradetron allows you to generate signals from outside tools... and control the strategy you create at Tradetron"

This means:
- ✅ Your Python code generates the trading decision (BUY/SELL)
- ✅ Your code sends that decision to Tradetron as a signal
- ✅ Tradetron executes the trade through a broker
- ✅ Tradetron manages the position (stop-loss, take-profit, etc.)

**You DON'T need to:**
- ❌ Connect Angel One API directly to your code
- ❌ Manage orders yourself
- ❌ Handle position management manually

**You DO need to:**
- ✅ Create a strategy on Tradetron
- ✅ Get the API token for that strategy
- ✅ Send signals from your code to Tradetron
- ✅ (For live trading) Connect a broker to Tradetron

## The Setup Process

### Phase 1: Paper Trading (What You Want Now)

1. **Create Strategy on Tradetron:**
   - Go to "Create" → "Strategy"
   - Create a simple strategy that accepts external signals
   - Configure it for MCX commodities

2. **Deploy in Paper Trading Mode:**
   - Click "Deploy" on your strategy
   - Select "Paper Trading"
   - Select "TT Paper Trading" as broker
   - **NO broker connection needed!**

3. **Get API Token:**
   - Find your strategy in "MY STRATEGIES"
   - Get the API OAuth token (as described above)

4. **Your Code Sends Signals:**
   - Your ML code generates signals
   - Sends to Tradetron webhook: `https://api.tradetron.tech/api`
   - Tradetron executes in paper trading account

### Phase 2: Live Trading (Later, When Ready)

1. **Connect Broker:**
   - Go to "Broker & Exchanges"
   - Add Angel One (or Maitra Commodities)
   - Enter your broker credentials

2. **Change Deployment:**
   - Edit your strategy deployment
   - Change from "Paper Trading" to "Live Trading"
   - Select your connected broker (Angel One)

3. **Same Code, Real Money:**
   - Your code stays the same
   - Same signals, same format
   - But now trades execute with real money

## Summary

**For Paper Trading:**
- ✅ Create strategy on Tradetron
- ✅ Deploy with "TT Paper Trading" broker
- ✅ Get API token from strategy
- ✅ Send signals from your code
- ❌ **NO need to connect Angel One**

**For Live Trading (Later):**
- ✅ Connect Angel One to Tradetron
- ✅ Change deployment to "Live Trading"
- ✅ Select Angel One as broker
- ✅ Same signals, real money

**Your code doesn't change** - it just sends signals to Tradetron, and Tradetron handles the rest!
