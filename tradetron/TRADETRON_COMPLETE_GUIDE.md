# Tradetron Paper Trading - Complete Step-by-Step Guide

## 📋 Table of Contents
1. [Overview](#overview)
2. [Account Setup](#account-setup)
3. [Strategy Creation](#strategy-creation)
4. [Strategy Deployment](#strategy-deployment)
5. [API Token Setup](#api-token-setup)
6. [Code Configuration](#code-configuration)
7. [Testing](#testing)
8. [Running Full Strategy](#running-full-strategy)
9. [Monitoring](#monitoring)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)

---

## Overview

### What is Tradetron?
Tradetron is a **strategy automation platform** (NOT a broker) that:
- Receives trading signals from your code via API
- Executes trades on your behalf through a connected broker
- Manages positions, stop-losses, and take-profits automatically

### How It Works
```
Your ML Code → Generates Signal → Sends to Tradetron API → Tradetron → Executes via Broker → Trade Done
```

### Paper Trading vs Live Trading

**Paper Trading (What You Want):**
- ✅ Uses "TT-PaperTrading" broker (built-in, no external broker needed)
- ✅ All trades are simulated with virtual money
- ✅ No real money at risk
- ✅ Perfect for testing your strategy

**Live Trading (Later):**
- Requires connecting a real broker (Angel One, etc.)
- Uses real money
- Same code, just different deployment settings

---

## Account Setup

### Step 1: Create Tradetron Account

1. **Go to Tradetron website:**
   - Visit: https://www.tradetron.tech (or your Tradetron URL)
   - Click "Sign Up" or "Register"

2. **Fill in registration form:**
   - Enter your email address
   - Create a password
   - Enter your name
   - Accept terms and conditions
   - Click "Register" or "Sign Up"

3. **Verify your email:**
   - Check your email inbox
   - Click the verification link sent by Tradetron
   - If you don't see it, check spam folder

4. **Log in to your account:**
   - Go back to Tradetron website
   - Enter your email and password
   - Click "Log In"

5. **Complete profile (if required):**
   - Fill in any additional profile information
   - Upload documents if needed (for live trading later)
   - Save your profile

**✅ You now have a Tradetron account!**

---

## Strategy Creation

### Step 2: Navigate to Strategy Creation

1. **Log in to Tradetron dashboard**
   - Enter your email and password
   - Click "Log In"

2. **Find the "Strategies" section:**
   - Look for "Strategies" in the top menu
   - Or click "Create" → "Strategy"
   - Or go to "My Strategies" section

3. **Click "Create New Strategy" button:**
   - This button is usually at the top-right
   - Or in the "Strategies" section
   - Labeled as "Create", "New Strategy", or "+ Create Strategy"

### Step 3: Configure Strategy Basic Settings

1. **Enter Strategy Name:**
   - Type: "Commodities ML Strategy" (or any name you prefer)
   - Make it descriptive so you can find it later
   - Click in the "Name" or "Strategy Name" field
   - Type your strategy name

2. **Select Strategy Type:**
   - Look for "Strategy Type" dropdown or selection
   - Choose: **"API Signal Based"** or **"External Signal"**
   - This allows your Python code to send signals
   - If you see "Webhook Based", that's also correct
   - **DO NOT** choose "Manual Trading" or "Automated Strategy" (those are different)

3. **Select Exchange:**
   - Find "Exchange" dropdown
   - Select: **"MCX"** (Multi Commodity Exchange)
   - This is for commodities trading
   - If MCX is not visible, check if you need to enable it in account settings

4. **Select Product Type:**
   - Find "Product Type" dropdown
   - Choose: **"INTRADAY"** (for day trading) or **"DELIVERY"** (for holding positions)
   - For commodities, INTRADAY is common
   - You can change this later if needed

### Step 4: Enable External Signal Reception

1. **Find "Signal Settings" or "API Settings" section:**
   - Scroll down in the strategy creation form
   - Look for a section about signals or API

2. **Enable "Accept External Signals via API":**
   - Find checkbox or toggle: "Accept External Signals via API"
   - Or "Enable API Signals"
   - Or "Allow External Signals"
   - **Check this box** or toggle it ON
   - This is CRITICAL - without this, your code can't send signals

3. **Configure Signal Format (if available):**
   - Some strategies may ask for signal format
   - Choose "Key-Value Pairs" or "Webhook Format"
   - Default is usually fine

### Step 5: Configure Risk Management (Optional but Recommended)

1. **Find "Risk Management" or "Stop Loss" section:**
   - Look for settings about stop-loss and take-profit

2. **Set Default Stop-Loss:**
   - Find "Stop Loss" or "Default Stop Loss" field
   - Enter: **2.0** (for 2% stop-loss)
   - Or enter a price value if it asks for price
   - This can be overridden by your signals

3. **Set Default Take-Profit:**
   - Find "Take Profit" or "Default Target" field
   - Enter: **5.0** (for 5% take-profit)
   - Or enter a price value if it asks for price
   - This can be overridden by your signals

4. **Set Position Size (if available):**
   - Find "Position Size" or "Lot Size" field
   - Leave default or set to 1 lot
   - Your code will specify quantity in signals

### Step 6: Save the Strategy

1. **Review all settings:**
   - Check that strategy name is correct
   - Verify exchange is MCX
   - Confirm "Accept External Signals via API" is enabled
   - Review risk management settings

2. **Click "Save" or "Create Strategy" button:**
   - Usually at the bottom of the form
   - Or top-right corner
   - Button may say "Save", "Create", "Submit", or "Create Strategy"

3. **Wait for confirmation:**
   - You should see a success message
   - Strategy should appear in "My Strategies" list
   - Note the strategy name - you'll need it later

**✅ Your strategy is now created!**

---

## Strategy Deployment

### Step 7: Find Your Strategy

1. **Go to "My Strategies" section:**
   - Click "Strategies" in top menu
   - Or click "My Strategies" link
   - You should see a list of your strategies

2. **Locate your strategy:**
   - Find "Commodities ML Strategy" (or whatever you named it)
   - It should show status as "Created" or "Not Deployed"

3. **Click on your strategy:**
   - Click the strategy name or row
   - This opens the strategy details page

### Step 8: Deploy Strategy

1. **Find "Deploy" button:**
   - Look for a button labeled "Deploy"
   - Usually near the top of the strategy page
   - Or in an "Actions" menu
   - Click it

2. **Deployment Settings Page Opens:**
   - You should see a form with deployment options
   - This is where you configure how the strategy runs

### Step 9: Configure Deployment Settings

1. **Select Execution Type:**
   - Find "Execution Type" dropdown or radio buttons
   - **IMPORTANT:** Select **"Paper Trading"** (NOT "Live Trading")
   - This ensures all trades are simulated
   - Paper Trading = fake money, safe testing
   - Live Trading = real money (don't select this yet!)

2. **Select Broker:**
   - Find "Broker" dropdown
   - **IMPORTANT:** Select **"TT-PaperTrading"** or **"TT Paper Trading"**
   - This is Tradetron's built-in paper trading broker
   - **DO NOT** select any real broker (Angel One, RMoney, etc.)
   - **DO NOT** try to connect a broker - it's not needed for paper trading
   - If you don't see "TT-PaperTrading", look for "Paper Trading Broker" or "Virtual Broker"

3. **Set Virtual Capital:**
   - Find "Capital" or "Virtual Capital" field
   - Enter: **₹1,00,000** (or any amount you want for testing)
   - This is fake money for paper trading
   - You can set it to any amount (₹50,000, ₹5,00,000, etc.)
   - This doesn't cost anything - it's virtual

4. **Review Other Settings (if any):**
   - Check if there are any other deployment options
   - Most defaults are fine for paper trading
   - Don't change anything unless you understand it

### Step 10: Confirm Deployment

1. **Review all deployment settings:**
   - Execution Type: Paper Trading ✅
   - Broker: TT-PaperTrading ✅
   - Capital: ₹1,00,000 (or your amount) ✅

2. **Click "Deploy" or "Confirm Deployment" button:**
   - Usually at the bottom of the form
   - Or a "Deploy" button at the top
   - Click it

3. **Wait for deployment confirmation:**
   - You should see a success message
   - Strategy status should change to "Deployed" or "Active"
   - May take a few seconds

4. **Verify deployment:**
   - Go back to "My Strategies"
   - Your strategy should show status as "Deployed" or "Active"
   - You should see "Paper Trading" indicator

**✅ Your strategy is now deployed in paper trading mode!**

---

## API Token Setup

### Step 11: Navigate to Strategy Settings

1. **Go to "My Strategies" section:**
   - Click "Strategies" → "My Strategies"
   - Or find it in the top menu

2. **Find your deployed strategy:**
   - Look for "Commodities ML Strategy" (or your strategy name)
   - It should show as "Deployed" or "Active"

3. **Open strategy menu:**
   - Look for three dots (⋯) next to your strategy
   - Or a gear icon (⚙️)
   - Or an "Actions" button
   - Or click on the strategy name to open details

### Step 12: Find API Token Option

1. **Look for API/Token options:**
   - In the strategy menu, look for:
     - "API OAUTH Token"
     - "API Link"
     - "API Token"
     - "Generate Token"
     - "API Settings"
   - This might be in a dropdown menu
   - Or in the strategy details page

2. **If you can't find it:**
   - Try clicking "Edit" on the strategy
   - Look in "Settings" or "Configuration" tab
   - Check "API" or "Developer" section
   - Contact Tradetron support if still not found

### Step 13: Generate API Token

1. **Click "API OAUTH Token" or "API Link":**
   - Click the option you found in Step 12
   - This opens a token generation page

2. **Click "Link" button:**
   - You should see a "Link" button
   - Or "Generate Token" button
   - Click it

3. **Click "Proceed" button:**
   - A confirmation dialog may appear
   - Click "Proceed" or "Confirm"
   - This generates your API token

4. **Copy the token:**
   - You should see a token displayed
   - It looks like: `ab4ee4c7-4413-4110-993e-cf9b9b927d4a`
   - Or a longer string of letters and numbers
   - **IMPORTANT:** Copy this token immediately
   - Click "Copy" button if available
   - Or select all text and copy (Ctrl+C)

5. **Save the token safely:**
   - Paste it in a text file temporarily
   - Or write it down
   - You'll need it in the next step
   - **DO NOT** share this token with anyone

**✅ You now have your API token!**

---

## Code Configuration

### Step 14: Locate Your .env File

1. **Navigate to your project folder:**
   - Open file explorer
   - Go to: `C:\Users\pc44\Desktop\Krishna Crypto\Crypto-Commodities`
   - This is your project root folder

2. **Find or create .env file:**
   - Look for a file named `.env` (may be hidden)
   - If it doesn't exist, create a new text file
   - Name it exactly: `.env` (with the dot at the beginning)
   - Make sure it's in the project root (same folder as your Python files)

3. **Open .env file:**
   - Right-click → "Open with" → "Notepad" or any text editor
   - Or double-click if it opens in your default editor

### Step 15: Add Tradetron API Token to .env

1. **Add the token line:**
   - In the .env file, add this line:
   ```bash
   TRADETRON_API_TOKEN=your-token-here
   ```
   - Replace `your-token-here` with the token you copied in Step 13
   - Example:
   ```bash
   TRADETRON_API_TOKEN=ab4ee4c7-4413-4110-993e-cf9b9b927d4a
   ```

2. **Check for other Tradetron settings (optional):**
   - You can also add (optional):
   ```bash
   TRADETRON_API_URL=https://api.tradetron.tech/api
   ```
   - This is usually not needed (default is fine)

3. **Save the file:**
   - Press Ctrl+S to save
   - Or File → Save
   - Make sure the file is saved as `.env` (not `.env.txt`)

4. **Verify the file:**
   - Close and reopen the file
   - Make sure your token is there
   - Make sure there are no extra spaces or quotes around the token

**✅ Your .env file is now configured!**

### Step 16: Update Your Code to Use TradetronClient

1. **Find your trading script:**
   - This might be `trade_all_commodities_auto.py`
   - Or `end_to_end_commodities.py`
   - Or any script that uses `AngelOneClient`

2. **Open the file in your code editor:**
   - Use VS Code, PyCharm, or any Python editor
   - Open the file

3. **Find the import statement:**
   - Look for a line like:
   ```python
   from trading.angelone_client import AngelOneClient
   ```
   - Or:
   ```python
   from trading import AngelOneClient
   ```

4. **Replace with TradetronClient:**
   - Change it to:
   ```python
   from tradetron.tradetron_client import TradetronClient
   ```
   - Or:
   ```python
   from trading import TradetronClient
   ```

5. **Find where client is created:**
   - Look for a line like:
   ```python
   client = AngelOneClient()
   ```
   - Or:
   ```python
   angelone_client = AngelOneClient()
   ```

6. **Replace with TradetronClient:**
   - Change it to:
   ```python
   client = TradetronClient()
   ```
   - Or:
   ```python
   tradetron_client = TradetronClient()
   ```

7. **Update variable names (if needed):**
   - If your code uses `angelone_client` variable name
   - You can keep it or change to `tradetron_client`
   - Just make sure all references are updated

8. **Save the file:**
   - Press Ctrl+S to save
   - Make sure there are no syntax errors

**✅ Your code is now configured to use Tradetron!**

---

## Testing

### Step 17: Run the Test Script

1. **Open terminal/command prompt:**
   - Press Win+R
   - Type: `cmd` or `powershell`
   - Press Enter
   - Or use VS Code terminal

2. **Navigate to project folder:**
   ```bash
   cd "C:\Users\pc44\Desktop\Krishna Crypto\Crypto-Commodities"
   ```
   - Press Enter

3. **Run the test script:**
   ```bash
   python tradetron/test_tradetron.py
   ```
   - Press Enter
   - Wait for the script to run

4. **Check the output:**
   - You should see:
     - ✅ "Connected to Tradetron"
     - ✅ "Signal sent successfully"
     - Order ID and status
   - If you see errors, go to Troubleshooting section

### Step 18: Verify in Tradetron Dashboard

1. **Log in to Tradetron dashboard:**
   - Go to Tradetron website
   - Log in with your credentials

2. **Go to "Deployed" section:**
   - Click "Deployed" in the top menu
   - Or "My Deployments"
   - Or find it in the dashboard

3. **Check "Orders" section:**
   - Look for "Orders" tab or section
   - You should see your test order
   - It should show:
     - Symbol: GOLDDEC24 (or your test symbol)
     - Side: BUY
     - Quantity: 1 lot
     - Status: Executed or Pending

4. **Check "Positions" section:**
   - Look for "Positions" tab or section
   - If order executed, you should see an open position
   - Shows:
     - Symbol
     - Quantity
     - Entry price
     - Current P&L

5. **Check "Notifications" or "Logs":**
   - Look for notification/log section
   - Should show order received and executed messages

**✅ If you see the order in dashboard, your setup is working!**

---

## Running Full Strategy

### Step 19: Prepare Your Strategy Script

1. **Make sure your code is updated:**
   - Verify TradetronClient is being used (Step 16)
   - Check .env file has TRADETRON_API_TOKEN (Step 15)
   - Make sure there are no syntax errors

2. **Check your commodities symbols:**
   - Make sure symbols match MCX format
   - Example: GOLDDEC24, SILVERDEC24
   - Check symbol_universe.py for enabled symbols

3. **Review risk settings:**
   - Check profit target percentage
   - Check stop-loss percentage
   - Make sure they're reasonable for testing

### Step 20: Run Your Full Strategy

1. **Open terminal:**
   - Navigate to project folder (Step 17.2)

2. **Run your strategy script:**
   ```bash
   python trade_all_commodities_auto.py --profit-target-pct 10.0 --stop-loss-pct 2.0 --interval 300
   ```
   - Or whatever your script name is
   - Adjust parameters as needed

3. **Monitor the output:**
   - Watch for connection messages
   - Check for signal sending confirmations
   - Look for any errors

4. **Let it run:**
   - The script will:
     - Fetch data
     - Generate features
     - Run predictions
     - Send signals to Tradetron
   - This happens in cycles (every few minutes)

**✅ Your strategy is now running in paper trading mode!**

---

## Monitoring

### Step 21: Monitor in Tradetron Dashboard

1. **Check "Deployed" section regularly:**
   - Log in to Tradetron
   - Go to "Deployed" section
   - Click on your strategy

2. **Monitor "Orders" tab:**
   - See all orders placed
   - Check execution status
   - View order details (price, quantity, time)

3. **Monitor "Positions" tab:**
   - See all open positions
   - Check current P&L
   - View entry prices
   - Monitor unrealized profit/loss

4. **Check "Performance" or "Analytics" (if available):**
   - View overall strategy performance
   - Check win rate
   - See total P&L
   - View drawdowns

5. **Review "Notifications" or "Logs":**
   - See all activity logs
   - Check for any errors
   - View signal reception confirmations

### Step 22: Monitor in Your Code Logs

1. **Check console output:**
   - Watch terminal/console where script is running
   - Look for:
     - Signal sent confirmations
     - Order IDs
     - Any error messages

2. **Check log files:**
   - Look in `logs/trading/` folder
   - Check trade logs
   - Review any error logs

3. **Monitor position manager:**
   - Check `data/positions/active_positions.json`
   - See what positions your code is tracking
   - Compare with Tradetron dashboard

---

## Troubleshooting

### Problem: "Tradetron authentication failed (401)"

**Solution:**
1. Check your .env file has TRADETRON_API_TOKEN
2. Verify token is correct (no extra spaces, quotes)
3. Get a fresh token from Tradetron dashboard (Step 13)
4. Make sure token is from the correct strategy

### Problem: "Signal sent but no order appears in dashboard"

**Solution:**
1. Check if strategy is deployed (Step 10)
2. Verify strategy is in "Paper Trading" mode (Step 9.1)
3. Check strategy is "Active" (not paused)
4. Verify symbol names match exactly (e.g., GOLDDEC24 not GOLD DEC24)
5. Check Tradetron dashboard for error messages
6. Look in "Notifications" section for details

### Problem: "Symbol not found" error

**Solution:**
1. Verify symbol format matches MCX format
   - ✅ Good: GOLDDEC24, SILVERDEC24
   - ❌ Bad: GOLD DEC24 (spaces), golddec24 (lowercase)
2. Check if symbol is available in MCX exchange
3. Ensure symbol is enabled in strategy settings
4. Verify symbol in your code matches Tradetron strategy

### Problem: "Orders not executing"

**Solution:**
1. Check Tradetron dashboard → Orders section
2. Verify strategy is not paused
3. Check if there are error messages
4. Verify paper trading account has sufficient virtual capital
5. Check if market is open (for live prices)
6. Review strategy settings for any restrictions

### Problem: "Can't find API token option"

**Solution:**
1. Make sure strategy is created first (Step 6)
2. Try strategy edit page → Settings/Configuration
3. Look in "API" or "Developer" section
4. Contact Tradetron support: support@tradetron.tech
5. Ask: "How do I get the API OAuth token for my strategy?"

### Problem: "Import error: cannot import TradetronClient"

**Solution:**
1. Make sure you're in the project root folder
2. Check tradetron folder exists: `tradetron/tradetron_client.py`
3. Verify Python path includes project folder
4. Try: `python -c "from tradetron.tradetron_client import TradetronClient; print('OK')"`

---

## FAQ

### Q: Do I need to connect Angel One to Tradetron for paper trading?
**A: NO!** For paper trading, Tradetron has its own built-in paper trading broker ("TT-PaperTrading"). You only need to connect Angel One when you're ready for live trading with real money.

### Q: How does Tradetron work vs direct broker API?
**A:**
- **Direct API (Angel One):** Your code → Direct order → Broker → Trade executed
- **Tradetron:** Your code → Signal → Tradetron → Broker (via Tradetron) → Trade executed
- Tradetron handles position management, stop-losses, etc. automatically

### Q: Is Tradetron the right platform?
**A:** Yes! It's perfect for paper trading. Free, easy to use, and works great for testing strategies before going live.

### Q: Can I use both Angel One and Tradetron?
**A:** Yes! Use Tradetron for paper trading (testing). For live trading, you can either:
- Use Tradetron with Angel One connected (Tradetron executes via Angel One)
- Or use Angel One directly (your current setup)

### Q: What if I can't find the API token?
**A:**
1. Make sure you've created a strategy first
2. Look in "MY STRATEGIES" → find your strategy → look for menu/actions button
3. Contact Tradetron support: support@tradetron.tech

### Q: How do I know if it's working?
**A:** Check Tradetron dashboard - you'll see all orders and positions in real-time in the "Deployed" section.

### Q: Can I change from paper trading to live trading later?
**A:** Yes! Just edit your strategy deployment:
1. Go to "My Strategies" → Your strategy
2. Click "Edit Deployment" or "Deployments"
3. Change "Execution Type" from "Paper Trading" to "Live Trading"
4. Select your connected broker (Angel One, etc.)
5. Confirm changes

### Q: What symbols can I trade?
**A:** Any MCX commodities symbols. Common ones:
- GOLDDEC24, GOLDFEB25 (Gold contracts)
- SILVERDEC24, SILVERFEB25 (Silver contracts)
- CRUDEOILDEC24 (Crude Oil)
- Check MCX website for available contracts

### Q: How do I stop the strategy?
**A:**
1. In your code: Press Ctrl+C in terminal
2. In Tradetron: Go to "Deployed" → Your strategy → Click "Pause" or "Stop"

### Q: What if my signals aren't working?
**A:**
1. Check strategy is deployed and active
2. Verify API token is correct
3. Check symbol format matches exactly
4. Review Tradetron dashboard for error messages
5. See Troubleshooting section above

---

## Summary Checklist

### On Tradetron Platform:
- [ ] Account created and verified
- [ ] Strategy created with "API Signal Based" type
- [ ] Exchange set to MCX
- [ ] "Accept External Signals via API" enabled
- [ ] Strategy deployed in "Paper Trading" mode
- [ ] "TT-PaperTrading" selected as broker
- [ ] Virtual capital set (e.g., ₹1,00,000)
- [ ] API OAuth Token obtained and copied

### In Your Code:
- [ ] TRADETRON_API_TOKEN added to .env file
- [ ] Code updated to use TradetronClient instead of AngelOneClient
- [ ] Test script runs successfully
- [ ] Test order appears in Tradetron dashboard
- [ ] Full strategy script ready to run

### Testing:
- [ ] Test script executed successfully
- [ ] Test order visible in Tradetron dashboard
- [ ] No authentication errors
- [ ] Signals being received by Tradetron

**Once all checkboxes are checked, you're ready to paper trade! 🚀**

---

## Next Steps After Paper Trading Works

1. **Test thoroughly** - Run for several days/weeks in paper mode
2. **Monitor performance** - Check P&L, win rate, drawdowns
3. **Adjust parameters** - Fine-tune stop-loss, take-profit, position sizing
4. **When ready for live trading:**
   - Change strategy deployment from "Paper Trading" to "Live Trading"
   - Connect your real broker account (e.g., Angel One) to Tradetron
   - Start with small position sizes
   - Monitor closely for first few days

---

## Support

**Tradetron Support:**
- Email: support@tradetron.tech
- Website: https://www.tradetron.tech

**Your Code Issues:**
- Check this guide's Troubleshooting section
- Review error messages carefully
- Check Tradetron dashboard for clues

---

**You're all set! Follow these steps exactly, and you'll be paper trading commodities in Tradetron. Good luck! 🎉**
