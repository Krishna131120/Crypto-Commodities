# Tradetron Paper Trading - Complete Flow Chart & Explanation

## Overview
This document explains the **complete flow** of paper trading commodities through Tradetron, from command execution to order placement and position monitoring.

---

## 📋 TABLE OF CONTENTS
1. [Command Execution](#1-command-execution)
2. [Initialization & Setup](#2-initialization--setup)
3. [Symbol Discovery](#3-symbol-discovery)
4. [Feature Loading & Model Prediction](#4-feature-loading--model-prediction)
5. [Risk Assessment & Position Sizing](#5-risk-assessment--position-sizing)
6. [Execution Engine Decision](#6-execution-engine-decision)
7. [TradetronClient Webhook Signal](#7-tradetronclient-webhook-signal)
8. [Tradetron Platform Processing](#8-tradetron-platform-processing)
9. [Position Monitoring Loop](#9-position-monitoring-loop)
10. [Exit Logic](#10-exit-logic)

---

## 1. COMMAND EXECUTION

### Entry Point: `trade_all_commodities_auto.py`

```bash
python trade_all_commodities_auto.py \
    --broker tradetron \
    --asset-type commodities \
    --profit-target-pct 1.0 \
    --stop-loss-pct 3.5 \
    --timeframe 1d \
    --horizon short \
    --interval 300
```

### What Happens:
1. **Script starts** → Parses command-line arguments
2. **Broker validation** → Checks if `--broker tradetron` is specified
3. **Environment check** → Validates `.env` file has:
   - `TRADETRON_API_TOKEN` (UUID)
   - `TRADETRON_AUTH_TOKEN` (webhook auth token)

### Code Location:
```python
# trade_all_commodities_auto.py, line ~250-280
args = parser.parse_args()

# Validate broker
if args.broker != "tradetron":
    raise ValueError("For Tradetron paper trading, use --broker tradetron")

# Setup TradetronClient
tradetron_client = setup_tradetron_client()  # Loads from .env
```

---

## 2. INITIALIZATION & SETUP

### Components Initialized:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. TradetronClient                                          │
│    - Loads TRADETRON_API_TOKEN from .env                    │
│    - Loads TRADETRON_AUTH_TOKEN from .env                   │
│    - Configures API endpoint URL                            │
│    - Validates credentials                                   │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. ExecutionEngine                                       │
│    - Receives TradetronClient instance                        │
│    - Configures TradingRiskConfig:                          │
│      * profit_target_pct = 1.0 (from CLI)                   │
│      * stop_loss_pct = 3.5% (from CLI)                      │
│      * max_notional_per_symbol_pct = 10%                    │
│      * max_total_equity_pct = 50%                           │
│    - Initializes PositionManager                            │
│    - Initializes SymbolLossTracker                          │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. PositionManager                                          │
│    - Loads active_positions.json                            │
│    - Restores any existing open positions                   │
│    - Sets up position tracking                              │
└─────────────────────────────────────────────────────────────┘
```

### Code Location:
```python
# trade_all_commodities_auto.py, line ~45-52
def setup_tradetron_client() -> TradetronClient:
    return TradetronClient()  # Loads from .env

# Line ~380-420
tradetron_client = setup_tradetron_client()
execution_engine = ExecutionEngine(
    client=tradetron_client,
    risk_config=risk_config,
    position_manager=position_manager
)
```

---

## 3. SYMBOL DISCOVERY

### Process Flow:

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Load Symbol Universe                                │
│   - Reads trading/symbol_universe.py                        │
│   - Filters by asset_type="commodities"                     │
│   - Filters by enabled=True                                 │
│   - Returns 30 enabled commodities                          │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Discover Tradable Symbols                           │
│   - For each commodity symbol:                              │
│     * Check if model exists:                                │
│       models/commodities/{symbol}/1d/short/                 │
│     * Load model summary.json                               │
│     * Verify model is trained and valid                     │
│   - Returns list of tradable symbols with:                  │
│     * data_symbol (e.g., "MCX_GOLDM")                       │
│     * trading_symbol (e.g., "GOLDM")                        │
│     * model metadata                                        │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Rank by Performance (Optional)                      │
│   - Evaluate model R² scores                                │
│   - Check confidence levels                                 │
│   - Rank commodities by performance                         │
│   - Select top N for trading                                │
└─────────────────────────────────────────────────────────────┘
```

### Code Location:
```python
# live_trader.py, line ~235-315
def discover_tradable_symbols(asset_type, timeframe, override_horizon):
    # 1. Get enabled symbols from universe
    all_assets = all_enabled() if asset_type == "crypto" else by_asset_type(asset_type)
    
    # 2. Check each symbol has a trained model
    for asset in all_assets:
        model_dir = horizon_dir(asset_type, asset.data_symbol, timeframe, horizon)
        if model_dir.exists():
            tradable.append({...})
    
    return tradable
```

### Example Symbols Discovered:
```
ALUMINI → MCX_ALUMINI → Trading Symbol: ALUMINI
CRUDEOILM → MCX_CRUDEOILM → Trading Symbol: CRUDEOILM
GOLD → GC=F → Trading Symbol: GOLD
COPPER → HG=F → Trading Symbol: COPPER
... (30 total)
```

---

## 4. FEATURE LOADING & MODEL PREDICTION

### Process Flow:

```
┌─────────────────────────────────────────────────────────────┐
│ For Each Tradable Symbol:                                   │
│                                                             │
│ Step 1: Load Latest Features                                │
│   - Path: data/features/commodities/{symbol}/1d/           │
│     features.json                                           │
│   - Reads latest feature row                               │
│   - Extracts: RSI, MACD, ATR, price, volume, etc.          │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Get Current Price                                   │
│   - Priority 1: Broker API (Tradetron doesn't provide)    │
│   - Priority 2: Position-based price (if position exists)   │
│   - Priority 3: Local data.json (Yahoo Finance)             │
│   - Priority 4: Last candle close price from features       │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Run Model Prediction                                │
│   - Loads models from:                                      │
│     models/commodities/{symbol}/1d/short/                   │
│   - Models: Lasso, Ridge, ElasticNet, XGBoost, LightGBM    │
│   - Runs InferencePipeline.predict()                        │
│   - Gets consensus:                                         │
│     * consensus_action: "long", "short", or "flat"          │
│     * consensus_return: predicted return %                  │
│     * confidence: average model confidence                  │
│     * model_agreement: how many models agree                │
└─────────────────────────────────────────────────────────────┘
```

### Code Location:
```python
# live_trader.py, line ~38-68
def load_feature_row(asset_type, symbol, timeframe):
    feature_path = Path("data/features") / asset_type / symbol / timeframe / "features.json"
    # Load and parse features

# Line ~71-150
def get_current_price_from_features(asset_type, symbol, timeframe):
    # Priority-based price fetching

# Line ~300-450
for symbol_info in tradable_symbols:
    features = load_feature_row(...)
    current_price = get_current_price_from_features(...)
    consensus = inference_pipeline.predict(features)
```

### Example Consensus Output:
```json
{
  "consensus_action": "long",
  "consensus_return": 0.015,
  "confidence": 0.75,
  "model_agreement_ratio": 0.8,
  "predicted_price": 72000.0,
  "current_price": 71000.0
}
```

---

## 5. RISK ASSESSMENT & POSITION SIZING

### Process Flow:

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Check Existing Position                             │
│   - PositionManager.get_position(symbol)                    │
│   - If position exists:                                     │
│     * Check status: "open", "closed"                        │
│     * Check entry_price, quantity, profit_target            │
│     * Calculate unrealized P/L                              │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Risk Checks                                          │
│   - SymbolLossTracker.can_trade(symbol)                     │
│     * Checks consecutive losses                             │
│     * Checks daily loss limit                               │
│     * Checks cooldown period                                │
│   - Momentum Filter                                         │
│     * RSI overbought/oversold check                         │
│     * Recent price movement check                           │
│   - Position Limit                                          │
│     * Max 10% equity per symbol                             │
│     * Max 50% total equity deployed                         │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Calculate Position Size                             │
│   - For MCX commodities:                                    │
│     * Get lot size (e.g., GOLDM = 100g)                     │
│     * Calculate notional: equity * max_notional_pct         │
│     * Convert to lots: notional / (price * lot_size)        │
│     * Round to nearest lot                                  │
│   - Example:                                                │
│     * Equity: Rs. 100,000                                   │
│     * Max per symbol: 10% = Rs. 10,000                      │
│     * GOLDM price: Rs. 50,000/lot                           │
│     * Quantity: 10,000 / 50,000 = 0.2 lots → 0 lots        │
│     * (Minimum 1 lot, so use 1 lot = Rs. 50,000)           │
└─────────────────────────────────────────────────────────────┘
```

### Code Location:
```python
# trading/execution_engine.py, line ~145-400
def execute_from_consensus(self, asset, consensus, current_price):
    # Check existing position
    existing_pos = self.position_manager.get_position(...)
    
    # Risk checks
    if not self.loss_tracker.can_trade(symbol):
        return {"decision": "skip", "reason": "symbol_blocked"}
    
    # Momentum filter
    if momentum_filter_rejects_entry(...):
        return {"decision": "skip", "reason": "momentum_filter"}
    
    # Position sizing
    desired_notional = equity * self.risk.max_notional_per_symbol_pct
    qty_lots = calculate_mcx_lot_quantity(desired_notional, current_price, lot_size)
```

---

## 6. EXECUTION ENGINE DECISION

### Decision Tree:

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: consensus_action, existing_position                  │
│                                                             │
│ Decision Logic:                                             │
│                                                             │
│ 1. IF consensus_action == "flat" AND existing_position:    │
│      → EXIT position (if meets exit criteria)              │
│                                                             │
│ 2. IF consensus_action == "long" AND no position:          │
│      → ENTER LONG (after risk checks pass)                 │
│                                                             │
│ 3. IF consensus_action == "long" AND existing long:        │
│      → HOLD (no action needed)                             │
│                                                             │
│ 4. IF consensus_action == "short" AND existing long:       │
│      → EXIT (model flipped signal)                         │
│                                                             │
│ 5. IF profit_target reached:                                │
│      → EXIT (profit target hit)                            │
│                                                             │
│ 6. IF stop_loss hit:                                        │
│      → EXIT (stop-loss triggered)                          │
└─────────────────────────────────────────────────────────────┘
```

### Code Location:
```python
# trading/execution_engine.py, line ~145-600
def execute_from_consensus(...):
    # Determine desired side
    target_side = "long" if consensus_action == "long" else "flat"
    
    # Compare with existing position
    existing_side = "long" if existing_pos and existing_pos.status == "open" else "flat"
    
    # Decision logic
    if target_side == existing_side:
        decision = "hold"
    elif target_side == "flat" and existing_side != "flat":
        decision = "exit_position"
    elif target_side != "flat" and existing_side == "flat":
        decision = "enter_long"  # or enter_short
    else:
        decision = "exit_and_reverse"
```

---

## 7. TRADETRONCLIENT WEBHOOK SIGNAL

### When Decision = "enter_long":

```
┌─────────────────────────────────────────────────────────────┐
│ ExecutionEngine calls:                                      │
│   execution_engine.client.submit_order(...)                 │
│                                                             │
│ This routes to TradetronClient.submit_order()               │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Normalize Symbol                                    │
│   - Input: "COOPER MI" or "GOLDM"                           │
│   - Normalization: symbol.upper().replace(" ", "_")         │
│   - Output: "COOPER_MI" or "GOLDM"                          │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Build Webhook Payload                               │
│   {                                                          │
│     "auth-token": "your-tradetron-auth-token",              │
│     "{SYMBOL}_long": "1",           # Enable long           │
│     "{SYMBOL}_short": "0",          # Disable short         │
│     "{SYMBOL}_long_lots": "10",     # Quantity in lots      │
│     "{SYMBOL}_stop_loss": "172.69", # Optional              │
│     "{SYMBOL}_target": "174.48"     # Optional              │
│   }                                                          │
│                                                             │
│ Example for GOLDM:                                          │
│   {                                                          │
│     "auth-token": "abc123...",                              │
│     "GOLDM_long": "1",                                      │
│     "GOLDM_short": "0",                                     │
│     "GOLDM_long_lots": "1"                                  │
│   }                                                          │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Send HTTP POST Request                              │
│   - URL: https://api.tradetron.tech/v1/webhook/{api_token}  │
│   - Method: POST                                            │
│   - Headers: Content-Type: application/json                 │
│   - Body: signal_payload (JSON)                            │
│   - Response: { "status": "success", "message": "..." }     │
└─────────────────────────────────────────────────────────────┘
```

### Code Location:
```python
# tradetron/tradetron_client.py, line ~240-340
def submit_order(self, symbol, side, qty, ...):
    # Normalize symbol
    symbol_normalized = symbol.upper().replace(" ", "_").replace("-", "_")
    
    # Build payload
    signal_payload = {
        "auth-token": self.config.auth_token,
    }
    
    if side == "buy":
        signal_payload[f"{symbol_normalized}_long"] = "1"
        signal_payload[f"{symbol_normalized}_short"] = "0"
        signal_payload[f"{symbol_normalized}_long_lots"] = str(qty_int)
    
    # Send webhook
    response = self._request("POST", "", json_body=signal_payload)
    
    return {
        "id": "tradetron_signal",
        "status": "accepted",
        "symbol": symbol.upper(),
        "qty": qty_int,
        "side": side.upper(),
    }
```

### Example Webhook Payload:
```json
{
  "auth-token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "GOLDM_long": "1",
  "GOLDM_short": "0",
  "GOLDM_long_lots": "1"
}
```

---

## 8. TRADETRON PLATFORM PROCESSING

### What Happens on Tradetron Side:

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Webhook Received                                    │
│   - Tradetron receives HTTP POST at webhook endpoint        │
│   - Validates auth-token                                    │
│   - Extracts strategy ID from URL                           │
│   - Parses signal payload                                   │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Signal Interpretation                               │
│   - Reads "{SYMBOL}_long": "1"                              │
│   - Reads "{SYMBOL}_long_lots": "10"                        │
│   - Maps symbol to MCX instrument (e.g., "GOLDM")           │
│   - Validates quantity (10 lots)                            │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Strategy Execution                                  │
│   - Checks Position Builder settings:                       │
│     * Trade Type: BUY                                       │
│     * Exchange: MCX                                         │
│     * Type: Futures                                         │
│     * Product: NRML                                         │
│     * Underlying: GOLDM                                     │
│     * Qty: 10 lots                                          │
│   - Executes order on "TT Paper Trading" broker             │
│   - Gets order confirmation                                 │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Order Execution (Paper Trading)                     │
│   - Paper trading account simulates execution               │
│   - No real money involved                                  │
│   - Gets filled price (simulated)                           │
│   - Returns order response                                  │
└─────────────────────────────────────────────────────────────┘
```

### Tradetron Strategy Configuration (From Position Builder):
```
Entry (S1E) Conditions:
  - (Empty - no conditions, just execute on signal)

Entry (S1E) Positions:
  - Buy/Sell: Buy
  - Underlying: GOLDM (or any commodity)
  - Strike: - (not used for futures)
  - Type: - (futures)
  - Expiry: tt_mcx_fut_expiry('Gol0')  (auto-current month)
  - Qty: 10 (from webhook signal)
```

---

## 9. POSITION MONITORING LOOP

### Continuous Monitoring Process:

```
┌─────────────────────────────────────────────────────────────┐
│ Main Trading Loop (Every --interval seconds)                │
│                                                             │
│ 1. Load all active positions from PositionManager                  │
│ 2. For each position:                                       │
│    a. Get current market price                              │
│    b. Calculate unrealized P/L                              │
│    c. Check profit target                                   │
│    d. Check stop-loss                                       │
│    e. Check model flip signal                               │
│    f. Check trailing stop                                   │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Position Check Logic:                                       │
│                                                             │
│ IF current_price >= profit_target_price:                    │
│   → EXIT (profit target hit)                                │
│                                                             │
│ IF current_price <= stop_loss_price:                        │
│   → EXIT (stop-loss hit)                                    │
│                                                             │
│ IF current_price dropped X% from peak:                      │
│   → EXIT (trailing stop)                                    │
│                                                             │
│ IF model consensus == "flat":                               │
│   → EXIT (model flip signal)                                │
│                                                             │
│ ELSE:                                                       │
│   → HOLD (continue monitoring)                              │
└─────────────────────────────────────────────────────────────┘
```

### Code Location:
```python
# live_trader.py, line ~700-900
def run_trading_cycle(...):
    while True:
        # Monitor existing positions
        open_positions = position_manager.get_open_positions()
        
        for position in open_positions:
            current_price = get_current_price(...)
            
            # Check profit target
            if current_price >= position.profit_target_price:
                execute_exit(position, "profit_target_hit")
            
            # Check stop-loss
            if current_price <= position.stop_loss_price:
                execute_exit(position, "stop_loss_hit")
            
            # Check trailing stop
            if current_price <= calculate_trailing_stop(position):
                execute_exit(position, "trailing_stop_from_peak")
        
        # Wait for next cycle
        time.sleep(interval)
```

---

## 10. EXIT LOGIC

### When Exit is Triggered:

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Prepare Exit Order                                  │
│   - Symbol: GOLDM                                           │
│   - Side: "sell" (to close long)                            │
│   - Quantity: 10 lots (same as entry)                       │
│   - Order Type: "market"                                    │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Send Exit Signal to Tradetron                       │
│   {                                                          │
│     "auth-token": "...",                                    │
│     "GOLDM_long": "0",        # Disable long                │
│     "GOLDM_short": "1",       # Enable short (or just close)│
│     "GOLDM_short_lots": "10"  # Or use close signal         │
│   }                                                          │
│                                                             │
│   OR Tradetron may use a "close" signal:                    │
│   {                                                          │
│     "auth-token": "...",                                    │
│     "GOLDM_close": "1"      # Close all positions           │
│   }                                                          │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Get Exit Price                                      │
│   - Tradetron executes exit order                           │
│   - Gets filled exit price                                  │
│   - Returns order response                                  │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Update Position                                     │
│   - Calculate realized P/L:                                 │
│     (exit_price - entry_price) * quantity                   │
│   - Update position status: "closed"                        │
│   - Save exit_price, exit_time, exit_reason                 │
│   - Update PositionManager                                  │
│   - Log trade to crypto_trades.jsonl                        │
│   - Update SymbolLossTracker                                │
└─────────────────────────────────────────────────────────────┘
```

### Code Location:
```python
# trading/execution_engine.py, line ~1200-1600
def _execute_exit(self, position, exit_reason):
    # Prepare exit order
    exit_order = self.client.submit_order(
        symbol=position.symbol,
        side="sell",  # Close long position
        qty=position.quantity,
        order_type="market"
    )
    
    # Get filled price
    exit_price = exit_order.get("filled_avg_price") or current_price
    
    # Calculate P/L
    realized_pl = (exit_price - position.entry_price) * position.quantity
    
    # Update position
    position.exit_price = exit_price
    position.exit_time = datetime.now()
    position.exit_reason = exit_reason
    position.status = "closed"
    position.realized_pl = realized_pl
    
    # Save
    self.position_manager.save_position(position)
    self.loss_tracker.record_trade(position)
```

---

## 🔄 COMPLETE FLOW DIAGRAM

```
START
  │
  ├─→ [1] Command: python trade_all_commodities_auto.py --broker tradetron
  │
  ├─→ [2] Initialize TradetronClient (loads .env)
  │   ├─→ Load TRADETRON_API_TOKEN
  │   └─→ Load TRADETRON_AUTH_TOKEN
  │
  ├─→ [3] Initialize ExecutionEngine with TradetronClient
  │
  ├─→ [4] Discover Tradable Symbols
  │   ├─→ Read symbol_universe.py
  │   ├─→ Filter: asset_type="commodities", enabled=True
  │   └─→ Check each has trained model
  │
  ├─→ [5] Main Trading Loop (every --interval seconds)
  │   │
  │   ├─→ [6] For each symbol:
  │   │   ├─→ Load features from features.json
  │   │   ├─→ Get current price
  │   │   └─→ Run model prediction (InferencePipeline)
  │   │
  │   ├─→ [7] Check existing positions (PositionManager)
  │   │
  │   ├─→ [8] For each position:
  │   │   ├─→ Check profit target
  │   │   ├─→ Check stop-loss
  │   │   ├─→ Check trailing stop
  │   │   └─→ If exit needed → Send exit signal
  │   │
  │   ├─→ [9] For each tradable symbol:
  │   │   ├─→ Risk checks (SymbolLossTracker, momentum filter)
  │   │   ├─→ Position sizing (calculate lots)
  │   │   └─→ If entry signal → Send entry signal
  │   │
  │   └─→ [10] Wait --interval seconds, then repeat
  │
  └─→ [11] Exit Signal Flow:
      │
      ├─→ ExecutionEngine.execute_from_consensus()
      │   └─→ Determines: "enter_long", "exit_position", or "hold"
      │
      ├─→ IF "enter_long":
      │   └─→ TradetronClient.submit_order(symbol, "buy", qty)
      │       ├─→ Normalize symbol: "GOLDM"
      │       ├─→ Build payload: {"GOLDM_long": "1", "GOLDM_long_lots": "10"}
      │       └─→ POST to Tradetron webhook
      │
      └─→ IF "exit_position":
          └─→ TradetronClient.submit_order(symbol, "sell", qty)
              ├─→ Normalize symbol: "GOLDM"
              ├─→ Build payload: {"GOLDM_long": "0", ...}
              └─→ POST to Tradetron webhook
```

---

## 📊 DATA FLOW SUMMARY

### Entry Flow:
```
Command → TradetronClient → ExecutionEngine → Model Prediction 
  → Risk Check → Position Sizing → Webhook Signal → Tradetron 
  → Paper Trading Execution → Position Saved
```

### Exit Flow:
```
Monitoring Loop → Check Position → Exit Criteria Met 
  → Exit Signal → Tradetron → Paper Trading Close 
  → P/L Calculated → Position Closed → Logged
```

---

## 🔑 KEY POINTS

1. **No Direct Broker API**: Tradetron uses webhook signals, not direct order API
2. **Signal-Based**: Orders are sent as key-value pairs in webhook payload
3. **Paper Trading**: All trades execute on "TT Paper Trading" broker (virtual)
4. **MCX Exchange**: All commodities trade on MCX (Multi Commodity Exchange)
5. **Lot-Based**: Commodities use lot-based quantities (e.g., 1 lot = 100g for GOLDM)
6. **Symbol Mapping**: Tradetron symbols must match Position Builder configuration
7. **Position Monitoring**: Continuous loop checks profit targets and stop-losses
8. **Automatic Exits**: System automatically exits when targets are hit

---

## ⚠️ IMPORTANT NOTES

1. **Tradetron Strategy Must Be Deployed**: Before running, ensure:
   - Strategy is created in Tradetron
   - Position Builder has all commodities configured
   - Strategy is deployed with "TT Paper Trading" broker
   - MCX exchange is selected

2. **Webhook URL**: The webhook endpoint is constructed from `TRADETRON_API_TOKEN`

3. **Symbol Normalization**: Symbols are normalized (spaces → underscores) before sending

4. **Lot Sizing**: Position size is calculated in lots, not notional amount

5. **No Real-Time Price Feed**: Tradetron doesn't provide live prices, so system uses:
   - Yahoo Finance data.json
   - Last candle close price from features
   - Position-based prices (if available)

---

## ✅ VERIFICATION CHECKLIST

Before running paper trading, verify:

- [ ] `.env` has `TRADETRON_API_TOKEN` and `TRADETRON_AUTH_TOKEN`
- [ ] Tradetron strategy is deployed with "TT Paper Trading" broker
- [ ] MCX exchange is selected in deployment
- [ ] Position Builder has all commodities with correct underlying names
- [ ] Models are trained for commodities you want to trade
- [ ] `symbol_universe.py` has all commodities enabled
- [ ] Webhook URL is accessible (Tradetron webhook endpoint)

---

**END OF FLOW CHART**
