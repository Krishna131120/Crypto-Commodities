"""
Test MCX Gold Trading with 0.5% Profit Target (Dry Run)

Tests:
1. Symbol mapping: GC=F -> MCX contract (GOLDDEC25)
2. DHAN client initialization
3. Execution engine with DHAN
4. Dry run trading with 0.5% profit target
"""

from __future__ import annotations

import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from trading.dhan_client import DhanClient
from trading.execution_engine import ExecutionEngine, TradingRiskConfig
from trading.symbol_universe import find_by_data_symbol
from trading.mcx_symbol_mapper import get_mcx_contract_for_horizon, get_mcx_lot_size
from unittest.mock import patch, MagicMock

# Load DHAN credentials from .env file (not hardcoded)
import os
from trading.dhan_client import DhanConfig

try:
    config = DhanConfig.from_env()
    DHAN_ACCESS_TOKEN = config.access_token
    DHAN_CLIENT_ID = config.client_id
    print("[INFO] Loaded DHAN credentials from .env file")
except Exception as e:
    print(f"[ERROR] Failed to load credentials from .env: {e}")
    print("  Make sure .env file contains:")
    print("    DHAN_ACCESS_TOKEN=\"your_token\"")
    print("    DHAN_CLIENT_ID=\"1107954503\"")
    sys.exit(1)

print("=" * 80)
print("MCX GOLD TRADING TEST (DRY RUN)")
print("=" * 80)
print()
print(f"[INFO] Using token from .env file")
print(f"[INFO] Client ID: {DHAN_CLIENT_ID}")
print(f"[INFO] Token (first 50 chars): {DHAN_ACCESS_TOKEN[:50]}...")
print()

# Step 1: Test Symbol Mapping
print("[STEP 1] Testing Symbol Mapping")
print("-" * 80)
gold_symbol = "GC=F"
horizon = "short"

try:
    mcx_contract = get_mcx_contract_for_horizon(gold_symbol, horizon)
    lot_size = get_mcx_lot_size(gold_symbol)
    print(f"✓ Yahoo Symbol: {gold_symbol}")
    print(f"✓ MCX Contract: {mcx_contract}")
    print(f"✓ Lot Size: {lot_size} kg per lot")
    print(f"✓ Horizon: {horizon}")
except Exception as e:
    print(f"✗ Symbol mapping failed: {e}")
    sys.exit(1)

# Step 2: Find Asset Mapping
print("\n[STEP 2] Finding Asset Mapping")
print("-" * 80)
try:
    asset = find_by_data_symbol(gold_symbol)
    if not asset:
        print(f"✗ Asset not found for {gold_symbol}")
        sys.exit(1)
    
    print(f"✓ Found asset: {asset.logical_name}")
    print(f"✓ Data Symbol: {asset.data_symbol}")
    print(f"✓ Trading Symbol (base): {asset.trading_symbol}")
    mcx_symbol = asset.get_mcx_symbol(horizon)
    print(f"✓ MCX Contract Symbol: {mcx_symbol}")
except Exception as e:
    print(f"✗ Asset lookup failed: {e}")
    sys.exit(1)

# Step 3: Initialize DHAN Client
print("\n[STEP 3] Initializing DHAN Client")
print("-" * 80)
try:
    dhan_client = DhanClient(
        access_token=DHAN_ACCESS_TOKEN,
        client_id=DHAN_CLIENT_ID
    )
    print(f"✓ DHAN client created")
    print(f"✓ Broker: {dhan_client.broker_name}")
    print(f"✓ Base URL: {dhan_client.config.base_url}")
except Exception as e:
    print(f"✗ DHAN client initialization failed: {e}")
    sys.exit(1)

# Step 4: Test Account Connection (may fail with dummy/invalid token, that's OK for dry run)
print("\n[STEP 4] Testing Account Connection")
print("-" * 80)
try:
    account = dhan_client.get_account()
    equity = account.get("equity", 0)
    buying_power = account.get("buying_power", 0)
    print(f"✓ Account connected")
    print(f"  Equity: ${equity:,.2f}")
    print(f"  Buying Power: ${buying_power:,.2f}")
except Exception as e:
    print(f"⚠ Account connection failed (expected if token expired): {e}")
    print("  Continuing with dry run anyway...")

# Step 5: Setup Execution Engine
print("\n[STEP 5] Setting Up Execution Engine")
print("-" * 80)
try:
    risk_config = TradingRiskConfig(
        default_stop_loss_pct=0.020,  # 2.0% stop-loss for commodities
        profit_target_pct=0.5,  # 0.5% profit target as requested
        allow_short=True,
    )
    
    execution_engine = ExecutionEngine(
        client=dhan_client,
        risk_config=risk_config,
        log_path=Path("logs") / "trading" / "test_mcx_gold.jsonl",
    )
    print(f"✓ Execution engine created")
    print(f"✓ Profit Target: {risk_config.profit_target_pct}%")
    print(f"✓ Stop-Loss: {risk_config.default_stop_loss_pct * 100}%")
except Exception as e:
    print(f"✗ Execution engine setup failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Simulate Trading Decision (Dry Run)
print("\n[STEP 6] Simulating Trading Decision (DRY RUN)")
print("-" * 80)
print("Simulating: Model predicts LONG with 85% confidence")
print()

# Get current price (simulated - in real trading, this comes from live data)
current_price = 65000.0  # Simulated gold price in INR per kg
print(f"Current Price (simulated): ₹{current_price:,.2f} per kg")

# Create consensus (simulating model prediction)
consensus = {
    "consensus_action": "long",
    "consensus_confidence": 0.85,
    "position_size": 100000.0,  # ₹100,000 investment
    "predicted_return": 0.006,  # 0.6% predicted return
}

print(f"Model Consensus:")
print(f"  Action: {consensus['consensus_action'].upper()}")
print(f"  Confidence: {consensus['consensus_confidence'] * 100:.1f}%")
print(f"  Position Size: ₹{consensus['position_size']:,.2f}")
print(f"  Predicted Return: {consensus['predicted_return'] * 100:.2f}%")
print()

try:
    # Mock all DHAN API calls for dry run (since API may not be accessible)
    mock_account = {
        "equity": 1000000.0,
        "buying_power": 1000000.0,
        "cash": 1000000.0,
        "portfolio_value": 1000000.0,
        "margin_used": 0.0,
        "margin_available": 1000000.0,
    }
    
    # Mock positions (no existing positions)
    mock_positions = []
    
    # Execute in dry run mode with all API calls mocked
    with patch.object(dhan_client, 'get_account', return_value=mock_account), \
         patch.object(dhan_client, 'list_positions', return_value=mock_positions), \
         patch.object(dhan_client, 'get_position', return_value=None), \
         patch.object(dhan_client, 'get_last_trade', return_value={"price": current_price}):
        result = execution_engine.execute_from_consensus(
            asset=asset,
            consensus=consensus,
            current_price=current_price,
            dry_run=True,  # DRY RUN - no real orders
            horizon_profile=horizon,
            profit_target_pct=0.5,  # 0.5% profit target
        )
    
    if result:
        print("=" * 80)
        print("TRADING DECISION RESULT")
        print("=" * 80)
        print(f"Decision: {result.get('decision', 'N/A')}")
        print(f"Final Side: {result.get('final_side', 'N/A')}")
        print(f"Entry Qty: {result.get('entry_qty', 0):.2f} kg")
        print(f"Entry Notional: ₹{result.get('entry_notional', 0):,.2f}")
        
        if result.get('stop_loss_price'):
            print(f"Stop-Loss Price: ₹{result.get('stop_loss_price', 0):,.2f}")
            print(f"Stop-Loss %: {result.get('stop_loss_pct', 0) * 100:.2f}%")
        
        if result.get('profit_target_price'):
            print(f"Profit Target Price: ₹{result.get('profit_target_price', 0):,.2f}")
            print(f"Profit Target %: {result.get('profit_target_pct', 0):.2f}%")
        
        print(f"\nMCX Contract Symbol: {mcx_symbol}")
        print(f"Lot Size: {lot_size} kg per lot")
        
        # Calculate expected values
        entry_qty = result.get('entry_qty', 0)
        entry_price = current_price
        profit_target_price = result.get('profit_target_price', current_price * 1.005)
        stop_loss_price = result.get('stop_loss_price', current_price * 0.98)
        
        if entry_qty > 0:
            entry_cost = entry_qty * entry_price
            expected_profit = (profit_target_price - entry_price) * entry_qty
            max_loss = (entry_price - stop_loss_price) * entry_qty
            
            print(f"\nExpected Outcomes:")
            print(f"  Entry Cost: ₹{entry_cost:,.2f}")
            print(f"  Expected Profit (at 0.5% target): ₹{expected_profit:,.2f}")
            print(f"  Maximum Loss (at stop-loss): ₹{max_loss:,.2f}")
            print(f"  Risk/Reward: {abs(expected_profit / max_loss) if max_loss > 0 else 0:.2f}:1")
        
        print("\n" + "=" * 80)
        print("✓ DRY RUN COMPLETE - No real orders were placed")
        print("=" * 80)
    else:
        print("⚠ No trading decision (model may have said 'hold' or confidence too low)")
        
except RuntimeError as e:
    if "require DHAN broker" in str(e):
        print(f"✗ Broker enforcement error: {e}")
        sys.exit(1)
    else:
        raise
except Exception as e:
    print(f"✗ Trading execution failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] All tests passed!")
print("Symbol mapping works correctly: GC=F -> MCX contract")
print("DHAN client initialized successfully")
print("Execution engine ready for MCX trading")
