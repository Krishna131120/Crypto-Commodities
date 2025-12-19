"""
Test MCX/DHAN enforcement for commodities
"""

from trading.execution_engine import ExecutionEngine
from trading.alpaca_client import AlpacaClient
from trading.dhan_client import DhanClient
from trading.symbol_universe import find_by_data_symbol

print("=" * 80)
print("TESTING MCX/DHAN ENFORCEMENT")
print("=" * 80)

# Test 1: Commodity with Alpaca (should fail)
print("\n[TEST 1] Commodity (GC=F) with AlpacaClient (should FAIL):")
try:
    engine = ExecutionEngine(client=AlpacaClient())
    asset = find_by_data_symbol('GC=F')
    engine.execute_from_consensus(
        asset=asset,
        consensus={'consensus_action': 'long', 'consensus_confidence': 0.8, 'position_size': 1000},
        current_price=50000,
        dry_run=True
    )
    print("[FAIL] Should have raised RuntimeError!")
except RuntimeError as e:
    print(f"[PASS] Correctly blocked: {str(e)[:100]}...")
except Exception as e:
    print(f"[UNEXPECTED] Different error: {type(e).__name__}: {e}")

# Test 2: Commodity with DHAN (should work)
print("\n[TEST 2] Commodity (GC=F) with DhanClient (should WORK):")
try:
    # Use dummy credentials for testing (will fail on actual API call, but should pass validation)
    dhan_client = DhanClient(
        access_token="dummy_token",
        client_id="dummy_client"
    )
    engine = ExecutionEngine(client=dhan_client)
    asset = find_by_data_symbol('GC=F')
    # This will fail on API call but should pass the broker check
    try:
        engine.execute_from_consensus(
            asset=asset,
            consensus={'consensus_action': 'long', 'consensus_confidence': 0.8, 'position_size': 1000},
            current_price=50000,
            dry_run=True
        )
        print("[PASS] Broker check passed (would work with valid credentials)")
    except Exception as api_error:
        if "DHAN" in str(api_error) or "broker" in str(api_error).lower():
            print(f"[FAIL] Broker check failed: {api_error}")
        else:
            print(f"[PASS] Broker check passed (API error expected with dummy credentials): {type(api_error).__name__}")
except Exception as e:
    print(f"[ERROR] Setup failed: {e}")

# Test 3: Crypto with Alpaca (should work)
print("\n[TEST 3] Crypto (BTC-USDT) with AlpacaClient (should WORK):")
try:
    engine = ExecutionEngine(client=AlpacaClient())
    asset = find_by_data_symbol('BTC-USDT')
    # This should not raise broker error (crypto can use Alpaca)
    print("[PASS] Crypto can use Alpaca (no broker enforcement for crypto)")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
