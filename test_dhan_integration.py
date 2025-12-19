"""
Test DHAN MCX Integration

Quick test to verify DHAN client and MCX symbol mapping work correctly.
"""

from trading.dhan_client import DhanClient
from trading.mcx_symbol_mapper import get_mcx_contract_for_horizon, get_mcx_lot_size, round_to_lot_size

# Test DHAN client creation
print("=" * 80)
print("TESTING DHAN CLIENT")
print("=" * 80)

access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzY2MDM1ODEwLCJpYXQiOjE3NjU5NDk0MTAsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTA3OTU0NTAzIn0.PUuKWaMTT-5zKBgPMpFft-xh93v85sJ4h7-xPWiRJyHtLNVM9wdU7GrCwg4SZOmOY0jGuZlzJePhcux633KPjQ"
client_id = "1107954503"

try:
    client = DhanClient(access_token=access_token, client_id=client_id)
    print(f"[PASS] DHAN client created successfully")
    print(f"  Broker: {client.broker_name}")
    print(f"  Base URL: {client.config.base_url}")
except Exception as e:
    print(f"[FAIL] Failed to create DHAN client: {e}")
    exit(1)

# Test MCX symbol mapping
print("\n" + "=" * 80)
print("TESTING MCX SYMBOL MAPPING")
print("=" * 80)

test_symbols = ["GC=F", "CL=F", "SI=F", "BZ=F"]

for symbol in test_symbols:
    try:
        short_contract = get_mcx_contract_for_horizon(symbol, "short")
        intraday_contract = get_mcx_contract_for_horizon(symbol, "intraday")
        long_contract = get_mcx_contract_for_horizon(symbol, "long")
        lot_size = get_mcx_lot_size(symbol)
        
        print(f"\n{symbol}:")
        print(f"  Short-term contract:   {short_contract}")
        print(f"  Intraday contract:     {intraday_contract}")
        print(f"  Long-term contract:    {long_contract}")
        print(f"  Lot size:              {lot_size}")
        
        # Test lot size rounding
        test_qty = 1.5 * lot_size
        rounded = round_to_lot_size(test_qty, symbol)
        print(f"  Test rounding:          {test_qty:.2f} -> {rounded} lots")
        
    except Exception as e:
        print(f"[FAIL] {symbol}: {e}")

print("\n" + "=" * 80)
print("INTEGRATION TEST COMPLETE")
print("=" * 80)
