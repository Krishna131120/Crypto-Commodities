"""
Quick diagnostic script to test Angel One price fetching.
Run this to see the exact error.
"""
from trading.angelone_client import AngelOneClient
from trading.mcx_symbol_mapper import get_mcx_contract_symbol

print("=" * 80)
print("ANGEL ONE PRICE FETCH DIAGNOSTIC")
print("=" * 80)

# Test 1: Authentication
print("\n[TEST 1] Testing Authentication...")
try:
    client = AngelOneClient()
    print("[OK] Authentication: SUCCESS")
    print(f"   Token: {client._access_token[:30]}..." if hasattr(client, '_access_token') and client._access_token else "   Token: None")
except Exception as e:
    print(f"[ERROR] Authentication: FAILED")
    print(f"   Error: {e}")
    import traceback
    print(f"   Details: {traceback.format_exc()}")
    exit(1)

# Test 2: MCX Symbol Mapping
print("\n[TEST 2] Testing MCX Symbol Mapping...")
test_symbols = ["GC=F", "MCX_GOLDPETAL"]
for symbol in test_symbols:
    try:
        mcx_symbol = get_mcx_contract_symbol(symbol)
        print(f"[OK] {symbol} -> {mcx_symbol}")
    except Exception as e:
        print(f"[ERROR] {symbol}: {e}")

# Test 3: Price Fetching
print("\n[TEST 2.5] Testing Symbol Token Lookup...")
try:
    from trading.angelone_client import AngelOneClient
    client = AngelOneClient()
    
    # Test token lookup for Gold
    print("\n   Testing GC=F (MCX: GOLDDEC25)...")
    token = client._get_symbol_token("GOLDDEC25", "MCX")
    if token:
        print(f"   [OK] Token found: {token}")
    else:
        print(f"   [WARN] Token not found - scrip master may need download")
        print(f"   [INFO] Scrip master will be downloaded on first use")
    
    # Test token lookup for Gold Petal
    print("\n   Testing MCX_GOLDPETAL (MCX: GOLDPETALDEC25)...")
    token2 = client._get_symbol_token("GOLDPETALDEC25", "MCX")
    if token2:
        print(f"   [OK] Token found: {token2}")
    else:
        print(f"   [WARN] Token not found - scrip master may need download")
except Exception as e:
    print(f"   [ERROR] Token lookup failed: {e}")
    import traceback
    traceback.print_exc()

print("\n[TEST 3] Testing Price Fetching...")
for symbol in test_symbols:
    try:
        mcx_symbol = get_mcx_contract_symbol(symbol)
        print(f"\n   Testing {symbol} (MCX: {mcx_symbol})...")
        
        last_trade = client.get_last_trade(mcx_symbol, max_retries=2, retry_delay=1.0)
        
        if last_trade:
            price = last_trade.get("price") or last_trade.get("p") or last_trade.get("ltp")
            print(f"   [OK] Price: Rs {price}")
        else:
            print(f"   [ERROR] No price returned")
            
            # Try to see what the API actually returns
            print(f"   [DEBUG] Attempting direct API call...")
            try:
                body = {
                    "mode": "LTP",
                    "exchangeTokens": {
                        "MCX": [mcx_symbol.upper()]
                    }
                }
                response = client._request("POST", "/rest/secure/marketData/quote", json_body=body)
                print(f"   [DEBUG] API Response: {response}")
            except Exception as api_e:
                print(f"   [DEBUG] API Error: {api_e}")
                import traceback
                print(f"   [DEBUG] Traceback: {traceback.format_exc()}")
                
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        import traceback
        print(f"   Details: {traceback.format_exc()}")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

