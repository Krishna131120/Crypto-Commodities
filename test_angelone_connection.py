"""
Test AngelOne connection and data fetching.
"""
import sys
from pathlib import Path

print("=" * 80)
print("ANGELONE CONNECTION TEST")
print("=" * 80)

# Check .env file
env_file = Path(".env")
print(f"\n1. Checking .env file...")
print(f"   .env exists: {env_file.exists()}")

if env_file.exists():
    content = env_file.read_text(encoding="utf-8")
    has_angel = "ANGELONE" in content.upper() or "ANGEL_ONE" in content.upper()
    print(f"   Contains AngelOne config: {has_angel}")
    
    if has_angel:
        lines = [l.strip() for l in content.split('\n') 
                 if ('ANGELONE' in l.upper() or 'ANGEL_ONE' in l.upper()) 
                 and not l.strip().startswith('#') 
                 and '=' in l]
        print(f"   Found {len(lines)} AngelOne config lines:")
        for line in lines[:5]:
            key = line.split('=')[0].strip()
            val = line.split('=')[1].strip()[:20] if '=' in line else ""
            print(f"     {key} = {val}...")
    else:
        print("   [WARN] No AngelOne credentials found in .env")
        print("   Required: ANGEL_ONE_API_KEY, ANGEL_ONE_CLIENT_ID, ANGEL_ONE_PASSWORD")
else:
    print("   [WARN] .env file not found")

# Test AngelOne client initialization
print(f"\n2. Testing AngelOne client initialization...")
try:
    from trading.angelone_client import AngelOneClient, AngelOneAuthError
    
    try:
        client = AngelOneClient()
        print("   [OK] AngelOneClient initialized successfully")
        
        # Test authentication
        print(f"\n3. Testing authentication...")
        try:
            account = client.get_account()
            print("   [OK] Authentication successful")
            print(f"   Account equity: {account.get('equity', 'N/A')}")
        except AngelOneAuthError as auth_err:
            print(f"   [ERROR] Authentication failed: {auth_err}")
            print("   Possible causes:")
            print("     - Invalid API key, client ID, or password")
            print("     - TOTP not enabled in AngelOne account")
            print("     - IP not whitelisted in AngelOne SmartAPI")
        except Exception as e:
            print(f"   [ERROR] Connection error: {e}")
            print("   Possible causes:")
            print("     - Network timeout (check internet connection)")
            print("     - AngelOne API is down")
            print("     - IP not whitelisted")
        
        # Test symbol token lookup
        print(f"\n4. Testing symbol token lookup...")
        test_symbols = ["GOLDJAN26", "GOLDFEB26", "SILVERJAN26", "CRUDEOILJAN26"]
        for symbol in test_symbols:
            token = client._get_symbol_token(symbol, "MCX")
            if token:
                print(f"   [OK] {symbol}: Token = {token}")
            else:
                print(f"   [FAIL] {symbol}: Token not found")
        
        # Test historical data fetch
        print(f"\n5. Testing historical data fetch...")
        try:
            candles = client.get_historical_candles(
                symbol="GOLDJAN26",
                timeframe="1d",
                years=1.0
            )
            if candles:
                print(f"   [OK] Successfully fetched {len(candles)} candles for GOLDJAN26")
                print(f"   Date range: {candles[0]['timestamp']} to {candles[-1]['timestamp']}")
            else:
                print(f"   [WARN] No candles returned (symbol might not exist or expired)")
        except Exception as e:
            print(f"   [ERROR] Historical fetch failed: {e}")
        
    except AngelOneAuthError as e:
        print(f"   [ERROR] Failed to initialize: {e}")
        print("   Check your .env file has:")
        print("     ANGEL_ONE_API_KEY=...")
        print("     ANGEL_ONE_CLIENT_ID=...")
        print("     ANGEL_ONE_PASSWORD=...")
    except Exception as e:
        print(f"   [ERROR] Initialization error: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"   [ERROR] Failed to import AngelOneClient: {e}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)
print("1. Ensure .env file has AngelOne credentials:")
print("   ANGEL_ONE_API_KEY=your_api_key")
print("   ANGEL_ONE_CLIENT_ID=your_client_id")
print("   ANGEL_ONE_PASSWORD=your_trading_password")
print("   ANGEL_ONE_TOTP_SECRET=your_totp_secret (optional)")
print("\n2. Whitelist your IP in AngelOne SmartAPI dashboard")
print("\n3. Enable TOTP in your AngelOne account")
print("\n4. Download fresh scrip master if symbol tokens not found:")
print("   https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json")
print("\n5. Check if contracts exist (JAN26 might be expired, try FEB26)")
print("=" * 80)
