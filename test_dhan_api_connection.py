"""
Test DHAN API Connection

Tests actual API connectivity with provided credentials.
"""

from __future__ import annotations

import sys
import json
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from trading.dhan_client import DhanClient, DhanAuthError

# Try to import jwt, but handle if not available
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    try:
        import PyJWT as jwt
        JWT_AVAILABLE = True
    except ImportError:
        JWT_AVAILABLE = False
        print("⚠ JWT library not available - cannot decode token expiry")

# Your DHAN credentials
DHAN_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzY2MDM1ODEwLCJpYXQiOjE3NjU5NDk0MTAsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTA3OTU0NTAzIn0.PUuKWaMTT-5zKBgPMpFft-xh93v85sJ4h7-xPWiRJyHtLNVM9wdU7GrCwg4SZOmOY0jGuZlzJePhcux633KPjQ"
DHAN_CLIENT_ID = "1107954503"

print("=" * 80)
print("DHAN API CONNECTION TEST")
print("=" * 80)
print()

# Step 1: Decode and check token
print("[STEP 1] Checking Access Token")
print("-" * 80)
if JWT_AVAILABLE:
    try:
        # Decode JWT without verification (just to read the payload)
        decoded = jwt.decode(DHAN_ACCESS_TOKEN, options={"verify_signature": False})
        print(f"✓ Token decoded successfully")
        print(f"  Issuer: {decoded.get('iss', 'N/A')}")
        print(f"  Client ID: {decoded.get('dhanClientId', 'N/A')}")
        print(f"  Token Type: {decoded.get('tokenConsumerType', 'N/A')}")
        
        # Check expiry
        exp = decoded.get('exp')
        iat = decoded.get('iat')
        if exp:
            exp_time = datetime.fromtimestamp(exp)
            iat_time = datetime.fromtimestamp(iat)
            now = datetime.now()
            
            print(f"  Issued At: {iat_time}")
            print(f"  Expires At: {exp_time}")
            print(f"  Current Time: {now}")
            
            if now >= exp_time:
                print(f"  ⚠️  TOKEN IS EXPIRED! (Expired {(now - exp_time).days} days ago)")
            else:
                days_left = (exp_time - now).days
                hours_left = (exp_time - now).seconds // 3600
                print(f"  ✓ Token is valid (expires in {days_left} days, {hours_left} hours)")
        else:
            print(f"  ⚠️  No expiry found in token")
            
    except Exception as e:
        print(f"✗ Token decode failed: {e}")
else:
    print("⚠ JWT library not available - skipping token expiry check")
    print(f"  Token (first 50 chars): {DHAN_ACCESS_TOKEN[:50]}...")
    print(f"  Client ID: {DHAN_CLIENT_ID}")

# Step 2: Initialize DHAN Client
print("\n[STEP 2] Initializing DHAN Client")
print("-" * 80)
try:
    dhan_client = DhanClient(
        access_token=DHAN_ACCESS_TOKEN,
        client_id=DHAN_CLIENT_ID
    )
    print(f"✓ DHAN client created")
    print(f"  Broker: {dhan_client.broker_name}")
    print(f"  Base URL: {dhan_client.config.base_url}")
    print(f"  Client ID: {dhan_client.config.client_id}")
    print(f"  Token (first 50 chars): {DHAN_ACCESS_TOKEN[:50]}...")
except Exception as e:
    print(f"✗ DHAN client initialization failed: {e}")
    sys.exit(1)

# Step 3: Test Account API (GET /funds)
print("\n[STEP 3] Testing Account API (GET /funds)")
print("-" * 80)
try:
    account = dhan_client.get_account()
    print(f"✓ Account API connected successfully!")
    print(f"  Response keys: {list(account.keys())}")
    print(f"  Equity: ₹{account.get('equity', 0):,.2f}")
    print(f"  Buying Power: ₹{account.get('buying_power', 0):,.2f}")
    print(f"  Cash: ₹{account.get('cash', 0):,.2f}")
    print(f"  Portfolio Value: ₹{account.get('portfolio_value', 0):,.2f}")
    if account.get('margin_used'):
        print(f"  Margin Used: ₹{account.get('margin_used', 0):,.2f}")
    if account.get('margin_available'):
        print(f"  Margin Available: ₹{account.get('margin_available', 0):,.2f}")
except DhanAuthError as e:
    print(f"✗ Authentication failed: {e}")
    print("  → Token may be expired or invalid")
except Exception as e:
    print(f"✗ Account API failed: {e}")
    print(f"  Error type: {type(e).__name__}")
    if hasattr(e, 'response'):
        print(f"  Response status: {getattr(e, 'status_code', 'N/A')}")

# Step 4: Test Positions API (GET /positions)
print("\n[STEP 4] Testing Positions API (GET /positions)")
print("-" * 80)
try:
    positions = dhan_client.list_positions()
    print(f"✓ Positions API connected successfully!")
    print(f"  Number of positions: {len(positions)}")
    if positions:
        print(f"  Positions:")
        for pos in positions[:5]:  # Show first 5
            print(f"    - {pos.get('symbol', 'N/A')}: {pos.get('qty', 0)} @ ₹{pos.get('avg_entry_price', 0):,.2f}")
    else:
        print(f"  No open positions")
except Exception as e:
    print(f"✗ Positions API failed: {e}")
    print(f"  Error type: {type(e).__name__}")

# Step 5: Test Market Data API (POST /marketfeed/ltp)
print("\n[STEP 5] Testing Market Data API (POST /marketfeed/ltp)")
print("-" * 80)
try:
    # Test with a common MCX symbol (GOLD)
    test_symbol = "GOLDDEC25"  # MCX gold contract
    last_trade = dhan_client.get_last_trade(test_symbol)
    if last_trade and last_trade.get('price'):
        print(f"✓ Market Data API connected successfully!")
        print(f"  Symbol: {test_symbol}")
        print(f"  Last Traded Price: ₹{last_trade.get('price', 0):,.2f}")
    else:
        print(f"⚠ Market Data API responded but no price found")
        print(f"  Response: {last_trade}")
except Exception as e:
    print(f"✗ Market Data API failed: {e}")
    print(f"  Error type: {type(e).__name__}")
    print(f"  Note: This might fail if symbol format is wrong or market is closed")

# Step 6: Test Order API (should fail in dry run, but test endpoint)
print("\n[STEP 6] Testing Order API Endpoint (Dry Run)")
print("-" * 80)
print("Note: We won't actually place an order, just test if endpoint is reachable")
print("(Skipping actual order placement to avoid accidental trades)")

# Summary
print("\n" + "=" * 80)
print("CONNECTION TEST SUMMARY")
print("=" * 80)

# Check token validity
if JWT_AVAILABLE:
    try:
        decoded = jwt.decode(DHAN_ACCESS_TOKEN, options={"verify_signature": False})
        exp = decoded.get('exp')
        if exp:
            exp_time = datetime.fromtimestamp(exp)
            now = datetime.now()
            if now >= exp_time:
                print("❌ TOKEN STATUS: EXPIRED")
                print(f"   Token expired on: {exp_time}")
                print(f"   You need to generate a new access token from DHAN")
            else:
                print("✓ TOKEN STATUS: VALID")
                print(f"   Token expires on: {exp_time}")
    except:
        print("⚠ TOKEN STATUS: Could not verify")
else:
    print("⚠ TOKEN STATUS: Could not verify (JWT library not available)")

print("\nAPI ENDPOINT STATUS:")
print("  GET /funds (Account): Tested above")
print("  GET /positions (Positions): Tested above")
print("  POST /marketfeed/ltp (Market Data): Tested above")
print("  POST /orders (Place Order): Not tested (requires valid token)")
print("  DELETE /orders/{id} (Cancel Order): Not tested")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)
print("1. If token is expired, generate a new one from DHAN web portal")
print("2. Verify the API endpoints match DHAN API v2 documentation")
print("3. Check if your account has MCX trading enabled")
print("4. Ensure your IP is whitelisted (DHAN requires static IP for orders)")
print("=" * 80)
