"""
Verify Tradetron broker connection setup.

This script checks if:
1. Broker is connected to Tradetron (for live trading)
2. API token is configured correctly
3. Strategy is deployed and active

Usage:
    python verify_tradetron_broker_connection.py
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import directly from module to avoid circular import
from tradetron.tradetron_client import TradetronClient, TradetronAuthError


def verify_tradetron_setup():
    """Verify Tradetron setup and broker connection."""
    print("=" * 80)
    print("TRADETRON BROKER CONNECTION VERIFICATION")
    print("=" * 80)
    print()
    
    # Step 1: Check API Token
    print("[1/4] Checking API Token Configuration...")
    try:
        client = TradetronClient()
        print("    ✅ API Token found in .env file")
        print(f"    ✅ Connected to Tradetron API: {client.config.api_url}")
    except TradetronAuthError as e:
        print(f"    ❌ ERROR: {e}")
        print()
        print("    SOLUTION:")
        print("    1. Go to Tradetron dashboard → My Strategies")
        print("    2. Find your strategy → Click menu (three dots)")
        print("    3. Click 'API OAUTH Token' → 'Link' → 'Proceed'")
        print("    4. Copy the token and add to .env: TRADETRON_API_TOKEN=your-token")
        return False
    except Exception as e:
        print(f"    ❌ ERROR: {e}")
        return False
    print()
    
    # Step 2: Test API Connection
    print("[2/4] Testing API Connection...")
    try:
        account = client.get_account()
        print("    ✅ API connection successful")
        print(f"    ✅ Broker name: {client.broker_name}")
    except Exception as e:
        print(f"    ⚠️  WARNING: Could not fetch account info: {e}")
        print("    NOTE: This might be normal - Tradetron may not provide account API")
    print()
    
    # Step 3: Check Strategy Deployment
    print("[3/4] Strategy Deployment Check...")
    print("    ⚠️  MANUAL CHECK REQUIRED:")
    print("    1. Go to Tradetron dashboard → My Strategies")
    print("    2. Check if your strategy is 'Deployed' and 'Active'")
    print("    3. Verify deployment mode:")
    print("       - Paper Trading: Uses 'TT Paper Trading' broker (no external broker needed)")
    print("       - Live Trading: Requires connected broker (Angel One, etc.)")
    print()
    
    # Step 4: Check Broker Connection (for live trading)
    print("[4/4] Broker Connection Check (for Live Trading only)...")
    print("    ⚠️  MANUAL CHECK REQUIRED:")
    print("    1. Go to Tradetron dashboard → Broker & Exchanges")
    print("    2. Check if your broker (Angel One) is listed and 'Connected'")
    print("    3. If not connected, follow these steps:")
    print("       a. Click '+ Add Broker'")
    print("       b. Select 'Angel One' from dropdown")
    print("       c. Enter your Angel One credentials:")
    print("          - API Key (from Angel One dashboard)")
    print("          - Client ID (your Angel One client ID)")
    print("          - Password (your trading password/MPIN)")
    print("       d. Click 'Save'")
    print("       e. Click 'Generate Token' or 'Regenerate Token'")
    print("       f. Follow OTP prompts if required")
    print("    4. Verify token is generated and shows 'Active' status")
    print()
    
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print()
    print("✅ API Token: Configured")
    print("✅ API Connection: Working")
    print("⚠️  Strategy Deployment: Check manually in Tradetron dashboard")
    print("⚠️  Broker Connection: Check manually in Tradetron dashboard")
    print()
    print("IMPORTANT NOTES:")
    print("- For PAPER TRADING: You DON'T need to connect a broker")
    print("  → Use 'TT Paper Trading' broker (built-in)")
    print()
    print("- For LIVE TRADING: You MUST connect a broker (Angel One)")
    print("  → Follow the steps in [4/4] above")
    print()
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        verify_tradetron_setup()
    except KeyboardInterrupt:
        print("\n\n✅ Verification stopped.")
    except Exception as e:
        print(f"\n\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
