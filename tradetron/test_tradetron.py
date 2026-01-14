"""
Simple test script for Tradetron paper trading.

This script tests the TradetronClient connection and sends a test signal.
Use this to verify your setup before running the full strategy.

Usage:
    python tradetron/test_tradetron.py
    OR
    cd tradetron && python test_tradetron.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from tradetron.tradetron_client import TradetronClient


def test_tradetron():
    """Test Tradetron connection and send a test signal."""
    print("=" * 80)
    print("TRADETRON PAPER TRADING TEST")
    print("=" * 80)
    print()
    
    try:
        # Initialize client (loads from .env)
        print("[1/3] Initializing Tradetron client...")
        client = TradetronClient()
        print(f"    ✅ Connected to Tradetron (broker: {client.broker_name})")
        print()
        
        # Test account info
        print("[2/3] Fetching account information...")
        account = client.get_account()
        print(f"    ✅ Account info retrieved")
        print(f"    Note: Actual balance shown in Tradetron dashboard")
        print(f"    Placeholder data: {account}")
        print()
        
        # Test signal (small quantity for testing)
        print("[3/3] Sending test BUY signal...")
        print("    Symbol: GOLDDEC24")
        print("    Quantity: 1 lot")
        print("    Side: BUY")
        print("    Order Type: MARKET")
        print()
        
        result = client.submit_order(
            symbol="GOLDDEC24",
            qty=1,  # 1 lot
            side="buy",
            order_type="market"
        )
        
        print("    ✅ Signal sent successfully!")
        print()
        print("=" * 80)
        print("TEST RESULTS")
        print("=" * 80)
        print(f"Order ID: {result.get('id', 'N/A')}")
        print(f"Status: {result.get('status', 'N/A')}")
        print(f"Symbol: {result.get('symbol', 'N/A')}")
        print(f"Message: {result.get('message', 'N/A')}")
        print()
        print("💡 IMPORTANT: Check Tradetron dashboard to verify:")
        print("   1. Order was received and placed")
        print("   2. Order appears in 'Orders' section")
        print("   3. Position appears in 'Positions' section (if executed)")
        print("   4. All in PAPER TRADING account (not live)")
        print()
        print("=" * 80)
        
        return result
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR")
        print("=" * 80)
        print(f"Test failed: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check .env file has TRADETRON_API_TOKEN set")
        print("2. Verify token is correct (get from Tradetron dashboard)")
        print("3. Ensure strategy is deployed and active")
        print("4. Check strategy is in 'Paper Trading' mode")
        print("5. Verify symbol format matches strategy configuration")
        print()
        print("=" * 80)
        raise


if __name__ == "__main__":
    test_tradetron()
