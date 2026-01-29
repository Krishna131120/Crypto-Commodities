import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from binance.binance_client import BinanceClient

def test_connection():
    try:
        print("Initializing Binance Client...")
        client = BinanceClient()
        print(f"Base URL: {client.config.base_url}")
        print(f"Testnet: {client.config.testnet}")
        
        print("Fetching account info...")
        account = client.get_account()
        print(f"Connection Successful!")
        print(f"Equity: {account['equity']}")
        print(f"Cash: {account['cash']}")
        
        print("\nListing positions...")
        positions = client.list_positions()
        if not positions:
            print("No open positions found.")
        else:
            for pos in positions:
                print(f"- {pos['symbol']}: {pos['qty']} (Value: {pos['market_value']})")
                
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_connection()
