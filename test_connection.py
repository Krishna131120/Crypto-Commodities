from trading.angelone_client import AngelOneClient

try:
    client = AngelOneClient()
    account = client.get_account()
    print("SUCCESS! Connected to Angel One")
    print(f"Equity: {account.get('equity', 0)}")
    print(f"Buying Power: {account.get('buying_power', 0)}")
    print("\nAll credentials are working! You can start trading.")
except Exception as e:
    print(f"ERROR: {e}")
