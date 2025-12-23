"""
View Angel One Access Token

This script shows the current access token and its expiry information.
The access token is generated dynamically when you authenticate.
"""

from trading.angelone_client import AngelOneClient
from datetime import datetime

def view_token():
    """Display the current access token and expiry information."""
    try:
        print("=" * 80)
        print("ANGEL ONE ACCESS TOKEN INFORMATION")
        print("=" * 80)
        print()
        
        # Initialize client (this will authenticate and get token)
        print("Authenticating with Angel One...")
        client = AngelOneClient()
        print("✓ Authentication successful!")
        print()
        
        # Access the token (it's a private attribute, but we can access it)
        access_token = client._access_token
        refresh_token = client._refresh_token
        token_expiry = client._token_expiry
        
        if access_token:
            print("ACCESS TOKEN:")
            print(f"  {access_token}")
            print()
            print(f"Token Length: {len(access_token)} characters")
            print()
            
            if token_expiry:
                expiry_time = datetime.fromtimestamp(token_expiry)
                current_time = datetime.now()
                time_remaining = expiry_time - current_time
                
                print("TOKEN EXPIRY:")
                print(f"  Expires At: {expiry_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Time Remaining: {time_remaining}")
                print()
            
            if refresh_token:
                print("REFRESH TOKEN:")
                print(f"  {refresh_token[:50]}... (truncated)")
                print()
            
            print("=" * 80)
            print("NOTE: This token is automatically refreshed when it expires.")
            print("You don't need to manually update it.")
            print("=" * 80)
        else:
            print("❌ No access token found. Authentication may have failed.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Make sure your .env file has:")
        print("  - ANGEL_ONE_API_KEY")
        print("  - ANGEL_ONE_CLIENT_ID")
        print("  - ANGEL_ONE_PASSWORD")
        print("  - ANGEL_ONE_TOTP_SECRET")

if __name__ == "__main__":
    view_token()
