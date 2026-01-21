#!/usr/bin/env python3
"""
Check if Tradetron tokens are configured in .env file.
"""

from pathlib import Path

def check_tradetron_tokens():
    """Check if Tradetron tokens are set in .env file."""
    print("=" * 60)
    print("TRADETRON TOKEN CONFIGURATION CHECK")
    print("=" * 60)
    print()
    
    # Check .env file directly
    env_file = Path(".env")
    tokens_found = {}
    
    if env_file.exists():
        print("[1] Reading .env file...")
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    
                    if key == "TRADETRON_AUTH_TOKEN":
                        tokens_found["TRADETRON_AUTH_TOKEN"] = val
                    elif key == "TRADETRON_API_TOKEN":
                        tokens_found["TRADETRON_API_TOKEN"] = val
        except Exception as e:
            print(f"    ⚠️  Error reading .env file: {e}")
    else:
        print("[1] [FAIL] .env file not found")
        print()
        print("Action Required:")
        print("  Create a .env file in the project root with:")
        print("  TRADETRON_AUTH_TOKEN=your-token-here")
        return
    
    print()
    print("[2] Token Status:")
    print()
    
    # Check AUTH_TOKEN (REQUIRED)
    auth_token = tokens_found.get("TRADETRON_AUTH_TOKEN")
    if auth_token:
        print("  [OK] TRADETRON_AUTH_TOKEN (REQUIRED): SET")
        print(f"     Length: {len(auth_token)} characters")
        if len(auth_token) > 30:
            print(f"     Preview: {auth_token[:30]}...")
        else:
            print(f"     Value: {auth_token}")
    else:
        print("  [FAIL] TRADETRON_AUTH_TOKEN (REQUIRED): NOT SET")
    
    print()
    
    # Check API_TOKEN (OPTIONAL)
    api_token = tokens_found.get("TRADETRON_API_TOKEN")
    if api_token:
        print("  [OK] TRADETRON_API_TOKEN (OPTIONAL): SET")
        print(f"     Length: {len(api_token)} characters")
        if len(api_token) > 30:
            print(f"     Preview: {api_token[:30]}...")
        else:
            print(f"     Value: {api_token}")
    else:
        print("  [WARN] TRADETRON_API_TOKEN (OPTIONAL): NOT SET")
        print("     (This is optional - only AUTH_TOKEN is required)")
    
    print()
    print("=" * 60)
    
    if auth_token:
        print("[OK] Status: READY - Required token (AUTH_TOKEN) is configured")
        print()
        print("You can now use TradetronClient in your code!")
    else:
        print("[FAIL] Status: MISSING - Required token (AUTH_TOKEN) is not configured")
        print()
        print("Action Required:")
        print("  1. Go to Tradetron dashboard → My Strategies")
        print("  2. Select your strategy → API Settings")
        print("  3. Copy the 'API OAuth Token' (this is your AUTH_TOKEN)")
        print("  4. Add to .env file:")
        print("     TRADETRON_AUTH_TOKEN=your-token-here")
        print()
        print("Optional (for reference only):")
        print("  TRADETRON_API_TOKEN=your-api-token-here")
        print()
        print("Note: AUTH_TOKEN is required and is used in API requests.")
        print("      API_TOKEN is optional and used for reference only.")
    
    print("=" * 60)

if __name__ == "__main__":
    check_tradetron_tokens()
