"""Check DHAN token expiry"""

import base64
import json
from datetime import datetime

token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzY2MDM1ODEwLCJpYXQiOjE3NjU5NDk0MTAsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTA3OTU0NTAzIn0.PUuKWaMTT-5zKBgPMpFft-xh93v85sJ4h7-xPWiRJyHtLNVM9wdU7GrCwg4SZOmOY0jGuZlzJePhcux633KPjQ"

parts = token.split('.')
payload = parts[1]

# Add padding if needed
padding = 4 - len(payload) % 4
if padding != 4:
    payload += '=' * padding

decoded_bytes = base64.urlsafe_b64decode(payload)
decoded = json.loads(decoded_bytes.decode())

exp = decoded.get('exp')
iat = decoded.get('iat')
now = datetime.now()

print("=" * 80)
print("DHAN TOKEN EXPIRY CHECK")
print("=" * 80)
print(f"Issued At: {datetime.fromtimestamp(iat) if iat else 'N/A'}")
print(f"Expires At: {datetime.fromtimestamp(exp) if exp else 'N/A'}")
print(f"Current Time: {now}")
print(f"Client ID: {decoded.get('dhanClientId', 'N/A')}")

if exp:
    exp_time = datetime.fromtimestamp(exp)
    if now >= exp_time:
        hours_expired = (now - exp_time).total_seconds() / 3600
        print(f"\n[EXPIRED] TOKEN STATUS: EXPIRED")
        print(f"   Expired {hours_expired:.1f} hours ago")
        print(f"   You need to generate a new token from web.dhan.co")
        print(f"   Steps:")
        print(f"   1. Go to https://web.dhan.co")
        print(f"   2. Login to your account")
        print(f"   3. Navigate to Profile > DhanHQ Trading APIs")
        print(f"   4. Generate a new Access Token (valid for 24 hours)")
    else:
        days_left = (exp_time - now).days
        hours_left = (exp_time - now).seconds // 3600
        print(f"\n[VALID] TOKEN STATUS: VALID")
        print(f"   Expires in {days_left} days, {hours_left} hours")
else:
    print(f"\n[UNKNOWN] TOKEN STATUS: Could not determine expiry")

print("=" * 80)
