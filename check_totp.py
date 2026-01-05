import os
import re
import pyotp

print(f"{'='*60}")
print("DIAGNOSTIC: CHECKING .env FILE FOR TOTP ERRORS")
print(f"{'='*60}")

env_path = ".env"

if not os.path.exists(env_path):
    print("❌ ERROR: .env file NOT FOUND at", os.path.abspath(env_path))
    exit(1)

print(f"✅ Found .env file at: {os.path.abspath(env_path)}")

totp_secret = None
raw_line = None

# Read exactly how the main script does
with open(env_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("ANGEL_ONE_TOTP_SECRET"):
            raw_line = line
            if "=" in line:
                key, val = line.split("=", 1)
                val = val.strip()
                # Remove quotes
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                elif val.startswith("'") and val.endswith("'"):
                    val = val[1:-1]
                totp_secret = val
            break

if not totp_secret:
    print("❌ ERROR: ANGEL_ONE_TOTP_SECRET not found in .env")
    exit(1)

# Mask the secret for security in output, show first/last chars
mask_len = max(0, len(totp_secret) - 8)
masked = totp_secret[:4] + "*" * mask_len + totp_secret[-4:] if len(totp_secret) > 8 else "****"

print(f"\n📝 Loaded Secret: {masked} (Length: {len(totp_secret)})")

# --- CHECK 1: INVALID CHARACTERS ---
# Base32 allows: A-Z and 2-7
# INVALID: 0, 1, 8, 9, special chars, lowercase (strictly speaking, though pyotp might handle lower)

invalid_chars = []
for i, char in enumerate(totp_secret):
    if char == " ":
        invalid_chars.append(f"Position {i+1}: SPACE")
    elif not re.match(r"[A-Z2-7]", char):
        # Check specific common errors
        desc = f"'{char}'"
        if char == '0': desc += " (Zero - prohibited, use 'O' if needed?)"
        if char == '1': desc += " (One - prohibited, use 'I' or 'L'?)"
        if char == '8': desc += " (Eight - prohibited)"
        if char == '9': desc += " (Nine - prohibited)"
        if char.islower(): desc += " (Lowercase - usually must be UPPERCASE)"
        
        invalid_chars.append(f"Position {i+1}: {desc}")

if invalid_chars:
    print("\n❌ CRITICAL: Found INVALID characters in your secret!")
    print("   Base32 secrets can ONLY verify: Uppercase A-Z and digits 2-7")
    print("-" * 40)
    for err in invalid_chars:
        print(f"   ► {err}")
    print("-" * 40)
else:
    print("\n✅ Character check: All characters look valid (A-Z, 2-7)")

# --- CHECK 2: PYOTP VERIFICATION ---
print("\n🔄 Testing pyotp generation...")
try:
    totp = pyotp.TOTP(totp_secret)
    token = totp.now()
    print(f"✅ SUCCESS! Generated token: {token}")
    print("\nYour .env file is correct. If the main script fails, something else is wrong.")
except Exception as e:
    print(f"❌ FAILURE: pyotp could not generate token.")
    print(f"   Error: {e}")
