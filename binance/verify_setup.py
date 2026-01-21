"""Verify Binance folder structure is set up correctly."""

import sys
from pathlib import Path

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

binance_root = Path(__file__).parent

print("=" * 80)
print("BINANCE FOLDER STRUCTURE VERIFICATION")
print("=" * 80)
print()

print(f"Binance root: {binance_root}")
print(f"  Exists: {binance_root.exists()}")
print()

# Check directories
dirs_to_check = [
    ("logs", binance_root / "logs"),
    ("data/positions", binance_root / "data" / "positions"),
]

print("Directories:")
for name, path in dirs_to_check:
    exists = path.exists()
    status = "[OK]" if exists else "[MISSING]"
    print(f"  {status} {name}/: {exists}")

print()

# Check files
files_to_check = [
    "__init__.py",
    "binance_client.py",
    "live_trader_binance.py",
    "README.md",
    "SETUP_COMPLETE.md",
]

print("Files:")
for filename in files_to_check:
    path = binance_root / filename
    exists = path.exists()
    status = "[OK]" if exists else "[MISSING]"
    print(f"  {status} {filename}: {exists}")

print()

# Test imports
print("Import Tests:")
try:
    from binance.binance_client import BinanceClient
    print("  [OK] BinanceClient imports successfully")
except Exception as e:
    print(f"  [ERROR] BinanceClient import failed: {e}")

try:
    from binance.live_trader_binance import discover_tradable_symbols
    print("  [OK] live_trader_binance imports successfully")
except Exception as e:
    print(f"  [ERROR] live_trader_binance import failed: {e}")

print()
print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
