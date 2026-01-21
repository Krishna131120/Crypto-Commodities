"""Analyze MCX contract format from scrip master."""
import json
import re
from pathlib import Path
from collections import defaultdict

sm_path = Path("data/cache/angelone_scrip_master.json")
with open(sm_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Find MCX futures contracts
mcx_symbols = [
    s for s in data 
    if isinstance(s, dict) 
    and s.get("exch_seg", "").upper() == "MCX" 
    and "FUT" in s.get("symbol", "").upper()
]

print("=" * 80)
print("MCX CONTRACT FORMAT ANALYSIS")
print("=" * 80)

# Analyze patterns
patterns = defaultdict(list)
for s in mcx_symbols:
    symbol = s.get("symbol", "")
    if symbol:
        # Extract base symbol (before day/month/year)
        # Pattern: BASESYMBOL + DAY(2digits) + MONTH(3letters) + YEAR(2digits) + FUT
        match = re.match(r"^([A-Z]+)(\d{2})([A-Z]{3})(\d{2})FUT$", symbol)
        if match:
            base, day, month, year = match.groups()
            patterns[base].append((day, month, year, symbol))

print("\nContract Format Pattern:")
print("  Format: {BASE_SYMBOL}{DAY:02d}{MONTH}{YEAR:02d}FUT")
print("  Example: GOLD05FEB26FUT = GOLD + 05 (day) + FEB (month) + 26 (year) + FUT")
print()

# Show patterns for common symbols
common_bases = ["GOLD", "GOLDM", "GOLDGUINEA", "GOLDPETAL", "SILVER", "SILVERM", "SILVERMIC", "CRUDEOIL", "CRUDEOILM"]
for base in common_bases:
    if base in patterns:
        print(f"\n{base} contracts:")
        for day, month, year, full_symbol in sorted(patterns[base], key=lambda x: (x[2], x[1], x[0]))[:10]:
            print(f"  {full_symbol} (Day: {day}, Month: {month}, Year: {year})")

# Find most common expiry days
print("\n" + "=" * 80)
print("MOST COMMON EXPIRY DAYS")
print("=" * 80)
day_counts = defaultdict(int)
for base, contracts in patterns.items():
    for day, month, year, _ in contracts:
        day_counts[day] += 1

for day, count in sorted(day_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  Day {day}: {count} contracts")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("MCX contracts use format: {BASE}{DAY:02d}{MONTH}{YEAR:02d}FUT")
print("Common expiry days: 05, 27, 30, 31 (varies by commodity)")
print("Need to update get_mcx_contract_symbol() to generate correct format")
print("=" * 80)
