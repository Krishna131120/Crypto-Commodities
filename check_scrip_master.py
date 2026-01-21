"""Check scrip master for MCX contract formats."""
import json
from pathlib import Path

sm_path = Path("data/cache/angelone_scrip_master.json")
print("=" * 80)
print("CHECKING SCRIP MASTER FOR MCX CONTRACTS")
print("=" * 80)
print(f"\nScrip master exists: {sm_path.exists()}")

if not sm_path.exists():
    print("\n[ERROR] Scrip master not found!")
    print("Download from: https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json")
    sys.exit(1)

with open(sm_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Find MCX futures contracts
mcx_symbols = [
    s for s in data 
    if isinstance(s, dict) 
    and s.get("exch_seg", "").upper() == "MCX" 
    and "FUT" in s.get("symbol", "").upper()
]

print(f"\nFound {len(mcx_symbols)} MCX FUT contracts")

# Check Gold contracts
gold_symbols = [
    s for s in mcx_symbols 
    if "GOLD" in s.get("symbol", "").upper() or "GOLD" in s.get("name", "").upper()
]

print(f"\nGold-related symbols: {len(gold_symbols)}")
print("Sample Gold contracts (first 15):")
for s in gold_symbols[:15]:
    symbol = s.get("symbol", "N/A")
    name = s.get("name", "N/A")
    token = s.get("token", "N/A")
    print(f"  Symbol: {symbol:25s} | Name: {name:30s} | Token: {token}")

# Check Silver contracts
silver_symbols = [
    s for s in mcx_symbols 
    if "SILVER" in s.get("symbol", "").upper() or "SILVER" in s.get("name", "").upper()
]

print(f"\nSilver-related symbols: {len(silver_symbols)}")
print("Sample Silver contracts (first 15):")
for s in silver_symbols[:15]:
    symbol = s.get("symbol", "N/A")
    name = s.get("name", "N/A")
    token = s.get("token", "N/A")
    print(f"  Symbol: {symbol:25s} | Name: {name:30s} | Token: {token}")

# Check what format is used
print("\n" + "=" * 80)
print("CONTRACT FORMAT ANALYSIS")
print("=" * 80)

if gold_symbols:
    sample = gold_symbols[0]
    sample_symbol = sample.get("symbol", "")
    print(f"\nSample Gold contract format: '{sample_symbol}'")
    print("\nPatterns found:")
    formats = set()
    for s in gold_symbols[:20]:
        sym = s.get("symbol", "")
        if sym:
            formats.add(sym)
    print(f"  Unique formats: {len(formats)}")
    for fmt in sorted(list(formats))[:10]:
        print(f"    - {fmt}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print("1. Check if contract format matches (e.g., 'GOLD26JANFUT' vs 'GOLDJAN26')")
print("2. Contracts might use different month codes (e.g., '26JAN' instead of 'JAN26')")
print("3. Download fresh scrip master if contracts not found")
print("4. Check if contracts have expired (JAN26 might be past expiry)")
print("=" * 80)
