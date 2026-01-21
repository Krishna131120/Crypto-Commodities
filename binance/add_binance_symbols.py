"""
Fetch all Binance trading pairs and add them to symbol_universe.py.

This script:
1. Fetches all active spot trading pairs from Binance API
2. Filters for USDT pairs (most common quote currency)
3. Adds them to symbol_universe.py with proper mapping
4. Handles duplicates (won't re-add existing symbols)
"""

import json
import requests
from pathlib import Path
from typing import List, Dict, Set, Any, Optional
import sys

# Add parent directory to path (binance folder is inside project root)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def fetch_binance_trading_pairs() -> List[Dict[str, Any]]:
    """Fetch all active spot trading pairs from Binance."""
    print("Fetching trading pairs from Binance API...")
    
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Filter for active spot trading pairs with USDT as quote asset
        pairs = []
        for symbol_info in data.get("symbols", []):
            status = symbol_info.get("status", "")
            
            # Binance uses permissionSets (nested array) instead of permissions
            permission_sets = symbol_info.get("permissionSets", [])
            has_spot = False
            if permission_sets and len(permission_sets) > 0:
                # Check first permission set (most common)
                has_spot = "SPOT" in permission_sets[0]
            else:
                # Fallback: check permissions field (older API format)
                permissions = symbol_info.get("permissions", [])
                has_spot = "SPOT" in permissions
            
            # Also check isSpotTradingAllowed (newer field)
            is_spot_allowed = symbol_info.get("isSpotTradingAllowed", False)
            has_spot = has_spot or is_spot_allowed
            
            # Only include TRADING status and SPOT permission
            if status == "TRADING" and has_spot:
                base_asset = symbol_info.get("baseAsset", "")
                quote_asset = symbol_info.get("quoteAsset", "")
                symbol = symbol_info.get("symbol", "")
                
                # Only include USDT pairs (can be extended to USDC, BTC, etc.)
                if quote_asset == "USDT":
                    pairs.append({
                        "symbol": symbol,  # e.g., "BTCUSDT"
                        "base_asset": base_asset,  # e.g., "BTC"
                        "quote_asset": quote_asset,  # e.g., "USDT"
                        "status": status,
                    })
        
        print(f"Found {len(pairs)} active USDT trading pairs")
        return sorted(pairs, key=lambda x: x["symbol"])
    
    except Exception as e:
        print(f"Error fetching Binance pairs: {e}")
        return []


def get_existing_symbols() -> Set[str]:
    """Get set of existing data_symbols from symbol_universe.py."""
    symbol_file = project_root / "trading" / "symbol_universe.py"
    
    if not symbol_file.exists():
        return set()
    
    existing = set()
    try:
        content = symbol_file.read_text(encoding="utf-8")
        # Look for data_symbol patterns: "BTC-USDT", "ETH-USDT", etc.
        import re
        # Match patterns like data_symbol="BTC-USDT"
        pattern = r'data_symbol=["\']([^"\']+-USDT)["\']'
        matches = re.findall(pattern, content)
        existing.update(matches)
    except Exception as e:
        print(f"Warning: Could not read existing symbols: {e}")
    
    return existing


def convert_to_logical_name(base_asset: str) -> str:
    """Convert asset symbol to logical name (lowercase, readable)."""
    # Handle special cases
    name_map = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "BNB": "binance_coin",
        "SOL": "solana",
        "XRP": "ripple",
        "ADA": "cardano",
        "DOGE": "dogecoin",
        "TRX": "tron",
        "DOT": "polkadot",
        "MATIC": "polygon",
        "LTC": "litecoin",
        "SHIB": "shiba_inu",
        "AVAX": "avalanche",
        "UNI": "uniswap",
        "ATOM": "cosmos",
        "ETC": "ethereum_classic",
        "LINK": "chainlink",
        "XLM": "stellar",
        "ALGO": "algorand",
        "VET": "vechain",
        "ICP": "internet_computer",
        "FIL": "filecoin",
        "AAVE": "aave",
        "EOS": "eos",
        "THETA": "theta",
        "AXS": "axie_infinity",
        "MANA": "decentraland",
        "SAND": "sandbox",
        "APE": "apecoin",
        "GMT": "stepn",
        "GALA": "gala",
    }
    
    if base_asset in name_map:
        return name_map[base_asset]
    
    # Default: lowercase
    return base_asset.lower()


def generate_asset_mapping_code(pair: Dict[str, Any], existing: Set[str]) -> Optional[str]:
    """Generate AssetMapping code for a trading pair."""
    symbol = pair["symbol"]  # e.g., "BTCUSDT"
    base_asset = pair["base_asset"]  # e.g., "BTC"
    
    # Convert to data_symbol format: BTCUSDT -> BTC-USDT
    data_symbol = f"{base_asset}-USDT"
    
    # Skip if already exists
    if data_symbol in existing:
        return None
    
    # Convert to trading_symbol: For Binance, it's the same as Binance symbol
    # But we use a format compatible with our system: BTCUSDT
    trading_symbol = symbol  # Keep Binance format
    
    logical_name = convert_to_logical_name(base_asset)
    
    # Generate AssetMapping entry
    code = f"""    AssetMapping(
        logical_name="{logical_name}",
        asset_type="crypto",
        data_symbol="{data_symbol}",
        trading_symbol="{trading_symbol}",
        timeframe="1d",
        horizon_profile="short",
        enabled=True,
    ),"""
    
    return code


def add_symbols_to_universe(new_symbols: List[str]) -> int:
    """Add new AssetMapping entries to symbol_universe.py."""
    # symbol_universe.py is in trading/ folder (parent of binance/)
    symbol_file = project_root / "trading" / "symbol_universe.py"
    
    if not symbol_file.exists():
        print(f"Error: {symbol_file} not found!")
        return 0
    
    # Read current file
    content = symbol_file.read_text(encoding="utf-8")
    
    # Find the end of UNIVERSE list (before the closing bracket)
    # Look for the last AssetMapping entry before ]
    lines = content.split("\n")
    
    # Find insertion point (before closing ] of UNIVERSE)
    # Look for the line with just "]" that closes the UNIVERSE list
    insert_idx = -1
    
    # First, find where UNIVERSE list starts
    universe_start = -1
    for i, line in enumerate(lines):
        if "UNIVERSE: List[AssetMapping] = [" in line or "UNIVERSE = [" in line:
            universe_start = i
            break
    
    if universe_start < 0:
        print("Error: Could not find UNIVERSE list start")
        return 0
    
    # Now find the closing ] - it should be after the last AssetMapping
    # Look for "] followed by blank line and function definitions
    for i in range(len(lines) - 1, universe_start, -1):
        stripped = lines[i].strip()
        if stripped == "]":
            # Check if next non-empty line starts a function
            for j in range(i + 1, min(i + 5, len(lines))):
                next_stripped = lines[j].strip()
                if next_stripped and (next_stripped.startswith("def ") or next_stripped.startswith("def ")):
                    insert_idx = i
                    break
            if insert_idx > 0:
                break
    
    if insert_idx < 0:
        print("Error: Could not find UNIVERSE list in symbol_universe.py")
        return 0
    
    # Insert new symbols before closing bracket
    new_content = "\n".join(lines[:insert_idx])
    new_content += "\n"
    new_content += "    # Binance trading pairs (auto-generated)\n"
    new_content += "\n".join(new_symbols)
    new_content += "\n"
    new_content += "\n".join(lines[insert_idx:])
    
    # Write back
    symbol_file.write_text(new_content, encoding="utf-8")
    return len(new_symbols)


def main():
    print("=" * 80)
    print("ADDING BINANCE TRADING PAIRS TO SYMBOL UNIVERSE")
    print("=" * 80)
    print()
    
    # Fetch all trading pairs
    pairs = fetch_binance_trading_pairs()
    if not pairs:
        print("No trading pairs found. Exiting.")
        return
    
    print(f"\nTotal pairs from Binance: {len(pairs)}")
    
    # Get existing symbols
    existing = get_existing_symbols()
    print(f"Existing symbols in universe: {len(existing)}")
    
    # Generate code for new symbols
    new_symbols = []
    skipped = 0
    
    for pair in pairs:
        code = generate_asset_mapping_code(pair, existing)
        if code:
            new_symbols.append(code)
            existing.add(f"{pair['base_asset']}-USDT")  # Mark as added to avoid duplicates
        else:
            skipped += 1
    
    print(f"\nNew symbols to add: {len(new_symbols)}")
    print(f"Already exist (skipped): {skipped}")
    
    if not new_symbols:
        print("\nNo new symbols to add. All pairs already in universe.")
        return
    
    # Show sample
    print("\nSample new symbols:")
    for code in new_symbols[:10]:
        print(f"  {code.split('logical_name=')[1].split(',')[0]}")
    if len(new_symbols) > 10:
        print(f"  ... and {len(new_symbols) - 10} more")
    
    # Add to file
    print("\nAdding symbols to symbol_universe.py...")
    added = add_symbols_to_universe(new_symbols)
    
    print(f"\n[OK] Successfully added {added} new symbols to symbol_universe.py")
    print(f"[INFO] Total symbols in universe: {len(existing) + added}")
    print("\nNext steps:")
    print("  1. Review symbol_universe.py to verify symbols")
    print("  2. Run data ingestion: python pipeline_runner.py --ingest --crypto <symbols>")
    print("  3. Train models: python train_models.py --crypto <symbols>")
    print("  4. Start trading: python live_trader.py --broker binance")
    print("=" * 80)


if __name__ == "__main__":
    main()
