"""
Script to check active positions in both the local JSON file and Alpaca API.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from trading.alpaca_client import AlpacaClient


def load_json_positions(file_path: Path) -> Dict[str, Any]:
    """Load positions from JSON file."""
    if not file_path.exists():
        return {}
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_active_json_positions(positions: Dict[str, Any]) -> Dict[str, Any]:
    """Filter positions to only include those with status 'open'."""
    active = {}
    for symbol, pos_data in positions.items():
        if pos_data.get("status") == "open":
            active[symbol] = pos_data
    return active


def check_alpaca_positions() -> List[Dict[str, Any]]:
    """Check active positions in Alpaca."""
    try:
        client = AlpacaClient()
        alpaca_positions = client.list_positions()
        
        if not alpaca_positions:
            return []
        
        # Filter out positions with zero quantity
        active = []
        for pos in alpaca_positions:
            qty = float(pos.get("qty", 0) or 0)
            if qty != 0:
                active.append(pos)
        
        return active
    except Exception as e:
        print(f"Error checking Alpaca positions: {e}")
        return []


def main():
    """Main function to check and compare positions."""
    print("=" * 80)
    print("ACTIVE POSITIONS CHECK")
    print("=" * 80)
    print()
    
    # Check local JSON file
    json_path = Path("data/positions/active_positions.json")
    print(f"[1] Checking local JSON file: {json_path}")
    print("-" * 80)
    
    all_positions = load_json_positions(json_path)
    active_json_positions = get_active_json_positions(all_positions)
    
    if active_json_positions:
        print(f"[OK] Found {len(active_json_positions)} active position(s) in JSON file:")
        for symbol, pos_data in active_json_positions.items():
            print(f"  - {symbol}:")
            print(f"    - Side: {pos_data.get('side', 'N/A')}")
            print(f"    - Quantity: {pos_data.get('quantity', 'N/A')}")
            print(f"    - Entry Price: ${pos_data.get('entry_price', 'N/A')}")
            print(f"    - Entry Time: {pos_data.get('entry_time', 'N/A')}")
            print(f"    - Cost Basis: ${pos_data.get('total_cost_basis', 'N/A')}")
            if pos_data.get('profit_target_price'):
                print(f"    - Profit Target: ${pos_data.get('profit_target_price', 'N/A')}")
            if pos_data.get('stop_loss_price'):
                print(f"    - Stop Loss: ${pos_data.get('stop_loss_price', 'N/A')}")
            print()
    else:
        print("  [X] No active positions found in JSON file")
        print()
    
    # Check Alpaca
    print("[2] Checking Alpaca API for active positions")
    print("-" * 80)
    
    alpaca_positions = check_alpaca_positions()
    
    if alpaca_positions:
        print(f"[OK] Found {len(alpaca_positions)} active position(s) in Alpaca:")
        for pos in alpaca_positions:
            symbol = pos.get("symbol", "N/A")
            qty = pos.get("qty", 0)
            side = "long" if float(qty) > 0 else "short"
            avg_entry_price = pos.get("avg_entry_price", "N/A")
            current_price = pos.get("current_price", "N/A")
            market_value = pos.get("market_value", "N/A")
            unrealized_pl = pos.get("unrealized_pl", "N/A")
            unrealized_plpc = pos.get("unrealized_plpc", "N/A")
            
            print(f"  - {symbol}:")
            print(f"    - Side: {side}")
            print(f"    - Quantity: {qty}")
            print(f"    - Avg Entry Price: ${avg_entry_price}")
            print(f"    - Current Price: ${current_price}")
            print(f"    - Market Value: ${market_value}")
            print(f"    - Unrealized P/L: ${unrealized_pl} ({unrealized_plpc}%)")
            print()
    else:
        print("  [X] No active positions found in Alpaca")
        print()
    
    # Compare
    print("[3] Comparison")
    print("-" * 80)
    
    json_symbols = set(active_json_positions.keys())
    alpaca_symbols = {pos.get("symbol", "").upper() for pos in alpaca_positions}
    
    if json_symbols == alpaca_symbols:
        print("[OK] JSON and Alpaca positions match!")
    else:
        only_in_json = json_symbols - alpaca_symbols
        only_in_alpaca = alpaca_symbols - json_symbols
        
        if only_in_json:
            print(f"[WARN] Positions only in JSON (not in Alpaca): {', '.join(only_in_json)}")
            print("  -> These positions may have been closed in Alpaca but not updated in JSON")
        
        if only_in_alpaca:
            print(f"[WARN] Positions only in Alpaca (not in JSON): {', '.join(only_in_alpaca)}")
            print("  -> These positions exist in Alpaca but are not tracked in JSON")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
