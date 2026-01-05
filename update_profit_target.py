"""
Simple script to update profit target for an existing position
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from trading.position_manager import PositionManager

def main():
    if len(sys.argv) < 3:
        print("Usage: python update_profit_target.py <SYMBOL> <NEW_PROFIT_TARGET_PCT>")
        print("Example: python update_profit_target.py BTCUSD 0.5")
        print("         (Changes BTC position profit target to 0.5%)")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    try:
        new_target = float(sys.argv[2])
    except ValueError:
        print(f"ERROR: Invalid profit target: {sys.argv[2]}")
        print("Profit target must be a number (e.g., 0.5 for 0.5%)")
        sys.exit(1)
    
    pm = PositionManager()
    position = pm.get_position(symbol)
    
    if not position:
        print(f"ERROR: No position found for {symbol}")
        print("Available positions:")
        all_positions = pm.get_all_positions()
        if all_positions:
            for pos in all_positions:
                print(f"  - {pos.symbol}: {pos.profit_target_pct:.2f}% target")
        else:
            print("  (No open positions)")
        sys.exit(1)
    
    if position.status != "open":
        print(f"ERROR: Position {symbol} is not open (status: {position.status})")
        sys.exit(1)
    
    # Show current status
    print("=" * 80)
    print(f"UPDATING PROFIT TARGET FOR {symbol}")
    print("=" * 80)
    print(f"\nCurrent Position:")
    print(f"  Entry Price: ${position.entry_price:,.2f}")
    print(f"  Current Target: {position.profit_target_pct:.2f}%")
    print(f"  Target Price: ${position.profit_target_price:,.2f}")
    print(f"  Stop-Loss: {position.stop_loss_pct:.2f}%")
    print(f"  Stop-Loss Price: ${position.stop_loss_price:,.2f}")
    
    # Update
    updated = pm.update_profit_target(symbol, new_target)
    
    if updated:
        print(f"\n[SUCCESS] Profit target updated!")
        print(f"  New Target: {updated.profit_target_pct:.2f}%")
        print(f"  New Target Price: ${updated.profit_target_price:,.2f}")
        print(f"  Stop-Loss unchanged: {updated.stop_loss_pct:.2f}%")
        print("\nThe bot will now monitor using the new profit target.")
    else:
        print("\n[ERROR] Failed to update profit target")
        sys.exit(1)

if __name__ == "__main__":
    main()

