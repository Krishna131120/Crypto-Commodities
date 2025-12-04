#!/usr/bin/env python3
"""
Display available horizon profiles and their trading behavior.

Run this script to see:
- What horizons are available (intraday, short, long)
- How each horizon trades (position sizes, stop-losses, thresholds)
- How selling works for each horizon
"""

from ml.horizons import print_horizon_summary

if __name__ == "__main__":
    print_horizon_summary()
    print()
    print("=" * 80)
    print("QUICK REFERENCE: How Selling Works")
    print("=" * 80)
    print()
    print("For EACH horizon independently:")
    print()
    print("  📈 LONG signal:")
    print("     → BUY BTCUSD (up to horizon's position size limit)")
    print("     → Position opens in your Alpaca account")
    print()
    print("  📊 FLAT/HOLD signal:")
    print("     → SELL entire position (if you're long)")
    print("     → BUY to close (if you're short)")
    print("     → You go back to flat (no position)")
    print()
    print("  📉 SHORT signal:")
    print("     → SELL BTCUSD to open short (if shorting enabled)")
    print("     → If you're already long: SELL to close long, then SELL to open short")
    print("     → When SHORT flips to FLAT/LONG: BUY to close the short")
    print()
    print("=" * 80)
    print()
    print("Note: Each horizon trades independently with its own:")
    print("  - Position size limits (intraday: 5%, short: 10%, long: 18%)")
    print("  - Stop-loss distances (intraday: 1.5%, short: 2%, long: 2.5%)")
    print("  - Confidence thresholds (intraday: 12%, short: 10%, long: 8%)")
    print()
    print("This means you can have multiple positions for the same symbol")
    print("if you're trading different horizons simultaneously.")
    print()

