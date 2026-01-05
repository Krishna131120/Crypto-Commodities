"""
Quick reference: How to run crypto trading with 15-second cycles
"""
print("=" * 80)
print("RUNNING CRYPTO TRADING WITH 15-SECOND CYCLES")
print("=" * 80)

print("\n[COMMAND]")
print("  python end_to_end_crypto.py \\")
print("    --crypto-symbols BTC-USDT \\")
print("    --profit-target 0.5 \\")
print("    --stop-loss-pct 3.0 \\")
print("    --interval 15 \\")
print("    --horizon intraday")

print("\n[WHAT THIS DOES]")
print("  - Runs trading cycles every 15 seconds")
print("  - Trades BTC with 0.5% profit target")
print("  - Uses 3% stop-loss")
print("  - Uses intraday horizon (short-term trading)")

print("\n[WARNING]")
print("  - 15 seconds is below the recommended 30 seconds")
print("  - May cause API rate limiting issues")
print("  - Use at your own risk")
print("  - System will show a warning but continue")

print("\n[ALTERNATIVE: SAFER OPTION]")
print("  Use --interval 30 for recommended minimum")
print("  This reduces risk of API rate limiting")

print("\n[TO RUN NOW]")
print("  Copy the command above and run it in your terminal")

