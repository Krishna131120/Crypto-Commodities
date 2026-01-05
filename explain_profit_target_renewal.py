"""
Explain what profit target is used when system re-enters after exit
"""
print("=" * 80)
print("PROFIT TARGET AFTER RE-ENTRY: DOES IT CHANGE?")
print("=" * 80)

print("\n[QUESTION] When the system re-enters a new position after exiting:")
print("  - Does it use the SAME profit target percentage?")
print("  - Or does it change/reset to a different value?")

print("\n[ANSWER] THE SAME PROFIT TARGET PERCENTAGE IS USED")
print("  - The profit target percentage you specify (e.g., 0.5%) is FIXED")
print("  - It is used for EVERY new position")
print("  - It does NOT change or reset")
print("  - Each new position calculates its target price from the NEW entry price")
print("  - But uses the SAME percentage you specified")

print("\n" + "=" * 80)
print("HOW IT WORKS")
print("=" * 80)

print("\n[INITIAL SETUP]")
print("  You run: python end_to_end_crypto.py --profit-target 0.5")
print("  This sets: profit_target_pct = 0.5% (FIXED for entire session)")

print("\n[TRADE 1]")
print("  Entry Price: $88,000")
print("  Profit Target %: 0.5% (from your command)")
print("  Profit Target Price: $88,000 * 1.005 = $88,440")
print("  Exit: When price hits $88,440")

print("\n[TRADE 2 - AFTER RE-ENTRY]")
print("  Entry Price: $88,500 (new market price)")
print("  Profit Target %: 0.5% (SAME as before - does NOT change)")
print("  Profit Target Price: $88,500 * 1.005 = $88,942.50")
print("  Exit: When price hits $88,942.50")

print("\n[TRADE 3 - AFTER RE-ENTRY]")
print("  Entry Price: $89,000 (new market price)")
print("  Profit Target %: 0.5% (SAME as before - does NOT change)")
print("  Profit Target Price: $89,000 * 1.005 = $89,445")
print("  Exit: When price hits $89,445")

print("\n" + "=" * 80)
print("CODE FLOW")
print("=" * 80)

print("\n[STEP 1] Command Line")
print("  end_to_end_crypto.py: --profit-target 0.5")
print("  -> args.profit_target = 0.5")

print("\n[STEP 2] Every Trading Cycle")
print("  end_to_end_crypto.py line 405:")
print("    profit_target_pct=args.profit_target  # Always 0.5")
print("  -> Passed to run_trading_cycle()")

print("\n[STEP 3] Execution Engine")
print("  execution_engine.py line 158:")
print("    effective_profit_target = profit_target_pct  # 0.5")
print("  -> Used for EVERY new position")

print("\n[STEP 4] New Position Entry")
print("  execution_engine.py line 1323:")
print("    profit_target_price = current_price * (1.0 + 0.5 / 100.0)")
print("  -> Calculates target price using SAME 0.5%")

print("\n" + "=" * 80)
print("KEY POINTS")
print("=" * 80)

print("\n[FIXED PERCENTAGE]")
print("  - Profit target PERCENTAGE is set once at startup")
print("  - It remains the SAME for all trades in the session")
print("  - You cannot change it without restarting the script")

print("\n[DYNAMIC TARGET PRICE]")
print("  - Each new position has a DIFFERENT target PRICE")
print("  - Target price = New Entry Price * (1 + profit_target_pct / 100)")
print("  - Example: If you enter at $90,000 with 0.5% target:")
print("    Target Price = $90,000 * 1.005 = $90,450")
print("  - Example: If you enter at $91,000 with 0.5% target:")
print("    Target Price = $91,000 * 1.005 = $91,455")

print("\n[WHY THIS DESIGN?]")
print("  - Consistent risk/reward ratio for every trade")
print("  - Each trade has the same profit percentage goal")
print("  - Prevents accidentally using different targets")
print("  - Simple and predictable behavior")

print("\n" + "=" * 80)
print("EXAMPLE: 2-DAY TRADING WITH 0.5% TARGET")
print("=" * 80)

print("\n[DAY 1 - Morning]")
print("  Trade 1:")
print("    Entry: $88,000")
print("    Target: $88,440 (0.5% = $440 profit)")
print("    Exit: $88,440 -> Profit locked in")

print("\n  Trade 2 (re-entry):")
print("    Entry: $88,450 (new price)")
print("    Target: $88,892.25 (0.5% = $442.25 profit)")
print("    Exit: $88,892.25 -> Profit locked in")

print("\n[DAY 1 - Afternoon]")
print("  Trade 3 (re-entry):")
print("    Entry: $89,000 (new price)")
print("    Target: $89,445 (0.5% = $445 profit)")
print("    Exit: $89,445 -> Profit locked in")

print("\n[DAY 2]")
print("  Trade 4 (re-entry):")
print("    Entry: $90,000 (new price)")
print("    Target: $90,450 (0.5% = $450 profit)")
print("    Exit: $90,450 -> Profit locked in")

print("\n[NOTICE]")
print("  - Each trade uses 0.5% (SAME percentage)")
print("  - But target PRICE changes based on entry price")
print("  - Higher entry = Higher target price, but same % profit")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\n[PROFIT TARGET AFTER RE-ENTRY]")
print("  Percentage: SAME as you specified (e.g., 0.5%)")
print("  Target Price: Calculated from NEW entry price")
print("  Formula: New Entry Price * (1 + profit_target_pct / 100)")

print("\n[FOR YOUR 2-DAY BTC TRADING]")
print("  If you set: --profit-target 0.5")
print("  Then EVERY trade will:")
print("    - Use 0.5% profit target")
print("    - Calculate target price from current entry price")
print("    - Exit when target is hit")
print("    - Re-enter with NEW 0.5% target if model predicts entry")

print("\n[IMPORTANT]")
print("  - Profit target percentage is FIXED for entire session")
print("  - To change it, you must restart the script with new value")
print("  - Each new position gets its own target price (based on entry)")
print("  - But all use the SAME percentage you specified")

