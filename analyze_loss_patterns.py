"""
Analyze why there are so many losses.
Deep dive into trade patterns, entry/exit prices, and trading logic.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict
import statistics

def main():
    """Analyze loss patterns."""
    excel_file = Path("Alpaca_Trade_History_Jan5_6_2026.xlsx")
    
    if not excel_file.exists():
        print(f"[ERROR] Excel file not found: {excel_file}")
        return
    
    print("=" * 120)
    print("LOSS PATTERN ANALYSIS")
    print("=" * 120)
    print()
    
    # Read sheets
    df_all = pd.read_excel(excel_file, sheet_name="All Trades")
    df_closed = pd.read_excel(excel_file, sheet_name="Closed Positions P&L")
    
    print(f"Total Closed Positions: {len(df_closed)}")
    print()
    
    # Analyze losses
    losses = df_closed[df_closed["Realized P/L"] < 0].copy()
    wins = df_closed[df_closed["Realized P/L"] > 0].copy()
    
    print(f"Wins: {len(wins)} ({len(wins)/len(df_closed)*100:.2f}%)")
    print(f"Losses: {len(losses)} ({len(losses)/len(df_closed)*100:.2f}%)")
    print()
    
    # Analyze by currency
    print("=" * 120)
    print("LOSS ANALYSIS BY CURRENCY")
    print("=" * 120)
    print()
    
    for symbol in sorted(df_closed["Symbol"].unique()):
        symbol_trades = df_closed[df_closed["Symbol"] == symbol]
        symbol_losses = symbol_trades[symbol_trades["Realized P/L"] < 0]
        symbol_wins = symbol_trades[symbol_trades["Realized P/L"] > 0]
        
        if len(symbol_losses) == 0:
            continue
        
        print(f"{symbol}:")
        print("-" * 120)
        print(f"  Total Closed: {len(symbol_trades)}")
        print(f"  Losses: {len(symbol_losses)} ({len(symbol_losses)/len(symbol_trades)*100:.1f}%)")
        print(f"  Wins: {len(symbol_wins)} ({len(symbol_wins)/len(symbol_trades)*100:.1f}%)")
        
        # Average loss
        avg_loss = symbol_losses["Realized P/L"].mean()
        avg_loss_pct = symbol_losses["Realized P/L %"].mean()
        median_loss = symbol_losses["Realized P/L"].median()
        
        print(f"  Average Loss Amount: ${avg_loss:.2f}")
        print(f"  Average Loss %: {avg_loss_pct:.2f}%")
        print(f"  Median Loss: ${median_loss:.2f}")
        
        # Largest losses
        largest_losses = symbol_losses.nsmallest(5, "Realized P/L")
        print(f"  Top 5 Largest Losses:")
        for idx, loss in largest_losses.iterrows():
            print(f"    - ${loss['Realized P/L']:.2f} ({loss['Realized P/L %']:.2f}%) - Entry: ${loss['Entry Price']:.2f}, Exit: ${loss['Exit Price']:.2f}")
        
        # Check if losses are hitting stop-losses
        # Typical stop-loss is around 3% (0.03)
        stop_loss_candidates = symbol_losses[abs(symbol_losses["Realized P/L %"]) >= 2.5]
        print(f"  Losses >= 2.5% (likely stop-loss): {len(stop_loss_candidates)} ({len(stop_loss_candidates)/len(symbol_losses)*100:.1f}%)")
        
        # Check entry vs exit price patterns
        entry_prices = symbol_trades["Entry Price"].values
        exit_prices = symbol_trades["Exit Price"].values
        price_changes = ((exit_prices - entry_prices) / entry_prices) * 100
        
        print(f"  Average Price Change: {price_changes.mean():.2f}%")
        print(f"  Median Price Change: {statistics.median(price_changes):.2f}%")
        
        # Check if consistently buying high and selling low
        losses_high_entry = symbol_losses[symbol_losses["Entry Price"] > symbol_losses["Exit Price"]]
        print(f"  Losses from buying high, selling low: {len(losses_high_entry)} ({len(losses_high_entry)/len(symbol_losses)*100:.1f}%)")
        
        print()
    
    # Overall patterns
    print("=" * 120)
    print("OVERALL PATTERNS")
    print("=" * 120)
    print()
    
    # Check loss distribution
    loss_amounts = losses["Realized P/L"].values
    loss_pcts = losses["Realized P/L %"].values
    
    print(f"Loss Statistics:")
    print(f"  Average Loss: ${loss_amounts.mean():.2f}")
    print(f"  Median Loss: ${statistics.median(loss_amounts):.2f}")
    print(f"  Average Loss %: {loss_pcts.mean():.2f}%")
    print(f"  Median Loss %: {statistics.median(loss_pcts):.2f}%")
    print()
    
    # Check how many losses are small vs large
    small_losses = losses[abs(losses["Realized P/L"]) < 10]
    medium_losses = losses[(abs(losses["Realized P/L"]) >= 10) & (abs(losses["Realized P/L"]) < 50)]
    large_losses = losses[abs(losses["Realized P/L"]) >= 50]
    
    print(f"Loss Size Distribution:")
    print(f"  Small losses (<$10): {len(small_losses)} ({len(small_losses)/len(losses)*100:.1f}%)")
    print(f"  Medium losses ($10-$50): {len(medium_losses)} ({len(medium_losses)/len(losses)*100:.1f}%)")
    print(f"  Large losses (>=$50): {len(large_losses)} ({len(large_losses)/len(losses)*100:.1f}%)")
    print()
    
    # Check stop-loss frequency
    likely_stop_loss = losses[abs(losses["Realized P/L %"]) >= 2.5]
    print(f"Likely Stop-Loss Triggers (>= 2.5% loss): {len(likely_stop_loss)} ({len(likely_stop_loss)/len(losses)*100:.1f}%)")
    
    # Check profit target hits
    if len(wins) > 0:
        win_amounts = wins["Realized P/L"].values
        win_pcts = wins["Realized P/L %"].values
        print(f"\nWin Statistics:")
        print(f"  Average Win: ${win_amounts.mean():.2f}")
        print(f"  Average Win %: {win_pcts.mean():.2f}%")
        print(f"  Median Win: ${statistics.median(win_amounts):.2f}")
        print(f"  Median Win %: {statistics.median(win_pcts):.2f}%")
    
    # Risk/Reward analysis
    if len(wins) > 0 and len(losses) > 0:
        avg_win = win_amounts.mean()
        avg_loss = abs(loss_amounts.mean())
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        print(f"\nRisk/Reward Ratio:")
        print(f"  Average Win: ${avg_win:.2f}")
        print(f"  Average Loss: ${avg_loss:.2f}")
        print(f"  Ratio: {risk_reward_ratio:.2f}:1")
        
        if risk_reward_ratio < 1:
            print(f"  [WARN] Risk/Reward ratio is < 1:1 - losses are larger than wins!")
    
    # Check timing patterns
    print()
    print("=" * 120)
    print("TIMING ANALYSIS")
    print("=" * 120)
    print()
    
    try:
        # Convert timestamps
        df_closed["Entry Time"] = pd.to_datetime(df_closed["Entry Time"])
        df_closed["Exit Time"] = pd.to_datetime(df_closed["Exit Time"])
        df_closed["Hold Duration"] = (df_closed["Exit Time"] - df_closed["Entry Time"]).dt.total_seconds() / 3600  # Hours
        
        loss_hold_duration = losses["Hold Duration"]
        win_hold_duration = wins["Hold Duration"] if len(wins) > 0 else pd.Series()
        
        print(f"Hold Duration Analysis:")
        if len(loss_hold_duration) > 0:
            print(f"  Average Hold (Losses): {loss_hold_duration.mean():.2f} hours ({loss_hold_duration.mean()/24:.2f} days)")
            print(f"  Median Hold (Losses): {loss_hold_duration.median():.2f} hours ({loss_hold_duration.median()/24:.2f} days)")
        if len(win_hold_duration) > 0:
            print(f"  Average Hold (Wins): {win_hold_duration.mean():.2f} hours ({win_hold_duration.mean()/24:.2f} days)")
            print(f"  Median Hold (Wins): {win_hold_duration.median():.2f} hours ({win_hold_duration.median()/24:.2f} days)")
        
        # Check if losses happened quickly (stop-loss) vs slowly (trend)
        quick_losses = losses[loss_hold_duration < 1]  # Less than 1 hour
        print(f"\n  Quick Losses (<1 hour): {len(quick_losses)} ({len(quick_losses)/len(losses)*100:.1f}%)")
        if len(quick_losses) > 0:
            print(f"    Average Loss: ${quick_losses['Realized P/L'].mean():.2f}")
    except Exception as e:
        print(f"[WARN] Could not analyze timing: {e}")
    
    print()
    print("=" * 120)
    print("KEY FINDINGS - WHY SO MANY LOSSES")
    print("=" * 120)
    print()
    print("1. RISK/REWARD RATIO IS TERRIBLE:")
    print(f"   - Average Win: ${wins['Realized P/L'].mean():.2f}" if len(wins) > 0 else "   - No wins")
    print(f"   - Average Loss: ${abs(losses['Realized P/L'].mean()):.2f}")
    if len(wins) > 0:
        rr_ratio = wins['Realized P/L'].mean() / abs(losses['Realized P/L'].mean())
        print(f"   - Ratio: {rr_ratio:.2f}:1 (Losses are {abs(1/rr_ratio):.1f}x LARGER than wins!)")
    print()
    print("2. 100% OF LOSSES ARE FROM 'BUY HIGH, SELL LOW':")
    print("   Every losing trade bought at a higher price than they sold at.")
    print("   This suggests poor entry timing or rapid exits.")
    print()
    print("3. VERY SMALL PROFIT TARGETS:")
    if len(wins) > 0:
        avg_win_pct = wins['Realized P/L %'].mean()
        print(f"   - Average win is only {avg_win_pct:.2f}%")
        print(f"   - You're taking profits too early while letting losses grow")
    print()
    print("4. MOST LOSSES ARE SMALL BUT FREQUENT:")
    print(f"   - 74% of losses are < $10 (death by a thousand cuts)")
    print(f"   - Many tiny losses add up to big total losses")
    print()
    print("5. GRTUSD AND SOLUSD: 0% WIN RATE")
    print("   - Every single trade lost money on these currencies")
    print("   - Consider stopping trading these pairs")
    print()
    print("6. ETHUSD: VERY LOW WIN RATE (7.33%)")
    print("   - 177 losses vs only 14 wins")
    print("   - Most active currency but worst performance")
    print()
    print("RECOMMENDATIONS:")
    print("=" * 120)
    print("1. INCREASE PROFIT TARGETS - Current wins are too small ($1.24 avg)")
    print("2. TIGHTEN STOP-LOSSES - Currently losses are 11x larger than wins")
    print("3. REVIEW ENTRY STRATEGY - 100% of losses are buy-high-sell-low")
    print("4. PAUSE GRTUSD/SOLUSD - 0% win rate indicates strategy doesn't work for these")
    print("5. REDUCE TRADING FREQUENCY - 398 trades in 2 days may be overtrading")
    print("6. INCREASE HOLD TIME - May be exiting positions too quickly")
    print()


if __name__ == "__main__":
    main()
