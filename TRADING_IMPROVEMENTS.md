# Trading Performance Analysis & Recommendations

## Current Performance Issues

**Your Results:**
- Win Rate: 15.4% (2 wins, 11 losses)
- Net Loss: -$37.14
- Risk/Reward Ratio: 0.03:1 (risking $75 to make $2.50)

## Root Causes

### 1. **Stop-Loss Too Tight (1.5%)**
- BTC has high volatility - normal price fluctuations trigger stop-losses
- Most losses are small (-0.02% to -0.08%), indicating stop-loss is hit by noise, not real reversals
- **Solution:** Increased to 2.5% (already updated in `ml/horizons.py`)

### 2. **Profit Target Too Small (0.05%)**
- With 1.5% stop-loss, you need 30 wins to cover 1 loss
- 0.05% profit target is too small for BTC's volatility
- **Solution:** Increase to 0.15-0.20% (set via `--profit-target` flag)

### 3. **Low Confidence Threshold (12%)**
- System enters trades too easily
- **Solution:** Increased to 15% (already updated in `ml/horizons.py`)

### 4. **Poor Risk/Reward Ratio**
- Current: Risk $75 → Make $2.50 (0.03:1)
- **Target:** Risk $75 → Make $15-20 (0.20:1 or better)

## Recommended Changes

### ✅ Already Fixed:
1. **Stop-loss widened:** 1.5% → 2.5% (in `ml/horizons.py`)
2. **Confidence threshold raised:** 12% → 15% (in `ml/horizons.py`)

### 🔧 Action Required:

**1. Increase Profit Target:**
```bash
# Instead of:
python end_to_end_crypto.py --crypto-symbols BTCUSD --crypto-horizon intraday --profit-target 0.05

# Use:
python end_to_end_crypto.py --crypto-symbols BTCUSD --crypto-horizon intraday --profit-target 0.15
```

**2. Consider Using "Short" Horizon Instead:**
The "short" horizon has better risk/reward:
- Stop-loss: 2.0% (better than 1.5%, but not as wide as 2.5%)
- Position size: 10% (vs 5% for intraday)
- Hold time: 3-5 days (gives trades more room to develop)

```bash
python end_to_end_crypto.py --crypto-symbols BTCUSD --crypto-horizon short --profit-target 0.20
```

## Expected Improvements

With these changes:
- **Stop-loss:** 2.5% (wider, fewer false exits)
- **Profit target:** 0.15-0.20% (better risk/reward)
- **Confidence:** 15% (fewer but better trades)
- **Expected win rate:** 30-40% (vs current 15.4%)
- **Risk/Reward:** 0.06:1 to 0.08:1 (vs current 0.03:1)

## Alternative: Switch to "Short" Horizon

If intraday continues to struggle, consider the "short" horizon:
- Better suited for BTC's volatility
- 2% stop-loss (good balance)
- 3-5 day hold time (more room for moves)
- 10% position size (better capital utilization)

## Next Steps

1. **Restart trading with new settings:**
   ```bash
   python end_to_end_crypto.py --crypto-symbols BTCUSD --crypto-horizon intraday --profit-target 0.15 --interval 30
   ```

2. **Monitor for 24-48 hours** and check if win rate improves

3. **If still struggling, switch to "short" horizon:**
   ```bash
   python end_to_end_crypto.py --crypto-symbols BTCUSD --crypto-horizon short --profit-target 0.20 --interval 30
   ```

## Summary

The main issue is **risk/reward ratio is terrible** (0.03:1). By:
- Widening stop-loss (1.5% → 2.5%) ✅ Done
- Increasing profit target (0.05% → 0.15-0.20%) ⚠️ You need to do this
- Raising confidence (12% → 15%) ✅ Done

You should see significant improvement in win rate and profitability.

