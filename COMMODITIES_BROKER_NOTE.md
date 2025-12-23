# Commodities Trading - Broker Selection

## Current Situation

### Alpaca Limitations
- ❌ **Alpaca does NOT support commodity futures** (GC=F, CL=F, etc.)
- ⚠️ **Current workaround**: Uses ETF proxies:
  - GC=F (Gold futures) → GLD (SPDR Gold Shares ETF)
  - CL=F (Crude oil futures) → USO (United States Oil Fund ETF)
  - SI=F (Silver futures) → SLV (iShares Silver Trust ETF)
  - PL=F (Platinum futures) → PPLT (Aberdeen Physical Platinum Shares ETF)

### Problems with ETF Proxies
1. **Price tracking errors**: ETFs don't perfectly track futures prices
2. **Different trading hours**: ETFs trade during stock market hours, futures trade 24/5
3. **Different margin requirements**: Futures have different leverage than ETFs
4. **Basis risk**: ETF prices can diverge from underlying futures
5. **Liquidity differences**: Futures markets may have better liquidity

## Recommended Solution

### Use Angel One for Commodities
✅ **Angel One supports actual commodity futures**
- Direct futures trading (GC=F, CL=F, etc.)
- Proper margin requirements
- Futures trading hours
- Better price accuracy

### Use Alpaca for Crypto
✅ **Alpaca supports crypto** (BTC, ETH, etc.)
- Direct crypto trading
- Good liquidity
- Well-established API

## Implementation

### Validation Script
The validation script now **auto-selects the correct broker**:
- Commodities → DHAN (default)
- Crypto → Alpaca (default)

You can override with `--broker` flag if needed.

### Code Changes
- `verify_live_trading_readiness.py` - Auto-selects DHAN for commodities
- `ml/live_trading_readiness.py` - Auto-selects broker based on asset type
- Broker abstraction allows easy switching

## Migration Path

### Current (ETF Proxies on Alpaca)
```python
# Works but not ideal
engine = ExecutionEngine(client=AlpacaClient())
# GC=F trades as GLD ETF
```

### Future (Futures on Angel One)
```python
# Ideal for commodities
from trading.angelone_client import AngelOneClient
engine = ExecutionEngine(client=AngelOneClient())
# GC=F trades as actual gold futures
```

## Recommendation

1. **For commodities**: Complete DHAN integration and use DHAN broker
2. **For crypto**: Continue using Alpaca (works well)
3. **Update validation**: Already done - auto-selects correct broker

## Notes

- The current ETF proxy approach works for **testing and paper trading**
- For **live commodities trading**, DHAN is the correct choice
- The broker abstraction layer makes switching easy
- No model logic changes needed - only broker selection
