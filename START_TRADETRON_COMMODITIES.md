# Start Paper Trading Commodities on Tradetron

## ✅ You're Ready!

Your setup is complete:
- ✅ Tradetron tokens configured (AUTH_TOKEN and API_TOKEN)
- ✅ Script already uses TradetronClient
- ✅ Same trading logic as Angel One

## 🚀 Quick Start

### Option 1: Start Trading Immediately (Skip Training)

If you already have trained models:

```bash
python trade_all_commodities_auto.py \
  --profit-target-pct 5.0 \
  --stop-loss-pct 2.0 \
  --skip-training \
  --skip-ranking \
  --interval 60
```

### Option 2: Full Setup (Train + Rank + Trade)

This will:
1. Train models for all commodities
2. Rank them by performance
3. Trade the best one

```bash
python trade_all_commodities_auto.py \
  --profit-target-pct 5.0 \
  --stop-loss-pct 2.0 \
  --interval 60
```

### Option 3: Dry Run First (Test Without Real Orders)

Test the system first:

```bash
python trade_all_commodities_auto.py \
  --profit-target-pct 5.0 \
  --stop-loss-pct 2.0 \
  --dry-run \
  --skip-training \
  --skip-ranking
```

## 📋 Command Options

- `--profit-target-pct 5.0` - Exit when profit reaches 5% (REQUIRED)
- `--stop-loss-pct 2.0` - Exit when loss reaches 2% (default: 2.0%)
- `--interval 60` - Check every 60 seconds (default: 60)
- `--skip-training` - Use existing models (skip training phase)
- `--skip-ranking` - Use first tradable commodity (skip ranking)
- `--dry-run` - Test mode (no real orders)
- `--timeframe 1d` - Use daily timeframe (default: 1d)
- `--commodities-horizon short` - Trading horizon: intraday/short/long (default: short)

## ⚠️ Important Notes

1. **Paper Trading**: This uses Tradetron's paper trading broker (TT-PaperTrading)
2. **No Angel One Needed**: For paper trading, you don't need Angel One credentials
3. **Same Logic**: Uses the exact same trading logic as Angel One setup
4. **Position Monitoring**: Automatically monitors existing positions
5. **Profit Target**: REQUIRED - positions will exit when profit target is hit
6. **Stop-Loss**: STRICT - positions will exit when stop-loss is hit

## 📊 What Happens

1. **Connects to Tradetron** - Uses your AUTH_TOKEN from .env
2. **Checks Existing Positions** - Monitors any open positions first
3. **Gets Predictions** - Uses your ML models to predict price movements
4. **Places Orders** - Sends signals to Tradetron (paper trading)
5. **Monitors Positions** - Tracks profit targets and stop-losses
6. **Exits Automatically** - Closes positions when targets are hit

## 🔍 Monitor Your Trades

- **Tradetron Dashboard**: Check orders and positions in real-time
- **Log Files**: `logs/trading/commodities_trades.jsonl`
- **Positions File**: `data/positions/active_positions.json`

## 🛑 Stop Trading

Press `Ctrl+C` to stop. Positions will remain open (not liquidated automatically).

## 📝 Example Commands

### Start with 1% profit target (conservative):
```bash
python trade_all_commodities_auto.py --profit-target-pct 1.0 --stop-loss-pct 2.0 --skip-training --skip-ranking
```

### Start with 10% profit target (aggressive):
```bash
python trade_all_commodities_auto.py --profit-target-pct 10.0 --stop-loss-pct 2.0 --skip-training --skip-ranking
```

### Test first (dry run):
```bash
python trade_all_commodities_auto.py --profit-target-pct 5.0 --stop-loss-pct 2.0 --dry-run --skip-training --skip-ranking
```

## ✅ Ready to Start!

Your Tradetron connection is verified. You can start paper trading commodities now!
