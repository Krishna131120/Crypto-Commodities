# Features and Training Data Documentation

## Overview
This document describes the features and data used for model training in the crypto/commodities prediction system.

---

## 1. Input Data (Raw Candles)

### Data Sources
- **Crypto**: Binance, KuCoin, OKX (with fallback mechanisms)
- **Commodities**: Yahoo Finance (yfinance)
- **Format**: OHLCV (Open, High, Low, Close, Volume) candles

### Data Structure
Each candle contains:
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume
- `timestamp`: Timestamp (UTC)

### Data Split
- **Training**: 75% of historical data
- **Validation**: 12.5% of historical data
- **Test**: 12.5% of historical data
- **Gap Days**: 1 day gap between splits (to prevent data leakage)

---

## 2. Target Variable

### Primary Target: `target_return`
- **Definition**: Future price return over a configurable horizon (default: 30 days for "long" profile)
- **Formula**: `(future_close_price / current_close_price) - 1.0`
- **Type**: Continuous regression target (percentage return)

### Target Generation Process
1. **Forward Shift**: Close price shifted forward by `horizon` bars (e.g., 30 days)
2. **Smoothing**: Optional rolling mean smoothing (default: 10 bars) to reduce noise
3. **Directional Labels**: Optional binary/ternary classification (long/short/hold)
4. **Quantile Bins**: Optional quantile-based ordinal classification

### Horizon Profiles & Defaults
- **Intraday**: 1-2 bar horizon, smoothing window 2, tight thresholds for scalping.
- **Short-Term**: 4-bar horizon, smoothing window 4, ±0.12% directional threshold (default for ≤1d timeframes).
- **Long**: 30-bar horizon, smoothing window 10, ±0.30% directional threshold (used for weekly/monthly data).

`pipeline_runner.py` and `train_models.py` now auto-map each timeframe to the closest sensible profile if you don't pass `--crypto-horizon/--commodities-horizon`. For example, `1d` and `4h` runs default to the short profile to avoid accidentally training a 30-bar target for a 4-bar decision horizon.

### Example
- Current price: $100
- Price in 30 days: $110
- `target_return` = (110/100) - 1 = 0.10 (10% return)

---

## 3. Feature Categories

The system computes **200+ features** organized into 6 main categories:

### 3.1 Moving Averages (15+ features)
- **Simple Moving Averages (SMA)**: 5, 10, 20, 50, 100, 200 periods
- **Exponential Moving Averages (EMA)**: 9, 12, 26, 50, 100, 200 periods
- **Weighted Moving Average (WMA)**: 14 periods
- **Hull Moving Average**: 21 periods
- **Volume-Weighted Moving Average (VWMA)**: 20 periods
- **Volume-Weighted Average Price (VWAP)**: Session-based
- **Session VWAP**: Session-specific VWAP

### 3.2 Momentum Indicators (20+ features)
- **MACD**: MACD line, signal line, histogram
- **RSI (Relative Strength Index)**: 7, 14, 21 periods
- **Rate of Change (ROC)**: 1, 7, 10, 14 periods
- **Momentum**: 1, 7, 14 periods
- **TRIX**: Triple Exponential Average
- **TSI**: True Strength Index
- **Stochastic Oscillator**: %K and %D
- **Williams %R**: Momentum oscillator
- **CCI (Commodity Channel Index)**: 20 periods
- **ADX (Average Directional Index)**: 14 periods
- **DI+ and DI-**: Directional indicators
- **RSI Band Cross**: Binary flag for RSI > 70 or < 30
- **ROC Signal**: 5-period rate of change

### 3.3 Volatility Indicators (20+ features)
- **ATR (Average True Range)**: 7, 14 periods
- **Bollinger Bands**: Upper, lower bands (20 periods, 2 std dev)
- **Bollinger Bandwidth**: Band width relative to middle
- **Bollinger %b**: Price position within bands
- **Keltner Channels**: Upper and lower channels
- **Donchian Channels**: High and low channels (20 periods)
- **Rolling Volatility**: 7, 14, 30 periods
- **Realized Volatility**: 30-period realized vol
- **Historical Volatility**: Annualized volatility
- **Rolling Skew**: 30-period return skewness
- **Rolling Kurtosis**: 30-period return kurtosis
- **Volatility Skew**: Short-term vs long-term vol
- **Volatility Clustering**: Volatility persistence measure
- **Z-Scores**: 7 and 30-period standardized returns
- **Normalized Returns**: Standardized return series
- **Log Returns**: Natural log of price ratios
- **Period Returns**: 1, 3, 7-period returns
- **Cumulative Return**: 30-period cumulative return
- **Rolling Sharpe Ratio**: 30 and 90-period risk-adjusted returns
- **Max Drawdown**: 30-period maximum drawdown
- **Drawdown Duration**: Length of drawdown periods

### 3.4 Volume Indicators (10+ features)
- **OBV (On-Balance Volume)**: Cumulative volume indicator
- **CMF (Chaikin Money Flow)**: 20-period money flow
- **MFI (Money Flow Index)**: 14-period volume-weighted RSI
- **Ease of Movement**: Price-volume relationship
- **Force Index**: Price-volume momentum
- **Accumulation/Distribution Line**: Volume-weighted price movement
- **Chaikin Oscillator**: A/D line momentum
- **Volume Change**: 1 and 7-period volume changes
- **Volume Ratio**: Current volume / 20-period average
- **Volume Surge Flag**: Binary flag for volume > 1.5x average

### 3.5 Price Structure Indicators (30+ features)
- **Price Lags**: Close price lagged 1, 2, 3 periods
- **Close Lags**: Close price lagged 1-10 periods
- **OHLC Lags**: Open, high, low lagged 1 period
- **Typical Price**: (High + Low + Close) / 3
- **Median Price**: (High + Low) / 2
- **Price Range**: High - Low
- **Intraday Range**: Close - Open
- **Gap Open**: Open - Previous Close
- **Gap Percentage**: Gap relative to previous close
- **Donchian Breakout Flag**: Binary flag for channel breakouts
- **MA Crossover Signal**: SMA 50 vs SMA 200 crossover
- **MACD Crossover Signal**: MACD vs signal line crossover
- **ROC Signal**: Rate of change signal
- **Volatility Expansion Flag**: ATR expansion indicator
- **Ichimoku Cloud**: Tenkan, Kijun, Senkou Span A/B
- **Parabolic SAR**: Stop and reverse indicator
- **Price Channel Slope**: Trend slope measure
- **Fractals**: High and low fractal points
- **Pivot Points**: Pivot, R1, R2, S1, S2 levels
- **Fibonacci Retracements**: 0.236, 0.382, 0.5, 0.618, 0.786 levels
- **Seasonality**: Monthly, day of month, day of week, hour of day
- **Holiday Flag**: Binary holiday indicator

### 3.6 Signals and Flags (5+ features)
- **Feature Interaction Terms**: Volatility × Momentum interactions
- **Ensemble Feature Aggregates**: Combined RSI, MACD, Z-score signals

---

## 4. Context Features (Optional)

### 4.1 Macro Economic Indicators
- **DXY Index**: US Dollar Index
- **VIX Index**: Volatility Index
- **US 10Y Yield**: 10-year Treasury yield
- **US 2Y Yield**: 2-year Treasury yield
- **TIP ETF**: Treasury Inflation-Protected Securities
- **Real Yield Spread**: 10Y yield vs TIP returns
- **Yield Curve (2s10s)**: 10Y - 2Y yield spread
- **5-period returns** for each macro indicator
- **60-period Z-scores** for each macro indicator

### 4.2 Volatility Indices
- **GVZ**: Gold Volatility Index
- **OVX**: Crude Oil Volatility Index

### 4.3 Spread Features
- **Inter-commodity spreads**: For related commodities (e.g., GC=F vs SI=F)
- **Cross-asset relationships**: Commodity-to-commodity correlations

### 4.4 Regime Features
- **Market regime indicators**: Bull/bear/neutral market states
- **Volatility regime**: High/low volatility periods

### 4.5 Intraday Aggregates
- **Multi-timeframe features**: Aggregated from 4h, 1h timeframes
- **30-period lookback**: Rolling aggregates from shorter timeframes

---

## 5. Feature Selection and Processing

### 5.1 Feature Filtering
1. **Variance Filtering**: Remove low-variance features (threshold: `LOW_VARIANCE_THRESHOLD`)
2. **Importance-Based Selection**: Top 180 features by importance (minimum share: 0.0008)
3. **Correlation-Based Reduction**: Remove highly correlated features (threshold: `HIGH_FEATURE_CORR`)
4. **Target Correlation Ranking**: Rank features by correlation with target
5. **Redundancy Pruning**: Remove redundant features (correlation > threshold)

### 5.2 Required Features (Always Included)
These features are always kept for interpretability:
- `RSI_14`
- `MACD_histogram`
- `SMA_50`
- `SMA_200`
- `ATR_14`
- `Volume_Ratio`

### 5.3 Feature Scaling
- **Scaler**: `RobustScaler` (robust to outliers)
- **Applied to**: All numeric features before training
- **Purpose**: Normalize features for model stability (especially for RL models)

### 5.4 Final Feature Count
- **Maximum Features**: 200 features per model
- **Typical Count**: 150-200 features after selection
- **Minimum**: 6 required action signal fields

---

## 6. Data Preprocessing

### 6.1 Missing Data Handling
- **Forward Fill**: Fill missing values with previous valid value
- **Backward Fill**: Fill remaining NaNs with next valid value
- **Zero Fill**: Replace any remaining NaNs with 0.0
- **Row Dropping**: Drop rows with insufficient data (warm-up period)

### 6.2 Constant Feature Removal
- **Detection**: Features with ≤ 1 unique value
- **Exception**: Required action features are never removed
- **Action**: Drop constant features before training

### 6.3 Data Validation
- **Schema Validation**: Ensure required features exist
- **Target Validation**: Ensure targets contain no NaNs
- **Index Validation**: Ensure chronological ordering
- **Split Validation**: Ensure proper train/val/test boundaries

---

## 7. Training Data Flow

```
Raw OHLCV Candles
    ↓
Feature Calculation (200+ features)
    ↓
Target Generation (target_return)
    ↓
Context Feature Addition (optional)
    ↓
Feature Selection & Filtering (→ 150-200 features)
    ↓
Feature Scaling (RobustScaler)
    ↓
Train/Val/Test Split
    ↓
Model Training
```

---

## 8. Model-Specific Considerations

### 8.1 Tree-Based Models (Random Forest, LightGBM, XGBoost)
- Use all selected features
- Feature importance used for selection
- Handle missing values internally

### 8.2 Deep Q-Network (DQN)
- Requires scaled features
- Uses feature importance for state representation
- Sensitive to feature normalization

### 8.3 Stacked Ensemble & Consensus Guard
- Combines predictions from all live models with R²/MAE-based weights (models flagged for overfitting receive a 65% weight penalty).
- Applies a neutral-return guard: if the weighted return magnitude is below the dynamic volatility threshold the action is forced to `HOLD` and the predicted return is clamped to 0%.
- Persists both percentage (`predicted_return_pct`) and decimal (`predicted_return`) outputs so downstream monitors stay in sync.

---

## 9. Feature Storage

### 9.1 Feature Files
- **Location**: `data/features/{asset_type}/{symbol}/{timeframe}/features.json`
- **Format**: JSON with feature names, values, timestamps, and status
- **Update Frequency**: Updated with each new candle

### 9.2 Feature Schema Version
- **Current Version**: 2
- **Purpose**: Track feature calculation changes for compatibility

---

## 10. Summary

### Key Statistics
- **Total Features Computed**: 200+
- **Features After Selection**: 150-200
- **Required Features**: 6 (always included)
- **Target Variable**: `target_return` (future return percentage)
- **Data Split**: 75% train / 12.5% val / 12.5% test
- **Feature Scaling**: RobustScaler
- **Feature Selection**: Importance + correlation-based

### Feature Categories Breakdown
1. **Moving Averages**: ~15 features
2. **Momentum Indicators**: ~20 features
3. **Volatility Indicators**: ~20 features
4. **Volume Indicators**: ~10 features
5. **Price Structure**: ~30 features
6. **Signals/Flags**: ~5 features
7. **Context Features**: ~20-30 features (optional)

---

## Notes

- Features are computed using only **backward-looking data** (no future data leakage)
- All features are **time-aligned** with the target variable
- Feature selection is **deterministic** and reproducible
- Context features are **optional** and can be disabled via `ContextFeatureConfig`
- The system automatically handles **missing data** and **feature versioning**

