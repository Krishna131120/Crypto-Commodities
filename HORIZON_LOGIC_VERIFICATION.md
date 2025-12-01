# Horizon Logic Verification

This document verifies that all three horizons (intraday, short, long) are working correctly.

## Horizon Definitions

### 1. Intraday (1 bar)
- **Horizon bars**: 1
- **Directional threshold**: 0.0005 (0.05%)
- **Smoothing window**: 2
- **Neutral band**: 0.0002 (0.02%)

### 2. Short-Term (4 bars)
- **Horizon bars**: 4
- **Directional threshold**: 0.0012 (0.12%)
- **Smoothing window**: 4
- **Neutral band**: 0.0006 (0.06%)

### 3. Long-Term (30 bars)
- **Horizon bars**: 30
- **Directional threshold**: 0.003 (0.30%)
- **Smoothing window**: 10
- **Neutral band**: 0.0015 (0.15%)

## Prediction Limits by Horizon

### Clamping Limits (ml/inference.py)
- **Intraday**: ±5% max
- **Short**: ±10% max
- **Long**: ±20% max

### Sanity Check Limits (core/mcp_adapter.py)
- **Intraday**: ±3% max (typical daily volatility)
- **Short**: ±8% max
- **Long**: ±20% max

### DQN Clamping (core/mcp_adapter.py)
- **Intraday**: ±3% max
- **Short**: ±8% max
- **Long**: ±20% max

## Logic Flow Verification

### ✅ 1. Horizon Selection
- **Location**: `core/mcp_adapter.py:398-400`
- **Logic**: Normalizes horizon profile and gets config from `PROFILE_BASES`
- **Status**: ✅ Correct

### ✅ 2. Action Threshold
- **Location**: `core/mcp_adapter.py:478-481`
- **Logic**: Uses `horizon_config.directional_threshold` (horizon-aware)
- **Status**: ✅ Fixed - Now uses horizon-specific threshold

### ✅ 3. Model Clamping
- **Location**: `ml/inference.py:139`
- **Logic**: Uses `_clamp()` with `horizon_profile` parameter
- **Status**: ✅ Correct - Horizon-aware clamping applied

### ✅ 4. Consensus Sanity Check
- **Location**: `ml/inference.py:155-163`
- **Logic**: For intraday, caps at 3% daily volatility
- **Status**: ✅ Correct - Only applies to intraday

### ✅ 5. Final Sanity Check
- **Location**: `core/mcp_adapter.py:632-653`
- **Logic**: Horizon-aware limits based on `horizon_bars`
- **Status**: ✅ Correct - Uses horizon_bars to determine limits

### ✅ 6. DQN Horizon-Aware Clamping
- **Location**: `core/mcp_adapter.py:502-507`
- **Logic**: Clamps DQN predictions based on `horizon_bars`
- **Status**: ✅ Correct - Uses horizon_bars for clamping

### ✅ 7. Neutral Guard
- **Location**: `ml/inference.py:200-214`
- **Logic**: Uses `dynamic_threshold` (loaded from model summary, which is horizon-specific)
- **Status**: ✅ Correct - Model's dynamic_threshold is set during training for that horizon

## Potential Issues Found and Fixed

### Issue 1: Action Threshold Not Horizon-Aware
- **Problem**: Was using `pipeline.dynamic_threshold` which might not match horizon
- **Fix**: Now uses `horizon_config.directional_threshold` directly
- **Status**: ✅ Fixed

## Summary

All three horizons are correctly configured with:
1. ✅ Proper horizon-specific thresholds
2. ✅ Horizon-aware clamping limits
3. ✅ Horizon-aware sanity checks
4. ✅ Horizon-aware DQN processing
5. ✅ Correct horizon_bars calculation

The logic is consistent across all three horizons and properly handles the different prediction ranges for each timeframe.

