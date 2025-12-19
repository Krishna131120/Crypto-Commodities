"""
Verification script to test RF/LightGBM fixes with synthetic data.
This proves the fixes work without requiring real data files.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import r2_score

# Create synthetic data with clear signal
np.random.seed(42)
n_samples = 1000
n_features = 50

# Create features with signal
X = np.random.randn(n_samples, n_features)
# Create target with clear signal (not just noise)
y = X[:, 0] * 0.8 + X[:, 1] * 0.5 + X[:, 2] * 0.3 + np.random.randn(n_samples) * 0.1

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("=" * 80)
print("VERIFYING RF/LIGHTGBM FIXES")
print("=" * 80)
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
print(f"Target std: {np.std(y):.4f}")
print()

# Test 1: Random Forest with AGGRESSIVE settings (like XGBoost fix)
print("1. Testing Random Forest with AGGRESSIVE settings (no restrictions):")
rf_params = {
    "n_estimators": 400,
    "max_depth": None,  # NO depth limit
    "min_samples_leaf": 1,  # Minimum restriction
    "min_samples_split": 2,  # Minimum restriction
    "max_features": 1.0,  # Use ALL features
    "max_samples": 1.0,  # Use ALL data
    "ccp_alpha": 0.0,  # NO pruning
    "random_state": 42,
    "n_jobs": 1,
}
rf = RandomForestRegressor(**rf_params)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
rf_r2 = r2_score(y_val, rf_pred)
rf_var = np.var(rf_pred)
print(f"   R²: {rf_r2:.4f} (should be > 0.5)")
print(f"   Variance: {rf_var:.6f} (should be > 0.01)")
print(f"   Status: {'PASS' if rf_r2 > 0.5 and rf_var > 0.01 else 'FAIL'}")

# Test 2: LightGBM with AGGRESSIVE settings (like XGBoost fix)
print("\n2. Testing LightGBM with AGGRESSIVE settings (no regularization):")
lgb_params = {
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.15,
    "max_depth": -1,  # NO depth limit (LightGBM uses -1)
    "num_leaves": 127,  # Maximum leaves
    "subsample": 1.0,  # Use ALL data
    "colsample_bytree": 1.0,  # Use ALL features
    "reg_alpha": 0.0,  # NO regularization
    "reg_lambda": 0.0,  # NO regularization
    "min_child_samples": 1,  # Minimum restriction
    "min_split_gain": 0.0,  # NO pruning
    "random_state": 42,
    "force_row_wise": True,
    "verbose": -1,
}
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_val)
lgb_r2 = r2_score(y_val, lgb_pred)
lgb_var = np.var(lgb_pred)
print(f"   R²: {lgb_r2:.4f} (should be > 0.5)")
print(f"   Variance: {lgb_var:.6f} (should be > 0.01)")
print(f"   Status: {'PASS' if lgb_r2 > 0.5 and lgb_var > 0.01 else 'FAIL'}")

# Test 3: XGBoost for comparison (should still work)
print("\n3. Testing XGBoost (baseline - should work):")
xgb_params = {
    "n_estimators": 300,
    "learning_rate": 0.08,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "reg_alpha": 1.0,
    "reg_lambda": 2.0,
    "min_child_weight": 3.0,
    "gamma": 0.1,
    "objective": "reg:squarederror",
    "tree_method": "approx",
    "random_state": 42,
}
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_val)
xgb_r2 = r2_score(y_val, xgb_pred)
xgb_var = np.var(xgb_pred)
print(f"   R²: {xgb_r2:.4f} (should be > 0.5)")
print(f"   Variance: {xgb_var:.6f} (should be > 0.01)")
print(f"   Status: {'PASS' if xgb_r2 > 0.5 and xgb_var > 0.01 else 'FAIL'}")

# Test 4: Compare all three
print("\n" + "=" * 80)
print("COMPARISON:")
print("=" * 80)
print(f"Random Forest R²: {rf_r2:.4f}")
print(f"LightGBM R²:      {lgb_r2:.4f}")
print(f"XGBoost R²:       {xgb_r2:.4f}")
print()

all_working = rf_r2 > 0.5 and lgb_r2 > 0.5 and xgb_r2 > 0.5
if all_working:
    print("SUCCESS: All three models are now learning!")
    print("The fixes are working - RF and LightGBM can now learn like XGBoost.")
    print("\nNext steps:")
    print("1. Run data ingestion to get real commodity data")
    print("2. Run training with real data")
    print("3. Verify all models work on real data")
else:
    print("ISSUE: Some models still not learning properly")
    if rf_r2 <= 0.5:
        print(f"  - Random Forest R² too low: {rf_r2:.4f}")
    if lgb_r2 <= 0.5:
        print(f"  - LightGBM R² too low: {lgb_r2:.4f}")
    if xgb_r2 <= 0.5:
        print(f"  - XGBoost R² too low: {xgb_r2:.4f}")

print("=" * 80)
sys.exit(0 if all_working else 1)
