"""
Diagnostic script to verify hyperopt fixes are working.
Tests if models can actually learn with the new parameter ranges.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

# Test data - create a simple dataset with signal
np.random.seed(42)
n_samples = 500
n_features = 20

# Create features with some signal
X = np.random.randn(n_samples, n_features)
# Create target with signal (not just noise)
y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.2

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("=" * 60)
print("TESTING FIXED PARAMETER RANGES")
print("=" * 60)

# Test 1: Random Forest with new defaults
print("\n1. Testing Random Forest with NEW defaults:")
rf_params = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "max_features": 0.8,
    "max_samples": 0.9,
    "ccp_alpha": 0.0,
    "random_state": 42,
    "n_jobs": 1,
}
rf = RandomForestRegressor(**rf_params)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
rf_var = np.var(rf_pred)
rf_r2 = 1 - np.sum((y_val - rf_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
print(f"   Variance: {rf_var:.6f} (should be > 0.01)")
print(f"   R²: {rf_r2:.4f} (should be > 0)")
print(f"   Status: {'PASS' if rf_var > 0.01 and rf_r2 > 0 else 'FAIL'}")

# Test 2: LightGBM with new defaults
print("\n2. Testing LightGBM with NEW defaults:")
lgb_params = {
    "boosting_type": "gbdt",
    "n_estimators": 400,
    "learning_rate": 0.1,
    "max_depth": 5,
    "num_leaves": 31,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "min_child_samples": 10,
    "min_split_gain": 0.01,
    "random_state": 42,
    "force_row_wise": True,
    "verbose": -1,
}
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_val)
lgb_var = np.var(lgb_pred)
lgb_r2 = 1 - np.sum((y_val - lgb_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
print(f"   Variance: {lgb_var:.6f} (should be > 0.01)")
print(f"   R²: {lgb_r2:.4f} (should be > 0)")
print(f"   Status: {'PASS' if lgb_var > 0.01 and lgb_r2 > 0 else 'FAIL'}")

# Test 3: XGBoost with new defaults
print("\n3. Testing XGBoost with NEW defaults:")
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
    "random_state": 42,
    "tree_method": "approx",
}
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_val)
xgb_var = np.var(xgb_pred)
xgb_r2 = 1 - np.sum((y_val - xgb_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
print(f"   Variance: {xgb_var:.6f} (should be > 0.01)")
print(f"   R²: {xgb_r2:.4f} (should be > 0)")
print(f"   Status: {'PASS' if xgb_var > 0.01 and xgb_r2 > 0 else 'FAIL'}")

# Test 4: Test hyperopt ranges (sample a few random combinations)
print("\n4. Testing hyperopt ranges (sampling random combinations):")
import random
random.seed(42)

# RF ranges
rf_trials = []
for _ in range(5):
    params = {
        "n_estimators": random.randint(200, 400),
        "max_depth": random.randint(3, 15),
        "min_samples_leaf": random.randint(1, 10),
        "min_samples_split": random.randint(2, 20),
        "max_features": random.uniform(0.5, 1.0),
        "max_samples": random.uniform(0.6, 1.0),
        "ccp_alpha": random.uniform(0.0, 0.01),
        "random_state": 42,
        "n_jobs": 1,
    }
    rf_test = RandomForestRegressor(**params)
    rf_test.fit(X_train, y_train)
    pred = rf_test.predict(X_val)
    var = np.var(pred)
    rf_trials.append(var > 0.01)

print(f"   Random Forest: {sum(rf_trials)}/5 trials produced variance > 0.01")
print(f"   Status: {'PASS' if sum(rf_trials) >= 4 else 'FAIL'}")

# LGBM ranges
lgb_trials = []
for _ in range(5):
    max_depth = random.randint(2, 8)
    max_leaves = min(2 ** max_depth, 63)
    min_leaves = max(4, max_leaves - 20)
    num_leaves = random.randint(min_leaves, max_leaves)
    params = {
        "boosting_type": "gbdt",
        "n_estimators": random.randint(100, 500),
        "learning_rate": random.uniform(0.01, 0.2),
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "subsample": random.uniform(0.5, 1.0),
        "colsample_bytree": random.uniform(0.5, 1.0),
        "min_child_samples": random.randint(5, 30),
        "reg_alpha": random.uniform(0.0, 5.0),
        "reg_lambda": random.uniform(0.0, 6.0),
        "min_split_gain": random.uniform(0.0, 0.1),
        "random_state": 42,
        "force_row_wise": True,
        "verbose": -1,
    }
    lgb_test = lgb.LGBMRegressor(**params)
    lgb_test.fit(X_train, y_train)
    pred = lgb_test.predict(X_val)
    var = np.var(pred)
    lgb_trials.append(var > 0.01)

print(f"   LightGBM: {sum(lgb_trials)}/5 trials produced variance > 0.01")
print(f"   Status: {'PASS' if sum(lgb_trials) >= 4 else 'FAIL'}")

# XGBoost ranges
xgb_trials = []
for _ in range(5):
    params = {
        "n_estimators": random.randint(100, 500),
        "learning_rate": random.uniform(0.01, 0.15),
        "max_depth": random.randint(2, 8),
        "subsample": random.uniform(0.5, 1.0),
        "colsample_bytree": random.uniform(0.5, 1.0),
        "min_child_weight": random.uniform(1.0, 10.0),
        "gamma": random.uniform(0.0, 0.5),
        "reg_alpha": random.uniform(0.0, 6.0),
        "reg_lambda": random.uniform(0.0, 8.0),
        "objective": "reg:squarederror",
        "tree_method": "approx",
        "random_state": 42,
    }
    xgb_test = xgb.XGBRegressor(**params)
    xgb_test.fit(X_train, y_train)
    pred = xgb_test.predict(X_val)
    var = np.var(pred)
    xgb_trials.append(var > 0.01)

print(f"   XGBoost: {sum(xgb_trials)}/5 trials produced variance > 0.01")
print(f"   Status: {'PASS' if sum(xgb_trials) >= 4 else 'FAIL'}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
all_passed = (
    rf_var > 0.01 and rf_r2 > 0 and
    lgb_var > 0.01 and lgb_r2 > 0 and
    xgb_var > 0.01 and xgb_r2 > 0 and
    sum(rf_trials) >= 4 and
    sum(lgb_trials) >= 4 and
    sum(xgb_trials) >= 4
)
if all_passed:
    print("ALL TESTS PASSED - Fixes are working correctly!")
    print("  If you're still getting bad results, the issue is likely:")
    print("  1. Data quality (no signal in your actual data)")
    print("  2. Feature engineering (features don't predict target)")
    print("  3. Target definition (target is too noisy)")
else:
    print("SOME TESTS FAILED - Fixes may need adjustment")
sys.exit(0 if all_passed else 1)
