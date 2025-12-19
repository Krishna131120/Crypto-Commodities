"""
Test the actual training functions to verify fixes are applied.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import the actual training functions
sys.path.insert(0, '.')
from ml.trainers import train_random_forest, train_lightgbm, train_xgboost

# Create synthetic data with clear signal
np.random.seed(42)
n_samples = 1000
n_features = 50

# Create features with signal
X = np.random.randn(n_samples, n_features)
# Create target with clear signal
y = X[:, 0] * 0.8 + X[:, 1] * 0.5 + X[:, 2] * 0.3 + np.random.randn(n_samples) * 0.1

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to DataFrames/Series (as expected by training functions)
X_train_df = pd.DataFrame(X_train)
X_val_df = pd.DataFrame(X_val)
y_train_series = pd.Series(y_train)
y_val_series = pd.Series(y_val)

print("=" * 80)
print("TESTING ACTUAL TRAINING FUNCTIONS")
print("=" * 80)
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
print()

# Test 1: Random Forest using actual training function
print("1. Testing train_random_forest() function:")
try:
    rf_model, rf_result = train_random_forest(
        X_train_df, y_train_series, X_val_df, y_val_series,
        refit_on_full=False
    )
    rf_r2 = rf_result.r2 if hasattr(rf_result, 'r2') else None
    rf_pred = rf_model.predict(X_val_df)
    rf_r2_calc = 1 - np.sum((y_val - rf_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
    rf_var = np.var(rf_pred)
    print(f"   R²: {rf_r2_calc:.4f} (should be > 0.5)")
    print(f"   Variance: {rf_var:.6f} (should be > 0.01)")
    print(f"   Status: {'PASS' if rf_r2_calc > 0.5 and rf_var > 0.01 else 'FAIL'}")
    rf_ok = rf_r2_calc > 0.5 and rf_var > 0.01
except Exception as e:
    print(f"   ERROR: {e}")
    rf_ok = False

# Test 2: LightGBM using actual training function
print("\n2. Testing train_lightgbm() function:")
try:
    lgb_model, lgb_result = train_lightgbm(
        X_train_df, y_train_series, X_val_df, y_val_series,
        refit_on_full=False
    )
    lgb_r2 = lgb_result.r2 if hasattr(lgb_result, 'r2') else None
    lgb_pred = lgb_model.predict(X_val_df)
    lgb_r2_calc = 1 - np.sum((y_val - lgb_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
    lgb_var = np.var(lgb_pred)
    print(f"   R²: {lgb_r2_calc:.4f} (should be > 0.5)")
    print(f"   Variance: {lgb_var:.6f} (should be > 0.01)")
    print(f"   Status: {'PASS' if lgb_r2_calc > 0.5 and lgb_var > 0.01 else 'FAIL'}")
    lgb_ok = lgb_r2_calc > 0.5 and lgb_var > 0.01
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    lgb_ok = False

# Test 3: XGBoost using actual training function
print("\n3. Testing train_xgboost() function:")
try:
    xgb_model, xgb_result = train_xgboost(
        X_train_df, y_train_series, X_val_df, y_val_series,
        refit_on_full=False
    )
    xgb_r2 = xgb_result.r2 if hasattr(xgb_result, 'r2') else None
    xgb_pred = xgb_model.predict(X_val_df)
    xgb_r2_calc = 1 - np.sum((y_val - xgb_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
    xgb_var = np.var(xgb_pred)
    print(f"   R²: {xgb_r2_calc:.4f} (should be > 0.5)")
    print(f"   Variance: {xgb_var:.6f} (should be > 0.01)")
    print(f"   Status: {'PASS' if xgb_r2_calc > 0.5 and xgb_var > 0.01 else 'FAIL'}")
    xgb_ok = xgb_r2_calc > 0.5 and xgb_var > 0.01
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    xgb_ok = False

# Summary
print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)
all_ok = rf_ok and lgb_ok and xgb_ok
if all_ok:
    print("SUCCESS: All training functions are working correctly!")
    print("The fixes are properly integrated into the training pipeline.")
else:
    print("ISSUE: Some training functions need attention")
    if not rf_ok:
        print("  - train_random_forest() needs fixing")
    if not lgb_ok:
        print("  - train_lightgbm() needs fixing")
    if not xgb_ok:
        print("  - train_xgboost() needs fixing")

print("=" * 80)
sys.exit(0 if all_ok else 1)
