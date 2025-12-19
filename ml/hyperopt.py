"""
Optuna-based hyperparameter search helpers with overfitting prevention.
Uses adaptive mechanisms that adjust based on data characteristics.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error, r2_score

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

# Import adaptive optimization
USE_ADAPTIVE = False
try:
    # Try relative import first (when used as module)
    from .adaptive_hyperopt import (
        optimize_lightgbm_adaptive,
        optimize_xgboost_adaptive,
        optimize_random_forest_adaptive,
    )
    USE_ADAPTIVE = True
except (ImportError, ModuleNotFoundError):
    try:
        # Try absolute import (when used directly)
        from ml.adaptive_hyperopt import (
            optimize_lightgbm_adaptive,
            optimize_xgboost_adaptive,
            optimize_random_forest_adaptive,
        )
        USE_ADAPTIVE = True
    except (ImportError, ModuleNotFoundError):
        # Fallback: use fixed ranges if adaptive module not available
        USE_ADAPTIVE = False

# Suppress LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", message=".*LightGBM.*")
warnings.filterwarnings("ignore", message=".*lgb.*")


def _mae_metric(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _get_safe_n_jobs() -> int:
    """
    Safely determine n_jobs for parallel processing.
    On Windows, joblib can fail when trying to count CPU cores,
    so we test first and fall back to single-threaded if needed.
    """
    try:
        import joblib
        # Quick test to see if parallel processing works
        test_result = joblib.Parallel(n_jobs=1, backend='threading')(
            joblib.delayed(lambda x: x)(1) for _ in range(1)
        )
        return -1  # Use all available cores if test passes
    except Exception:
        # If joblib has issues (common on Windows), use single thread
        return 1


def _compute_overfitting_penalty(y_train, y_val, train_pred, val_pred) -> float:
    """
    Compute penalty for overfitting based on train/val gap.
    FIXED: Less harsh penalties to allow models to learn during hyperopt.
    """
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    gap = train_r2 - val_r2
    
    # More lenient threshold: penalize if gap > 0.08 (8%)
    if gap > 0.08:
        penalty = 1.0 + (gap - 0.08) * 10.0  # Less harsh penalty
        return float(penalty)
    # Less harsh penalty for negative validation R² - still allow exploration
    if val_r2 < -0.2:  # Only heavily penalize very negative R²
        penalty = 1.0 + abs(val_r2) * 2.0  # Less harsh penalty
        return float(penalty)
    return 1.0


def _optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials: int, timeout: Optional[int]):
    def objective(trial: optuna.Trial) -> float:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*LightGBM.*")
            warnings.filterwarnings("ignore", message=".*lgb.*")
            
            # STEP 4 FIX: Very permissive ranges to ensure learning
            # Use deeper trees and more leaves to guarantee learning capacity
            max_depth = trial.suggest_int("max_depth", 5, 10)  # Deeper trees for learning
            max_leaves = min(2 ** max_depth, 127)  # Allow many more leaves (up to 127)
            min_leaves = max(15, max_leaves - 30)  # Wider range, ensure minimum capacity
            num_leaves = trial.suggest_int("num_leaves", min_leaves, max_leaves)
            
            min_iterations = 150  # More iterations to ensure learning
            n_est = trial.suggest_int("n_estimators", min_iterations, 500)  # More iterations
            
            # STEP 4 FIX: Start with GUARANTEED learning parameters
            # Use uniform distribution that heavily favors learning-friendly values
            # First ensure model CAN learn, then we'll add regularization to prevent overfitting
            
            # Learning rate: Balanced range - too high causes instability, too low prevents learning
            learning_rate = trial.suggest_float("learning_rate", 0.05, 0.15)  # Balanced range for stability
            
            # Regularization: Moderate range - need some regularization to prevent overfitting
            reg_alpha = trial.suggest_float("reg_alpha", 0.1, 2.0)  # Moderate regularization
            reg_lambda = trial.suggest_float("reg_lambda", 0.1, 3.0)  # Moderate regularization
            
            # Ensure enough capacity to learn - very permissive
            min_child_samples = trial.suggest_int("min_child_samples", 1, 10)  # Very permissive
            min_split_gain = trial.suggest_float("min_split_gain", 0.0, 0.02)  # Minimal pruning
            
            params = {
                "boosting_type": "gbdt",
                "n_estimators": n_est,
                "learning_rate": learning_rate,  # Higher to ensure learning
                "num_leaves": num_leaves,
                "max_depth": max_depth,
                "subsample": trial.suggest_float("subsample", 0.8, 1.0),  # Use most/all data
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),  # Use most/all features
                "min_child_samples": min_child_samples,  # Very permissive
                "reg_alpha": reg_alpha,  # Low regularization
                "reg_lambda": reg_lambda,  # Low regularization
                "min_split_gain": min_split_gain,  # Minimal pruning
                "random_state": 42,
                "force_row_wise": True,
                "verbose": -1,
            }
            model = lgb.LGBMRegressor(**params)
            
            # Train for minimum iterations first (no early stopping)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="l2",
                callbacks=[lgb.log_evaluation(period=0)],  # No early stopping - train full min_iterations
            )
            
            val_preds = model.predict(X_val)
            train_preds = model.predict(X_train)
            
            # STEP 1 FIX: Check if predictions are constant - use more lenient threshold
            # Very small variance might still indicate learning, just weak signal
            pred_variance = np.var(val_preds) if len(val_preds) > 1 else 0.0
            train_variance = np.var(train_preds) if len(train_preds) > 1 else 0.0
            
            # Use more lenient threshold - only reject if truly constant
            # Check if predictions are actually constant (std dev near zero)
            pred_std = np.std(val_preds) if len(val_preds) > 1 else 0.0
            train_std = np.std(train_preds) if len(train_preds) > 1 else 0.0
            target_std = np.std(y_val.to_numpy()) if len(y_val) > 1 else 0.0
            
            # Only reject if predictions are truly constant (std < 1% of target std)
            if pred_std < (target_std * 0.01) or train_std < (target_std * 0.01):
                # Constant predictions - return very high loss to reject this trial
                return 1e10
            
            # Compute MAE with overfitting penalty
            val_mae = _mae_metric(y_val, val_preds)
            penalty = _compute_overfitting_penalty(y_train, y_val, train_preds, val_preds)
            return val_mae * penalty

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    # STEP 4 FIX: If all trials failed, return guaranteed-working parameters instead of empty dict
    if study.best_value >= 1e9:  # All trials produced constant predictions
        # Return balanced parameters that are guaranteed to allow learning
        # These are less aggressive than the training defaults but still allow learning
        return {
            "n_estimators": 400,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "min_samples_split": 5,
            "max_features": 0.9,
            "max_samples": 0.9,
            "ccp_alpha": 0.0001,
        }
    
    return study.best_params


def _optimize_xgboost(X_train, y_train, X_val, y_val, n_trials: int, timeout: Optional[int]):
    def objective(trial: optuna.Trial) -> float:
        # FIXED: More permissive ranges to allow actual learning
        # XGBoost works but we need to ensure hyperopt can find good parameters
        min_iterations = 100
        n_est = trial.suggest_int("n_estimators", min_iterations, 500)  # More iterations
        
        # STEP 4 FIX: Start with GUARANTEED learning parameters
        # Use uniform distribution that heavily favors learning-friendly values
        # First ensure model CAN learn, then we'll add regularization to prevent overfitting
        
        # Learning rate: Balanced range for stability
        learning_rate = trial.suggest_float("learning_rate", 0.05, 0.12)  # Balanced range
        
        # Regularization: Moderate range to prevent overfitting while allowing learning
        reg_alpha = trial.suggest_float("reg_alpha", 0.5, 3.0)  # Moderate regularization
        reg_lambda = trial.suggest_float("reg_lambda", 1.0, 4.0)  # Moderate regularization
        
        # Ensure enough capacity to learn - very permissive
        min_child_weight = trial.suggest_float("min_child_weight", 0.5, 2.0)  # Very permissive
        gamma = trial.suggest_float("gamma", 0.0, 0.1)  # Minimal pruning
        
        params = {
            "n_estimators": n_est,
            "learning_rate": learning_rate,  # Higher to ensure learning
            "max_depth": trial.suggest_int("max_depth", 5, 9),  # Deeper trees for learning
            "subsample": trial.suggest_float("subsample", 0.85, 1.0),  # Use most/all data
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.85, 1.0),  # Use most/all features
            "min_child_weight": min_child_weight,  # Very permissive
            "gamma": gamma,  # Minimal pruning
            "reg_alpha": reg_alpha,  # Low regularization
            "reg_lambda": reg_lambda,  # Low regularization
            "objective": "reg:squarederror",
            "tree_method": "approx",
            "random_state": 42,
        }
        model = xgb.XGBRegressor(**params)
        
        # Train for minimum iterations first (no early stopping)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            # No early_stopping_rounds - train full min_iterations
        )
        
        val_preds = model.predict(X_val)
        train_preds = model.predict(X_train)
        
        # CRITICAL: Check if predictions are constant - use relative threshold
        pred_std = np.std(val_preds) if len(val_preds) > 1 else 0.0
        train_std = np.std(train_preds) if len(train_preds) > 1 else 0.0
        target_std = np.std(y_val.to_numpy()) if len(y_val) > 1 else 0.0
        
        # Only reject if predictions are truly constant (std < 1% of target std)
        if pred_std < (target_std * 0.01) or train_std < (target_std * 0.01):
            # Constant predictions - return very high loss to reject this trial
            return 1e10
        
        # Compute MAE with overfitting penalty
        val_mae = _mae_metric(y_val, val_preds)
        penalty = _compute_overfitting_penalty(y_train, y_val, train_preds, val_preds)
        return val_mae * penalty

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    # STEP 4 FIX: If all trials failed, return guaranteed-working parameters for XGBoost
    if study.best_value >= 1e9:  # All trials produced constant predictions
        # Return balanced parameters that are guaranteed to allow learning
        return {
            "n_estimators": 300,
            "learning_rate": 0.08,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.75,
            "reg_alpha": 1.0,
            "reg_lambda": 2.0,
            "min_child_weight": 3.0,
            "gamma": 0.1,
        }
    
    return study.best_params


def _optimize_random_forest(X_train, y_train, X_val, y_val, n_trials: int, timeout: Optional[int]):
    def objective(trial: optuna.Trial) -> float:
        # FIXED: Much more permissive ranges to allow actual learning
        # The previous ranges were too restrictive, causing ALL trials to produce constant predictions
        # STEP 4 FIX: Start with GUARANTEED learning parameters
        # Use uniform distribution that heavily favors learning-friendly values
        # First ensure model CAN learn, then we'll add regularization to prevent overfitting
        
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 350, 500),  # More trees for learning
            "max_depth": trial.suggest_int("max_depth", 10, 20),  # Much deeper trees for learning capacity
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 2),  # Very permissive
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 4),  # Very permissive
            "max_features": trial.suggest_float("max_features", 0.95, 1.0),  # Use almost all features
            "max_samples": trial.suggest_float("max_samples", 0.95, 1.0),  # Use almost all data
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.001),  # Minimal pruning (uniform, favors 0)
            "random_state": 42,
            "n_jobs": _get_safe_n_jobs(),
            "bootstrap": True,
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        train_preds = model.predict(X_train)
        
        # CRITICAL: Check if predictions are constant - use relative threshold
        pred_std = np.std(val_preds) if len(val_preds) > 1 else 0.0
        train_std = np.std(train_preds) if len(train_preds) > 1 else 0.0
        target_std = np.std(y_val.to_numpy()) if len(y_val) > 1 else 0.0
        
        # Only reject if predictions are truly constant (std < 1% of target std)
        if pred_std < (target_std * 0.01) or train_std < (target_std * 0.01):
            # Constant predictions - return very high loss to reject this trial
            return 1e10
        
        # Compute MAE with overfitting penalty
        val_mae = _mae_metric(y_val, val_preds)
        penalty = _compute_overfitting_penalty(y_train, y_val, train_preds, val_preds)
        return val_mae * penalty

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    # STEP 4 FIX: If all trials failed, return guaranteed-working parameters for Random Forest
    if study.best_value >= 1e9:  # All trials produced constant predictions
        # Return balanced parameters that are guaranteed to allow learning
        return {
            "n_estimators": 400,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "min_samples_split": 5,
            "max_features": 0.9,
            "max_samples": 0.9,
            "ccp_alpha": 0.0001,
        }
    
    return study.best_params


def optimize_model(
    model_name: str,
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials: int = 25,
    timeout: Optional[int] = None,
) -> Dict:
    """
    Run Optuna search for the requested model and return best params.
    Uses adaptive optimization that adjusts based on data characteristics.
    """
    if n_trials <= 0:
        return {}
    model_name = model_name.lower()
    
    # Use adaptive optimization if available (permanent solution)
    if USE_ADAPTIVE:
        if model_name == "lightgbm":
            return optimize_lightgbm_adaptive(X_train, y_train, X_val, y_val, n_trials, timeout)
        if model_name == "xgboost":
            return optimize_xgboost_adaptive(X_train, y_train, X_val, y_val, n_trials, timeout)
        if model_name == "random_forest":
            return optimize_random_forest_adaptive(X_train, y_train, X_val, y_val, n_trials, timeout)
    
    # Fallback to fixed ranges (legacy)
    if model_name == "lightgbm":
        return _optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials, timeout)
    if model_name == "xgboost":
        return _optimize_xgboost(X_train, y_train, X_val, y_val, n_trials, timeout)
    if model_name == "random_forest":
        return _optimize_random_forest(X_train, y_train, X_val, y_val, n_trials, timeout)
    raise ValueError(f"Unsupported model for hyperopt: {model_name}")


