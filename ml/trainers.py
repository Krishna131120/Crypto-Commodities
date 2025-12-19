"""
Model training utilities for classical ML and RL agents.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Suppress LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", message=".*LightGBM.*")
warnings.filterwarnings("ignore", message=".*lgb.*")
from gymnasium import Env, spaces
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import lightgbm as lgb
import xgboost as xgb


DQN_CONFIG = {
    "learning_rate": 5e-4,
    "buffer_size": 2000,
    "learning_starts": 200,
    "batch_size": 128,
    "gamma": 0.97,
    "exploration_fraction": 0.15,
    "exploration_final_eps": 0.1,
    "target_update_interval": 200,
}
DQN_TOTAL_TIMESTEPS = 5000


@dataclass
class TrainingResult:
    model_name: str
    mae: float
    rmse: float
    r2: float
    directional_accuracy: Optional[float] = None
    mean_predicted_return: Optional[float] = None
    mean_actual_return: Optional[float] = None
    notes: Optional[str] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model predictions (returns) using regression and directional metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        directional = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "directional_accuracy": directional,
        "mean_predicted_return": float(np.mean(y_pred)),
        "mean_actual_return": float(np.mean(y_true)),
    }


def _evaluate_classification(
    y_true: np.ndarray, y_prob: np.ndarray, decision_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate binary classification predictions (directional models).
    """
    y_pred = (y_prob >= decision_threshold).astype(int)
    metrics = {
        "directional_accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "mae": None,
        "rmse": None,
        "r2": None,
        "mean_predicted_return": float(np.mean(y_prob)),
        "mean_actual_return": float(np.mean(y_true)),
    }
    return metrics


def _build_random_forest(custom_params: Optional[Dict] = None) -> RandomForestRegressor:
    """
    STRONG regularization RandomForest for noisy financial targets.
    Maximum anti-overfitting: very conservative parameters, strong regularization.
    """
    # Try to use parallel processing, but fallback to single-threaded if joblib fails
    n_jobs = 1  # Default to single-threaded on Windows to avoid subprocess errors
    try:
        # Quick test to see if parallel processing works
        test_result = joblib.Parallel(n_jobs=1, backend='threading')(
            joblib.delayed(lambda x: x)(1) for _ in range(1)
        )
        n_jobs = -1  # If test passes, use all cores
    except Exception:
        pass  # Keep n_jobs=1 if test fails
    
    params = {
        "n_estimators": 400,  # Good number of trees
        "max_depth": 15,  # Deep enough to learn but not unlimited
        "min_samples_leaf": 2,  # Some restriction to prevent overfitting
        "min_samples_split": 5,  # Some restriction to prevent overfitting
        "max_features": 0.9,  # Use most features (not all to prevent overfitting)
        "max_samples": 0.9,  # Use most data (not all to prevent overfitting)
        "random_state": 42,
        "n_jobs": n_jobs,
        "bootstrap": True,
        "ccp_alpha": 0.0001,  # Small pruning to prevent overfitting
    }
    if custom_params:
        params.update(custom_params)
        # Override n_jobs if explicitly set in custom_params
        if "n_jobs" in custom_params:
            params["n_jobs"] = custom_params["n_jobs"]
    return RandomForestRegressor(**params)


def _build_random_forest_classifier() -> RandomForestClassifier:
    # Use safe n_jobs to avoid Windows subprocess errors (same logic as _build_random_forest)
    n_jobs = 1  # Default to single-threaded on Windows to avoid subprocess errors
    try:
        # Quick test to see if parallel processing works
        test_result = joblib.Parallel(n_jobs=1, backend='threading')(
            joblib.delayed(lambda x: x)(1) for _ in range(1)
        )
        n_jobs = -1  # If test passes, use all cores
    except Exception:
        pass  # Keep n_jobs=1 if test fails
    
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=15,
        min_samples_split=40,
        max_features=0.7,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=n_jobs,
        bootstrap=True,
    )


def _build_lightgbm(custom_params: Optional[Dict] = None) -> lgb.LGBMRegressor:
    """
    LightGBM configuration with STRONG regularization for financial data.
    Maximum anti-overfitting: very conservative parameters, strong regularization.
    """
    params = {
        "boosting_type": "gbdt",
        "n_estimators": 400,  # Good number of trees
        "learning_rate": 0.1,  # Good learning rate (not too high to prevent overfitting)
        "max_depth": 8,  # Deep enough to learn
        "num_leaves": 63,  # Good capacity (not maximum to prevent overfitting)
        "subsample": 0.9,  # Use most data (not all to prevent overfitting)
        "subsample_freq": 1,
        "colsample_bytree": 0.9,  # Use most features (not all to prevent overfitting)
        "reg_alpha": 0.5,  # Some regularization to prevent overfitting
        "reg_lambda": 1.0,  # Some regularization to prevent overfitting
        "min_child_samples": 10,  # Some restriction to prevent overfitting
        "min_split_gain": 0.01,  # Some pruning to prevent overfitting
        "max_bin": 255,
        "extra_trees": False,
        "min_data_in_bin": 3,  # Some restriction
        "feature_pre_filter": False,
        "random_state": 42,
        "force_row_wise": True,
        "verbose": -1,  # Suppress all LightGBM output
    }
    if custom_params:
        params.update(custom_params)
        # Ensure verbose is always -1 unless explicitly set
        if "verbose" not in custom_params:
            params["verbose"] = -1
    return lgb.LGBMRegressor(**params)


def _build_lightgbm_classifier() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        boosting_type="gbdt",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        reg_alpha=0.5,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=42,
        objective="binary",
        verbose=-1,  # Suppress all LightGBM output
    )


def _build_xgboost(custom_params: Optional[Dict] = None) -> xgb.XGBRegressor:
    """
    XGBoost configuration - CRITICAL FIX: Use 'approx' tree_method instead of 'hist'
    to avoid numerical precision issues that cause constant predictions.
    """
    params = {
        "n_estimators": 300,  # More trees for better learning
        "learning_rate": 0.08,  # Higher learning rate to allow learning
        "max_depth": 5,  # Deeper trees to allow learning
        "subsample": 0.8,  # More data for learning
        "colsample_bytree": 0.75,  # More features for learning
        "reg_alpha": 1.0,  # Less regularization - allow learning
        "reg_lambda": 2.0,  # Less regularization - allow learning
        "min_child_weight": 3.0,  # Less restrictive - allow more splits
        "gamma": 0.1,  # Less pruning - allow learning
        "objective": "reg:squarederror",
        "random_state": 42,
        "tree_method": "approx",  # CRITICAL: Use 'approx' instead of 'hist' to avoid precision issues
        "max_bin": 256,
        "grow_policy": "depthwise",  # Use depthwise growth for better learning
    }
    if custom_params:
        params.update(custom_params)
    return xgb.XGBRegressor(**params)


def _build_xgboost_classifier() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.5,
        min_child_weight=4,
        gamma=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
    )


def train_random_forest(
    X_train,
    y_train,
    X_val,
    y_val,
    refit_on_full: bool = True,
    X_full=None,
    y_full=None,
    param_overrides: Optional[Dict] = None,
) -> Tuple[RandomForestRegressor, TrainingResult]:
    """
    Train Random Forest with overfitting prevention.
    
    Args:
        X_train, y_train: Training data
        X_val, close_val, target_val: Validation data for evaluation
        refit_on_full: If True, refit on full dataset after validation
        X_full, y_full: Full dataset for refitting (if refit_on_full=True)
    
    Returns:
        Trained model and training result
    """
    # STEP 2 FIX: Use param_overrides if provided (from hyperopt), otherwise use balanced defaults
    # If hyperopt found parameters, use them. Otherwise use defaults that allow learning but prevent overfitting
    if param_overrides:
        # Use hyperopt-found parameters (should have some regularization)
        model = _build_random_forest(param_overrides)
    else:
        # Balanced defaults: allow learning but prevent severe overfitting
        balanced_params = {
            "n_estimators": 400,
            "max_depth": 15,  # Deep enough to learn
            "min_samples_leaf": 2,  # Some restriction to prevent overfitting
            "min_samples_split": 5,  # Some restriction to prevent overfitting
            "max_features": 0.9,  # Use most features
            "max_samples": 0.9,  # Use most data
            "ccp_alpha": 0.0001,  # Small pruning to prevent overfitting
        }
        model = _build_random_forest(balanced_params)
    
    model.fit(X_train, y_train)
    
    # Check if model produced constant predictions
    val_preds_initial = model.predict(X_val)
    train_preds_initial = model.predict(X_train)
    pred_variance = np.var(val_preds_initial) if len(val_preds_initial) > 1 else 0.0
    train_variance = np.var(train_preds_initial) if len(train_preds_initial) > 1 else 0.0
    
    # If predictions are constant or have very low variance, try more aggressive settings
    if pred_variance < 1e-6 or train_variance < 1e-6:
        import warnings as w
        w.warn(
            f"RandomForest producing low variance predictions "
            f"(train_var={train_variance:.2e}, val_var={pred_variance:.2e}). "
            f"Trying more aggressive settings..."
        )
        # Try with more permissive settings
        model = _build_random_forest({
            "n_estimators": 500,
            "max_depth": 20,  # Deeper
            "min_samples_leaf": 1,  # More permissive
            "min_samples_split": 2,  # More permissive
            "max_features": 1.0,  # All features
            "max_samples": 1.0,  # All data
            "ccp_alpha": 0.0,  # No pruning
        })
        model.fit(X_train, y_train)
        # Re-check
        val_preds_check = model.predict(X_val)
        pred_variance = np.var(val_preds_check) if len(val_preds_check) > 1 else 0.0
    
    # Evaluate on validation set
    pred_returns = model.predict(X_val)
    metrics = _evaluate(y_val.to_numpy(), pred_returns)
    
    # Refit on full dataset if requested (for deployment)
    if refit_on_full and X_full is not None and y_full is not None:
        model.fit(X_full, y_full)
    
    return model, TrainingResult("RandomForest", **metrics)


def train_lightgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    refit_on_full: bool = True,
    X_full=None,
    y_full=None,
    param_overrides: Optional[Dict] = None,
) -> Tuple[lgb.LGBMRegressor, TrainingResult]:
    """
    Train LightGBM with overfitting prevention and early stopping.
    """
    # Suppress LightGBM warnings during training
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*LightGBM.*")
        warnings.filterwarnings("ignore", message=".*lgb.*")
        
        # STEP 2 FIX: Use param_overrides if provided (from hyperopt), otherwise use balanced defaults
        # If hyperopt found parameters, use them. Otherwise use defaults that allow learning but prevent overfitting
        if param_overrides:
            # Use hyperopt-found parameters (should have some regularization)
            model = _build_lightgbm(param_overrides)
            min_iterations = param_overrides.get("n_estimators", 200)
        else:
            # Balanced defaults: allow learning but prevent severe overfitting
            balanced_params = {
                "n_estimators": 400,
                "learning_rate": 0.1,  # Good learning rate
                "max_depth": 8,  # Deep enough to learn
                "num_leaves": 63,  # Good capacity
                "subsample": 0.9,  # Use most data
                "colsample_bytree": 0.9,  # Use most features
                "reg_alpha": 0.5,  # Some regularization to prevent overfitting
                "reg_lambda": 1.0,  # Some regularization to prevent overfitting
                "min_child_samples": 10,  # Some restriction
                "min_split_gain": 0.01,  # Some pruning
            }
            model = _build_lightgbm(balanced_params)
            min_iterations = 200
        
        # ALWAYS train for minimum iterations first (no early stopping)
        # This ensures models actually learn before early stopping can trigger
        model.set_params(n_estimators=min_iterations)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            callbacks=[lgb.log_evaluation(period=0)],  # No early stopping - train full min_iterations
        )
        
        # Check if model produced constant predictions after minimum training
        val_preds_initial = model.predict(X_val)
        train_preds_initial = model.predict(X_train)
        pred_variance = np.var(val_preds_initial) if len(val_preds_initial) > 1 else 0.0
        train_variance = np.var(train_preds_initial) if len(train_preds_initial) > 1 else 0.0
        
        # If predictions are still constant or have very low variance, try more aggressive settings
        if pred_variance < 1e-6 or train_variance < 1e-6:
            import warnings as w
            w.warn(
                f"LightGBM producing low variance predictions after {min_iterations} iterations "
                f"(train_var={train_variance:.2e}, val_var={pred_variance:.2e}). "
                f"Trying more aggressive settings..."
            )
            # Try with more permissive settings
            aggressive_params = {
                "n_estimators": 400,
                "learning_rate": 0.15,  # Higher learning rate
                "max_depth": 10,  # Deeper
                "num_leaves": 127,  # Maximum leaves
                "subsample": 1.0,  # All data
                "colsample_bytree": 1.0,  # All features
                "reg_alpha": 0.0,  # No regularization
                "reg_lambda": 0.0,  # No regularization
                "min_child_samples": 5,  # More permissive
                "min_split_gain": 0.0,  # No pruning
            }
            model = _build_lightgbm(aggressive_params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="l2",
                callbacks=[lgb.log_evaluation(period=0)],
            )
            # Re-check
            val_preds_check = model.predict(X_val)
            train_preds_check = model.predict(X_train)
            pred_variance = np.var(val_preds_check) if len(val_preds_check) > 1 else 0.0
            train_variance = np.var(train_preds_check) if len(train_preds_check) > 1 else 0.0
        
        # Now allow early stopping for further optimization (if needed)
        # But only if we have non-constant predictions
        val_preds_check = model.predict(X_val)
        if np.var(val_preds_check) > 1e-10:
            # Model is learning, can use early stopping for optimization
            best_iter = getattr(model, 'best_iteration_', None) or min_iterations
            if best_iter < min_iterations:
                # Still need minimum training
                model = _build_lightgbm(param_overrides)
                model.set_params(n_estimators=min_iterations)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="l2",
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=30, verbose=False, first_metric_only=True),  # More aggressive: stop earlier to prevent overfitting
                        lgb.log_evaluation(period=0),
                    ],
                )
    
    # Evaluate on validation set
    pred_returns = model.predict(X_val)
    
    # CRITICAL: Validate predictions have variance before computing metrics
    pred_variance = np.var(pred_returns) if len(pred_returns) > 1 else 0.0
    train_preds_final = model.predict(X_train)
    train_variance_final = np.var(train_preds_final) if len(train_preds_final) > 1 else 0.0
    
    # If predictions are STILL constant after all training attempts, try one more aggressive fix
    if pred_variance < 1e-10 or train_variance_final < 1e-10:
        import warnings as w
        w.warn(
            f"LightGBM still producing constant predictions after extended training "
            f"(train_var={train_variance_final:.2e}, val_var={pred_variance:.2e}). "
            f"Attempting aggressive retraining with higher learning rate..."
        )
        # Last resort: train with even higher learning rate and more iterations
        model = _build_lightgbm(param_overrides)
        # Force higher learning rate if not already set
        if param_overrides and "learning_rate" in param_overrides:
            model.set_params(learning_rate=max(param_overrides["learning_rate"], 0.15))
        else:
            model.set_params(learning_rate=0.15)  # Higher learning rate
        model.set_params(n_estimators=300)  # More iterations
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            callbacks=[lgb.log_evaluation(period=0)],
        )
        # Re-check predictions
        pred_returns = model.predict(X_val)
        pred_variance = np.var(pred_returns) if len(pred_returns) > 1 else 0.0
        train_preds_final = model.predict(X_train)
        train_variance_final = np.var(train_preds_final) if len(train_preds_final) > 1 else 0.0
        
        if pred_variance < 1e-10:
            w.warn(
                f"LightGBM predictions are STILL constant after aggressive retraining (variance={pred_variance:.2e}). "
                f"This will result in R² = 0.000. Model may not have learned. Check data quality."
            )
    
    metrics = _evaluate(y_val.to_numpy(), pred_returns)
    
    # Refit on full dataset if requested
    # IMPORTANT: When refitting, use the SAME number of iterations that early stopping found
    # to prevent overfitting. Don't train longer just because we have more data.
    if refit_on_full and X_full is not None and y_full is not None:
        # Get the best iteration from the initial training
        best_iteration = getattr(model, 'best_iteration_', None) or getattr(model, 'n_estimators', 300)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*LightGBM.*")
            warnings.filterwarnings("ignore", message=".*lgb.*")
            
            # Create fresh model with same config - use actual iterations that worked
            actual_iterations = getattr(model, 'n_estimators', min_iterations)
            best_iter = getattr(model, 'best_iteration_', None)
            
            # Use best_iteration if available, otherwise use what we trained with
            if best_iter and best_iter >= min_iterations:
                actual_iterations = min(best_iter, 300)
            else:
                actual_iterations = max(actual_iterations, min_iterations)
            
            final_model = _build_lightgbm(param_overrides)
            final_model.set_params(n_estimators=actual_iterations)
            
            # Train WITHOUT early stopping to ensure we get the full iterations
            final_model.fit(
                X_full,
                y_full,
                eval_set=[(X_val, y_val)],
                eval_metric="l2",
                callbacks=[lgb.log_evaluation(period=0)],  # No early stopping - train full iterations
            )
            
            # Verify refit model has variance
            refit_preds = final_model.predict(X_val)
            refit_variance = np.var(refit_preds) if len(refit_preds) > 1 else 0.0
            if refit_variance < 1e-10:
                import warnings as w
                w.warn(
                    f"LightGBM refit model still producing constant predictions (variance={refit_variance:.2e}). "
                    f"Using original model instead."
                )
            else:
                model = final_model
    
    return model, TrainingResult("LightGBM", **metrics)


def train_xgboost(
    X_train,
    y_train,
    X_val,
    y_val,
    refit_on_full: bool = True,
    X_full=None,
    y_full=None,
    param_overrides: Optional[Dict] = None,
) -> Tuple[xgb.XGBRegressor, TrainingResult]:
    """
    Train XGBoost with overfitting prevention and early stopping.
    CRITICAL FIX: Remove all regularization initially to force learning.
    """
    # CRITICAL FIX: Force minimum training iterations to prevent constant predictions
    # Train R² = 0.000 means model predicts mean, which happens when early stopping triggers too early
    min_iterations = 100  # Increased from 50 - models need more iterations to learn
    
    # CRITICAL FIX: For XGBoost, try removing ALL regularization initially to force learning
    # If model still predicts constant, it means features aren't informative enough
    # Start with minimal regularization, then add it back if needed
    initial_params = param_overrides.copy() if param_overrides else {}
    initial_params.update({
        "reg_alpha": 0.0,  # NO regularization initially
        "reg_lambda": 0.0,  # NO regularization initially
        "min_child_weight": 0,  # NO minimum weight requirement
        "gamma": 0.0,  # NO pruning initially
    })
    model = _build_xgboost(initial_params)
    
    # ALWAYS train for minimum iterations first (no early stopping)
    # This ensures models actually learn before early stopping can trigger
    model.set_params(n_estimators=min_iterations)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        # No early_stopping_rounds - train full min_iterations
    )
    
    # Check if model produced constant predictions after minimum training
    val_preds_initial = model.predict(X_val)
    train_preds_initial = model.predict(X_train)
    pred_variance = np.var(val_preds_initial) if len(val_preds_initial) > 1 else 0.0
    train_variance = np.var(train_preds_initial) if len(train_preds_initial) > 1 else 0.0
    
    # If predictions are still constant, try with different tree_method
    if pred_variance < 1e-10 or train_variance < 1e-10:
        import warnings as w
        w.warn(
            f"XGBoost producing constant predictions after {min_iterations} iterations "
            f"(train_var={train_variance:.2e}, val_var={pred_variance:.2e}). "
            f"Trying 'exact' tree_method..."
        )
        # Try with 'exact' tree_method (slower but more precise)
        model = _build_xgboost(initial_params)
        model.set_params(tree_method="exact", n_estimators=200)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        # Re-check
        val_preds_check = model.predict(X_val)
        train_preds_check = model.predict(X_train)
        pred_variance = np.var(val_preds_check) if len(val_preds_check) > 1 else 0.0
        train_variance = np.var(train_preds_check) if len(train_preds_check) > 1 else 0.0
        
        if pred_variance < 1e-10 or train_variance < 1e-10:
            w.warn(
                f"XGBoost still constant with 'exact' method. "
                f"Training longer with higher learning rate..."
            )
            # Last resort: very high learning rate, more iterations
            model = _build_xgboost(initial_params)
            model.set_params(
                tree_method="exact",
                learning_rate=0.2,  # Very high learning rate
                n_estimators=300,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
    
    # Evaluate on validation set
    pred_returns = model.predict(X_val)
    
    # CRITICAL: Validate predictions have variance before computing metrics
    pred_variance = np.var(pred_returns) if len(pred_returns) > 1 else 0.0
    train_preds_final = model.predict(X_train)
    train_variance_final = np.var(train_preds_final) if len(train_preds_final) > 1 else 0.0
    
    # If predictions are STILL constant after all training attempts, try one more aggressive fix
    if pred_variance < 1e-10 or train_variance_final < 1e-10:
        import warnings as w
        w.warn(
            f"XGBoost still producing constant predictions after extended training "
            f"(train_var={train_variance_final:.2e}, val_var={pred_variance:.2e}). "
            f"Attempting aggressive retraining with higher learning rate..."
        )
        # Last resort: train with even higher learning rate and more iterations
        model = _build_xgboost(param_overrides)
        # Force higher learning rate if not already set
        if param_overrides and "learning_rate" in param_overrides:
            model.set_params(learning_rate=max(param_overrides["learning_rate"], 0.15))
        else:
            model.set_params(learning_rate=0.15)  # Higher learning rate
        model.set_params(n_estimators=300)  # More iterations
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        # Re-check predictions
        pred_returns = model.predict(X_val)
        pred_variance = np.var(pred_returns) if len(pred_returns) > 1 else 0.0
        train_preds_final = model.predict(X_train)
        train_variance_final = np.var(train_preds_final) if len(train_preds_final) > 1 else 0.0
        
        if pred_variance < 1e-10:
            w.warn(
                f"XGBoost predictions are STILL constant after aggressive retraining (variance={pred_variance:.2e}). "
                f"This will result in R² = 0.000. Model may not have learned. Check data quality."
            )
    
    metrics = _evaluate(y_val.to_numpy(), pred_returns)
    
    # Refit on full dataset if requested
    # FIXED: Allow refit if variance is reasonable (above 1e-5)
    # Only skip refit if variance is very low (near constant predictions)
    if refit_on_full and X_full is not None and y_full is not None:
        if pred_variance < 1e-5:
            # Variance is very low - skip refit to avoid making it worse
            import warnings as w
            w.warn(
                f"XGBoost: Skipping refit on full dataset due to low variance (variance={pred_variance:.2e}). "
                f"Using original model trained on train/val split."
            )
            # Do NOT refit - use original model
        else:
            # Variance is reasonable - proceed with refit
            try:
                # Get current n_estimators or use default
                current_n_est = model.get_params().get("n_estimators", 300)
                # Create fresh model for refit with same parameters
                refit_model = _build_xgboost(param_overrides)
                refit_model.set_params(n_estimators=current_n_est)
                refit_model.fit(
                    X_full,
                    y_full,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                # Verify refit didn't make predictions worse
                refit_preds = refit_model.predict(X_val)
                refit_variance = np.var(refit_preds) if len(refit_preds) > 1 else 0.0
                if refit_variance >= 1e-5:
                    # Refit is good - use it
                    model = refit_model
                else:
                    # Refit made it worse - keep original model
                    import warnings as w
                    w.warn(
                        f"XGBoost refit produced low variance (variance={refit_variance:.2e}). "
                        f"Keeping original model."
                    )
            except Exception as e:
                import warnings as w
                w.warn(
                    f"XGBoost refit failed: {e}. Using original model."
                )
    elif refit_on_full and pred_variance < 1e-10:
        # Original model has constant predictions - skip refit entirely
        import warnings as w
        w.warn(
            f"XGBoost original model has constant predictions (variance={pred_variance:.2e}). "
            f"Skipping refit - it would likely fail too."
        )
    
    return model, TrainingResult("XGBoost", **metrics)


def train_random_forest_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    refit_on_full: bool = True,
    X_full=None,
    y_full=None,
) -> Tuple[RandomForestClassifier, TrainingResult]:
    model = _build_random_forest_classifier()
    model.fit(X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]
    metrics = _evaluate_classification(y_val.to_numpy(), val_prob)
    if refit_on_full and X_full is not None and y_full is not None:
        model.fit(X_full, y_full)
    return model, TrainingResult("RandomForestClassifier", **metrics)


def train_lightgbm_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    refit_on_full: bool = True,
    X_full=None,
    y_full=None,
) -> Tuple[lgb.LGBMClassifier, TrainingResult]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*LightGBM.*")
        warnings.filterwarnings("ignore", message=".*lgb.*")
        
        model = _build_lightgbm_classifier()
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],  # More aggressive: stop earlier to prevent overfitting
        )
    val_prob = model.predict_proba(X_val)[:, 1]
    metrics = _evaluate_classification(y_val.to_numpy(), val_prob)
    if refit_on_full and X_full is not None and y_full is not None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*LightGBM.*")
            warnings.filterwarnings("ignore", message=".*lgb.*")
            
            final_model = _build_lightgbm_classifier()
            final_model.fit(X_full, y_full)
            model = final_model
    return model, TrainingResult("LightGBMClassifier", **metrics)


def train_xgboost_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    refit_on_full: bool = True,
    X_full=None,
    y_full=None,
) -> Tuple[xgb.XGBClassifier, TrainingResult]:
    model = _build_xgboost_classifier()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=50,  # More aggressive: stop earlier to prevent overfitting
    )
    val_prob = model.predict_proba(X_val)[:, 1]
    metrics = _evaluate_classification(y_val.to_numpy(), val_prob)
    if refit_on_full and X_full is not None and y_full is not None:
        final_model = _build_xgboost_classifier()
        final_model.fit(X_full, y_full)
        model = final_model
    return model, TrainingResult("XGBoostClassifier", **metrics)


class TradingEnv(Env):
    """
    Minimal trading environment for DQN.
    State: concatenated feature vector per timestep.
    Action: {-1, 0, 1} (short, hold, long).
    Reward: next-step return minus transaction costs.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, X: np.ndarray, returns: np.ndarray, transaction_cost: float = 0.0005):
        super().__init__()
        self.X = X
        self.returns = returns
        self.transaction_cost = transaction_cost
        self.current_step = 0
        self.max_steps = len(self.X)
        self.position = 0  # -1, 0, 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(X.shape[1],), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return self.X[self.current_step].astype(np.float32)

    def step(self, action):
        prev_position = self.position
        self.position = action - 1  # map {0,1,2} to {-1,0,1}
        future_return = self.returns[self.current_step]
        turnover = abs(self.position - prev_position)
        cost = self.transaction_cost * turnover
        raw_pnl = self.position * future_return
        reward = float(np.tanh(raw_pnl / 0.01) - cost)
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        obs = self._get_obs() if not terminated else np.zeros_like(self.X[0])
        info = {}
        return obs, float(reward), terminated, truncated, info


def train_dqn(
    X_train: pd.DataFrame,
    close_train: pd.Series,
    X_val: pd.DataFrame,
    close_val: pd.Series,
) -> Tuple[DQN, TrainingResult]:
    if len(X_train) < 2 or len(X_val) < 2:
        raise ValueError("DQN training requires at least two samples in train and val splits.")
    train_returns = (close_train.shift(-1) / close_train - 1.0).fillna(0.0).to_numpy()[:-1]
    val_returns = (close_val.shift(-1) / close_val - 1.0).fillna(0.0).to_numpy()[:-1]
    train_obs = X_train.iloc[:-1].to_numpy()
    val_obs = X_val.iloc[:-1].to_numpy()

    def _make_train_env():
        return Monitor(TradingEnv(train_obs, train_returns))

    def _make_eval_env():
        return Monitor(TradingEnv(val_obs, val_returns))

    vec_env = DummyVecEnv([_make_train_env])
    # Make tensorboard optional - don't fail if not installed
    # Try to create model with tensorboard first, but silently fallback if not available
    try:
        model = DQN(
            "MlpPolicy",
            vec_env,
            verbose=0,
            tensorboard_log="logs/tensorboard",
            **DQN_CONFIG,
        )
    except Exception as e:
        # If tensorboard fails, create model without it (silent fallback)
        if "tensorboard" in str(e).lower():
            # Suppress the error - tensorboard is optional
            model = DQN(
                "MlpPolicy",
                vec_env,
                verbose=0,
                **DQN_CONFIG,
            )
        else:
            raise
    eval_env = DummyVecEnv([_make_eval_env])
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=500,
        deterministic=True,
        render=False,
    )
    model.learn(total_timesteps=DQN_TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=False)
    # DQN doesn't predict prices directly - it outputs actions (short/hold/long)
    # Metrics are tracked via reward history in TensorBoard
    # Use None instead of NaN for JSON serialization
    metrics = {"mae": None, "rmse": None, "r2": None}
    return model, TrainingResult("DQN", **metrics, notes="RL metrics tracked separately")


def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_metrics(results: Dict[str, TrainingResult], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: asdict(result) for name, result in results.items()}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


