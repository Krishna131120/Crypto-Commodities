"""
Optuna-based hyperparameter search helpers with overfitting prevention.
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

# Suppress LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", message=".*LightGBM.*")
warnings.filterwarnings("ignore", message=".*lgb.*")


def _mae_metric(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _compute_overfitting_penalty(y_train, y_val, train_pred, val_pred) -> float:
    """
    Compute penalty for overfitting based on train/val gap.
    Returns penalty factor (1.0 = no penalty, >1.0 = penalty applied).
    """
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    gap = train_r2 - val_r2
    
    # Penalize if gap > 0.10 (10%)
    if gap > 0.10:
        # Exponential penalty: gap of 0.15 = 1.5x penalty, gap of 0.20 = 2.0x penalty
        penalty = 1.0 + (gap - 0.10) * 10.0
        return float(penalty)
    return 1.0


def _optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials: int, timeout: Optional[int]):
    def objective(trial: optuna.Trial) -> float:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*LightGBM.*")
            warnings.filterwarnings("ignore", message=".*lgb.*")
            
            # VERY conservative parameter ranges to prevent overfitting
            max_depth = trial.suggest_int("max_depth", 2, 4)  # Further reduced: max 4 instead of 5
            # Constrain num_leaves based on max_depth to prevent overfitting
            max_leaves = min(2 ** max_depth, 24)  # Further reduced: cap at 24 instead of 32
            num_leaves = trial.suggest_int("num_leaves", 8, max_leaves)
            
            params = {
                "boosting_type": "gbdt",
                "n_estimators": trial.suggest_int("n_estimators", 150, 400),  # Further reduced: max 400 instead of 600
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.03, log=True),  # Further reduced: max 0.03 instead of 0.05
                "num_leaves": num_leaves,
                "max_depth": max_depth,
                "subsample": trial.suggest_float("subsample", 0.65, 0.80),  # Further reduced: tighter range
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.75),  # Further reduced: tighter range
                "min_child_samples": trial.suggest_int("min_child_samples", 25, 50),  # Increased min: 25 instead of 20
                "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 5.0),  # Increased min: 1.0 instead of 0.5
                "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 6.0),  # Increased min: 2.0 instead of 1.0
                "min_split_gain": trial.suggest_float("min_split_gain", 0.02, 0.1),  # Increased min: 0.02 instead of 0.01
                "random_state": 42,
                "force_row_wise": True,
                "verbose": -1,  # Suppress LightGBM output
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="l2",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20, verbose=False),  # Further reduced: 20 instead of 30
                    lgb.log_evaluation(period=0),  # Disable evaluation logging
                ],
            )
            val_preds = model.predict(X_val)
            train_preds = model.predict(X_train)
            
            # Compute MAE with overfitting penalty
            val_mae = _mae_metric(y_val, val_preds)
            penalty = _compute_overfitting_penalty(y_train, y_val, train_preds, val_preds)
            return val_mae * penalty

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params


def _optimize_xgboost(X_train, y_train, X_val, y_val, n_trials: int, timeout: Optional[int]):
    def objective(trial: optuna.Trial) -> float:
        # VERY conservative parameter ranges to prevent overfitting
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 400),  # Further reduced: max 400 instead of 600
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.03, log=True),  # Further reduced: max 0.03 instead of 0.05
            "max_depth": trial.suggest_int("max_depth", 2, 4),  # Further reduced: max 4 instead of 5
            "subsample": trial.suggest_float("subsample", 0.65, 0.80),  # Further reduced: tighter range
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.75),  # Further reduced: tighter range
            "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 12.0),  # Increased min: 5.0 instead of 3.0
            "gamma": trial.suggest_float("gamma", 0.1, 0.35),  # Increased min: 0.1 instead of 0.05
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 5.0),  # Increased min: 1.0 instead of 0.5
            "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 6.0),  # Increased min: 2.0 instead of 1.0
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 42,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=20,  # Further reduced: 20 instead of 30
        )
        val_preds = model.predict(X_val)
        train_preds = model.predict(X_train)
        
        # Compute MAE with overfitting penalty
        val_mae = _mae_metric(y_val, val_preds)
        penalty = _compute_overfitting_penalty(y_train, y_val, train_preds, val_preds)
        return val_mae * penalty

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params


def _optimize_random_forest(X_train, y_train, X_val, y_val, n_trials: int, timeout: Optional[int]):
    def objective(trial: optuna.Trial) -> float:
        # VERY conservative parameter ranges to prevent overfitting
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 350),  # Further reduced: max 350 instead of 500
            "max_depth": trial.suggest_int("max_depth", 3, 5),  # Further reduced: max 5 instead of 6
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 30, 70),  # Increased min: 30 instead of 20
            "min_samples_split": trial.suggest_int("min_samples_split", 50, 180),  # Increased min: 50 instead of 30
            "max_features": trial.suggest_float("max_features", 0.3, 0.6),  # Further reduced: max 0.6 instead of 0.7
            "max_samples": trial.suggest_float("max_samples", 0.6, 0.80),  # Further reduced: max 0.80 instead of 0.85
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.002, 0.015),  # Increased min: 0.002 instead of 0.001
            "random_state": 42,
            "n_jobs": -1,
            "bootstrap": True,
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        train_preds = model.predict(X_train)
        
        # Compute MAE with overfitting penalty
        val_mae = _mae_metric(y_val, val_preds)
        penalty = _compute_overfitting_penalty(y_train, y_val, train_preds, val_preds)
        return val_mae * penalty

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
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
    """
    if n_trials <= 0:
        return {}
    model_name = model_name.lower()
    if model_name == "lightgbm":
        return _optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials, timeout)
    if model_name == "xgboost":
        return _optimize_xgboost(X_train, y_train, X_val, y_val, n_trials, timeout)
    if model_name == "random_forest":
        return _optimize_random_forest(X_train, y_train, X_val, y_val, n_trials, timeout)
    raise ValueError(f"Unsupported model for hyperopt: {model_name}")


