"""
Optuna-based hyperparameter search helpers.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

# Suppress LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", message=".*LightGBM.*")
warnings.filterwarnings("ignore", message=".*lgb.*")


def _mae_metric(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials: int, timeout: Optional[int]):
    def objective(trial: optuna.Trial) -> float:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*LightGBM.*")
            warnings.filterwarnings("ignore", message=".*lgb.*")
            
            params = {
                "boosting_type": "gbdt",
                "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 64),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),
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
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),  # Disable evaluation logging
                ],
            )
            preds = model.predict(X_val)
        return _mae_metric(y_val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params


def _optimize_xgboost(X_train, y_train, X_val, y_val, n_trials: int, timeout: Optional[int]):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 0.4),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
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
            early_stopping_rounds=50,
        )
        preds = model.predict(X_val)
        return _mae_metric(y_val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params


def _optimize_random_forest(X_train, y_train, X_val, y_val, n_trials: int, timeout: Optional[int]):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 200),
            "max_features": trial.suggest_float("max_features", 0.3, 0.9),
            "max_samples": trial.suggest_float("max_samples", 0.5, 0.95),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.01),
            "random_state": 42,
            "n_jobs": -1,
            "bootstrap": True,
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return _mae_metric(y_val, preds)

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


