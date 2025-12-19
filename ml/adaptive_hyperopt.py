"""
Adaptive hyperparameter optimization that adjusts based on data characteristics.
This provides a permanent, balanced solution that adapts to data quality.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error, r2_score

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor


def _analyze_data_characteristics(
    X_train, y_train, X_val, y_val
) -> Dict[str, float]:
    """
    Analyze data characteristics to adaptively set hyperparameter ranges.
    Returns dict with signal_strength, noise_level, feature_richness, data_size_factor.
    """
    # Signal strength: How well can we predict? (baseline R² from simple model)
    from sklearn.linear_model import Ridge
    baseline = Ridge(alpha=1.0)
    baseline.fit(X_train, y_train)
    baseline_r2 = r2_score(y_val, baseline.predict(X_val))
    signal_strength = max(0.0, min(1.0, baseline_r2 + 0.2))  # Normalize to 0-1
    
    # Noise level: How much variance in target?
    target_std = float(np.std(y_train))
    target_mean_abs = float(np.abs(y_train).mean())
    noise_level = min(1.0, target_std / (target_mean_abs + 1e-10)) if target_mean_abs > 0 else 1.0
    
    # Feature richness: How many useful features?
    n_features = X_train.shape[1]
    n_samples = X_train.shape[0]
    feature_richness = min(1.0, n_features / max(50, n_samples / 10))  # Normalize
    
    # Data size factor: How much data do we have?
    data_size_factor = min(1.0, n_samples / 500)  # Normalize to 500 samples = 1.0
    
    return {
        "signal_strength": signal_strength,
        "noise_level": noise_level,
        "feature_richness": feature_richness,
        "data_size_factor": data_size_factor,
    }


def _get_adaptive_ranges(
    model_name: str, data_chars: Dict[str, float]
) -> Dict[str, Tuple[float, float]]:
    """
    Get adaptive hyperparameter ranges based on data characteristics.
    Returns dict of (min, max) tuples for each hyperparameter.
    """
    ss = data_chars["signal_strength"]  # 0-1, higher = better signal
    nl = data_chars["noise_level"]  # 0-1, higher = more noise
    fr = data_chars["feature_richness"]  # 0-1, higher = more features
    ds = data_chars["data_size_factor"]  # 0-1, higher = more data
    
    # Base ranges (balanced, proven to work)
    if model_name == "lightgbm":
        # Learning rate: Lower for noisy data, higher for clean signal
        lr_min = 0.05 + (nl * 0.02)  # 0.05-0.07
        lr_max = 0.08 + (ss * 0.02)  # 0.08-0.10
        
        # Depth: Deeper for more data, shallower for noisy data
        depth_min = 3
        depth_max = 3 + int(ds * 1.5) + int((1 - nl) * 0.5)  # 3-5
        
        # Regularization: Higher for noisy data, lower for clean signal
        reg_alpha_min = 0.8 + (nl * 1.2)  # 0.8-2.0
        reg_alpha_max = 1.5 + (nl * 1.5)  # 1.5-3.0
        reg_lambda_min = 1.5 + (nl * 1.0)  # 1.5-2.5
        reg_lambda_max = 2.5 + (nl * 1.5)  # 2.5-4.0
        
        # Subsample: More data for noisy, less for clean
        subsample_min = 0.72 + (nl * 0.05)  # 0.72-0.77
        subsample_max = 0.80 + ((1 - nl) * 0.05)  # 0.80-0.85
        
        return {
            "learning_rate": (lr_min, lr_max),
            "max_depth": (depth_min, depth_max),
            "num_leaves": (8, min(31, 2 ** depth_max)),
            "n_estimators": (100, 250),
            "subsample": (subsample_min, subsample_max),
            "colsample_bytree": (0.70, 0.80),
            "min_child_samples": (12, 20),
            "reg_alpha": (reg_alpha_min, reg_alpha_max),
            "reg_lambda": (reg_lambda_min, reg_lambda_max),
            "min_split_gain": (0.01, 0.04),
        }
    
    elif model_name == "xgboost":
        # Learning rate: Balanced range
        lr_min = 0.05 + (nl * 0.01)  # 0.05-0.06
        lr_max = 0.07 + (ss * 0.01)  # 0.07-0.08
        
        # Depth: Moderate, adaptive
        depth_min = 2
        depth_max = 2 + int(ds * 1.0) + int((1 - nl) * 0.5)  # 2-4
        
        # Regularization: Adaptive to noise
        reg_alpha_min = 1.5 + (nl * 1.5)  # 1.5-3.0
        reg_alpha_max = 3.0 + (nl * 2.0)  # 3.0-5.0
        reg_lambda_min = 2.5 + (nl * 1.5)  # 2.5-4.0
        reg_lambda_max = 4.0 + (nl * 2.0)  # 4.0-6.0
        
        return {
            "learning_rate": (lr_min, lr_max),
            "max_depth": (depth_min, depth_max),
            "n_estimators": (100, 250),
            "subsample": (0.70, 0.80),
            "colsample_bytree": (0.65, 0.75),
            "min_child_weight": (3.0, 7.0),
            "gamma": (0.10, 0.25),
            "reg_alpha": (reg_alpha_min, reg_alpha_max),
            "reg_lambda": (reg_lambda_min, reg_lambda_max),
        }
    
    elif model_name == "random_forest":
        # Depth: Adaptive to data size and noise
        depth_min = 5 + int(ds * 1.0)  # 5-6
        depth_max = 7 + int(ds * 1.5) + int((1 - nl) * 1.0)  # 7-10
        
        # Regularization: Adaptive to noise
        ccp_alpha_min = 0.0001 + (nl * 0.0002)  # 0.0001-0.0003
        ccp_alpha_max = 0.0003 + (nl * 0.0003)  # 0.0003-0.0006
        
        return {
            "n_estimators": (200, 300),
            "max_depth": (depth_min, depth_max),
            "min_samples_leaf": (2, 5),
            "min_samples_split": (5, 10),
            "max_features": (0.65, 0.80),
            "max_samples": (0.75, 0.90),
            "ccp_alpha": (ccp_alpha_min, ccp_alpha_max),
        }
    
    raise ValueError(f"Unknown model: {model_name}")


def _compute_adaptive_overfitting_penalty(
    y_train, y_val, train_pred, val_pred, data_chars: Dict[str, float]
) -> float:
    """
    Adaptive overfitting penalty that adjusts threshold based on data characteristics.
    """
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    gap = train_r2 - val_r2
    
    # Adaptive threshold: Stricter for clean data, more lenient for noisy data
    noise_level = data_chars["noise_level"]
    adaptive_threshold = 0.03 + (noise_level * 0.02)  # 0.03-0.05
    
    if gap > adaptive_threshold:
        # Penalty scales with gap and data quality
        penalty_multiplier = 20.0 + ((1 - noise_level) * 10.0)  # 20-30
        penalty = 1.0 + (gap - adaptive_threshold) * penalty_multiplier
        return float(penalty)
    
    if val_r2 < 0:
        penalty = 1.0 + abs(val_r2) * 3.0
        return float(penalty)
    
    return 1.0


def optimize_lightgbm_adaptive(
    X_train, y_train, X_val, y_val, n_trials: int = 20, timeout: Optional[int] = None
) -> Dict:
    """Adaptive LightGBM optimization."""
    data_chars = _analyze_data_characteristics(X_train, y_train, X_val, y_val)
    ranges = _get_adaptive_ranges("lightgbm", data_chars)
    
    def objective(trial: optuna.Trial) -> float:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            max_depth = trial.suggest_int("max_depth", int(ranges["max_depth"][0]), int(ranges["max_depth"][1]))
            max_leaves = min(2 ** max_depth, int(ranges["num_leaves"][1]))
            min_leaves = max(8, max_leaves - 5)
            num_leaves = trial.suggest_int("num_leaves", min_leaves, max_leaves)
            
            params = {
                "boosting_type": "gbdt",
                "n_estimators": trial.suggest_int("n_estimators", int(ranges["n_estimators"][0]), int(ranges["n_estimators"][1])),
                "learning_rate": trial.suggest_float("learning_rate", ranges["learning_rate"][0], ranges["learning_rate"][1], log=True),
                "num_leaves": num_leaves,
                "max_depth": max_depth,
                "subsample": trial.suggest_float("subsample", ranges["subsample"][0], ranges["subsample"][1]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", ranges["colsample_bytree"][0], ranges["colsample_bytree"][1]),
                "min_child_samples": trial.suggest_int("min_child_samples", int(ranges["min_child_samples"][0]), int(ranges["min_child_samples"][1])),
                "reg_alpha": trial.suggest_float("reg_alpha", ranges["reg_alpha"][0], ranges["reg_alpha"][1]),
                "reg_lambda": trial.suggest_float("reg_lambda", ranges["reg_lambda"][0], ranges["reg_lambda"][1]),
                "min_split_gain": trial.suggest_float("min_split_gain", ranges["min_split_gain"][0], ranges["min_split_gain"][1]),
                "random_state": 42,
                "force_row_wise": True,
                "verbose": -1,
            }
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="l2", callbacks=[lgb.log_evaluation(period=0)])
            
            val_preds = model.predict(X_val)
            train_preds = model.predict(X_train)
            
            pred_variance = np.var(val_preds) if len(val_preds) > 1 else 0.0
            if pred_variance < 1e-10:
                return 1e10
            
            val_mae = mean_absolute_error(y_val, val_preds)
            penalty = _compute_adaptive_overfitting_penalty(y_train, y_val, train_preds, val_preds, data_chars)
            return val_mae * penalty
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params


def optimize_xgboost_adaptive(
    X_train, y_train, X_val, y_val, n_trials: int = 20, timeout: Optional[int] = None
) -> Dict:
    """Adaptive XGBoost optimization."""
    data_chars = _analyze_data_characteristics(X_train, y_train, X_val, y_val)
    ranges = _get_adaptive_ranges("xgboost", data_chars)
    
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", int(ranges["n_estimators"][0]), int(ranges["n_estimators"][1])),
            "learning_rate": trial.suggest_float("learning_rate", ranges["learning_rate"][0], ranges["learning_rate"][1], log=True),
            "max_depth": trial.suggest_int("max_depth", int(ranges["max_depth"][0]), int(ranges["max_depth"][1])),
            "subsample": trial.suggest_float("subsample", ranges["subsample"][0], ranges["subsample"][1]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", ranges["colsample_bytree"][0], ranges["colsample_bytree"][1]),
            "min_child_weight": trial.suggest_float("min_child_weight", ranges["min_child_weight"][0], ranges["min_child_weight"][1]),
            "gamma": trial.suggest_float("gamma", ranges["gamma"][0], ranges["gamma"][1]),
            "reg_alpha": trial.suggest_float("reg_alpha", ranges["reg_alpha"][0], ranges["reg_alpha"][1]),
            "reg_lambda": trial.suggest_float("reg_lambda", ranges["reg_lambda"][0], ranges["reg_lambda"][1]),
            "objective": "reg:squarederror",
            "tree_method": "approx",
            "random_state": 42,
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        val_preds = model.predict(X_val)
        pred_variance = np.var(val_preds) if len(val_preds) > 1 else 0.0
        if pred_variance < 1e-10:
            return 1e10
        
        train_preds = model.predict(X_train)
        val_mae = mean_absolute_error(y_val, val_preds)
        penalty = _compute_adaptive_overfitting_penalty(y_train, y_val, train_preds, val_preds, data_chars)
        return val_mae * penalty
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params


def optimize_random_forest_adaptive(
    X_train, y_train, X_val, y_val, n_trials: int = 20, timeout: Optional[int] = None
) -> Dict:
    """Adaptive Random Forest optimization."""
    data_chars = _analyze_data_characteristics(X_train, y_train, X_val, y_val)
    ranges = _get_adaptive_ranges("random_forest", data_chars)
    
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", int(ranges["n_estimators"][0]), int(ranges["n_estimators"][1])),
            "max_depth": trial.suggest_int("max_depth", int(ranges["max_depth"][0]), int(ranges["max_depth"][1])),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", int(ranges["min_samples_leaf"][0]), int(ranges["min_samples_leaf"][1])),
            "min_samples_split": trial.suggest_int("min_samples_split", int(ranges["min_samples_split"][0]), int(ranges["min_samples_split"][1])),
            "max_features": trial.suggest_float("max_features", ranges["max_features"][0], ranges["max_features"][1]),
            "max_samples": trial.suggest_float("max_samples", ranges["max_samples"][0], ranges["max_samples"][1]),
            "ccp_alpha": trial.suggest_float("ccp_alpha", ranges["ccp_alpha"][0], ranges["ccp_alpha"][1]),
            "random_state": 42,
            "n_jobs": 1,  # Safe default
            "bootstrap": True,
        }
        
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        val_preds = model.predict(X_val)
        pred_variance = np.var(val_preds) if len(val_preds) > 1 else 0.0
        if pred_variance < 1e-10:
            return 1e10
        
        train_preds = model.predict(X_train)
        val_mae = mean_absolute_error(y_val, val_preds)
        penalty = _compute_adaptive_overfitting_penalty(y_train, y_val, train_preds, val_preds, data_chars)
        return val_mae * penalty
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params
