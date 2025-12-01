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
    Conservative RandomForest tuned for noisy financial targets with stronger regularization.
    """
    params = {
        "n_estimators": 300,
        "max_depth": 4,
        "min_samples_leaf": 20,
        "min_samples_split": 40,
        "max_features": 0.5,
        "max_samples": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "bootstrap": True,
        "ccp_alpha": 0.001,
    }
    if custom_params:
        params.update(custom_params)
    return RandomForestRegressor(**params)


def _build_random_forest_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=15,
        min_samples_split=40,
        max_features=0.7,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
    )


def _build_lightgbm(custom_params: Optional[Dict] = None) -> lgb.LGBMRegressor:
    """
    LightGBM configuration with stronger regularization for financial data.
    """
    params = {
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 3,
        "num_leaves": 15,
        "subsample": 0.75,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "min_child_samples": 20,
        "min_split_gain": 0.02,
        "max_bin": 255,
        "extra_trees": False,
        "min_data_in_bin": 3,
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
    XGBoost configuration with stronger regularization for financial data.
    """
    params = {
        "n_estimators": 500,
        "learning_rate": 0.025,
        "max_depth": 3,
        "subsample": 0.75,
        "colsample_bytree": 0.65,
        "reg_alpha": 1.0,
        "reg_lambda": 2.5,
        "min_child_weight": 5,
        "gamma": 0.1,
        "objective": "reg:squarederror",
        "random_state": 42,
        "tree_method": "hist",
        "max_bin": 256,
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
    model = _build_random_forest(param_overrides)
    model.fit(X_train, y_train)
    
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
        
        model = _build_lightgbm(param_overrides)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),  # Increased from 50
                lgb.log_evaluation(period=0),  # Disable verbose output
            ],
        )
    
    # Evaluate on validation set
    pred_returns = model.predict(X_val)
    metrics = _evaluate(y_val.to_numpy(), pred_returns)
    
    # Refit on full dataset if requested (without early stopping to use all data)
    if refit_on_full and X_full is not None and y_full is not None:
        # Use a fresh model with same config but no early stopping for final fit
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*LightGBM.*")
            warnings.filterwarnings("ignore", message=".*lgb.*")
            
            final_model = _build_lightgbm(param_overrides)
            final_model.fit(
                X_full,
                y_full,
                eval_set=[(X_val, y_val)],
                eval_metric="l2",
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
            )
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
    """
    model = _build_xgboost(param_overrides)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=100,  # Increased from 50 for more patience
    )
    
    # Evaluate on validation set
    pred_returns = model.predict(X_val)
    metrics = _evaluate(y_val.to_numpy(), pred_returns)
    
    # Refit on full dataset if requested
    if refit_on_full and X_full is not None and y_full is not None:
        # Use a fresh model with same config for final fit
        final_model = _build_xgboost(param_overrides)
        final_model.fit(
            X_full,
            y_full,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=100,
        )
        model = final_model
    
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
            callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)],
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
        early_stopping_rounds=75,
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
    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=0,
        tensorboard_log="logs/tensorboard",
        **DQN_CONFIG,
    )
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


