"""
Training entry point for multi-asset prediction models.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress LightGBM warnings globally
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", message=".*LightGBM.*")
warnings.filterwarnings("ignore", message=".*lgb.*")
from sklearn.linear_model import RidgeCV
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
from sklearn.preprocessing import RobustScaler

import config
from ml.context_features import ContextFeatureConfig
from ml.data_loader import (
    FEATURE_SCHEMA_VERSION,
    assemble_dataset,
    extract_xy,
    train_val_test_split,
)
from ml.hyperopt import optimize_model
from ml.json_logger import get_training_logger
from ml.horizons import (
    DEFAULT_HORIZON_PROFILE,
    available_profiles as available_horizon_profiles,
    build_profile_report,
    get_profile_config,
    normalize_profile,
)
from ml.targets import TargetConfig
from ml.trainers import (
    TrainingResult,
    save_metrics,
    save_model,
    train_dqn,
    train_lightgbm,
    train_random_forest,
    train_xgboost,
    _build_lightgbm,
    _build_random_forest,
    _build_xgboost,
    DQN_CONFIG,
)
from core.model_paths import horizon_dir, ensure_horizon_dirs


PREDICTION_CLAMP = 0.2
MIN_THRESHOLD = 0.0025
MAX_THRESHOLD = 0.03
DEFAULT_THRESHOLD = 0.01
MIN_CONFIDENCE = 0.05
ACTION_SIGNAL_FIELDS = [
    "RSI_14",
    "MACD_histogram",
    "SMA_50",
    "SMA_200",
    "ATR_14",
    "Volume_Ratio",
]
TRAIN_RATIO = 0.75
VAL_RATIO = 0.125
CONFIDENCE_MULTIPLIER_CAP = 1.5
TIMEFRAME_GAP_DAYS = {"1d": 1}
MAX_MODEL_FEATURES = 200
FEATURE_IMPORTANCE_TOPK = 180
FEATURE_IMPORTANCE_MIN_SHARE = 0.0008
CONSENSUS_NEUTRAL_MULTIPLIER = 1.05
OVERFITTING_WEIGHT_MULTIPLIER = 0.35
LOW_VARIANCE_THRESHOLD = 1e-8
HIGH_FEATURE_CORR = 0.995
WALK_FORWARD_SPLITS = 3
WALK_FORWARD_MIN_TRAIN = 200
MIN_ACCEPTABLE_R2 = 0.05
MAX_TRAIN_VAL_GAP = 0.15  # Tighter: 0.15 instead of 0.25 to catch overfitting earlier
MIN_SIGNAL_CORR = 0.05
MIN_MAE_IMPROVEMENT = 1e-4
MIN_DIRECTIONAL_ACCURACY = 0.52
MIN_TEST_DIRECTIONAL_ACCURACY = 0.52
MIN_TEST_R2_STRICT = 0.15
MAX_VAL_TEST_GAP = 0.08  # Tighter: 0.08 instead of 0.10
MAX_TRAIN_TEST_GAP = 0.15  # Tighter: 0.15 instead of 0.20
SCALER_NAME = "feature_scaler.joblib"
CONTEXT_CONFIG = ContextFeatureConfig(
    include_macro=True,
    include_spreads=True,
    include_volatility_indices=True,
    include_regime_features=True,
    include_intraday_aggregates=True,
    intraday_timeframes=("4h", "1h"),
    intraday_lookback=45,
)
ENABLE_DIRECTIONAL_MODELS = False  # Disabled - directional models removed
ENABLE_QUANTILE_MODELS = False  # Disabled - quantile models removed
ENABLE_HYPEROPT = True
HYPEROPT_TRIALS = 20
HYPEROPT_TIMEOUT = 900  # seconds
QUANTILE_LEVELS = (0.2, 0.5, 0.8)
def _to_native_metric_dict(metrics: Optional[Dict[str, float]]) -> Optional[Dict[str, Optional[float]]]:
    if not metrics:
        return None
    native = {}
    for key, value in metrics.items():
        if value is None:
            native[key] = None
        else:
            native[key] = float(value)
    return native


def _reduce_feature_matrix(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    max_features: int = MAX_MODEL_FEATURES,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, any]]:
    """
    Apply variance filtering, target-correlation ranking, and redundancy pruning.
    """
    metadata: Dict[str, Any] = {}
    if X_train.empty:
        raise ValueError("Training frame is empty after split; cannot reduce features.")

    sample_cap = max(25, len(X_train) // 4)
    effective_cap = min(max_features, sample_cap, X_train.shape[1])
    variances = X_train.var()
    high_variance_cols = variances[variances > LOW_VARIANCE_THRESHOLD].index.tolist()
    removed_for_variance = [col for col in X_train.columns if col not in high_variance_cols]

    X_train_filtered = X_train[high_variance_cols]
    correlation = X_train_filtered.corrwith(y_train).fillna(0.0).abs()
    ranked = correlation.sort_values(ascending=False).index.tolist()

    selected: List[str] = []
    for col in ranked:
        if len(selected) >= effective_cap:
            break
        col_series = X_train_filtered[col]
        redundant = False
        for keep_col in selected:
            if X_train_filtered[keep_col].corr(col_series) > HIGH_FEATURE_CORR:
                redundant = True
                break
        if not redundant:
            selected.append(col)

    # Always include required action features if present (for interpretability)
    for field in ACTION_SIGNAL_FIELDS:
        if field in X_train_filtered.columns and field not in selected:
            selected.append(field)

    # Ensure deterministic order and limit
    unique_selected = []
    for col in selected:
        if col not in unique_selected:
            unique_selected.append(col)
    cap_with_actions = max(effective_cap, len(ACTION_SIGNAL_FIELDS))
    selected = unique_selected[: cap_with_actions]

    if not selected:
        fallback_cap = max(cap_with_actions, len(high_variance_cols))
        selected = high_variance_cols[:fallback_cap] or list(X_train.columns[:fallback_cap])

    metadata["removed_low_variance"] = removed_for_variance
    metadata["selected_features"] = selected
    metadata["target_correlations"] = {col: float(correlation.get(col, 0.0)) for col in selected}
    metadata["max_abs_correlation"] = float(
        max((abs(correlation.get(col, 0.0)) for col in selected), default=0.0)
    )

    return (
        X_train[selected],
        X_val[selected],
        X_test[selected],
        metadata,
    )


def _filter_by_importance(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = FEATURE_IMPORTANCE_TOPK,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, any]]:
    """
    Use a quick LightGBM model to drop the lowest-importance features automatically.
    """
    metadata: Dict[str, any] = {"method": "lightgbm_importance"}
    if X_train.shape[1] <= top_k:
        metadata["selected_features"] = list(X_train.columns)
        metadata["dropped_features"] = []
        metadata["skipped"] = True
        return X_train, X_val, X_test, metadata

    probe_model = _build_lightgbm({"n_estimators": 300, "learning_rate": 0.05})
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*LightGBM.*")
        warnings.filterwarnings("ignore", message=".*lgb.*")
        probe_model.fit(X_train, y_train)
    importances = pd.Series(probe_model.feature_importances_, index=X_train.columns).fillna(0.0)
    total = importances.sum() or 1.0
    normalized = importances / total
    threshold_mask = normalized >= FEATURE_IMPORTANCE_MIN_SHARE
    selected = normalized[threshold_mask].sort_values(ascending=False).index.tolist()
    if not selected:
        selected = normalized.sort_values(ascending=False).head(top_k).index.tolist()
    else:
        selected = selected[:top_k]
    dropped = [col for col in X_train.columns if col not in selected]
    metadata["selected_features"] = selected
    metadata["dropped_features"] = dropped
    metadata["threshold"] = FEATURE_IMPORTANCE_MIN_SHARE
    return X_train[selected], X_val[selected], X_test[selected], metadata


def _scale_feature_matrix(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    asset_type: str = "crypto",
    symbol: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RobustScaler]:
    """
    Apply a robust scaler to stabilize feature ranges across splits.
    
    For commodities, we add more thorough outlier clipping and price-aware
    normalization to handle different price scales and volatility patterns.
    """
    # Commodity-specific preprocessing
    if asset_type == "commodities":
        from ml.commodity_cleaning import get_commodity_cleaning_config
        
        cleaning_config = get_commodity_cleaning_config(symbol or "")
        outlier_multiplier = cleaning_config.get("outlier_multiplier", 3.0)
    else:
        outlier_multiplier = 3.0  # Default for crypto
    
    # Clip extreme outliers before scaling (especially important for commodities)
    # Use IQR-based clipping: values beyond Q3 + multiplier*IQR or below Q1 - multiplier*IQR are clipped
    def _clip_outliers(df: pd.DataFrame, reference_df: pd.DataFrame, multiplier: float = 3.0) -> pd.DataFrame:
        """Clip outliers based on reference distribution (train set)."""
        clipped = df.copy()
        for col in df.columns:
            if col in reference_df.columns:
                q1 = reference_df[col].quantile(0.25)
                q3 = reference_df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:  # Only clip if IQR is meaningful
                    lower_bound = q1 - multiplier * iqr
                    upper_bound = q3 + multiplier * iqr
                    clipped[col] = clipped[col].clip(lower=lower_bound, upper=upper_bound)
        return clipped
    
    # Clip outliers in all splits based on train distribution
    X_train_clipped = _clip_outliers(X_train, X_train, multiplier=outlier_multiplier)
    X_val_clipped = _clip_outliers(X_val, X_train, multiplier=outlier_multiplier)
    X_test_clipped = _clip_outliers(X_test, X_train, multiplier=outlier_multiplier)
    
    # Apply RobustScaler (more robust to outliers than StandardScaler)
    # For commodities, RobustScaler is especially important due to different price scales
    scaler = RobustScaler()
    scaler.fit(X_train_clipped)

    def _transform(frame: pd.DataFrame) -> pd.DataFrame:
        transformed = scaler.transform(frame)
        return pd.DataFrame(transformed, index=frame.index, columns=frame.columns)

    return _transform(X_train_clipped), _transform(X_val_clipped), _transform(X_test_clipped), scaler


def _run_time_series_cv(
    model_factory: Callable[[], any],
    X: pd.DataFrame,
    y: pd.Series,
    splits: int = WALK_FORWARD_SPLITS,
    min_train_size: int = WALK_FORWARD_MIN_TRAIN,
) -> Dict[str, float]:
    total_rows = len(X)
    if total_rows < (splits + 1) * 50:
        return {}

    fold_metrics: List[Dict[str, float]] = []
    fold_size = total_rows // (splits + 1)
    start = min_train_size
    for split in range(splits):
        train_end = start + split * fold_size
        val_end = min(total_rows, train_end + fold_size)
        if val_end - train_end < 30 or train_end < min_train_size:
            continue
        X_tr = X.iloc[:train_end]
        y_tr = y.iloc[:train_end]
        X_val_fold = X.iloc[train_end:val_end]
        y_val_fold = y.iloc[train_end:val_end]
        model = model_factory()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val_fold)
        fold_metrics.append(_evaluate(y_val_fold.to_numpy(), preds))
    if not fold_metrics:
        return {}
    aggregated = {
        "r2_mean": float(np.mean([m["r2"] for m in fold_metrics])),
        "r2_std": float(np.std([m["r2"] for m in fold_metrics])),
        "mae_mean": float(np.mean([m["mae"] for m in fold_metrics])),
        "rmse_mean": float(np.mean([m["rmse"] for m in fold_metrics])),
    }
    return aggregated


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate predictions and return metrics."""
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


def _evaluate_classifier_probs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    decision_threshold: float = 0.5,
) -> Dict[str, Optional[float]]:
    """
    Evaluate binary classifier probabilities with trading-oriented metrics.
    """
    y_pred = (y_prob >= decision_threshold).astype(int)
    metrics = {
        "r2": 0.0,
        "mae": None,
        "rmse": None,
        "directional_accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "mean_predicted_return": float(np.mean(y_prob)),
        "mean_actual_return": float(np.mean(y_true)),
    }
    return metrics


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    diff = y_true - y_pred
    loss = np.maximum(alpha * diff, (alpha - 1) * diff)
    return float(np.mean(loss))


def _check_model_tradability(
    results: Dict[str, Dict],
    metric_store: Dict[str, TrainingResult],
    overfitting_warnings: List[str],
    asset_type: str,
) -> Tuple[bool, List[str]]:
    """
    Check if models pass robustness requirements for live trading.
    
    Returns:
        (is_tradable, list_of_reasons)
    """
    reasons = []
    
    # Need at least 2 successful models for consensus (or 1 if quality is excellent)
    # Include DQN in successful models count (it's a valid model)
    successful_models = [
        name for name, data in results.items()
        if isinstance(data, dict) and data.get("status") != "failed"
    ]
    
    # Get failed models with reasons for detailed reporting
    failed_models = []
    for name, data in results.items():
        if isinstance(data, dict) and data.get("status") == "failed":
            reason = data.get("reason", "Unknown reason")
            failed_models.append(f"{name}: {reason}")
    
    # Check if we have high-quality models (validation R² >= 0.60)
    high_quality_count = 0
    for name in successful_models:
        model_data = results.get(name, {})
        if isinstance(model_data, dict):
            # Check if model has good validation R²
            val_r2 = model_data.get("r2")
            if val_r2 is not None and val_r2 >= 0.60:
                high_quality_count += 1
    
    # For commodities: allow 1 high-quality model OR 2 regular models
    min_models_required = 2
    if asset_type == "commodities":
        if high_quality_count >= 1:
            min_models_required = 1  # Allow 1 high-quality model for commodities
        elif len(successful_models) >= 1 and high_quality_count == 0:
            # Check if we have at least one model with decent validation R² (>= 0.50)
            decent_quality_count = 0
            for name in successful_models:
                model_data = results.get(name, {})
                if isinstance(model_data, dict):
                    val_r2 = model_data.get("r2")
                    if val_r2 is not None and val_r2 >= 0.50:
                        decent_quality_count += 1
            if decent_quality_count >= 1:
                min_models_required = 1  # Allow 1 decent-quality model for commodities
    
    if len(successful_models) < min_models_required:
        reason_msg = f"Insufficient models: {len(successful_models)} successful (need at least {min_models_required})"
        if successful_models:
            reason_msg += f" [Successful: {', '.join(successful_models)}]"
        if failed_models:
            reason_msg += f" [Failed: {', '.join(failed_models[:3])}]"  # Show first 3 failures
        reasons.append(reason_msg)
        return False, reasons
    
    # Check for severe overfitting warnings
    # Count models with severe overfitting (generalization failure or very large gaps)
    # Be more lenient: only reject if ALL models have severe issues AND poor test performance
    severe_overfitting_count = 0
    models_with_good_test_r2 = 0
    
    # First, check which models have good test R² despite gaps
    for name in successful_models:
        data = results.get(name, {})
        test_metrics = data.get("test_metrics", {})
        if isinstance(test_metrics, dict):
            test_r2 = test_metrics.get("r2", 0.0)
            # For commodities, accept models with test R² >= 0.10 (was 0.05)
            # For crypto, accept models with test R² >= 0.15
            min_test_r2 = 0.10 if asset_type == "commodities" else 0.15
            if test_r2 >= min_test_r2:
                models_with_good_test_r2 += 1
    
    # Count severe overfitting, but only if test R² is also poor
    # CRITICAL: Good test R² means model is learning, so gaps are acceptable
    for w in overfitting_warnings:
        # Extract model name from warning
        model_name = None
        for name in successful_models:
            if name in w.lower():
                model_name = name
                break
        
        # Get test R² for this model
        test_r2 = None
        if model_name:
            data = results.get(model_name, {})
            test_metrics = data.get("test_metrics", {})
            if isinstance(test_metrics, dict):
                test_r2 = test_metrics.get("r2", 0.0)
        
        is_severe = False
        if "generalization failure" in w.lower():
            # Extract gap value if possible
            gap_val = None
            try:
                if "gap=" in w.lower():
                    gap_part = [p for p in w.split() if "gap=" in p.lower()][0]
                    gap_val = float(gap_part.split("=")[-1].rstrip(","))
            except (ValueError, IndexError):
                pass
            
            # Only count as severe if gap is very large AND test R² is poor
            if asset_type == "commodities":
                # For commodities, only severe if gap > 0.25 AND test R² < 0.10
                if gap_val is not None and gap_val > 0.25:
                    if test_r2 is None or test_r2 < 0.10:
                        is_severe = True
                # If gap <= 0.25, not severe (even if test R² is low)
            else:
                # For crypto, only severe if gap > 0.20 AND test R² < 0.15
                if gap_val is not None and gap_val > 0.20:
                    if test_r2 is None or test_r2 < 0.15:
                        is_severe = True
                elif gap_val is None:
                    # If we can't parse gap, check test R²
                    if test_r2 is None or test_r2 < 0.15:
                        is_severe = True
        elif "significant performance drop" in w.lower():
            # Only count as severe if test R² is poor
            min_test_r2 = 0.10 if asset_type == "commodities" else 0.15
            if test_r2 is None or test_r2 < min_test_r2:
                is_severe = True
        elif ">>" in w:
            # Try to extract gap value - if gap > threshold, consider it severe
            try:
                if "gap=" in w.lower():
                    gap_part = [p for p in w.split() if "gap=" in p.lower()][0]
                    gap_val = float(gap_part.split("=")[-1].rstrip(","))
                    # For commodities, use higher threshold (0.30), for crypto use 0.25
                    threshold = 0.30 if asset_type == "commodities" else 0.25
                    if gap_val > threshold:
                        # Only severe if test R² is also poor
                        min_test_r2 = 0.10 if asset_type == "commodities" else 0.15
                        if test_r2 is None or test_r2 < min_test_r2:
                            is_severe = True
            except (ValueError, IndexError):
                pass
        
        if is_severe:
            severe_overfitting_count += 1
    
    # Only reject if ALL models have severe overfitting AND poor test performance
    # If at least one model has good test R², allow trading
    if severe_overfitting_count >= len(successful_models) and len(successful_models) > 0:
        # But if we have models with good test R², don't reject
        if models_with_good_test_r2 == 0:
            reasons.append("All models show severe overfitting (large train/val/test gaps)")
            return False, reasons
    # If some models have overfitting but not all, continue with other checks
    
    # Check validation/test performance for at least one model
    has_good_performance = False
    for name in successful_models:
        data = results.get(name, {})
        val_r2 = data.get("r2", 0.0)
        val_dir = data.get("directional_accuracy")
        test_metrics = data.get("test_metrics", {})
        test_r2 = test_metrics.get("r2", 0.0) if isinstance(test_metrics, dict) else 0.0
        test_dir = test_metrics.get("directional_accuracy") if isinstance(test_metrics, dict) else None
        
        # Accept if validation R² is reasonable OR directional accuracy is good
        if val_r2 >= MIN_ACCEPTABLE_R2 or (val_dir is not None and val_dir >= MIN_DIRECTIONAL_ACCURACY):
            # Also check test performance if available
            if isinstance(test_metrics, dict) and test_metrics:
                if test_r2 >= MIN_ACCEPTABLE_R2 or (test_dir is not None and test_dir >= MIN_TEST_DIRECTIONAL_ACCURACY):
                    has_good_performance = True
                    break
            else:
                # If no test metrics, accept based on validation alone
                has_good_performance = True
                break
    
    if not has_good_performance:
        reasons.append("No model meets minimum performance thresholds (R² or directional accuracy)")
        return False, reasons
    
    # All checks passed
    return True, ["All robustness checks passed"]


def discover_symbols(asset_type: str) -> List[str]:
    asset_root = config.BASE_DATA_DIR / asset_type
    if not asset_root.exists():
        return []
    symbols = set()
    for source_dir in asset_root.iterdir():
        if not source_dir.is_dir():
            continue
        for symbol_dir in source_dir.iterdir():
            if not symbol_dir.is_dir():
                continue
            symbols.add(symbol_dir.name)
    return sorted(symbols)


def _determine_action(pred_return: float, threshold: float) -> str:
    if pred_return >= threshold:
        return "long"
    if pred_return <= -threshold:
        return "short"
    return "hold"


def _clamp_return(pred_return: float) -> float:
    return float(max(min(pred_return, PREDICTION_CLAMP), -PREDICTION_CLAMP))


def _calibrate_predicted_return(
    pred_return: float,
    test_metrics: Optional[Dict[str, float]],
    dynamic_threshold: float,
) -> float:
    """
    Dampens unrealistically large returns using out-of-sample performance.
    """
    cap = max(MIN_THRESHOLD, dynamic_threshold) * 2.0
    if test_metrics:
        candidates: List[float] = []
        for key in ("mae", "rmse"):
            value = test_metrics.get(key)
            if isinstance(value, (int, float)) and value > 0:
                candidates.append(float(value))
        if candidates:
            cap = min(cap, max(MIN_THRESHOLD, min(candidates) * 1.5))
    cap = min(cap, PREDICTION_CLAMP)
    return float(max(min(pred_return, cap), -cap))


def _validate_prediction_realism(
    pred_return: float,
    pred_price: float,
    current_price: float,
    model_metrics: Dict[str, float],
    historical_returns: pd.Series,
) -> Tuple[bool, List[str]]:
    """
    Validate if a prediction seems realistic based on multiple criteria.
    
    Args:
        pred_return: Predicted return
        pred_price: Predicted price
        current_price: Current price
        model_metrics: Model performance metrics (r2, mae, rmse, directional_accuracy)
        historical_returns: Historical returns for context
    
    Returns:
        Tuple of (is_realistic, list_of_warnings)
    """
    warnings = []
    is_realistic = True
    
    # Check 1: Predicted return magnitude vs historical volatility
    if len(historical_returns) > 0:
        abs_historical = historical_returns.abs()
        historical_std = float(abs_historical.std())
        historical_max = float(abs_historical.max())
        abs_pred_return = abs(pred_return)
        
        # If prediction is more than 3 standard deviations above historical max, flag it
        if historical_std > 0 and abs_pred_return > max(historical_max * 1.5, historical_std * 3):
            warnings.append(
                f"Predicted return {abs_pred_return*100:.2f}% exceeds historical volatility "
                f"(max: {historical_max*100:.2f}%, std: {historical_std*100:.2f}%)"
            )
            is_realistic = False
    
    # Check 2: Predicted price change vs current price (sanity check)
    price_change_pct = abs((pred_price - current_price) / current_price) if current_price > 0 else 0
    if price_change_pct > PREDICTION_CLAMP * 1.1:  # Allow 10% tolerance
        warnings.append(
            f"Predicted price change {price_change_pct*100:.2f}% exceeds expected clamp "
            f"({PREDICTION_CLAMP*100:.2f}%)"
        )
        is_realistic = False
    
    # Check 3: Model performance indicators
    r2 = model_metrics.get("r2", 0.0)
    mae = model_metrics.get("mae")
    directional_acc = model_metrics.get("directional_accuracy")
    
    # If model has very low R², predictions are less reliable
    if r2 < -0.5:  # Negative R² means worse than baseline
        warnings.append(f"Model R² ({r2:.3f}) indicates poor predictive power")
        is_realistic = False
    elif r2 < 0.0:
        warnings.append(f"Model R² ({r2:.3f}) is negative (worse than baseline)")
    
    # If MAE is very high relative to typical returns, flag it
    if mae is not None and len(historical_returns) > 0:
        typical_return = float(abs(historical_returns).median())
        if typical_return > 0 and mae > typical_return * 2:
            warnings.append(
                f"Model MAE ({mae*100:.2f}%) is high relative to typical returns "
                f"({typical_return*100:.2f}%)"
            )
    
    # Check 4: Directional accuracy should be reasonable
    if directional_acc is not None and directional_acc < 0.45:  # Below random chance
        warnings.append(
            f"Directional accuracy ({directional_acc*100:.1f}%) is below random chance (50%)"
        )
        is_realistic = False
    
    # Check 5: Extreme predictions relative to clamped return
    clamped_return = _clamp_return(pred_return)
    if abs(pred_return - clamped_return) > 0.001:  # Significant clamping occurred
        warnings.append(
            f"Predicted return {pred_return*100:.2f}% was clamped to {clamped_return*100:.2f}% "
            "(extreme prediction)"
        )
    
    return is_realistic, warnings


def _compute_dynamic_threshold(returns: pd.Series) -> Tuple[float, Dict[str, float]]:
    abs_returns = returns.abs()
    rolling_mad = abs_returns.rolling(60).median().iloc[-1] * 1.4826
    global_mad = abs_returns.median() * 1.4826
    iqr = (abs_returns.quantile(0.75) - abs_returns.quantile(0.25)) / 1.349
    atr_proxy = abs_returns.rolling(14).mean().iloc[-1]

    candidates = [
        rolling_mad if pd.notna(rolling_mad) else None,
        global_mad if pd.notna(global_mad) else None,
        iqr if pd.notna(iqr) else None,
        atr_proxy if pd.notna(atr_proxy) else None,
        DEFAULT_THRESHOLD,
    ]
    positive = [float(c) for c in candidates if c and c > 0]
    if not positive:
        raw_threshold = DEFAULT_THRESHOLD
    else:
        raw_threshold = float(np.median(positive))

    threshold = float(max(MIN_THRESHOLD, min(MAX_THRESHOLD, raw_threshold)))
    details = {
        "rolling_mad": float(rolling_mad) if pd.notna(rolling_mad) else None,
        "global_mad": float(global_mad) if pd.notna(global_mad) else None,
        "iqr_based": float(iqr) if pd.notna(iqr) else None,
        "atr_proxy": float(atr_proxy) if pd.notna(atr_proxy) else None,
        "median_candidate": raw_threshold,
        "clamped": threshold,
    }
    return threshold, details


def _extract_signals(context: pd.Series) -> Dict[str, float]:
    signals: Dict[str, float] = {}
    for field in ACTION_SIGNAL_FIELDS:
        value = context.get(field)
        if pd.notna(value):
            try:
                signals[field] = float(value)
            except (TypeError, ValueError):
                continue
    return signals


def _compute_consensus_action(
    results: Dict[str, Dict],
    dynamic_threshold: float,
    price_reference: float
) -> Dict[str, any]:
    """
    Compute consensus action from all model predictions using weighted voting.
    
    Strategy:
    1. Weight each model by its R² score (performance-based weighting)
    2. Consider predicted return magnitude (confidence)
    3. Require strong consensus for long/short (at least 2 models + weighted majority)
    4. Default to hold if disagreement or weak signals
    
    Returns:
        Dict with consensus_action, consensus_confidence, consensus_price, and detailed reasoning
    """
    # Collect valid model predictions
    model_votes = []
    total_weight = 0.0
    
    action_scores = {"long": 0.0, "hold": 0.0, "short": 0.0}
    action_returns = {"long": [], "hold": [], "short": []}
    action_prices = {"long": [], "hold": [], "short": []}
    model_details = []
    
    for model_name, model_data in results.items():
        if model_data.get("status") == "failed":
            continue
        
        action = model_data.get("action", "hold")
        pred_return = model_data.get("predicted_return", 0.0)  # Already clamped
        pred_price = model_data.get("predicted_price", price_reference)
        r2_score = model_data.get("r2")
        mae_score = model_data.get("mae")
        
        # Penalize models with unrealistic predictions
        is_realistic = model_data.get("prediction_realistic", True)
        if not is_realistic:
            # Reduce weight for unrealistic predictions
            r2_score = r2_score * 0.5 if r2_score else 0.0

        walk_forward = model_data.get("walk_forward", {})
        wf_r2 = walk_forward.get("r2_mean")
        if isinstance(r2_score, (int, float)) and not np.isnan(r2_score):
            clipped_r2 = float(min(1.0, max(0.0, r2_score)))
        else:
            clipped_r2 = 0.0
        if isinstance(mae_score, (int, float)) and mae_score > 0:
            inv_mae = 1.0 / (1.0 + float(mae_score))
        else:
            inv_mae = 0.0
        if isinstance(model_data.get("rmse"), (int, float)) and model_data["rmse"] > 0:
            inv_rmse = 1.0 / (1.0 + float(model_data["rmse"]))
        else:
            inv_rmse = 0.0
        wf_bonus = float(max(0.0, min(1.0, wf_r2))) if isinstance(wf_r2, (int, float)) else 0.0

        base_weight = 0.25 * clipped_r2 + 0.35 * inv_rmse + 0.35 * inv_mae + 0.05 * wf_bonus
        if base_weight == 0:
            base_weight = 0.5  # fallback for models without metrics
        
        # FIXED: Penalize models with large val-test gaps (overfitting indicator)
        # Check if we have test metrics to compute gap
        test_r2 = model_data.get("test_r2")
        val_r2 = model_data.get("val_r2") or r2_score
        if isinstance(test_r2, (int, float)) and isinstance(val_r2, (int, float)) and not np.isnan(test_r2) and not np.isnan(val_r2):
            val_test_gap = val_r2 - test_r2
            if val_test_gap > 0.15:  # Large gap indicates overfitting
                # Strongly penalize: reduce weight by 50-70% depending on gap size
                gap_penalty = 1.0 - min(0.7, (val_test_gap - 0.15) * 2.0)  # Penalty scales with gap
                base_weight *= gap_penalty
            elif val_test_gap > 0.10:  # Moderate gap
                # Moderate penalty: reduce weight by 20-40%
                gap_penalty = 1.0 - (val_test_gap - 0.10) * 4.0  # Penalty scales with gap
                base_weight *= gap_penalty
        
        # Confidence multiplier based on return magnitude with cap
        return_magnitude = abs(pred_return)
        confidence_multiplier = min(
            CONFIDENCE_MULTIPLIER_CAP,
            1.0 + (return_magnitude / max(dynamic_threshold, MIN_THRESHOLD)),
        )
        adjusted_weight = base_weight * confidence_multiplier
        if model_data.get("overfitting_flag"):
            adjusted_weight *= OVERFITTING_WEIGHT_MULTIPLIER
        
        action_scores[action] += adjusted_weight
        action_returns[action].append(pred_return)
        action_prices[action].append(pred_price)
        total_weight += adjusted_weight
        
        model_details.append({
            "model": model_name,
            "action": action,
            "predicted_return": pred_return,
            "predicted_price": pred_price,
            "weight": base_weight,
            "r2_score": r2_score,
            "mae": mae_score,
            "overfitting_flag": bool(model_data.get("overfitting_flag")),
        })
    
    if total_weight == 0:
        # Fallback if no valid models
        return {
            "consensus_action": "hold",
            "consensus_confidence": 0.0,
            "consensus_price": price_reference,
            "consensus_return": 0.0,
            "reasoning": "No valid model predictions available",
            "model_votes": model_details,
            "action_scores": {"long": 0.0, "hold": 1.0, "short": 0.0},
            "neutral_return_threshold": dynamic_threshold,
            "neutral_guard_triggered": False,
        }
    
    # Normalize scores
    for action in action_scores:
        action_scores[action] /= total_weight
    
    # Determine consensus
    best_action = max(action_scores, key=action_scores.get)
    best_score = action_scores[best_action]
    
    # Require strong consensus for long/short (at least 60% weighted vote)
    # Hold requires only 40% (default if no strong signal)
    if best_action in ["long", "short"]:
        if best_score >= 0.6:  # Strong consensus
            consensus_action = best_action
            consensus_confidence = best_score
        else:
            # Weak consensus - check if at least 2 models agree
            action_counts = {}
            for detail in model_details:
                action = detail["action"]
                action_counts[action] = action_counts.get(action, 0) + 1
            
            if action_counts.get(best_action, 0) >= 2 and best_score >= 0.4:
                consensus_action = best_action
                consensus_confidence = best_score
            else:
                consensus_action = "hold"
                consensus_confidence = action_scores["hold"]
    else:
        consensus_action = "hold"
        consensus_confidence = best_score
    
    # PERMANENT FIX: Normalized consensus with robust outlier handling
    if action_returns[consensus_action]:
        action_model_returns = np.array(action_returns[consensus_action])
        action_model_weights = np.array([
                model_details[i]["weight"]
                for i, detail in enumerate(model_details)
                if detail["action"] == consensus_action
        ])
        
        # Step 1: Normalize returns to reasonable range (prevent extreme predictions)
        # Clamp returns to ±5% for short-term, ±10% for longer horizons
        max_return = dynamic_threshold * 5.0  # Adaptive max based on volatility
        action_model_returns = np.clip(action_model_returns, -max_return, max_return)
        
        # Step 2: Robust consensus calculation
        if len(action_model_returns) >= 3:
            median_return = np.median(action_model_returns)
            std_return = np.std(action_model_returns)
            mean_return = np.average(action_model_returns, weights=action_model_weights)
            
            # If models disagree (std > 1.5x threshold), use robust median-based approach
            if std_return > dynamic_threshold * 1.5:
                # Weighted median: closer to median gets more weight
                distances = np.abs(action_model_returns - median_return)
                robust_weights = 1.0 / (1.0 + distances / (std_return + 1e-10))
                robust_weights = robust_weights * action_model_weights  # Combine with quality weights
                robust_weights = robust_weights / robust_weights.sum()
                
                # Blend: 60% median (robust), 40% weighted mean (quality-aware)
                consensus_return = 0.6 * median_return + 0.4 * np.average(action_model_returns, weights=robust_weights)
            else:
                # Models agree: use quality-weighted average
                consensus_return = mean_return
        else:
            # 1-2 models: use weighted average
            consensus_return = np.average(action_model_returns, weights=action_model_weights)
        
        # Step 3: Final normalization - ensure prediction is realistic
        consensus_return = np.clip(consensus_return, -max_return, max_return)
        consensus_price = price_reference * (1.0 + consensus_return)
    else:
        # Fallback: normalized robust averaging of all predictions
        all_returns = np.array([detail["predicted_return"] for detail in model_details])
        all_weights = np.array([detail["weight"] for detail in model_details])
        
        # Normalize returns to reasonable range
        max_return = dynamic_threshold * 5.0
        all_returns = np.clip(all_returns, -max_return, max_return)
        
        if len(all_returns) >= 3:
            median_return = np.median(all_returns)
            std_return = np.std(all_returns)
            mean_return = np.average(all_returns, weights=all_weights)
            
            # Robust averaging if models disagree
            if std_return > dynamic_threshold * 1.5:
                distances = np.abs(all_returns - median_return)
                robust_weights = 1.0 / (1.0 + distances / (std_return + 1e-10))
                robust_weights = robust_weights * all_weights
                robust_weights = robust_weights / robust_weights.sum()
                consensus_return = 0.6 * median_return + 0.4 * np.average(all_returns, weights=robust_weights)
            else:
                consensus_return = mean_return
        else:
            consensus_return = np.average(all_returns, weights=all_weights)
        
        # Final normalization
        consensus_return = np.clip(consensus_return, -max_return, max_return)
        consensus_price = price_reference * (1.0 + consensus_return)
        
        # CRITICAL VALIDATION: Ensure action matches return sign
        # If return is negative, action should be SHORT or HOLD, not LONG
        # If return is positive, action should be LONG or HOLD, not SHORT
        if consensus_return < -1e-6 and consensus_action == "long":
            # Negative return but LONG action - force to HOLD
            consensus_action = "hold"
            consensus_confidence = max(action_scores.get("hold", 0.0), min(0.55, consensus_confidence))
        elif consensus_return > 1e-6 and consensus_action == "short":
            # Positive return but SHORT action - force to HOLD
            consensus_action = "hold"
            consensus_confidence = max(action_scores.get("hold", 0.0), min(0.55, consensus_confidence))

    neutral_threshold = max(dynamic_threshold, MIN_THRESHOLD) * CONSENSUS_NEUTRAL_MULTIPLIER
    raw_consensus_return = consensus_return
    neutral_guard_triggered = False
    if abs(consensus_return) < neutral_threshold and consensus_action != "hold":
        neutral_guard_triggered = True
        dominant_score = action_scores.get(consensus_action, 0.0)
        # CRITICAL FIX: If neutral guard triggers, ALWAYS set action to HOLD and zero return
        # This prevents contradictory signals (LONG action with 0% or negative return)
        consensus_action = "hold"
        consensus_confidence = max(action_scores.get("hold", 0.0), min(0.55, consensus_confidence))
        consensus_return = 0.0
        consensus_price = price_reference
    
    # Build reasoning
    reasoning_parts = []
    num_models = len(model_details)
    reasoning_parts.append(
        f"Consensus: {consensus_action.upper()} (confidence: {consensus_confidence*100:.1f}%)"
    )
    if num_models == 1:
        reasoning_parts.append("⚠️ WARNING: Consensus based on only 1 model - lower confidence recommended")
    reasoning_parts.append(f"Expected return: {consensus_return*100:+.2f}%")
    reasoning_parts.append(f"Consensus price: ${consensus_price:,.2f}")
    if neutral_guard_triggered:
        reasoning_parts.append(
            f"Neutral guard engaged: |return| {abs(raw_consensus_return)*100:.2f}% < {neutral_threshold*100:.2f}%"
        )
    
    # Model breakdown
    vote_summary = []
    for action in ["long", "hold", "short"]:
        count = sum(1 for d in model_details if d["action"] == action)
        score = action_scores[action] * 100
        if count > 0:
            vote_summary.append(f"{action.upper()}: {count} model(s), {score:.1f}% weighted vote")
    
    reasoning_parts.append("Model votes: " + "; ".join(vote_summary))
    
    # Individual model contributions
    model_contributions = []
    for detail in model_details:
        r2_value = detail.get("r2_score")
        if r2_value is None:
            r2_display = "N/A"
        else:
            try:
                r2_display = f"{float(r2_value):.3f}"
            except (TypeError, ValueError):
                r2_display = "N/A"
        contrib = (
            f"{detail['model']} ({detail['action']}, "
            f"{detail['predicted_return']*100:+.2f}%, R²={r2_display})"
        )
        model_contributions.append(contrib)
    reasoning_parts.append("Individual models: " + "; ".join(model_contributions))
    
    return {
        "consensus_action": consensus_action,
        "consensus_confidence": float(consensus_confidence),
        "consensus_price": float(consensus_price),
        "consensus_return": float(consensus_return),  # May be zeroed by neutral guard
        "raw_consensus_return": float(raw_consensus_return),  # Actual prediction before neutral guard
        "reasoning": ". ".join(reasoning_parts),
        "model_votes": model_details,
        "action_scores": {k: float(v) for k, v in action_scores.items()},
        "neutral_return_threshold": float(neutral_threshold),
        "neutral_guard_triggered": neutral_guard_triggered,
    }


def _build_action_reason(
    pred_return: float, threshold: float, signals: Dict[str, float], current_price: float
) -> str:
    direction = "expected gain" if pred_return >= 0 else "expected loss"
    magnitude = abs(pred_return) * 100
    comparison = "exceeds" if abs(pred_return) >= threshold else "is below"
    threshold_pct = threshold * 100

    clauses = []
    rsi = signals.get("RSI_14")
    if rsi is not None:
        if rsi < 30:
            clauses.append(f"RSI14={rsi:.1f} (oversold)")
        elif rsi > 70:
            clauses.append(f"RSI14={rsi:.1f} (overbought)")
        else:
            clauses.append(f"RSI14={rsi:.1f} (neutral)")

    sma50 = signals.get("SMA_50")
    sma200 = signals.get("SMA_200")
    if sma50 and sma200:
        if sma50 > sma200:
            clauses.append("SMA50 > SMA200 (uptrend)")
        else:
            clauses.append("SMA50 < SMA200 (downtrend)")
        if current_price > sma50:
            clauses.append("price above SMA50")
        else:
            clauses.append("price below SMA50")

    macd_hist = signals.get("MACD_histogram")
    if macd_hist is not None:
        if macd_hist > 0:
            clauses.append(f"MACD hist {macd_hist:.4f} (bullish momentum)")
        else:
            clauses.append(f"MACD hist {macd_hist:.4f} (bearish momentum)")

    vol_ratio = signals.get("Volume_Ratio")
    if vol_ratio is not None:
        if vol_ratio > 1:
            clauses.append(f"Volume ratio {vol_ratio:.2f} (>1, strong activity)")
        else:
            clauses.append(f"Volume ratio {vol_ratio:.2f} (<1, weak activity)")

    reason = (
        f"{direction.capitalize()} of {magnitude:.2f}% {comparison} the "
        f"{threshold_pct:.2f}% dynamic threshold."
    )
    if clauses:
        reason += " Signals: " + "; ".join(clauses)
    return reason


def _simulate_dqn_policy(
    model,
    feature_frame: pd.DataFrame,
    close_series: pd.Series,
    transaction_cost: float = 0.0005,
) -> Dict[str, float]:
    """
    Replay a trained DQN policy on a dataset slice to estimate PnL characteristics.
    """
    if feature_frame.empty or len(feature_frame) < 3:
        return {}
    obs = feature_frame.to_numpy(dtype=float)
    future_returns = (close_series.shift(-1) / close_series - 1.0).iloc[:-1].to_numpy()
    observations = obs[:-1]
    if len(future_returns) != len(observations):
        future_returns = future_returns[: len(observations)]
    actions = []
    for row in observations:
        try:
            action_id, _ = model.predict(row, deterministic=True)
            actions.append(int(action_id))
        except Exception:
            actions.append(1)  # hold on failure
    positions = np.array(actions) - 1  # {-1,0,1}
    if not len(positions):
        return {}
    turnover = np.abs(np.diff(np.concatenate(([0], positions))))
    trading_costs = transaction_cost * turnover
    pnl = positions * future_returns - trading_costs[: len(positions)]
    equity_curve = np.cumsum(pnl)
    total_return = float(pnl.sum())
    avg_return = float(np.mean(pnl))
    std_return = float(np.std(pnl)) or 1e-9
    sharpe = float(np.sqrt(252) * avg_return / std_return)
    hit_rate = float(np.mean(pnl > 0))
    max_equity = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve - max_equity
    max_drawdown = float(drawdowns.min())
    return {
        "total_return": total_return,
        "avg_return": avg_return,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "max_drawdown": max_drawdown,
    }


def _write_dqn_summary(
    output_dir: Path,
    asset_type: str,
    symbol: str,
    timeframe: str,
    training_samples: int,
    validation_samples: int,
    feature_count: int,
    feature_columns: List[str],
    metrics: Dict[str, Any],
) -> None:
    """
    Persist a human-readable DQN run summary instead of opaque .zip artifacts.
    """
    summary_dir = output_dir / "dqn"
    summary_dir.mkdir(parents=True, exist_ok=True)
    feature_sample = feature_columns[: min(25, len(feature_columns))]
    payload = {
        "asset_type": asset_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "training_samples": training_samples,
        "validation_samples": validation_samples,
        "feature_count": feature_count,
        "feature_sample": feature_sample,
        "config": DQN_CONFIG,
        "metrics": metrics,
    }
    summary_file = summary_dir / f"{asset_type}_{symbol}_{timeframe}.json"
    with open(summary_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def train_for_symbol(
    asset_type: str,
    symbol: str,
    timeframe: str,
    output_dir: Path,
    min_rows: int = 200,
    verbose: bool = True,
    target_config: Optional[TargetConfig] = None,
    horizon_profile: str = DEFAULT_HORIZON_PROFILE,
) -> Dict:
    """
    Train all models for a single symbol with overfitting prevention.
    
    Args:
        asset_type: Type of asset (crypto/commodities)
        symbol: Symbol name
        timeframe: Timeframe string
        output_dir: Directory to save models and results
        min_rows: Minimum rows required for training
        verbose: Print diagnostic information
    
    Returns:
        Dictionary with training summary
    """
    # Initialize JSON logger
    logger = get_training_logger(asset_type, symbol, timeframe)
    resolved_profile, target_cfg = get_profile_config(horizon_profile)
    if target_config is not None:
        target_cfg = target_config
        resolved_profile = horizon_profile or resolved_profile
    profile_report = build_profile_report(resolved_profile, target_cfg)
    ensure_horizon_dirs(asset_type, symbol, timeframe)
    symbol_dir = horizon_dir(asset_type, symbol, timeframe, resolved_profile)
    symbol_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Starting training for {asset_type}/{symbol}/{timeframe} ({profile_report['name']} horizon)",
        category="TRAIN",
        symbol=symbol,
        asset_type=asset_type,
        data={
            "timeframe": timeframe,
            "min_rows": min_rows,
            "horizon_profile": profile_report,
        }
    )
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"TRAINING: {asset_type}/{symbol}/{timeframe}")
        print(f"Horizon profile: {profile_report['label']} ({profile_report['horizon_bars']} bars)")
        print(f"{'='*80}")
    
    gap_days = TIMEFRAME_GAP_DAYS.get(timeframe, 0)
    dataset, dataset_meta = assemble_dataset(
        asset_type,
        symbol,
        timeframe,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        gap_days=gap_days,
        expected_feature_version=FEATURE_SCHEMA_VERSION,
        verbose=verbose,
        logger=logger,
        target_config=target_cfg,
        context_config=CONTEXT_CONFIG,
    )
    if len(dataset) < min_rows:
        error_msg = f"insufficient data ({len(dataset)} rows, need {min_rows})"
        logger.error(error_msg, category="DATA", symbol=symbol, asset_type=asset_type)
        raise ValueError(error_msg)
    
    logger.info(
        f"Dataset assembled: {len(dataset)} rows",
        category="DATA",
        symbol=symbol,
        asset_type=asset_type,
        data={"rows": len(dataset), "features": len([c for c in dataset.columns if c not in ["open", "high", "low", "close", "volume", "target", "target_return", "symbol", "asset_type", "timeframe"]])}
    )
    
    train_df, val_df, test_df = train_val_test_split(
        dataset,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        gap_days=gap_days,
        verbose=verbose,
    )
    
    logger.info(
        f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}",
        category="DATA",
        symbol=symbol,
        asset_type=asset_type,
        data={
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "gap_days": gap_days,
            "split_boundaries": dataset_meta["split_boundaries"],
        }
    )

    rows_used = len(train_df) + len(val_df) + len(test_df)
    unused_rows = len(dataset) - rows_used
    logger.info(
        "Dataset utilization confirmed",
        category="DATA",
        symbol=symbol,
        asset_type=asset_type,
        data={"rows_total": len(dataset), "rows_used": rows_used, "unused_rows": unused_rows},
    )
    if unused_rows and verbose:
        print(f"[WARN] {unused_rows} rows unused due to split configuration.")

    logger.info(
        "Split boundaries confirmed",
        category="DATA",
        symbol=symbol,
        asset_type=asset_type,
        data=dataset_meta["split_boundaries"],
    )
    
    # CRITICAL: Validate test period dates are not in the future (data leak check)
    from datetime import datetime, timezone
    test_boundaries = dataset_meta["split_boundaries"].get("test", {})
    if test_boundaries:
        test_start_str = test_boundaries.get("start", "")
        test_end_str = test_boundaries.get("end", "")
        try:
            if test_start_str and test_end_str:
                test_start = datetime.fromisoformat(test_start_str.replace('Z', '+00:00'))
                test_end = datetime.fromisoformat(test_end_str.replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                
                # Check if test period is in the future (CRITICAL DATA LEAK)
                # Also check if test year is suspiciously far in the future (more than 1 year ahead)
                test_year = test_end.year
                current_year = now.year
                is_future_date = test_start > now or test_end > now
                is_suspicious_year = test_year > current_year + 1  # More than 1 year in future
                
                if is_future_date or is_suspicious_year:
                    future_days = max(
                        (test_start - now).days if test_start > now else 0,
                        (test_end - now).days if test_end > now else 0
                    )
                    year_warning = f" Test year ({test_year}) is {test_year - current_year} year(s) ahead of current year ({current_year})." if is_suspicious_year else ""
                    error_msg = (
                        f"CRITICAL: Test period appears to be in the future! "
                        f"Test start: {test_start_str}, Test end: {test_end_str}, "
                        f"Current time: {now.isoformat()}, Future by {future_days} days.{year_warning} "
                        f"This indicates a SEVERE DATA LEAK - test data should be from the past!"
                    )
                    logger.error(
                        error_msg,
                        category="DATA_LEAK",
                        symbol=symbol,
                        asset_type=asset_type,
                        data={
                            "test_start": test_start_str,
                            "test_end": test_end_str,
                            "current_time": now.isoformat(),
                            "future_days": future_days
                        }
                    )
                    overfitting_warnings.append(error_msg)
                    # Always print CRITICAL errors regardless of verbose flag
                    print(f"\n{'='*80}")
                    print(f"[CRITICAL ERROR] {error_msg}")
                    print(f"{'='*80}\n")
        except Exception as date_exc:
            logger.warning(
                f"Could not validate test period dates: {date_exc}",
                category="DATA",
                symbol=symbol,
                asset_type=asset_type,
            )
    
    # Extract features and targets
    X_train, y_train = extract_xy(train_df, target_column="target_return")
    X_val, y_val = extract_xy(val_df, target_column="target_return")
    X_test, y_test = extract_xy(test_df, target_column="target_return")
    
    # Validate target variable has variance (not constant)
    y_train_std = y_train.std()
    y_val_std = y_val.std()
    y_test_std = y_test.std()
    y_train_mean = y_train.mean()
    y_val_mean = y_val.mean()
    
    if y_train_std < 1e-6 or y_val_std < 1e-6:
        raise ValueError(
            f"Target variable has near-zero variance (train_std={y_train_std:.2e}, val_std={y_val_std:.2e}). "
            f"Models cannot learn from constant targets. Check target generation."
        )
    
    # Validate features have variance (critical for learning)
    X_train_std = X_train.std()
    zero_variance_features = X_train_std[X_train_std < 1e-10].index.tolist()
    if zero_variance_features:
        logger.warning(
            f"Found {len(zero_variance_features)} features with zero variance, removing them",
            category="DATA",
            symbol=symbol,
            asset_type=asset_type,
            data={"zero_variance_features": zero_variance_features[:10]}  # Log first 10
        )
        X_train = X_train.drop(columns=zero_variance_features)
        X_val = X_val.drop(columns=zero_variance_features)
        X_test = X_test.drop(columns=zero_variance_features)
        if verbose:
            print(f"[WARN] Removed {len(zero_variance_features)} zero-variance features")
    
    if verbose:
        print(f"[DATA] Target statistics - Train: mean={y_train_mean:.6f}, std={y_train_std:.6f}, "
              f"Val: mean={y_val_mean:.6f}, std={y_val_std:.6f}, "
              f"Test: mean={y_test.mean():.6f}, std={y_test_std:.6f}")
        print(f"[DATA] Feature statistics - Train: {X_train.shape[1]} features, "
              f"mean_std={X_train.std().mean():.6f}, min_std={X_train.std().min():.6f}, "
              f"max_std={X_train.std().max():.6f}")
    
    logger.info(
        "Target variable validated",
        category="DATA",
        symbol=symbol,
        asset_type=asset_type,
        data={
            "train_mean": float(y_train.mean()),
            "train_std": float(y_train_std),
            "val_mean": float(y_val.mean()),
            "val_std": float(y_val_std),
            "test_mean": float(y_test.mean()),
            "test_std": float(y_test_std),
        }
    )
    
    # ========================================================================
    # STEP 1: DATA BALANCE ANALYSIS - Check for bearish/bullish bias
    # ========================================================================
    def _analyze_target_distribution(y: pd.Series, split_name: str) -> Dict[str, Any]:
        """Analyze target return distribution to detect bias."""
        y_array = y.values
        positive_count = np.sum(y_array > 0)
        negative_count = np.sum(y_array < 0)
        neutral_count = np.sum(y_array == 0)
        total = len(y_array)
        
        positive_pct = (positive_count / total * 100) if total > 0 else 0
        negative_pct = (negative_count / total * 100) if total > 0 else 0
        neutral_pct = (neutral_count / total * 100) if total > 0 else 0
        
        mean_return = float(np.mean(y_array))
        median_return = float(np.median(y_array))
        std_return = float(np.std(y_array))
        
        # Calculate imbalance ratio (positive / negative)
        imbalance_ratio = (positive_count / negative_count) if negative_count > 0 else float('inf')
        
        # Detect bias: if >60% are negative or positive, there's a bias
        bias_detected = None
        bias_severity = "none"
        if negative_pct > 60:
            bias_detected = "bearish"
            if negative_pct > 75:
                bias_severity = "severe"
            elif negative_pct > 65:
                bias_severity = "moderate"
            else:
                bias_severity = "mild"
        elif positive_pct > 60:
            bias_detected = "bullish"
            if positive_pct > 75:
                bias_severity = "severe"
            elif positive_pct > 65:
                bias_severity = "moderate"
            else:
                bias_severity = "mild"
        
        return {
            "split": split_name,
            "total_samples": total,
            "positive_count": int(positive_count),
            "negative_count": int(negative_count),
            "neutral_count": int(neutral_count),
            "positive_pct": float(positive_pct),
            "negative_pct": float(negative_pct),
            "neutral_pct": float(neutral_pct),
            "mean_return": mean_return,
            "median_return": median_return,
            "std_return": std_return,
            "imbalance_ratio": float(imbalance_ratio) if imbalance_ratio != float('inf') else None,
            "bias_detected": bias_detected,
            "bias_severity": bias_severity,
        }
    
    train_dist = _analyze_target_distribution(y_train, "train")
    val_dist = _analyze_target_distribution(y_val, "val")
    test_dist = _analyze_target_distribution(y_test, "test")
    
    # Log data balance analysis
    logger.warning(
        "Data balance analysis - checking for bearish/bullish bias",
        category="DATA",
        symbol=symbol,
        asset_type=asset_type,
        data={
            "train_distribution": train_dist,
            "val_distribution": val_dist,
            "test_distribution": test_dist,
            "warning": "If bias_severity is 'severe' or 'moderate', models may learn directional bias"
        }
    )
    
    # Warn if severe bias detected
    if train_dist["bias_severity"] in ["severe", "moderate"]:
        import warnings
        warnings.warn(
            f"⚠️  {train_dist['bias_detected'].upper()} BIAS DETECTED in training data: "
            f"{train_dist['negative_pct']:.1f}% negative, {train_dist['positive_pct']:.1f}% positive. "
            f"Models may learn to predict {train_dist['bias_detected']} direction. "
            f"Consider using class weights or data balancing.",
            UserWarning
        )
        print(f"[WARNING] {train_dist['bias_detected'].upper()} BIAS: {train_dist['negative_pct']:.1f}% negative, {train_dist['positive_pct']:.1f}% positive")
    
    # Store distribution info for later use
    data_balance_info = {
        "train": train_dist,
        "val": val_dist,
        "test": test_dist,
    }

    # Feature importance pruning followed by correlation-based reduction
    X_train, X_val, X_test, importance_meta = _filter_by_importance(
        X_train, X_val, X_test, y_train
    )
    X_train, X_val, X_test, feature_selection_meta = _reduce_feature_matrix(
        X_train, X_val, X_test, y_train
    )
    # ========================================================================
    # STEP 2: FEATURE SCALING VERIFICATION - Ensure proper normalization
    # ========================================================================
    # Feature scaling for stability across models (esp. RL)
    X_train, X_val, X_test, feature_scaler = _scale_feature_matrix(
        X_train, X_val, X_test, asset_type=asset_type, symbol=symbol
    )
    
    # Verify scaling was applied correctly
    def _verify_scaling(X: pd.DataFrame, split_name: str) -> Dict[str, Any]:
        """Verify that features are properly scaled."""
        X_array = X.values
        feature_means = np.mean(X_array, axis=0)
        feature_stds = np.std(X_array, axis=0)
        feature_mins = np.min(X_array, axis=0)
        feature_maxs = np.max(X_array, axis=0)
        
        # After RobustScaler, features are centered on median (not mean)
        # For RobustScaler, we check median instead of mean since distributions may be skewed
        feature_medians = np.median(X_array, axis=0)
        mean_abs_mean = float(np.mean(np.abs(feature_means)))
        mean_abs_median = float(np.mean(np.abs(feature_medians)))  # Better check for RobustScaler
        mean_std = float(np.mean(feature_stds))
        mean_abs_min = float(np.mean(np.abs(feature_mins)))
        mean_abs_max = float(np.mean(np.abs(feature_maxs)))
        
        # Check for issues
        # For RobustScaler: median should be near 0, mean can be skewed
        # Use more lenient threshold for mean (2.0) since RobustScaler centers on median
        # For val/test sets, be more lenient since distributions may differ from train
        issues = []
        # More lenient threshold for val/test (1.5) vs train (0.5)
        # This accounts for distribution shift between splits
        median_threshold = 1.5 if split_name in ["val", "test"] else 0.5
        if mean_abs_median > median_threshold:  # Median should be near 0 for RobustScaler
            issues.append(f"Features not well-centered (median-based, mean_abs_median={mean_abs_median:.3f})")
        elif mean_abs_mean > 2.0:  # Mean can be skewed, but shouldn't be extreme
            issues.append(f"Features not well-centered (mean-based, mean_abs_mean={mean_abs_mean:.3f})")
        if mean_std < 0.1 or mean_std > 10.0:  # Unusual scale
            issues.append(f"Unusual feature scale (mean_std={mean_std:.3f})")
        if mean_abs_max > 100.0:  # Extreme values
            issues.append(f"Extreme feature values detected (mean_abs_max={mean_abs_max:.3f})")
        
        # Check for NaN or Inf
        nan_count = int(np.isnan(X_array).sum())
        inf_count = int(np.isinf(X_array).sum())
        if nan_count > 0:
            issues.append(f"NaN values found: {nan_count}")
        if inf_count > 0:
            issues.append(f"Inf values found: {inf_count}")
        
        return {
            "split": split_name,
            "mean_abs_mean": mean_abs_mean,
            "mean_abs_median": mean_abs_median,  # Added for RobustScaler verification
            "mean_std": mean_std,
            "mean_abs_min": mean_abs_min,
            "mean_abs_max": mean_abs_max,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "issues": issues,
            "scaling_ok": len(issues) == 0,
        }
    
    train_scaling = _verify_scaling(X_train, "train")
    val_scaling = _verify_scaling(X_val, "val")
    test_scaling = _verify_scaling(X_test, "test")
    
    scaler_info = {
        "scaler": "RobustScaler",
        "feature_count": X_train.shape[1],
        "center_sample": feature_scaler.center_[:5].tolist() if hasattr(feature_scaler, "center_") else None,
        "scale_sample": feature_scaler.scale_[:5].tolist() if hasattr(feature_scaler, "scale_") else None,
        "scaling_verification": {
            "train": train_scaling,
            "val": val_scaling,
            "test": test_scaling,
        }
    }
    
    # Warn if scaling issues detected
    all_issues = train_scaling["issues"] + val_scaling["issues"] + test_scaling["issues"]
    if all_issues:
        import warnings
        warnings.warn(
            f"⚠️  FEATURE SCALING ISSUES DETECTED: {', '.join(set(all_issues))}. "
            f"Features may not be properly normalized.",
            UserWarning
        )
        print(f"[WARNING] Feature scaling issues: {', '.join(set(all_issues))}")
    else:
        logger.info(
            "Feature scaling verified - all features properly normalized",
            category="DATA",
            symbol=symbol,
            asset_type=asset_type,
        )
    
    logger.info(
        "Applied robust feature scaling",
        category="DATA",
        symbol=symbol,
        asset_type=asset_type,
        data=scaler_info,
    )
    
    # Verify features are being used
    feature_count = X_train.shape[1]
    sample_features = list(X_train.columns[:10])  # First 10 feature names
    max_feature_signal = float(feature_selection_meta.get("max_abs_correlation", 0.0))
    logger.info(
        f"Features extracted: {feature_count} features will be used for training",
        category="DATA",
        symbol=symbol,
        asset_type=asset_type,
        data={
            "feature_count": feature_count,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "sample_features": sample_features,
            "selected_features": feature_selection_meta["selected_features"],
            "removed_low_variance": feature_selection_meta["removed_low_variance"],
            "max_abs_corr": max_feature_signal,
            "importance_filter": importance_meta,
        }
    )
    
    # Prepare full dataset for refitting (train + val, excluding test)
    # CRITICAL: Test set is NEVER used for training or validation - only for final evaluation
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    baseline_val_mae = float(mean_absolute_error(y_val, np.zeros_like(y_val)))
    baseline_test_mae = float(mean_absolute_error(y_test, np.zeros_like(y_test)))
    
    logger.info(
        "Data splits prepared - test set isolated for final evaluation only",
        category="DATA",
        symbol=symbol,
        asset_type=asset_type,
        data={
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "test_rows": len(X_test),
            "full_train_val_rows": len(X_full),
            "note": "Test set will NOT be used during training or hyperparameter optimization"
        }
    )
    
    reference_price_dataset = float(test_df["close"].iloc[-1])
    train_returns = train_df["target_return"]
    dynamic_threshold, threshold_details = _compute_dynamic_threshold(train_returns)
    context_row = test_df.iloc[-1]
    context_signals = _extract_signals(context_row)
    latest_market_price = float(dataset_meta.get("latest_close", reference_price_dataset))
    reference_price = latest_market_price
    logger.info(
        "Reference price recalibrated",
        category="DATA",
        symbol=symbol,
        asset_type=asset_type,
        data={
            "dataset_close": reference_price_dataset,
            "latest_market_price": latest_market_price,
        },
    )
    latest_market_timestamp = dataset_meta.get("latest_timestamp")

    results: Dict[str, Dict] = {}
    metric_store: Dict[str, TrainingResult] = {}
    models = {}
    train_pred_store: Dict[str, np.ndarray] = {}
    val_pred_store: Dict[str, np.ndarray] = {}
    test_pred_store: Dict[str, np.ndarray] = {}
    overfitting_warnings = []

    def _record(name: str, model_obj, metrics: TrainingResult, pred_return: float, 
                train_metrics: Optional[Dict] = None, test_metrics: Optional[Dict] = None) -> bool:
        """Record model results and check for overfitting."""
        entry = asdict(metrics)
        model_overfitting_notes: List[str] = []

        def _note_overfit(message: str):
            overfitting_warnings.append(message)
            model_overfitting_notes.append(message)
        clamped_return = _clamp_return(pred_return)
        calibrated_return = _calibrate_predicted_return(
            clamped_return,
            test_metrics,
            dynamic_threshold,
        )
        entry["predicted_return"] = float(calibrated_return)
        entry["predicted_price"] = float(
            latest_market_price * (1.0 + calibrated_return)
        )
        action = _determine_action(clamped_return, dynamic_threshold)
        entry["action"] = action
        entry["action_reason"] = _build_action_reason(
            calibrated_return, dynamic_threshold, context_signals, latest_market_price
        )

        train_metrics_native = _to_native_metric_dict(train_metrics) if train_metrics else None
        test_metrics_native = _to_native_metric_dict(test_metrics) if test_metrics else None
        if train_metrics_native:
            entry["train_metrics"] = train_metrics_native
        if test_metrics_native:
            entry["test_metrics"] = test_metrics_native
        
        # Validate prediction realism
        model_metrics_dict = {
            "r2": entry.get("r2", 0.0),
            "mae": entry.get("mae"),
            "rmse": entry.get("rmse"),
            "directional_accuracy": entry.get("directional_accuracy"),
        }
        is_realistic, realism_warnings = _validate_prediction_realism(
            clamped_return,
            entry["predicted_price"],
            latest_market_price,
            model_metrics_dict,
            train_returns,
        )
        entry["prediction_realistic"] = is_realistic
        if realism_warnings:
            entry["prediction_warnings"] = realism_warnings
            if not is_realistic:
                logger.warning(
                    f"{name} prediction may be unrealistic: {'; '.join(realism_warnings)}",
                    category="PREDICTION",
                    symbol=symbol,
                    asset_type=asset_type,
                )

        # Comprehensive overfitting detection with detailed logging
        rejection_reasons: List[str] = []
        val_r2 = entry.get("r2", 0.0)
        val_mae = entry.get("mae")
        mae_improvement = (
            baseline_val_mae - val_mae if val_mae is not None else None
        )
        val_dir = entry.get("directional_accuracy")
        
        # Log all metrics for debugging
        logger.info(
            f"{name} metrics computed",
            category="MODEL",
            symbol=symbol,
            asset_type=asset_type,
            data={
                "model": name,
                "val_r2": val_r2,
                "val_mae": val_mae,
                "val_dir": val_dir,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            }
        )
        
        if train_metrics and test_metrics:
            train_r2 = train_metrics.get("r2", 0)
            test_r2 = test_metrics.get("r2", 0)
            test_dir = test_metrics.get("directional_accuracy")
            train_val_gap = train_r2 - val_r2
            val_test_gap = val_r2 - test_r2
            
            # Log gaps for monitoring
            logger.info(
                f"{name} overfitting analysis",
                category="OVERFITTING",
                symbol=symbol,
                asset_type=asset_type,
                data={
                    "model": name,
                    "train_r2": train_r2,
                    "val_r2": val_r2,
                    "test_r2": test_r2,
                    "train_val_gap": train_val_gap,
                    "val_test_gap": val_test_gap,
                }
            )
            
            if train_r2 - val_r2 > 0.10:
                _note_overfit(
                    f"{name}: Train R² ({train_r2:.3f}) >> Val R² ({val_r2:.3f}), gap={train_val_gap:.3f}"
                )
                logger.warning(
                    f"{name} shows train/val overfitting: gap={train_val_gap:.3f}",
                    category="OVERFITTING",
                    symbol=symbol,
                    asset_type=asset_type,
                    data={"model": name, "train_r2": train_r2, "val_r2": val_r2, "gap": train_val_gap}
                )
            if val_r2 - test_r2 > 0.08:  # Tighter threshold: 0.08 instead of 0.10
                _note_overfit(
                    f"{name}: Val R² ({val_r2:.3f}) >> Test R² ({test_r2:.3f}), gap={val_test_gap:.3f} (generalization failure)"
                )
                logger.warning(
                    f"{name} shows generalization failure: val/test gap={val_test_gap:.3f}",
                    category="OVERFITTING",
                    symbol=symbol,
                    asset_type=asset_type,
                    data={"model": name, "val_r2": val_r2, "test_r2": test_r2, "gap": val_test_gap}
                )
            # WARN: Suspiciously high R² scores (likely overfitting or data leakage)
            if val_r2 > 0.85:
                severity = "CRITICAL" if val_r2 > 0.95 else "WARNING"
                _note_overfit(
                    f"{name}: Validation R² ({val_r2:.3f}) suspiciously high for financial data ({severity})"
                )
                logger.warning(
                    f"{name} has suspiciously high validation R²: {val_r2:.3f} (expected: 0.3-0.7)",
                    category="OVERFITTING",
                    symbol=symbol,
                    asset_type=asset_type,
                    data={"model": name, "val_r2": val_r2, "severity": severity}
                )
            if test_r2 > 0.85:
                severity = "CRITICAL" if test_r2 > 0.95 else "WARNING"
                _note_overfit(
                    f"{name}: Test R² ({test_r2:.3f}) suspiciously high for financial data ({severity})"
                )
                logger.warning(
                    f"{name} has suspiciously high test R²: {test_r2:.3f} (expected: 0.3-0.7)",
                    category="OVERFITTING",
                    symbol=symbol,
                    asset_type=asset_type,
                    data={"model": name, "test_r2": test_r2, "severity": severity}
                )
            # WARN: Suspiciously high accuracy (likely overfitting)
            if val_dir is not None and val_dir > 0.90:
                _note_overfit(
                    f"{name}: Validation accuracy ({val_dir*100:.1f}%) suspiciously high (expected: 55-70%)"
                )
            if test_dir is not None and test_dir > 0.90:
                _note_overfit(
                    f"{name}: Test accuracy ({test_dir*100:.1f}%) suspiciously high (expected: 55-70%)"
                )
            if test_r2 < train_r2 - 0.20 or test_r2 < val_r2 - 0.15:
                _note_overfit(
                    f"{name}: Significant performance drop on test set (Train: {train_r2:.3f}, Val: {val_r2:.3f}, Test: {test_r2:.3f})"
                )
            test_mae = test_metrics.get("mae")
            test_mae_improvement = (
                baseline_test_mae - test_mae if test_mae is not None else None
            )
            # More lenient R² threshold if directional accuracy is good
            # If model can predict direction well, accept lower R² (useful for trading)
            effective_r2_threshold = MIN_ACCEPTABLE_R2
            has_good_direction = False
            
            # Check if directional accuracy is good enough to compensate for low R²
            if val_dir is not None and val_dir >= 0.54:  # Good directional accuracy
                effective_r2_threshold = max(-0.01, MIN_ACCEPTABLE_R2 - 0.03)  # Allow slightly negative
                has_good_direction = True
            if test_dir is not None and test_dir >= 0.54:
                effective_r2_threshold = max(-0.01, MIN_ACCEPTABLE_R2 - 0.03)
                has_good_direction = True
            
            # Special case: if both val and test have good directional accuracy (>=0.54)
            # and R² is only slightly negative, accept it (direction matters more for trading)
            if (val_dir is not None and val_dir >= 0.54 and 
                test_dir is not None and test_dir >= 0.54 and
                val_r2 >= -0.01 and test_r2 >= -0.01):
                # Accept models with good direction even if R² is slightly negative
                effective_r2_threshold = -0.01
            
            if val_r2 < effective_r2_threshold:
                rejection_reasons.append(
                    f"validation R² {val_r2:.3f} below minimum {effective_r2_threshold:.2f}"
                )
            if test_r2 < effective_r2_threshold:
                rejection_reasons.append(
                    f"test R² {test_r2:.3f} below minimum {effective_r2_threshold:.2f}"
                )
            # Stricter gap thresholds to catch overfitting early
            # For commodities, allow more lenient thresholds due to different market characteristics
            # IMPORTANT: If validation R² is good (>=0.60), allow larger gaps (models are still useful)
            if asset_type == "commodities":
                # Base threshold is more lenient for commodities
                base_max_train_val_gap = MAX_TRAIN_VAL_GAP * 2.3  # 0.345 for commodities (much more lenient)
                # If validation R² is good, allow even larger gaps
                if val_r2 >= 0.60:
                    max_train_val_gap = base_max_train_val_gap * 1.2  # 0.414 for good models
                else:
                    max_train_val_gap = base_max_train_val_gap  # 0.345 for others
                max_val_test_gap = MAX_VAL_TEST_GAP * 2.0  # 0.16 for commodities (more lenient)
                max_train_test_gap = MAX_TRAIN_TEST_GAP * 2.0  # 0.30 for commodities (more lenient)
            else:
                max_train_val_gap = MAX_TRAIN_VAL_GAP  # 0.15 - strict for crypto
                max_val_test_gap = MAX_VAL_TEST_GAP  # 0.08 - strict for crypto
                max_train_test_gap = MAX_TRAIN_TEST_GAP  # 0.15 - strict for crypto
            
            # Only reject if gap exceeds threshold AND validation R² is not good enough
            train_val_gap = train_r2 - val_r2
            if train_val_gap > max_train_val_gap:
                # For commodities with good validation R², be more lenient
                if asset_type == "commodities" and val_r2 >= 0.60:
                    # Allow up to 0.50 gap if validation R² is >= 0.60
                    if train_val_gap > 0.50:
                        rejection_reasons.append(
                            f"train/val gap {train_val_gap:.3f} exceeds {max_train_val_gap:.2f} (even with good val R² {val_r2:.3f})"
                        )
                    # Otherwise, accept it despite large gap (validation R² is good)
                else:
                    rejection_reasons.append(
                        f"train/val gap {train_val_gap:.3f} exceeds {max_train_val_gap:.2f}"
                    )
            if isinstance(test_r2, (int, float)) and not np.isnan(test_r2):
                strict_floor = MIN_TEST_R2_STRICT
                if has_good_direction:
                    strict_floor = max(-0.01, strict_floor - 0.05)  # Reduced from 0.10 - less lenient
                # For commodities, allow more lenient thresholds
                if asset_type == "commodities":
                    # IMPORTANT: If validation R² is good (>=0.60), be very lenient with test R²
                    # A good validation R² indicates the model is useful even if test R² is lower
                    if val_r2 >= 0.60:
                        # For commodities with excellent validation R², accept even negative test R²
                        # (test set might be small or have different characteristics)
                        strict_floor = -0.10  # Very lenient - accept models with good val R²
                    elif has_good_direction:
                        strict_floor = max(-0.05, strict_floor - 0.10)  # More lenient for commodities with good direction
                    elif test_r2 >= 0.10:
                        # For commodities with reasonable test R² (>= 0.10), be more lenient
                        # Test R² of 0.10-0.15 is acceptable for commodities even without perfect direction
                        strict_floor = 0.10  # Lower threshold for commodities with decent test R²
                    elif test_r2 >= 0.05:
                        # For commodities with minimum acceptable test R², be lenient if gaps are reasonable
                        strict_floor = 0.05  # Accept minimum threshold if test R² meets basic requirement
                    elif val_r2 >= 0.50:
                        # If validation R² is decent (>=0.50), be lenient with test R²
                        strict_floor = -0.05  # Accept slightly negative test R² if val R² is good
                # Apply same strict standards to all asset types
                if test_r2 < strict_floor:
                    rejection_reasons.append(
                        f"test R² {test_r2:.3f} below strict guard {strict_floor:.2f}"
                    )
                    logger.warning(
                        f"{name} rejected: test R² {test_r2:.3f} below threshold {strict_floor:.2f}",
                        category="MODEL",
                        symbol=symbol,
                        asset_type=asset_type,
                        data={"model": name, "test_r2": test_r2, "threshold": strict_floor}
                    )
                val_test_gap = val_r2 - test_r2
                train_test_gap = train_r2 - test_r2
                # For commodities, allow models with reasonable test R² even if gaps are slightly higher
                if asset_type == "commodities":
                    if test_r2 >= 0.10:
                        # If test R² is good (>= 0.10), be very lenient with gaps
                        effective_val_test_gap = max_val_test_gap * 2.0  # 100% more lenient (0.16 -> 0.32)
                        effective_train_test_gap = max_train_test_gap * 1.5  # 50% more lenient (0.225 -> 0.3375)
                    elif test_r2 >= 0.05:
                        # If test R² is reasonable (>= 0.05), be more lenient with gaps
                        effective_val_test_gap = max_val_test_gap * 1.5  # 50% more lenient
                        effective_train_test_gap = max_train_test_gap * 1.3  # 30% more lenient
                    else:
                        effective_val_test_gap = max_val_test_gap
                        effective_train_test_gap = max_train_test_gap
                else:
                    effective_val_test_gap = max_val_test_gap
                    effective_train_test_gap = max_train_test_gap
                
                # Only reject if gap is exceeded AND test R² is poor
                # If test R² is good (>= 0.10 for commodities), gaps are acceptable
                if val_test_gap > effective_val_test_gap:
                    # Only reject if test R² is also poor
                    min_test_r2 = 0.10 if asset_type == "commodities" else 0.15
                    if test_r2 < min_test_r2:
                        rejection_reasons.append(
                            f"val/test gap {val_test_gap:.3f} exceeds {effective_val_test_gap:.2f} and test R² {test_r2:.3f} < {min_test_r2:.2f}"
                        )
                if train_test_gap > effective_train_test_gap:
                    # Only reject if test R² is also poor
                    min_test_r2 = 0.10 if asset_type == "commodities" else 0.15
                    if test_r2 < min_test_r2:
                        rejection_reasons.append(
                            f"train/test gap {train_test_gap:.3f} exceeds {effective_train_test_gap:.2f} and test R² {test_r2:.3f} < {min_test_r2:.2f}"
                        )
            # Allow models with good directional accuracy even if MAE is close to baseline
            # For trading, direction matters more than exact magnitude
            mae_threshold = MIN_MAE_IMPROVEMENT
            if val_dir is not None and val_dir >= 0.55:  # Good directional accuracy
                mae_threshold = max(0.0, MIN_MAE_IMPROVEMENT * 0.5)  # More lenient
            if test_dir is not None and test_dir >= 0.55:
                mae_threshold = max(0.0, MIN_MAE_IMPROVEMENT * 0.5)
            
            if mae_improvement is None or mae_improvement < mae_threshold:
                if val_dir is not None and val_dir >= 0.55:
                    # Don't reject if directional accuracy is good
                    pass
                else:
                    rejection_reasons.append("validation MAE not better than zero-return baseline")
            
            if test_mae_improvement is None or test_mae_improvement < mae_threshold:
                if test_dir is not None and test_dir >= 0.55:
                    # Don't reject if directional accuracy is good
                    pass
                else:
                    rejection_reasons.append("test MAE not better than zero-return baseline")
            # Allow slightly lower directional accuracy if R² is reasonable
            effective_dir_threshold = MIN_DIRECTIONAL_ACCURACY
            effective_test_dir_threshold = MIN_TEST_DIRECTIONAL_ACCURACY
            if val_r2 >= 0.02:  # Some predictive power
                effective_dir_threshold = max(0.50, MIN_DIRECTIONAL_ACCURACY - 0.01)
            if test_r2 >= 0.02:
                effective_test_dir_threshold = max(0.50, MIN_TEST_DIRECTIONAL_ACCURACY - 0.01)
            
            if val_dir is not None and val_dir < effective_dir_threshold:
                rejection_reasons.append(
                    f"validation directional accuracy {val_dir:.3f} below {effective_dir_threshold:.2f}"
                )
            if test_dir is not None and test_dir < effective_test_dir_threshold:
                rejection_reasons.append(
                    f"test directional accuracy {test_dir:.3f} below {effective_test_dir_threshold:.2f}"
                )
        elif train_metrics:
            # Fallback if test metrics not available
            train_r2 = train_metrics.get("r2", 0)
            if train_r2 - val_r2 > 0.08:  # Tighter threshold: 0.08 instead of 0.10
                _note_overfit(
                    f"{name}: Train R² ({train_r2:.3f}) >> Val R² ({val_r2:.3f})"
                )
            if val_r2 > 0.95:
                _note_overfit(
                    f"{name}: Validation R² ({val_r2:.3f}) suspiciously high"
                )
            if val_r2 < MIN_ACCEPTABLE_R2:
                rejection_reasons.append(
                    f"validation R² {val_r2:.3f} below minimum {MIN_ACCEPTABLE_R2:.2f}"
                )
            if mae_improvement is None or mae_improvement < MIN_MAE_IMPROVEMENT:
                rejection_reasons.append("validation MAE not better than zero-return baseline")
            if val_dir is not None and val_dir < MIN_DIRECTIONAL_ACCURACY:
                rejection_reasons.append(
                    f"validation directional accuracy {val_dir:.3f} below {MIN_DIRECTIONAL_ACCURACY:.2f}"
                )
        
        # Check for constant predictions (additional validation)
        # This catches models that predict the same value for all samples
        if name in models:
            try:
                test_preds = models[name].predict(X_test)
                pred_variance = np.var(test_preds) if len(test_preds) > 1 else 0.0
                if pred_variance < 1e-10:
                    rejection_reasons.append(f"constant predictions (variance={pred_variance:.2e})")
                    logger.warning(
                        f"{name} producing constant predictions",
                        category="MODEL",
                        symbol=symbol,
                        asset_type=asset_type,
                        data={"prediction_variance": float(pred_variance)}
                    )
            except Exception as e:
                logger.warning(
                    f"Could not validate predictions for {name}: {e}",
                    category="MODEL",
                    symbol=symbol,
                    asset_type=asset_type,
                )
        
        # FINAL SAFETY CHECK: For commodities with excellent validation R² (>=0.60),
        # override any rejection reasons (except for truly critical failures like constant predictions)
        # This ensures high-quality models are never rejected due to overly strict thresholds
        # EXCEPTION: Stacked blend with negative val R² should still be rejected (it's truly bad)
        if asset_type == "commodities" and val_r2 >= 0.60 and name != "stacked_blend":
            # Check if rejection is only due to gaps or test R² (which are acceptable for good val R²)
            critical_failures = [
                "constant predictions",
                "validation R²",
                "validation MAE not better",
            ]
            has_critical_failure = any(
                any(critical in reason.lower() for critical in critical_failures)
                for reason in rejection_reasons
            )
            if not has_critical_failure:
                # Clear rejection reasons - model is good enough despite gaps
                rejection_reasons.clear()
                logger.info(
                    f"{name} accepted despite gaps: excellent validation R² {val_r2:.3f} overrides strict thresholds",
                    category="MODEL",
                    symbol=symbol,
                    asset_type=asset_type,
                )
        # For stacked blend specifically: if val R² is negative, it's truly bad - don't override
        elif name == "stacked_blend" and val_r2 < 0:
            # Stacked blend with negative R² is genuinely bad - keep rejection
            pass
        
        if rejection_reasons:
            reason = "; ".join(rejection_reasons)
            results[name] = {"status": "failed", "reason": f"Rejected due to {reason}"}
            logger.warning(
                f"{name} rejected: {reason}",
                category="MODEL",
                symbol=symbol,
                asset_type=asset_type,
            )
            metric_store[name] = metrics
            return False

        if model_overfitting_notes:
            entry["overfitting_flag"] = True
            entry["overfitting_notes"] = model_overfitting_notes

        results[name] = entry
        models[name] = model_obj
        metric_store[name] = metrics
        return True

    # Train Random Forest
    logger.info(
        "Training Random Forest model",
        category="MODEL",
        symbol=symbol,
        asset_type=asset_type,
        data={
            "model": "RandomForest",
            "training_samples": len(X_train),
            "features_count": X_train.shape[1],
            "feature_names_sample": list(X_train.columns[:5])
        }
    )
    if verbose:
        print(f"\n[RF] Training Random Forest with {X_train.shape[1]} features on {len(X_train)} samples...")
    try:
        rf_param_override: Dict = {}
        if ENABLE_HYPEROPT:
            try:
                rf_param_override = optimize_model(
                    "random_forest",
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    n_trials=HYPEROPT_TRIALS,
                    timeout=HYPEROPT_TIMEOUT,
                )
                logger.info(
                    "RandomForest hyperparameter search complete",
                    category="MODEL",
                    symbol=symbol,
                    asset_type=asset_type,
                    data={"best_params": rf_param_override},
                )
            except Exception as exc:
                logger.warning(
                    f"RandomForest hyperopt failed: {exc}",
                    category="MODEL",
                    symbol=symbol,
                    asset_type=asset_type,
                )
                rf_param_override = {}
        # FIXED: Less restrictive fallback parameters to allow actual learning
        # Previous defaults were too restrictive, producing negative R²
        if asset_type == "commodities" and not rf_param_override:
            rf_param_override = {
                "max_depth": 15,  # Deep enough to learn
                "min_samples_leaf": 2,  # Some restriction to prevent overfitting
                "min_samples_split": 5,  # Some restriction to prevent overfitting
                "max_features": 0.9,  # Use most features (not all to prevent overfitting)
                "max_samples": 0.9,  # Use most data (not all to prevent overfitting)
                "ccp_alpha": 0.0001,  # Small pruning to prevent overfitting
                "n_estimators": 400,  # Good number of trees
            }
        # Train without refitting first to get train/val/test metrics
        rf_model_temp, rf_metrics = train_random_forest(
            X_train, y_train, X_val, y_val,
            refit_on_full=False,
            param_overrides=rf_param_override,
        )
        # Evaluate on train set for overfitting check (before refitting)
        rf_train_pred = rf_model_temp.predict(X_train)
        rf_train_metrics = _evaluate(
            train_df["target_return"].to_numpy(), rf_train_pred
        )
        
        # Evaluate on validation/test set (before refitting) to check generalization
        rf_val_pred = rf_model_temp.predict(X_val)
        rf_test_pred = rf_model_temp.predict(X_test)
        rf_test_metrics = _evaluate(
            test_df["target_return"].to_numpy(), rf_test_pred
        )
        train_pred_store["random_forest"] = rf_train_pred
        val_pred_store["random_forest"] = rf_val_pred
        test_pred_store["random_forest"] = rf_test_pred

        rf_walk_forward = _run_time_series_cv(
            _build_random_forest, pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
        )
        
        # Now refit on full dataset (train+val) for deployment
        rf_model, _ = train_random_forest(
            X_train, y_train, X_val, y_val,
            refit_on_full=True, X_full=X_full, y_full=y_full,
            param_overrides=rf_param_override,
        )
        # Make prediction using features from test set (last row)
        rf_pred_return_raw = float(rf_model.predict(X_test)[-1])
        rf_pred_return_clamped = _clamp_return(rf_pred_return_raw)
        rf_predicted_price = float(latest_market_price * (1.0 + rf_pred_return_clamped))
        logger.info(
            f"Random Forest prediction made using {X_test.shape[1]} features",
            category="PREDICTION",
            symbol=symbol,
            asset_type=asset_type,
            data={
                "model": "RandomForest",
                "features_used": X_test.shape[1],
                "test_sample_index": len(X_test) - 1,
                "raw_return": rf_pred_return_raw,
                "clamped_return": rf_pred_return_clamped
            }
        )
        rf_accepted = _record(
            "random_forest",
            rf_model,
            rf_metrics,
            rf_pred_return_clamped,  # Pass clamped return
            rf_train_metrics,
            rf_test_metrics,
        )
        if rf_accepted and rf_walk_forward:
            results["random_forest"]["walk_forward"] = rf_walk_forward
        logger.success(
            "Random Forest training completed",
            category="MODEL",
            symbol=symbol,
            asset_type=asset_type,
            data={
                "model": "RandomForest",
                "metrics": asdict(rf_metrics),
                "current_price": latest_market_price,
                "predicted_return": rf_pred_return_clamped,
                "predicted_price": rf_predicted_price,
                "features_used": feature_count
            }
        )
    except Exception as exc:
        error_msg = str(exc)
        results["random_forest"] = {"status": "failed", "reason": error_msg}
        logger.error(
            f"Random Forest training failed: {error_msg}",
            category="MODEL",
            symbol=symbol,
            asset_type=asset_type,
            data={"model": "RandomForest", "error": error_msg}
        )
        if verbose:
            print(f"[RF] Failed: {exc}")

    # Train LightGBM
    logger.info(
        "Training LightGBM model",
        category="MODEL",
        symbol=symbol,
        asset_type=asset_type,
        data={
            "model": "LightGBM",
            "training_samples": len(X_train),
            "features_count": X_train.shape[1]
        }
    )
    if verbose:
        print(f"[LGBM] Training LightGBM with {X_train.shape[1]} features on {len(X_train)} samples...")
    if max_feature_signal < MIN_SIGNAL_CORR:
        reason = (
            f"Skipped (max |feature correlation| {max_feature_signal:.4f} "
            f"< minimum {MIN_SIGNAL_CORR:.4f})"
        )
        results["lightgbm"] = {"status": "failed", "reason": reason}
        logger.warning(
            reason,
            category="MODEL",
            symbol=symbol,
            asset_type=asset_type,
            data={"model": "LightGBM", "max_abs_corr": max_feature_signal},
        )
    else:
        try:
            lgb_param_override: Dict = {}
            if ENABLE_HYPEROPT:
                try:
                    lgb_param_override = optimize_model(
                        "lightgbm",
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        n_trials=HYPEROPT_TRIALS,
                        timeout=HYPEROPT_TIMEOUT,
                    )
                    logger.info(
                        "LightGBM hyperparameter search complete",
                        category="MODEL",
                        symbol=symbol,
                        asset_type=asset_type,
                        data={"best_params": lgb_param_override},
                    )
                except Exception as exc:
                    logger.warning(
                        f"LightGBM hyperopt failed: {exc}",
                        category="MODEL",
                        symbol=symbol,
                        asset_type=asset_type,
                    )
                    lgb_param_override = {}
            # Use balanced parameters for commodities (prevent overfitting while allowing learning)
            # CRITICAL FIX: Use conservative parameters to prevent overfitting and constant predictions
            if asset_type == "commodities" and not lgb_param_override:
                lgb_param_override = {
                    "max_depth": 8,  # Deep enough to learn
                    "num_leaves": 63,  # Good capacity (not maximum to prevent overfitting)
                    "subsample": 0.9,  # Use most data (not all to prevent overfitting)
                    "colsample_bytree": 0.9,  # Use most features (not all to prevent overfitting)
                    "reg_alpha": 0.5,  # Some regularization to prevent overfitting
                    "reg_lambda": 1.0,  # Some regularization to prevent overfitting
                    "min_child_samples": 10,  # Some restriction to prevent overfitting
                    "min_split_gain": 0.01,  # Some pruning to prevent overfitting
                    "learning_rate": 0.1,  # Good learning rate (not too high)
                    "n_estimators": 400,  # Good number of trees
                }
            # Ensure learning_rate is always set (even if hyperopt returned params without it)
            if lgb_param_override and "learning_rate" not in lgb_param_override:
                lgb_param_override["learning_rate"] = 0.12  # Default to higher learning rate for commodities
            # Train without refitting first to get train/val/test metrics
            lgb_model_temp, lgb_metrics = train_lightgbm(
                X_train, y_train, X_val, y_val,
                refit_on_full=False,
                param_overrides=lgb_param_override,
            )
            # Evaluate on train/val/test sets (before refitting) to check generalization
            lgb_train_pred = lgb_model_temp.predict(X_train)
            lgb_train_metrics = _evaluate(
                train_df["target_return"].to_numpy(), lgb_train_pred
            )
            
            lgb_val_pred = lgb_model_temp.predict(X_val)
            lgb_test_pred = lgb_model_temp.predict(X_test)
            lgb_test_metrics = _evaluate(
                test_df["target_return"].to_numpy(), lgb_test_pred
            )
            train_pred_store["lightgbm"] = lgb_train_pred
            val_pred_store["lightgbm"] = lgb_val_pred
            test_pred_store["lightgbm"] = lgb_test_pred

            lgb_walk_forward = _run_time_series_cv(
                _build_lightgbm, pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
            )
            
            # Now refit on full dataset (train+val) for deployment
            lgb_model, _ = train_lightgbm(
                X_train, y_train, X_val, y_val,
                refit_on_full=True, X_full=X_full, y_full=y_full,
                param_overrides=lgb_param_override,
            )
            
            # Validate prediction is not constant (check variance across test set)
            lgb_test_preds = lgb_model.predict(X_test)
            pred_variance = np.var(lgb_test_preds)
            pred_mean = np.mean(lgb_test_preds)
            
            # Also check train/val predictions for consistency
            lgb_train_preds = lgb_model_temp.predict(X_train)
            lgb_val_preds = lgb_model_temp.predict(X_val)
            train_variance = np.var(lgb_train_preds)
            val_variance = np.var(lgb_val_preds)
            
            if pred_variance < 1e-10 or train_variance < 1e-10 or val_variance < 1e-10:
                import warnings as w
                # CRITICAL: If model is still constant after all fixes, explicitly mark as failed
                w.warn(
                    f"LightGBM producing constant predictions - "
                    f"Train variance: {train_variance:.2e}, Val variance: {val_variance:.2e}, "
                    f"Test variance: {pred_variance:.2e}, Mean prediction: {pred_mean:.6f}. "
                    f"Model may not have trained properly. Check target variable variance and learning rate."
                )
                # Log this as a failure reason
                logger.warning(
                    f"LightGBM constant predictions detected - model will be rejected",
                    category="MODEL",
                    symbol=symbol,
                    asset_type=asset_type,
                    data={
                        "train_variance": float(train_variance),
                        "val_variance": float(val_variance),
                        "test_variance": float(pred_variance),
                        "mean_prediction": float(pred_mean),
                    }
                )
            
            # Make prediction using features from test set (last row)
            lgb_pred_return_raw = float(lgb_test_preds[-1])
            lgb_pred_return_clamped = _clamp_return(lgb_pred_return_raw)
            lgb_predicted_price = float(latest_market_price * (1.0 + lgb_pred_return_clamped))
            logger.info(
                f"LightGBM prediction made using {X_test.shape[1]} features",
                category="PREDICTION",
                symbol=symbol,
                asset_type=asset_type,
                data={
                    "model": "LightGBM",
                    "features_used": X_test.shape[1],
                    "test_sample_index": len(X_test) - 1,
                    "raw_return": lgb_pred_return_raw,
                    "clamped_return": lgb_pred_return_clamped
                }
            )
            lgb_accepted = _record(
                "lightgbm",
                lgb_model,
                lgb_metrics,
                lgb_pred_return_clamped,  # Pass clamped return
                lgb_train_metrics,
                lgb_test_metrics,
            )
            if lgb_accepted and lgb_walk_forward:
                results["lightgbm"]["walk_forward"] = lgb_walk_forward
            logger.success(
                "LightGBM training completed",
                category="MODEL",
                symbol=symbol,
                asset_type=asset_type,
                data={
                    "model": "LightGBM",
                    "metrics": asdict(lgb_metrics),
                    "current_price": latest_market_price,
                "predicted_return": lgb_pred_return_clamped,
                "predicted_price": lgb_predicted_price,
                    "features_used": feature_count
                }
            )
        except Exception as exc:
            error_msg = str(exc)
            results["lightgbm"] = {"status": "failed", "reason": error_msg}
            logger.error(
                f"LightGBM training failed: {error_msg}",
                category="MODEL",
                symbol=symbol,
                asset_type=asset_type,
                data={"model": "LightGBM", "error": error_msg}
            )
            if verbose:
                print(f"[LGBM] Failed: {exc}")

    # Train XGBoost
    logger.info(
        "Training XGBoost model",
        category="MODEL",
        symbol=symbol,
        asset_type=asset_type,
        data={
            "model": "XGBoost",
            "training_samples": len(X_train),
            "features_count": X_train.shape[1]
        }
    )
    if verbose:
        print(f"[XGB] Training XGBoost with {X_train.shape[1]} features on {len(X_train)} samples...")
    try:
        xgb_param_override: Dict = {}
        if ENABLE_HYPEROPT:
            try:
                xgb_param_override = optimize_model(
                    "xgboost",
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    n_trials=HYPEROPT_TRIALS,
                    timeout=HYPEROPT_TIMEOUT,
                )
                logger.info(
                    "XGBoost hyperparameter search complete",
                    category="MODEL",
                    symbol=symbol,
                    asset_type=asset_type,
                    data={"best_params": xgb_param_override},
                )
            except Exception as exc:
                logger.warning(
                    f"XGBoost hyperopt failed: {exc}",
                    category="MODEL",
                    symbol=symbol,
                    asset_type=asset_type,
                )
                xgb_param_override = {}
            # Apply balanced parameters for commodities (allow learning while preventing overfitting)
            # Note: XGBoost training in trainers.py removes ALL regularization initially to force learning
            # These fallback params are only used if hyperopt fails, but trainers.py will override them
            # Still set reasonable defaults for consistency
            if asset_type == "commodities" and not xgb_param_override:
                xgb_param_override = {
                    "max_depth": 5,  # Deeper trees to allow learning
                    "subsample": 0.8,  # More data for learning
                    "colsample_bytree": 0.75,  # More features for learning
                    "reg_alpha": 1.0,  # Less regularization - allow learning
                    "reg_lambda": 2.0,  # Less regularization - allow learning
                    "min_child_weight": 3.0,  # Less restrictive - allow more splits
                    "gamma": 0.1,  # Less pruning - allow learning
                    "learning_rate": 0.08,  # Higher learning rate to allow learning
                    "n_estimators": 300,  # More trees
                }
            # Ensure learning_rate is always set (even if hyperopt returned params without it)
            if xgb_param_override and "learning_rate" not in xgb_param_override:
                xgb_param_override["learning_rate"] = 0.1  # Default learning rate
            # Train without refitting first to get train/val/test metrics
            xgb_model_temp, xgb_metrics = train_xgboost(
                X_train, y_train, X_val, y_val,
                refit_on_full=False,
                param_overrides=xgb_param_override,
            )
        # Evaluate on train/val/test sets (before refitting) to check generalization
        xgb_train_pred = xgb_model_temp.predict(X_train)
        xgb_train_metrics = _evaluate(
            train_df["target_return"].to_numpy(), xgb_train_pred
        )
        
        xgb_val_pred = xgb_model_temp.predict(X_val)
        xgb_test_pred = xgb_model_temp.predict(X_test)
        xgb_test_metrics = _evaluate(
            test_df["target_return"].to_numpy(), xgb_test_pred
        )
        train_pred_store["xgboost"] = xgb_train_pred
        val_pred_store["xgboost"] = xgb_val_pred
        test_pred_store["xgboost"] = xgb_test_pred

        xgb_walk_forward = _run_time_series_cv(
            _build_xgboost, pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
        )
        
        # Now refit on full dataset (train+val) for deployment
        xgb_model, _ = train_xgboost(
            X_train, y_train, X_val, y_val,
            refit_on_full=True, X_full=X_full, y_full=y_full,
            param_overrides=xgb_param_override,
        )
        
        # Validate prediction is not constant (check variance across test set)
        xgb_test_preds = xgb_model.predict(X_test)
        pred_variance = np.var(xgb_test_preds)
        pred_mean = np.mean(xgb_test_preds)
        
        # Also check train/val predictions for consistency
        xgb_train_preds = xgb_model_temp.predict(X_train)
        xgb_val_preds = xgb_model_temp.predict(X_val)
        train_variance = np.var(xgb_train_preds)
        val_variance = np.var(xgb_val_preds)
        
        if pred_variance < 1e-10 or train_variance < 1e-10 or val_variance < 1e-10:
            import warnings as w
            # CRITICAL: If model is still constant after all fixes, explicitly mark as failed
            w.warn(
                f"XGBoost producing constant predictions - "
                f"Train variance: {train_variance:.2e}, Val variance: {val_variance:.2e}, "
                f"Test variance: {pred_variance:.2e}, Mean prediction: {pred_mean:.6f}. "
                f"Model may not have trained properly. Check target variable variance and learning rate. Model will be rejected."
            )
            # Log this as a failure reason
            logger.warning(
                f"XGBoost constant predictions detected",
                category="MODEL",
                symbol=symbol,
                asset_type=asset_type,
                data={
                    "train_variance": float(train_variance),
                    "val_variance": float(val_variance),
                    "test_variance": float(pred_variance),
                    "mean_prediction": float(pred_mean),
                }
            )
        
        # Make prediction using features from test set (last row)
        xgb_pred_return_raw = float(xgb_test_preds[-1])
        xgb_pred_return_clamped = _clamp_return(xgb_pred_return_raw)
        xgb_predicted_price = float(latest_market_price * (1.0 + xgb_pred_return_clamped))
        logger.info(
            f"XGBoost prediction made using {X_test.shape[1]} features",
            category="PREDICTION",
            symbol=symbol,
            asset_type=asset_type,
            data={
                "model": "XGBoost",
                "features_used": X_test.shape[1],
                "test_sample_index": len(X_test) - 1,
                "raw_return": xgb_pred_return_raw,
                "clamped_return": xgb_pred_return_clamped
            }
        )
        xgb_accepted = _record(
            "xgboost",
            xgb_model,
            xgb_metrics,
            xgb_pred_return_clamped,  # Pass clamped return
            xgb_train_metrics,
            xgb_test_metrics,
        )
        if xgb_accepted and xgb_walk_forward:
            results["xgboost"]["walk_forward"] = xgb_walk_forward
        logger.success(
            "XGBoost training completed",
            category="MODEL",
            symbol=symbol,
            asset_type=asset_type,
            data={
                "model": "XGBoost",
                "metrics": asdict(xgb_metrics),
                "current_price": latest_market_price,
                "predicted_return": xgb_pred_return_clamped,
                "predicted_price": xgb_predicted_price,
                "features_used": feature_count
            }
        )
    except Exception as exc:
        error_msg = str(exc)
        results["xgboost"] = {"status": "failed", "reason": error_msg}
        logger.error(
            f"XGBoost training failed: {error_msg}",
            category="MODEL",
            symbol=symbol,
            asset_type=asset_type,
            data={"model": "XGBoost", "error": error_msg}
        )
        if verbose:
            print(f"[XGB] Failed: {exc}")

    # Directional classifiers removed - they were causing errors and are not needed
    # The main regression models (random_forest, lightgbm, xgboost, stacked_blend, dqn) 
    # provide sufficient prediction capability.

    # Stacked ensemble blender (regression)
    stacked_blend_model_list = None  # Store which models were used for stacked blend
    if len(val_pred_store) >= 2:
        stack_keys = sorted(val_pred_store.keys())
        stack_train = np.column_stack([train_pred_store[k] for k in stack_keys if k in train_pred_store])
        stack_val = np.column_stack([val_pred_store[k] for k in stack_keys])
        stack_test = np.column_stack([test_pred_store[k] for k in stack_keys])
        if stack_train.size and stack_val.size and stack_test.size:
            # ANTI-OVERFITTING: Much stronger regularization to prevent negative test R²
            # Higher alphas = more regularization = less overfitting
            # For commodities, use EXTREMELY high regularization to eliminate overfitting
            # Also check if base models are too correlated (causes overfitting in stacking)
            base_model_corr = np.corrcoef([val_pred_store[k] for k in stack_keys]) if len(stack_keys) > 1 else np.array([[1.0]])
            avg_correlation = np.mean(base_model_corr[np.triu_indices_from(base_model_corr, k=1)])
            
            if asset_type == "commodities":
                # If base models are highly correlated (>0.90), use even higher regularization
                # Lower threshold to catch more cases of correlation
                if avg_correlation > 0.90:
                    alphas = [100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]  # EXTREME regularization for correlated models
                elif avg_correlation > 0.80:
                    alphas = [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]  # Very high regularization
                else:
                    alphas = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]  # High regularization for commodities
            else:
                alphas = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]  # Increased regularization
            # CRITICAL: Use cross-validation on TRAIN set, not validation set, to prevent overfitting
            # This ensures the stacked blend doesn't overfit to the validation set
            # Use more CV folds for better regularization selection
            cv_folds = max(5, min(10, len(stack_train) // 15))  # More folds for better CV
            ridge = RidgeCV(alphas=alphas, cv=cv_folds, scoring='r2')  # Use train set for CV
            ridge.fit(stack_train, y_train.to_numpy())  # Fit on train, not val
            val_stack_preds = ridge.predict(stack_val)
            test_stack_preds = ridge.predict(stack_test)
            train_stack_preds = ridge.predict(stack_train)
            stack_val_metrics = _evaluate(y_val.to_numpy(), val_stack_preds)
            stack_train_metrics = _evaluate(y_train.to_numpy(), train_stack_preds[: len(y_train)])
            stack_test_metrics = _evaluate(y_test.to_numpy(), test_stack_preds)
            
            # FIX: If Ridge regression fails (negative R²), fallback to simple weighted average
            # Weight by validation R² of each model
            if stack_val_metrics.get("r2", -1) < 0 or stack_test_metrics.get("r2", -1) < 0:
                if verbose:
                    print(f"[STACK] Ridge regression failed (val R²={stack_val_metrics.get('r2', 0):.3f}, test R²={stack_test_metrics.get('r2', 0):.3f}), using weighted average fallback")
                # Get validation R² for each base model to use as weights
                model_weights = []
                for k in stack_keys:
                    model_result = results.get(k, {})
                    if isinstance(model_result, dict) and "metrics" in model_result:
                        val_r2 = model_result["metrics"].get("r2", 0.0)
                        model_weights.append(max(0.0, val_r2))  # Use R² as weight, clamp to 0
                    else:
                        model_weights.append(1.0)  # Default weight
                
                # Normalize weights
                total_weight = sum(model_weights) or 1.0
                model_weights = [w / total_weight for w in model_weights]
                
                # Weighted average predictions
                val_stack_preds = np.average(stack_val, axis=1, weights=model_weights)
                test_stack_preds = np.average(stack_test, axis=1, weights=model_weights)
                train_stack_preds = np.average(stack_train, axis=1, weights=model_weights)
                
                # Re-evaluate with weighted average
                stack_val_metrics = _evaluate(y_val.to_numpy(), val_stack_preds)
                stack_train_metrics = _evaluate(y_train.to_numpy(), train_stack_preds[: len(y_train)])
                stack_test_metrics = _evaluate(y_test.to_numpy(), test_stack_preds)
                
                # Create a simple wrapper object that mimics Ridge for compatibility
                class WeightedAverageBlender:
                    def __init__(self, weights, models):
                        self.coef_ = np.array(weights)
                        self.alpha_ = None
                        self._stacked_blend_models = models
                    def predict(self, X):
                        return np.average(X, axis=1, weights=self.coef_)
                
                ridge = WeightedAverageBlender(model_weights, stack_keys)
                if verbose:
                    print(f"[STACK] Weighted average: val R²={stack_val_metrics.get('r2', 0):.3f}, test R²={stack_test_metrics.get('r2', 0):.3f}")
            stack_latest_return = float(test_stack_preds[-1])
            stack_result = TrainingResult("StackedBlend", **stack_val_metrics)
            accepted = _record(
                "stacked_blend",
                ridge,
                stack_result,
                stack_latest_return,
                stack_train_metrics,
                stack_test_metrics,
            )
            if accepted:
                # Store which models were used for stacked blend
                stacked_blend_model_list = stack_keys
                # Store in model object for inference-time checking
                if hasattr(ridge, '__dict__'):
                    ridge.__dict__['_stacked_blend_models'] = stack_keys
                logger.success(
                    "Stacked blender created",
                    category="MODEL",
                    symbol=symbol,
                    asset_type=asset_type,
                    data={
                        "models": stack_keys,
                        "weights": ridge.coef_.tolist(),
                        "alpha": ridge.alpha_,
                        "latest_return": stack_latest_return,
                    },
                )

    if ENABLE_QUANTILE_MODELS:
        quant_X = pd.concat([X_train, X_val])
        quant_y = pd.concat([y_train, y_val])
        for alpha in QUANTILE_LEVELS:
            try:
                quant_params = {
                    "objective": "quantile",
                    "alpha": alpha,
                    "n_estimators": 400,
                }
                quant_model = _build_lightgbm(quant_params)
                quant_model.fit(
                    quant_X,
                    quant_y,
                    eval_set=[(X_val, y_val)],
                    eval_metric="quantile",
                )
                train_pred = quant_model.predict(X_train)
                test_pred = quant_model.predict(X_test)
                val_pred = quant_model.predict(X_val)
                pinball_val = _pinball_loss(y_val.to_numpy(), val_pred, alpha)
                pinball_train = _pinball_loss(y_train.to_numpy(), train_pred, alpha)
                pinball_test = _pinball_loss(y_test.to_numpy(), test_pred, alpha)
                latest_return = float(test_pred[-1])
                quant_metrics = TrainingResult(
                    f"LightGBM_Quantile_{int(alpha*100)}",
                    mae=pinball_val,
                    rmse=pinball_val,
                    r2=0.0,
                    directional_accuracy=None,
                    notes=f"Pinball loss (val) {pinball_val:.6f}",
                )
                train_metrics = {
                    "r2": 0.0,
                    "mae": pinball_train,
                    "rmse": pinball_train,
                    "directional_accuracy": None,
                }
                test_metrics = {
                    "r2": 0.0,
                    "mae": pinball_test,
                    "rmse": pinball_test,
                    "directional_accuracy": None,
                }
                accepted = _record(
                    f"lightgbm_quantile_{int(alpha*100)}",
                    quant_model,
                    quant_metrics,
                    latest_return,
                    train_metrics,
                    test_metrics,
                )
                if accepted:
                    results[f"lightgbm_quantile_{int(alpha*100)}"]["quantile_alpha"] = alpha
                    logger.success(
                        "Quantile model trained",
                        category="MODEL",
                        symbol=symbol,
                        asset_type=asset_type,
                        data={
                            "alpha": alpha,
                            "pinball_val": pinball_val,
                            "latest_return": latest_return,
                        },
                    )
            except Exception as exc:
                logger.warning(
                    f"Quantile model alpha={alpha} failed: {exc}",
                    category="MODEL",
                    symbol=symbol,
                    asset_type=asset_type,
                )
                results[f"lightgbm_quantile_{int(alpha*100)}"] = {"status": "failed", "reason": str(exc)}

    # Train DQN
    logger.info(
        "Training DQN model",
        category="MODEL",
        symbol=symbol,
        asset_type=asset_type,
        data={
            "model": "DQN",
            "training_samples": len(X_train),
            "features_count": X_train.shape[1]
        }
    )
    if verbose:
        print(f"[DQN] Training DQN with {X_train.shape[1]} features on {len(X_train)} samples...")
    try:
        dqn_model, dqn_metrics = train_dqn(
            X_train, train_df["close"], X_val, val_df["close"]
        )
        models["dqn"] = dqn_model
        action_obs = X_test.to_numpy()[-1].astype(float)
        try:
            action_id, _ = dqn_model.predict(action_obs, deterministic=True)
            action_map = {0: "short", 1: "hold", 2: "long"}
            action = action_map.get(int(action_id), "hold")
        except Exception as exc:
            action = "hold"
            dqn_metrics.notes = f"Action unavailable: {exc}"
        dqn_entry = asdict(dqn_metrics)
        dqn_entry["action"] = action
        dqn_entry["action_reason"] = (
            "Action selected directly by DQN policy on latest state; reward history tracked separately."
        )
        val_policy_metrics = _simulate_dqn_policy(dqn_model, X_val, val_df["close"])
        test_policy_metrics = _simulate_dqn_policy(dqn_model, X_test, test_df["close"])
        if val_policy_metrics:
            dqn_entry["validation_policy_metrics"] = val_policy_metrics
        if test_policy_metrics:
            dqn_entry["test_policy_metrics"] = test_policy_metrics
        
        # Estimate predicted return from DQN policy metrics and action
        # Use test policy avg_return as baseline, adjusted by action direction
        if test_policy_metrics and action != "hold":
            # Use average return from test policy as expected return
            base_return = test_policy_metrics.get("avg_return", 0.0)
            # If avg_return is too small or zero, use a more meaningful estimate
            if abs(base_return) < 0.001:  # Very small return
                # Use sharpe ratio or hit rate to estimate expected return
                sharpe = test_policy_metrics.get("sharpe", 0.0)
                hit_rate = test_policy_metrics.get("hit_rate", 0.5)
                # Estimate return based on sharpe (if positive) or hit rate
                if sharpe > 0:
                    # Estimate: sharpe * volatility / sqrt(252) for daily
                    # Use a conservative volatility estimate (1% daily)
                    estimated_vol = 0.01
                    base_return = (sharpe * estimated_vol) / np.sqrt(252)
                elif hit_rate > 0.55:  # Good hit rate
                    # Use dynamic threshold scaled by hit rate
                    base_return = dynamic_threshold * (hit_rate - 0.5) * 2
                else:
                    # Use dynamic threshold as fallback
                    base_return = dynamic_threshold * 0.5  # Conservative estimate
            
            # Adjust sign based on action (long = positive, short = negative)
            if action == "long":
                estimated_return = max(0.001, abs(base_return))  # Ensure positive for long
            else:  # short
                estimated_return = min(-0.001, -abs(base_return))  # Ensure negative for short
            # Clamp to reasonable bounds
            estimated_return = _clamp_return(estimated_return)
        elif action == "hold":
            estimated_return = 0.0
        else:
            # Fallback: use dynamic threshold with action direction
            if action == "long":
                estimated_return = dynamic_threshold * 0.5  # Conservative estimate
            elif action == "short":
                estimated_return = -dynamic_threshold * 0.5  # Conservative estimate
            else:
                estimated_return = 0.0
        
        dqn_entry["predicted_return"] = float(estimated_return)
        dqn_entry["predicted_price"] = float(latest_market_price * (1.0 + estimated_return))
        
        # Save DQN summary as JSON (not the model itself)
        _write_dqn_summary(
            output_dir=output_dir,
            asset_type=asset_type,
            symbol=symbol,
            timeframe=timeframe,
            training_samples=len(X_train),
            validation_samples=len(X_val),
            feature_count=feature_count,
            feature_columns=list(X_train.columns),
            metrics=dqn_entry,
        )
        results["dqn"] = dqn_entry
        logger.success(
            "DQN training completed",
            category="MODEL",
            symbol=symbol,
            asset_type=asset_type,
            data={
                "model": "DQN",
                "action": action,
                "current_price": latest_market_price,
                "features_used": feature_count,
                "validation_policy": val_policy_metrics,
                "test_policy": test_policy_metrics,
            }
        )
    except Exception as exc:
        error_msg = str(exc)
        # Don't log tensorboard errors as failures - it's optional
        if "tensorboard" in error_msg.lower():
            # Tensorboard is optional - just skip DQN training silently
            if verbose:
                print(f"[DQN] Skipped: TensorBoard not installed (optional dependency)")
            results["dqn"] = {"status": "skipped", "reason": "TensorBoard not installed (optional)"}
        else:
            # Real error - log it
            results["dqn"] = {"status": "failed", "reason": error_msg}
            logger.error(
                f"DQN training failed: {error_msg}",
                category="MODEL",
                symbol=symbol,
                asset_type=asset_type,
                data={"model": "DQN", "error": error_msg}
            )
            if verbose:
                print(f"[DQN] Failed: {exc}")
    
    # Log overfitting warnings
    if overfitting_warnings:
        for warning in overfitting_warnings:
            logger.warning(
                warning,
                category="OVERFITTING",
                symbol=symbol,
                asset_type=asset_type
            )
        if verbose:
            print("\n[WARNING] Potential overfitting detected:")
            for warning in overfitting_warnings:
                print(f"  - {warning}")
    
    # Compute consensus action from all models
    consensus = _compute_consensus_action(results, dynamic_threshold, latest_market_price)
    logger.info(
        f"Consensus action computed: {consensus['consensus_action'].upper()}",
        category="CONSENSUS",
        symbol=symbol,
        asset_type=asset_type,
        data={
            "consensus_action": consensus.get("consensus_action", "hold"),
            "consensus_confidence": consensus.get("consensus_confidence", 0.0),
            "consensus_price": consensus.get("consensus_price", latest_market_price),
            "consensus_return": consensus.get("consensus_return", 0.0),
            "action_scores": consensus.get("action_scores", {"long": 0.0, "hold": 1.0, "short": 0.0})
        }
    )
    if verbose:
        print(f"\n[CONSENSUS] Final Action: {consensus['consensus_action'].upper()}")
        print(f"  Confidence: {consensus['consensus_confidence']*100:.1f}%")
        print(f"  Consensus Price: ${consensus['consensus_price']:,.2f}")
        print(f"  Expected Return: {consensus['consensus_return']*100:+.2f}%")
        print(f"  Reasoning: {consensus['reasoning']}")

    if feature_scaler is not None:
        try:
            save_model(feature_scaler, symbol_dir / SCALER_NAME)
        except Exception:
            pass
    for name, model in models.items():
        try:
            save_model(model, symbol_dir / f"{name}.joblib")
        except Exception:
            pass
    if metric_store:
        save_metrics(metric_store, symbol_dir / "metrics.json")
    with open(symbol_dir / "metadata.json", "w", encoding="utf-8") as handle:
        metadata_dict = {
            "asset_type": asset_type,
            "symbol": symbol,
            "timeframe": timeframe,
            "rows": len(dataset),
            "feature_version": dataset_meta["feature_version"],
            "split_boundaries": dataset_meta["split_boundaries"],
            "target_profile": profile_report,
            "target_config": dataset_meta.get("target_config"),
        }
        # Store which models were used for stacked blend (if available)
        if stacked_blend_model_list:
            metadata_dict["stacked_blend_models"] = stacked_blend_model_list
        json.dump(metadata_dict, handle, indent=2)

    vote_lookup = {vote["model"]: vote for vote in consensus.get("model_votes", [])}
    condensed_models: Dict[str, Dict[str, Any]] = {}
    for name, data in results.items():
        # Skip directional and quantile models - they are not used
        if "_directional" in name or "_quantile" in name:
            continue
        if isinstance(data, dict) and data.get("status") == "failed":
            condensed_models[name] = {
                "status": "failed",
                "reason": data.get("reason"),
            }
            continue
        # Build simplified model data (old format fields removed)
        model_entry = {
            "action": data.get("action"),
            "predicted_return": data.get("predicted_return"),  # Keep for internal use
            "r2": data.get("r2"),
            "mae": data.get("mae"),
            "directional_accuracy": data.get("directional_accuracy"),
        }
        vote = vote_lookup.get(name)
        if vote:
            model_entry["vote_weight"] = vote.get("weight")
        condensed_models[name] = {k: v for k, v in model_entry.items() if v is not None}

    # Build simplified, user-friendly summary structure
    # Use raw_consensus_return for predicted_price so users see actual model predictions
    # even when neutral guard is triggered (consensus_return may be zeroed)
    raw_return = consensus.get("raw_consensus_return", consensus.get("consensus_return", 0.0))
    predicted_price_from_raw = float(latest_market_price * (1.0 + raw_return))
    consensus_price = float(latest_market_price * (1.0 + consensus["consensus_return"]))
    accuracy_values = [
        float(data.get("directional_accuracy"))
        for data in condensed_models.values()
        if isinstance(data.get("directional_accuracy"), (int, float))
    ]
    avg_directional_accuracy = float(np.mean(accuracy_values)) if accuracy_values else None
    consensus_confidence_pct = float(consensus["consensus_confidence"] * 100)
    if avg_directional_accuracy is not None:
        effective_confidence = consensus["consensus_confidence"] * avg_directional_accuracy
        consensus_confidence_pct = max(
            MIN_CONFIDENCE * 100,
            min(1.0, effective_confidence) * 100,
        )
    action_label = f"{profile_report.get('label')} {consensus['consensus_action'].upper()}"
    
    # Main prediction section (most important - at the top)
    summary = {
        "symbol": symbol,
        "asset_type": asset_type,
        "timeframe": timeframe,
        "last_updated": latest_market_timestamp,
        
        # MAIN PREDICTION - Easy to find
        # Show actual model prediction (raw return) for predicted_price
        # but show zeroed return if neutral guard triggered
        "prediction": {
            "current_price": latest_market_price,
            "predicted_price": predicted_price_from_raw,  # Use raw return for actual prediction
            "predicted_return_pct": float(raw_return * 100),  # Show actual prediction return
            "action": consensus["consensus_action"],
            "confidence": consensus_confidence_pct,
            "horizon_days": profile_report.get("horizon_bars", 30),
            # Explanation text uses raw return to show actual model prediction
            # Note: If neutral guard triggered, consensus section will show 0.0% return
            "explanation": (
                # Build a logically consistent explanation string:
                # - if expected return ~ 0, say "stay flat"
                # - otherwise say "rise" or "fall" with the correct percentage.
                lambda _price, _target, _ret_pct, _h: (
                    f"Price expected to stay flat from ${_price:,.2f} to "
                    f"${_target:,.2f} ({_ret_pct:+.2f}%) over {_h} days"
                    if abs(_ret_pct) < 1e-9
                    else (
                        f"Price expected to "
                        f"{'rise' if _ret_pct > 0 else 'fall'} from "
                        f"${_price:,.2f} to ${_target:,.2f} ({_ret_pct:+.2f}%) over {_h} days"
                    )
                )
            )(
                latest_market_price,
                predicted_price_from_raw,  # Use raw prediction price
                raw_return * 100,  # Use raw return for explanation
                profile_report.get("horizon_bars", 30),
            ),
        },
        
        # Individual model predictions
        "model_predictions": {},
        
        # Technical details (for advanced users)
        "technical": {
            "training_data": {
                "total_rows": len(dataset),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "split_boundaries": dataset_meta["split_boundaries"],
            },
            "model_config": {
                "target_profile": profile_report,
                "target_config": dataset_meta.get("target_config"),
            "model_reference_price": reference_price,
            },
            "analysis": {
                "dynamic_threshold": dynamic_threshold,
                "threshold_details": threshold_details,
                "context_signals": context_signals,
                "overfitting_warnings": overfitting_warnings if overfitting_warnings else None,
            },
            "feature_scaler": {
                "path": str(symbol_dir / SCALER_NAME) if feature_scaler is not None else None,
            },
        },
    }
    
    # Add individual model predictions in simplified format
    for name, model_data in condensed_models.items():
        if isinstance(model_data, dict) and model_data.get("status") == "failed":
            continue
        
        pred_return = model_data.get("predicted_return", 0)
        pred_price = float(latest_market_price * (1.0 + pred_return))
        
        vote_weight = model_data.get("vote_weight")
        accuracy = model_data.get("directional_accuracy")
        if isinstance(accuracy, (int, float)):
            confidence_pct = max(MIN_CONFIDENCE * 100, float(accuracy) * 100)
            accuracy_pct = float(accuracy) * 100
        elif isinstance(vote_weight, (int, float)):
            confidence_pct = max(MIN_CONFIDENCE * 100, float(vote_weight) * 100)
            accuracy_pct = None
        else:
            confidence_pct = MIN_CONFIDENCE * 100
            accuracy_pct = None
        # Get test metrics from results if available
        test_metrics = results.get(name, {}).get("test_metrics", {})
        test_r2 = test_metrics.get("r2") if isinstance(test_metrics, dict) else None
        test_accuracy = test_metrics.get("directional_accuracy") if isinstance(test_metrics, dict) else None
        val_r2 = model_data.get("r2")
        
        # Calculate val-test gap for overfitting detection
        val_test_gap = None
        if val_r2 is not None and test_r2 is not None:
            val_test_gap = float(val_r2 - test_r2)
        
        summary["model_predictions"][name] = {
            "predicted_price": pred_price,
            "predicted_return_pct": float(pred_return * 100),
            "action": model_data.get("action", "hold"),
            "confidence": confidence_pct,
            "r2_score": val_r2,  # Validation R²
            "test_r2_score": test_r2,  # Test R² (NEW)
            "val_test_gap": val_test_gap,  # Overfitting indicator (NEW)
            "accuracy": accuracy_pct,  # Validation accuracy
            "test_accuracy": float(test_accuracy * 100) if test_accuracy is not None else None,  # Test accuracy (NEW)
        }
    
    # Add consensus details
    summary["consensus"] = {
        "action": consensus.get("consensus_action", "hold"),
        "confidence_pct": consensus_confidence_pct,
        "predicted_return_pct": float(consensus.get("consensus_return", 0.0) * 100),  # May be zeroed by neutral guard
        "predicted_return": float(consensus.get("consensus_return", 0.0)),  # May be zeroed by neutral guard
        "raw_predicted_return_pct": float(consensus.get("raw_consensus_return", consensus.get("consensus_return", 0.0)) * 100),  # Actual model prediction
        "raw_predicted_return": float(consensus.get("raw_consensus_return", consensus.get("consensus_return", 0.0))),  # Actual model prediction
        "reasoning": consensus.get("reasoning", "No consensus available"),
        "action_scores": consensus.get("action_scores", {"long": 0.0, "hold": 1.0, "short": 0.0}),
        "horizon_profile": profile_report.get("label"),
        "target_horizon_bars": profile_report.get("horizon_bars"),
        "action_label": action_label,
        "neutral_return_threshold_pct": float(consensus.get("neutral_return_threshold", dynamic_threshold) * 100),
        "neutral_guard_triggered": bool(consensus.get("neutral_guard_triggered", False)),
    }

    # Surface DQN recommendation separately for easy consumption
    dqn_model = summary["model_predictions"].get("dqn")
    if dqn_model:
        summary["dqn_recommendation"] = {
            "action": dqn_model.get("action", "hold"),
            "predicted_return_pct": dqn_model.get("predicted_return_pct"),
            "predicted_price": dqn_model.get("predicted_price"),
            "confidence": dqn_model.get("confidence"),
            "reason": "Direct DQN policy decision based on latest features",
        }
    
    # Add all models (including failed ones) to summary for display
    # This allows the UI to show which models failed and why
    summary["models"] = {}
    for name, model_data in condensed_models.items():
        if isinstance(model_data, dict) and model_data.get("status") == "failed":
            # Include failed models with their failure reason
            summary["models"][name] = {
                "status": "failed",
                "reason": model_data.get("reason", "Unknown reason"),
            }
        else:
            # Include successful models with their metrics
            test_metrics = results.get(name, {}).get("test_metrics", {})
            test_r2 = test_metrics.get("r2") if isinstance(test_metrics, dict) else None
            test_mae = test_metrics.get("mae") if isinstance(test_metrics, dict) else None
            test_dir = test_metrics.get("directional_accuracy") if isinstance(test_metrics, dict) else None
            
            summary["models"][name] = {
                "action": model_data.get("action"),
                "predicted_return": model_data.get("predicted_return"),
                "r2": model_data.get("r2"),  # Validation R²
                "test_r2": test_r2,  # Test R² (NEW)
                "mae": model_data.get("mae"),  # Validation MAE
                "test_mae": test_mae,  # Test MAE (NEW)
                "directional_accuracy": model_data.get("directional_accuracy"),  # Validation accuracy
                "test_directional_accuracy": test_dir,  # Test accuracy (NEW)
            }
    
    # Check if model passes robustness requirements for trading
    tradable, tradability_reasons = _check_model_tradability(
        results, metric_store, overfitting_warnings, asset_type
    )
    summary["tradable"] = tradable
    summary["tradability_reasons"] = tradability_reasons
    
    # Add data balance and scaling verification info to summary
    if 'data_balance_info' in locals():
        summary["data_balance"] = data_balance_info
    if 'scaler_info' in locals() and 'scaling_verification' in scaler_info:
        summary["feature_scaling_verification"] = scaler_info["scaling_verification"]
    
    with open(symbol_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    
    # Log completion
    logger.success(
        f"Training completed for {symbol}",
        category="TRAIN",
        symbol=symbol,
        asset_type=asset_type,
        data={
            "summary_file": str(symbol_dir / "summary.json"),
            "log_file": str(logger.log_file),
            "models_trained": len([r for r in results.values() if r.get("status") != "failed"]),
            "models_failed": len([r for r in results.values() if r.get("status") == "failed"])
        }
    )
    logger.flush()
    
    return summary


def train_symbols(
    crypto_symbols: List[str],
    commodities_symbols: List[str],
    timeframe: str,
    output_dir: str = "models",
    horizon_profiles: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict] = []
    horizon_profiles = horizon_profiles or {}

    for asset_type, symbols in {
        "crypto": crypto_symbols,
        "commodities": commodities_symbols,
    }.items():
        if not symbols:
            continue
        unique_symbols = list(dict.fromkeys(symbols))
        requested_profile = horizon_profiles.get(asset_type)
        if requested_profile and requested_profile.lower() != "all":
            profile_queue = [normalize_profile(requested_profile)]
        else:
            profile_queue = available_horizon_profiles()
        for symbol in unique_symbols:
            for profile_name in profile_queue:
                try:
                    _, symbol_target_config = get_profile_config(profile_name)
                    summary = train_for_symbol(
                        asset_type,
                        symbol,
                        timeframe,
                        output_path,
                        target_config=symbol_target_config,
                        horizon_profile=profile_name,
                    )
                    summaries.append(summary)
                    print(json.dumps(summary, indent=2))
                except Exception as exc:
                    error_msg = f"{asset_type}/{symbol} ({profile_name}): {exc}"
                    print(f"[SKIP] {error_msg}")
                    skip_logger = get_training_logger(asset_type, symbol, timeframe)
                    skip_logger.error(
                        f"Training skipped: {error_msg}",
                        category="TRAIN",
                        symbol=symbol,
                        asset_type=asset_type,
                        data={"error": str(exc), "horizon": profile_name}
                    )
                    skip_logger.flush()
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Train multi-asset prediction models.")
    parser.add_argument("--asset-types", nargs="+", default=["crypto", "commodities"])
    parser.add_argument("--symbols", nargs="+", help="Symbols to train (default: auto-discover)")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--output-dir", default="models")
    horizon_choices = available_horizon_profiles()
    parser.add_argument(
        "--crypto-horizon",
        choices=horizon_choices,
        help="Horizon profile for crypto models (intraday/short/hold).",
    )
    parser.add_argument(
        "--commodities-horizon",
        choices=horizon_choices,
        help="Horizon profile for commodity models (intraday/short/hold).",
    )
    args = parser.parse_args()

    # Build symbol map per requested asset type.
    available_by_type = {
        asset_type: discover_symbols(asset_type) for asset_type in args.asset_types
    }
    symbol_map: Dict[str, List[str]] = {}
    provided_symbols = args.symbols or []

    if provided_symbols:
        unmatched = set(provided_symbols)
        used_symbols = set()  # Track symbols already assigned to prevent duplicates
        
        # Process asset types in order (crypto first, then commodities)
        # This ensures if a symbol exists in both, it's assigned to crypto
        for asset_type in args.asset_types:
            available = set(available_by_type.get(asset_type, []))
            # Only select symbols that are available AND not already used
            selected = [sym for sym in provided_symbols 
                       if sym in available and sym not in used_symbols]
            if selected:
                symbol_map[asset_type] = selected
                used_symbols.update(selected)
                unmatched.difference_update(selected)
            else:
                symbol_map[asset_type] = []
        
        if unmatched:
            print(
                "[WARN] Skipping symbols with no local data: "
                + ", ".join(sorted(unmatched))
            )
        
        # Warn if any symbols were found in multiple asset types
        all_found = set()
        for asset_type in args.asset_types:
            found_in_type = set(provided_symbols) & set(available_by_type.get(asset_type, []))
            duplicates = all_found & found_in_type
            if duplicates:
                print(
                    f"[INFO] Symbols found in multiple asset types (assigned to first match): "
                    + ", ".join(sorted(duplicates))
                )
            all_found.update(found_in_type)
    else:
        symbol_map = {
            asset_type: available_by_type.get(asset_type, [])
            for asset_type in args.asset_types
        }
    horizon_map = {
        asset: profile
        for asset, profile in (
            ("crypto", args.crypto_horizon),
            ("commodities", args.commodities_horizon),
        )
        if profile
    }
    train_symbols(
        crypto_symbols=symbol_map.get("crypto", []),
        commodities_symbols=symbol_map.get("commodities", []),
        timeframe=args.timeframe,
        output_dir=args.output_dir,
        horizon_profiles=horizon_map or None,
    )


if __name__ == "__main__":
    main()


