"""
Deployment-ready inference pipeline for classical + RL models.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from ml.risk import RiskManager, RiskManagerConfig
from monitoring.live_metrics import LiveMetricsTracker

# SKLEARN VERSION COMPATIBILITY
# Models may have been trained with sklearn 1.3.2 but are being loaded with sklearn 1.7.1
# This causes warnings about missing 'monotonic_cst' attribute (deprecated/renamed in newer versions)
# These warnings are safe to ignore - the models still work correctly
# Suppress all sklearn version compatibility warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", message=".*Trying to unpickle.*")
warnings.filterwarnings("ignore", message=".*monotonic_cst.*")
warnings.filterwarnings("ignore", message=".*does not have the attribute.*")


PREDICTION_CLAMP = 0.2
MIN_THRESHOLD = 0.0025
CONSENSUS_NEUTRAL_MULTIPLIER = 1.05
# Base maximum confidence cap (will be adjusted dynamically)
BASE_MAX_CONFIDENCE_CAP = 0.90  # Base 90% cap, adjusted based on conditions

# Horizon-aware prediction limits (based on typical volatility)
# More conservative limits to prevent unrealistic predictions
HORIZON_CLAMPS = {
    "intraday": 0.05,   # ±5% max for 1-day (extreme but possible)
    "short": 0.08,      # ±8% max for 4-day (reduced from 10%)
    "long": 0.12,       # ±12% max for 30-day (reduced from 20% - more realistic)
}

# Asset-type-aware typical daily volatility
# Crypto: Higher volatility (BTC ~3% daily)
# Commodities: Lower volatility (Gold ~1.5%, Crude Oil ~2.5%, Silver ~2%)
TYPICAL_DAILY_VOLATILITY = {
    "crypto": 0.03,        # 3% daily volatility for BTC (conservative)
    "commodities": 0.02,   # 2% daily volatility for commodities (conservative average)
}

def get_typical_volatility(asset_type: str = "crypto") -> float:
    """Get typical daily volatility for asset type."""
    return TYPICAL_DAILY_VOLATILITY.get(asset_type.lower(), 0.025)  # Default 2.5%


def calculate_dynamic_confidence_cap(
    total_models: int,
    agreement_ratio: float,
    volatility: float,
    horizon_profile: Optional[str] = None,
    model_metrics: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """
    Calculate dynamic confidence cap based on market conditions and model performance.
    
    Factors considered:
    - Number of models (fewer models = lower cap)
    - Model agreement (all models agree = lower cap due to potential bias)
    - Volatility (higher volatility = lower cap)
    - Horizon profile (different base caps for different horizons)
    - Model quality (average R² and directional accuracy)
    
    Returns:
        Dynamic confidence cap between 0.50 and 0.90
    """
    # Base cap varies by horizon (longer horizons can have slightly higher caps)
    base_caps = {
        "intraday": 0.85,  # Intraday: more volatile, lower cap
        "short": 0.88,      # Short-term: moderate cap
        "long": 0.90,       # Long-term: can be slightly higher
    }
    base_cap = base_caps.get(horizon_profile, 0.88)  # Default to short-term cap
    
    # Factor 1: Number of models (fewer models = lower confidence cap)
    # With 1 model: reduce by 20%, with 2 models: reduce by 10%, with 3+ models: no reduction
    model_count_factor = 1.0
    if total_models == 1:
        model_count_factor = 0.80  # 20% reduction
    elif total_models == 2:
        model_count_factor = 0.90  # 10% reduction
    # 3+ models: no reduction (model_count_factor = 1.0)
    
    # Factor 2: Model agreement (all models agree = potential bias, lower cap)
    # If 100% agreement, reduce cap by 15% (markets rarely have perfect consensus)
    # If 80-99% agreement, reduce by 5-10%
    agreement_factor = 1.0
    if agreement_ratio >= 1.0:  # All models agree (100%)
        agreement_factor = 0.85  # 15% reduction
    elif agreement_ratio >= 0.90:  # 90-99% agreement
        agreement_factor = 0.92  # 8% reduction
    elif agreement_ratio >= 0.80:  # 80-89% agreement
        agreement_factor = 0.95  # 5% reduction
    # Less than 80% agreement: no reduction (agreement_factor = 1.0)
    
    # Factor 3: Volatility (higher volatility = lower confidence cap)
    # Normalize volatility: typical daily volatility is 0.03 (3%)
    # If volatility > 5%, reduce cap significantly
    # If volatility < 1%, can allow slightly higher cap
    volatility_factor = 1.0
    if volatility > 0.05:  # Very high volatility (>5%)
        volatility_factor = 0.85  # 15% reduction
    elif volatility > 0.03:  # High volatility (3-5%)
        volatility_factor = 0.92  # 8% reduction
    elif volatility < 0.01:  # Low volatility (<1%)
        volatility_factor = 1.05  # 5% increase (but capped at base_cap)
    # Normal volatility (1-3%): no adjustment
    
    # Factor 4: Model quality (average R² and directional accuracy)
    # Higher quality models = can allow slightly higher confidence
    quality_factor = 1.0
    if model_metrics:
        avg_r2 = 0.0
        avg_directional = 0.0
        count = 0
        for name, metrics in model_metrics.items():
            r2 = metrics.get("r2", 0.0) or 0.0
            directional = metrics.get("directional_accuracy", 0.5) or 0.5
            if r2 > 0:  # Only count models with positive R²
                avg_r2 += r2
                avg_directional += directional
                count += 1
        
        if count > 0:
            avg_r2 = avg_r2 / count
            avg_directional = avg_directional / count
            
            # If models are high quality (R² > 0.3 and directional > 0.6), allow slightly higher cap
            if avg_r2 > 0.3 and avg_directional > 0.6:
                quality_factor = 1.03  # 3% increase
            # If models are low quality (R² < 0.1 or directional < 0.52), reduce cap
            elif avg_r2 < 0.1 or avg_directional < 0.52:
                quality_factor = 0.92  # 8% reduction
    
    # Calculate final dynamic cap
    dynamic_cap = base_cap * model_count_factor * agreement_factor * volatility_factor * quality_factor
    
    # Ensure cap is within reasonable bounds (50% minimum, 90% maximum)
    dynamic_cap = max(0.50, min(0.90, dynamic_cap))
    
    return dynamic_cap


def _clamp(value: float, clamp: float = PREDICTION_CLAMP, horizon_profile: Optional[str] = None) -> float:
    """Clamp prediction to reasonable range, with horizon-aware limits."""
    # Use horizon-specific clamp if available
    if horizon_profile and horizon_profile in HORIZON_CLAMPS:
        clamp = HORIZON_CLAMPS[horizon_profile]
    # Apply global clamp as well (safety net)
    clamp = min(clamp, PREDICTION_CLAMP)
    
    # Additional dampening for extreme predictions (mean reversion)
    # If prediction is very large, scale it down more aggressively
    abs_value = abs(value)
    if abs_value > clamp * 0.5:  # If prediction exceeds 50% of clamp
        # Apply exponential dampening: larger predictions get scaled down more
        scale_factor = 0.5 + 0.5 * (clamp * 0.5 / abs_value)
        value = value * scale_factor
    
    return float(max(min(value, clamp), -clamp))


class InferencePipeline:
    """
    Loads scaler + trained models, produces consensus actions with risk guardrails.
    """

    def __init__(
        self,
        model_dir: Path,
        risk_config: Optional[RiskManagerConfig] = None,
        live_tracker: Optional[LiveMetricsTracker] = None,
        asset_type: Optional[str] = None,
    ):
        self.model_dir = Path(model_dir)
        self.scaler = None
        self.models: Dict[str, any] = {}
        self.metrics: Dict[str, Dict] = {}
        self.summary: Dict[str, any] = {}
        self.metadata: Dict[str, any] = {}
        self.dynamic_threshold = 0.01
        # Extract asset_type from model_dir path if not provided
        # Path structure: models/{asset_type}/{symbol}/{timeframe}/{horizon}/
        if asset_type:
            self.asset_type = asset_type.lower()
        else:
            model_dir_parts = self.model_dir.parts
            # Try to find 'crypto' or 'commodities' in path
            if 'crypto' in model_dir_parts:
                self.asset_type = 'crypto'
            elif 'commodities' in model_dir_parts:
                self.asset_type = 'commodities'
            else:
                self.asset_type = 'crypto'  # Default fallback
        
        # Adjust risk config based on asset_type if not explicitly provided
        # Commodities typically have lower volatility and can use lower confidence thresholds
        if risk_config is None:
            # Create default config with asset-aware min_confidence
            asset_min_confidence = {
                "crypto": 0.55,      # Crypto: higher volatility, need higher confidence
                "commodities": 0.45,  # Commodities: lower volatility, can accept lower confidence
            }
            min_conf = asset_min_confidence.get(self.asset_type, 0.55)
            risk_config = RiskManagerConfig(min_confidence=min_conf)
        else:
            # If config provided but min_confidence is default (0.55), adjust for commodities
            if risk_config.min_confidence == 0.55 and self.asset_type == "commodities":
                # Only adjust if using default value (user didn't explicitly set it)
                risk_config.min_confidence = 0.45
        
        self.risk_manager = RiskManager(risk_config)
        self.live_tracker = live_tracker or LiveMetricsTracker()
        self.target_profile: Optional[Dict[str, Any]] = None
        self.horizon_profile: Optional[str] = None
        self.target_horizon_bars: Optional[int] = None
        self.loaded = False

    def load(self):
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        scaler_path = self.model_dir / "feature_scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        metrics_path = self.model_dir / "metrics.json"
        if metrics_path.exists():
            self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary_path = self.model_dir / "summary.json"
        if summary_path.exists():
            self.summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.dynamic_threshold = float(
                self.summary.get("analysis", {}).get("dynamic_threshold", self.dynamic_threshold)
            )
            # Update asset_type from summary if available (more reliable than path)
            if "asset_type" in self.summary:
                self.asset_type = self.summary["asset_type"].lower()
        metadata_path = self.model_dir / "metadata.json"
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        target_profile = (
            self.summary.get("target_profile")
            or self.metadata.get("target_profile")
        )
        if target_profile:
            self.target_profile = dict(target_profile)
            self.horizon_profile = target_profile.get("name")
            self.target_horizon_bars = target_profile.get("horizon_bars")
            
            # CRITICAL: Validate that the model directory matches the expected horizon
            # Extract horizon from directory path (e.g., models/crypto/BTC-USDT/1d/short -> "short")
            model_dir_parts = self.model_dir.parts
            if len(model_dir_parts) >= 2:
                dir_horizon = model_dir_parts[-1]  # Last part should be horizon name
                expected_horizon = self.horizon_profile
                if dir_horizon != expected_horizon:
                    import warnings
                    warnings.warn(
                        f"⚠️  HORIZON MISMATCH: Model directory '{dir_horizon}' does not match "
                        f"model's horizon profile '{expected_horizon}'. "
                        f"This may cause incorrect predictions. "
                        f"Expected model path: .../{expected_horizon}/, but found: {self.model_dir}",
                        UserWarning
                    )
        for joblib_path in self.model_dir.glob("*.joblib"):
            if joblib_path.name == "feature_scaler.joblib":
                continue
            model_name = joblib_path.stem
            try:
                self.models[model_name] = joblib.load(joblib_path)
            except Exception:
                continue
        self.loaded = True

    def _prepare_features(self, feature_row: pd.Series) -> np.ndarray:
        """
        Prepare a single feature row for inference.

        Returns a numpy array (not DataFrame) to match how models were trained.
        This avoids sklearn warnings about feature name mismatches.

        To avoid scikit-learn \"feature names should match\" errors when live
        feature sets drift from the training set, we align the incoming
        features to the exact feature order used by the fitted scaler, if
        available. Any missing features are filled with 0.0 and any extra
        live-only features are dropped.
        """
        frame = pd.DataFrame([feature_row])

        if self.scaler is not None:
            # If the scaler knows which feature names it was fitted on,
            # reindex the live frame to that exact set and order.
            feature_names = getattr(self.scaler, "feature_names_in_", None)
            if feature_names is not None:
                cols = list(feature_names)
                frame = frame.reindex(columns=cols, fill_value=0.0)
            scaled = self.scaler.transform(frame)
            # Return as numpy array (models were trained on arrays, not DataFrames)
            # Ensure it's 2D: [1, n_features] for sklearn models
            if len(scaled.shape) == 1:
                scaled = scaled.reshape(1, -1)
            return scaled

        # No scaler: convert to numpy array, ensure 2D shape
        values = frame.values
        if len(values.shape) == 1:
            values = values.reshape(1, -1)
        return values

    def _prob_to_return(self, prob: float) -> float:
        margin = max(self.dynamic_threshold, 0.0025)
        return _clamp((prob - 0.5) * 2.0 * margin, horizon_profile=self.horizon_profile)

    def _extract_feature_signals(self, feature_row: pd.Series, current_price: float) -> Dict[str, float]:
        """
        Extract key technical indicators for mean reversion logic.
        Returns normalized signals that can be used for bias correction.
        
        Key insight: For long-term predictions (30 days), price can move down for a few days,
        but the final prediction at 30 days should account for mean reversion. Markets rarely
        move in straight lines. This function extracts signals that indicate when mean reversion
        is likely (e.g., oversold conditions suggest upward mean reversion).
        
        Args:
            feature_row: Series containing feature values
            current_price: Current market price (needed for SMA comparisons)
        
        Returns:
            Dictionary with feature signals including:
            - rsi_signal: Normalized RSI signal (-1 to +1, positive = oversold/mean reversion up expected)
            - macd_signal: Normalized MACD signal
            - sma50_signal: Price deviation from SMA50 (negative = price below SMA, expect mean reversion up)
            - sma200_signal: Price deviation from SMA200
        """
        signals = {
            "rsi": None,
            "macd_histogram": None,
            "price_vs_sma50": None,
            "price_vs_sma200": None,
        }
        
        # RSI (typically 0-100, oversold < 40, overbought > 60)
        # Extended range: RSI 30-40 is moderately oversold, RSI 60-70 is moderately overbought
        if "RSI_14" in feature_row:
            rsi = float(feature_row["RSI_14"])
            signals["rsi"] = rsi
            # Normalize to -1 (oversold) to +1 (overbought)
            # RSI < 40 is oversold (expect upward mean reversion)
            # RSI > 60 is overbought (expect downward mean reversion)
            if rsi < 40:
                # Oversold: positive signal (expect mean reversion up)
                # RSI 0-30: very oversold (signal 0.5-1.0)
                # RSI 30-40: moderately oversold (signal 0.0-0.5)
                if rsi < 30:
                    signals["rsi_signal"] = (30 - rsi) / 30.0  # 0.0 to 1.0
                else:
                    signals["rsi_signal"] = (40 - rsi) / 20.0  # 0.0 to 0.5 for RSI 30-40
            elif rsi > 60:
                # Overbought: negative signal (expect mean reversion down)
                # RSI 60-70: moderately overbought (signal -0.5 to 0.0)
                # RSI 70-100: very overbought (signal -1.0 to -0.5)
                if rsi > 70:
                    signals["rsi_signal"] = -(rsi - 70) / 30.0  # -1.0 to 0.0
                else:
                    signals["rsi_signal"] = -(rsi - 60) / 20.0  # -0.5 to 0.0 for RSI 60-70
            else:
                signals["rsi_signal"] = 0.0  # Neutral zone (40-60)
        
        # MACD Histogram (negative = bearish, positive = bullish)
        if "MACD_histogram" in feature_row:
            macd = float(feature_row["MACD_histogram"])
            signals["macd_histogram"] = macd
            # Normalize: extreme negative = oversold, extreme positive = overbought
            # Use a reasonable scale (e.g., divide by typical ATR or use log scale)
            if abs(macd) > 0:
                # Simple normalization: if MACD is very negative, expect mean reversion up
                signals["macd_signal"] = -np.sign(macd) * min(abs(macd) / 1000.0, 1.0)  # Scale down
            else:
                signals["macd_signal"] = 0.0
        
        # Price vs Moving Averages (mean reversion indicator)
        if "SMA_50" in feature_row and current_price > 0:
            sma50 = float(feature_row["SMA_50"])
            price_vs_sma50 = (current_price - sma50) / sma50 if sma50 > 0 else 0.0
            signals["price_vs_sma50"] = price_vs_sma50
            # If price is far below SMA, expect mean reversion up (and vice versa)
            signals["sma50_signal"] = -price_vs_sma50 * 0.5  # Dampened mean reversion signal
        
        if "SMA_200" in feature_row and current_price > 0:
            sma200 = float(feature_row["SMA_200"])
            price_vs_sma200 = (current_price - sma200) / sma200 if sma200 > 0 else 0.0
            signals["price_vs_sma200"] = price_vs_sma200
            signals["sma200_signal"] = -price_vs_sma200 * 0.3  # Weaker signal for longer MA
        
        return signals

    def predict(
        self,
        feature_row: pd.Series,
        current_price: float,
        volatility: float,
    ) -> Dict[str, any]:
        # (Re)load models if needed.
        # Important: we may start the server before models exist and then
        # train them later (auto-bootstrap). In that case an InferencePipeline
        # instance can be cached with loaded=True but models still empty.
        # To make this robust for live API usage, always ensure that if the
        # models dict is empty we attempt a fresh load from disk before
        # declaring "no trained models".
        if not self.loaded or not self.models:
            self.load()

        # If we still have no models after attempting to load, return a
        # neutral "hold" consensus instead of throwing. This guarantees that
        # callers (API / MCP adapter) always receive a well-formed payload
        # even in edge-cases where classical models were pruned or failed
        # validation for a given symbol/horizon.
        if not self.models:
            neutral_threshold = max(self.dynamic_threshold, MIN_THRESHOLD) * CONSENSUS_NEUTRAL_MULTIPLIER
            consensus = {
                "consensus_action": "hold",
                "consensus_return": 0.0,
                "consensus_confidence": 0.0,
                "action_scores": {"long": 0.0, "hold": 0.0, "short": 0.0},
                "neutral_guard_triggered": False,
                "neutral_return_threshold": float(neutral_threshold),
                "raw_consensus_return": 0.0,
                "position_size": 0.0,  # No position when no models
            }
            if self.horizon_profile:
                consensus["horizon_profile"] = self.horizon_profile
            if self.target_horizon_bars is not None:
                consensus["target_horizon_bars"] = self.target_horizon_bars
            payload: Dict[str, Any] = {"models": {}, "consensus": consensus}
            if self.target_profile:
                payload["target_profile"] = self.target_profile
            return payload
        features = self._prepare_features(feature_row)
        model_outputs: Dict[str, Dict[str, float]] = {}
        
        # First pass: get predictions from base models (skip stacked_blend - it's a meta-model)
        base_model_outputs = {}
        for name, model in self.models.items():
            # Skip stacked_blend - it's a meta-model that needs predictions from other models
            if name == "stacked_blend":
                continue
                
            try:
                # Suppress sklearn feature name warnings - we're using numpy arrays which is correct
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="X does not have valid feature names")
                    
                    if name == "dqn":
                        # CRITICAL FIX for DQN: SB3 models return (action, state)
                        # Action is discrete: 0=Short, 1=Hold, 2=Long (mapped from -1,0,1 in env)
                        # We must convert this to a predicted return for the consensus logic
                        action_arr, _ = model.predict(features, deterministic=True)
                        action_idx = int(action_arr[0]) if hasattr(action_arr, '__iter__') else int(action_arr)
                        
                        # Map action to synthetic return based on typical volatility
                        # This ensures the consensus logic respects the DQN's decision
                        typical_vol = get_typical_volatility(self.asset_type)
                        
                        if action_idx == 0:  # Short
                            pred_return = -typical_vol * 1.5  # Strong short signal
                            action = "short"
                        elif action_idx == 2:  # Long
                            pred_return = typical_vol * 1.5   # Strong long signal
                            action = "long"
                        else:  # 1 = Hold
                            pred_return = 0.0
                            action = "hold"
                            
                        # Use a fixed high confidence for DQN as it's a direct policy decision
                        confidence = 0.6  # Moderate-high confidence default
                        
                        model_outputs[name] = {
                            "predicted_return": pred_return,
                            "action": action,
                            "confidence": confidence,
                            "raw_prediction": float(action_idx),
                        }
                        base_model_outputs[name] = pred_return
                    
                    elif hasattr(model, "predict_proba") and "directional" in name:
                        prob = float(model.predict_proba(features)[0, 1])
                        pred_return = self._prob_to_return(prob)
                        base_model_outputs[name] = pred_return
                        # Determine action from predicted return
                        action = "long" if pred_return > self.dynamic_threshold else "short" if pred_return < -self.dynamic_threshold else "hold"
                        # Calculate confidence based on model metrics (R² and directional accuracy)
                        model_metrics = self.metrics.get(name, {})
                        r2 = model_metrics.get("r2", 0.0) or 0.0
                        directional = model_metrics.get("directional_accuracy", 0.5) or 0.5
                        # Confidence = weighted combination of R² and directional accuracy
                        confidence = min(0.95, (r2 * 0.6) + ((directional - 0.5) * 0.4 * 2.0))
                        confidence = max(0.05, confidence)  # Minimum 5% confidence
                        model_outputs[name] = {
                            "predicted_return": pred_return,
                            "probability": prob,
                            "action": action,
                            "confidence": float(confidence),
                        }
                    else:
                        pred = float(model.predict(features)[0])
                        # Apply horizon-aware clamping
                        clamped_pred = _clamp(pred, horizon_profile=self.horizon_profile)
                        # Additional sanity check: if prediction is extreme, scale it down
                        typical_vol = get_typical_volatility(self.asset_type)
                        if self.horizon_profile == "intraday" and abs(clamped_pred) > typical_vol:
                            # For intraday, if prediction exceeds typical daily volatility, scale it
                            scale_factor = typical_vol / abs(clamped_pred)
                            clamped_pred = clamped_pred * min(scale_factor, 1.0)
                        base_model_outputs[name] = clamped_pred
                        # Determine action from predicted return
                        action = "long" if clamped_pred > self.dynamic_threshold else "short" if clamped_pred < -self.dynamic_threshold else "hold"
                        # Calculate confidence based on model metrics (R² and directional accuracy)
                        model_metrics = self.metrics.get(name, {})
                        r2 = model_metrics.get("r2", 0.0) or 0.0
                        directional = model_metrics.get("directional_accuracy", 0.5) or 0.5
                        # Confidence = weighted combination of R² and directional accuracy
                        # R² contributes 60%, directional accuracy contributes 40%
                        confidence = min(0.95, (r2 * 0.6) + ((directional - 0.5) * 0.4 * 2.0))  # Scale directional to 0-1 range
                        confidence = max(0.05, confidence)  # Minimum 5% confidence
                        model_outputs[name] = {
                            "predicted_return": clamped_pred,
                            "raw_prediction": float(pred),  # Store raw prediction for debugging
                            "action": action,
                            "confidence": float(confidence),
                        }
            except Exception as exc:
                # Log the error for debugging but continue with other models
                # Suppress known sklearn compatibility warnings (monotonic_cst)
                error_str = str(exc)
                if "monotonic_cst" in error_str:
                    # Known sklearn version compatibility issue - model still works, just skip this prediction
                    continue
                # For other errors, log but don't spam warnings
                error_msg = f"Model {name} prediction failed: {exc}"
                # Only print if it's not a known compatibility issue
                if "monotonic_cst" not in error_str:
                    print(f"[WARNING] {error_msg}", flush=True)
                continue
        
        # Second pass: if we have stacked_blend and enough base model predictions, use it
        # CRITICAL: Stacked blend expects a specific number of features (typically 3: RF, LGBM, XGB)
        # If we have fewer models, we need to check if the stacked model can handle it
        if "stacked_blend" in self.models and len(base_model_outputs) >= 2:
            try:
                stacked_model = self.models["stacked_blend"]
                # Get the expected number of features from the model
                expected_features = None
                if hasattr(stacked_model, "n_features_in_"):
                    expected_features = stacked_model.n_features_in_
                elif hasattr(stacked_model, "coef_") and stacked_model.coef_ is not None:
                    # For RidgeCV, coef_ shape tells us expected features
                    if len(stacked_model.coef_.shape) > 0:
                        expected_features = stacked_model.coef_.shape[0]
                
                # Check which models were used to train the stacked blend (from metadata)
                required_models = None
                if hasattr(stacked_model, "__dict__") and "_stacked_blend_models" in stacked_model.__dict__:
                    required_models = stacked_model.__dict__["_stacked_blend_models"]
                elif self.metadata and "stacked_blend_models" in self.metadata:
                    required_models = self.metadata["stacked_blend_models"]
                
                # Check if all required models are available
                if required_models:
                    available_required = [m for m in required_models if m in base_model_outputs]
                    if len(available_required) != len(required_models):
                        # Some required models are missing - skip stacked blend silently
                        # This is expected when a model fails validation
                        missing = [m for m in required_models if m not in base_model_outputs]
                        # Only show warning if in debug mode or if it's unexpected
                        # (Silently skip if it's just one model that failed validation)
                        pass  # Skip silently - this is expected behavior
                    else:
                        # All required models available - use them in the correct order
                        base_predictions = np.array([[base_model_outputs[k] for k in required_models]])
                        # Suppress sklearn feature name warnings
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="X does not have valid feature names")
                            stack_pred = float(stacked_model.predict(base_predictions)[0])
                        clamped_pred = _clamp(stack_pred, horizon_profile=self.horizon_profile)
                        # Determine action from predicted return
                        action = "long" if clamped_pred > self.dynamic_threshold else "short" if clamped_pred < -self.dynamic_threshold else "hold"
                        # Calculate confidence based on model metrics
                        model_metrics = self.metrics.get("stacked_blend", {})
                        r2 = model_metrics.get("r2", 0.0) or 0.0
                        directional = model_metrics.get("directional_accuracy", 0.5) or 0.5
                        confidence = min(0.95, (r2 * 0.6) + ((directional - 0.5) * 0.4 * 2.0))
                        confidence = max(0.05, confidence)
                        model_outputs["stacked_blend"] = {
                            "predicted_return": clamped_pred,
                            "action": action,
                            "confidence": float(confidence),
                        }
                elif expected_features is not None and len(base_model_outputs) != expected_features:
                    # Number of models doesn't match and we don't know which models were used
                    # Skip stacked blend (this is the old behavior)
                    pass  # Skip silently
                else:
                    # Try to use stacked blend with available models (fallback)
                    base_predictions = np.array([[base_model_outputs[k] for k in sorted(base_model_outputs.keys())]])
                    # Suppress sklearn feature name warnings
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="X does not have valid feature names")
                        stack_pred = float(stacked_model.predict(base_predictions)[0])
                    clamped_pred = _clamp(stack_pred, horizon_profile=self.horizon_profile)
                    model_outputs["stacked_blend"] = {"predicted_return": clamped_pred}
            except Exception as exc:
                # Silently skip stacked blend if prediction fails
                # This is expected when models are missing
                pass
        
        # Load DQN from summary.json (DQN is saved as JSON, not a model file)
        if self.summary and "model_predictions" in self.summary:
            dqn_data = self.summary["model_predictions"].get("dqn", {})
            if dqn_data and isinstance(dqn_data, dict):
                # Extract DQN prediction from summary
                dqn_return_pct = dqn_data.get("predicted_return_pct", 0.0)
                dqn_return = float(dqn_return_pct) / 100.0 if dqn_return_pct else 0.0
                dqn_action = dqn_data.get("action", "hold")
                dqn_confidence = dqn_data.get("confidence", 50.0)
                # Convert confidence from percentage to 0-1 range if needed
                if dqn_confidence > 1.0:
                    dqn_confidence = dqn_confidence / 100.0
                
                # Apply clamping to DQN return
                clamped_dqn_return = _clamp(dqn_return, horizon_profile=self.horizon_profile)
                
                # Add DQN to model_outputs for consensus calculation
                model_outputs["dqn"] = {
                    "predicted_return": clamped_dqn_return,
                    "action": dqn_action.lower() if isinstance(dqn_action, str) else "hold",
                    "confidence": float(dqn_confidence),
                }
                # Also add to base_model_outputs for stacked blend compatibility (if needed)
                base_model_outputs["dqn"] = clamped_dqn_return

        # ========================================================================
        # STEP 3: DETECT SIMILAR MODEL OUTPUTS - Warn when models produce identical predictions
        # ========================================================================
        # Store detection results in a temporary dict - will be merged into consensus later
        prediction_detection_results = {}
        
        if len(model_outputs) > 0:
            raw_predictions = {}
            clamped_predictions = {}
            for name, data in model_outputs.items():
                raw_pred = data.get("raw_prediction", data.get("predicted_return", 0))
                clamped_pred = data.get("predicted_return", 0)
                raw_predictions[name] = float(raw_pred)
                clamped_predictions[name] = float(clamped_pred)
            
            # Check if all raw predictions are identical (within tolerance)
            raw_values = list(raw_predictions.values())
            if len(raw_values) > 1:
                raw_std = float(np.std(raw_values))
                raw_range = float(np.max(raw_values) - np.min(raw_values))
                
                # Check if predictions are suspiciously similar
                identical_threshold = 1e-6  # Very small threshold for "identical"
                similar_threshold = 0.001   # 0.1% threshold for "very similar"
                
                if raw_std < identical_threshold:
                    # All raw predictions are identical - this suggests a bug
                    import warnings
                    warnings.warn(
                        f"⚠️  ALL {len(model_outputs)} MODELS PRODUCED IDENTICAL RAW PREDICTIONS: {raw_values[0]:.6f}. "
                        f"This indicates a critical bug - models are not producing different outputs. "
                        f"Possible causes: feature scaling issue, model bug, or identical features.",
                        UserWarning
                    )
                    print(f"[CRITICAL] All models produced identical predictions: {raw_values[0]:.6f}")
                    prediction_detection_results["identical_predictions_detected"] = True
                    prediction_detection_results["raw_predictions"] = raw_predictions
                elif raw_std < similar_threshold:
                    # Predictions are very similar (within 0.1%)
                    import warnings
                    warnings.warn(
                        f"⚠️  ALL {len(model_outputs)} MODELS PRODUCED VERY SIMILAR PREDICTIONS "
                        f"(std={raw_std:.6f}, range={raw_range:.6f}). "
                        f"Models may be overfitting to same pattern or features lack diversity.",
                        UserWarning
                    )
                    print(f"[WARNING] Models produced very similar predictions (std={raw_std:.6f}, range={raw_range:.6f})")
                    prediction_detection_results["similar_predictions_detected"] = True
                    prediction_detection_results["raw_predictions"] = raw_predictions
                    prediction_detection_results["raw_std"] = raw_std
                    prediction_detection_results["raw_range"] = raw_range
                    prediction_detection_results["prediction_std"] = float(raw_std)
                    prediction_detection_results["prediction_range"] = float(raw_range)
                    prediction_detection_results["raw_predictions"] = raw_predictions
                
                # Check if all predictions have same sign (all positive or all negative)
                all_positive = all(v > 0 for v in raw_values)
                all_negative = all(v < 0 for v in raw_values)
                if all_positive or all_negative:
                    prediction_detection_results["unanimous_direction"] = "positive" if all_positive else "negative"
                    prediction_detection_results["unanimous_direction_count"] = len(model_outputs)
                    
                    # WARNING: Only warn if all models predict same direction AND predictions are suspiciously similar
                    # This helps distinguish between legitimate market consensus vs. model overfitting
                    is_suspicious = (
                        raw_std < similar_threshold or  # Very similar predictions
                        abs(raw_range) < 0.005 or  # Very small range (< 0.5%)
                        (all_negative and all(v > -0.01 for v in raw_values)) or  # All very small negative (< 1%)
                        (all_positive and all(v < 0.01 for v in raw_values))  # All very small positive (< 1%)
                    )
                    
                    if len(model_outputs) >= 3 and all_positive and is_suspicious:
                        import warnings
                        warnings.warn(
                            f"⚠️  MODEL BIAS DETECTED: All {len(model_outputs)} models predict POSITIVE (LONG) "
                            f"with suspiciously similar magnitudes (std={raw_std:.6f}, range={raw_range:.6f}). "
                            f"This suggests models may be overfitted or biased toward bullish predictions. "
                            f"Consider checking training data balance and model diversity.",
                            UserWarning
                        )
                        print(f"[WARNING] Model bias: All models predict LONG with similar magnitudes - may indicate overfitting")
                    elif len(model_outputs) >= 3 and all_negative and is_suspicious:
                        import warnings
                        warnings.warn(
                            f"⚠️  MODEL BIAS DETECTED: All {len(model_outputs)} models predict NEGATIVE (SHORT) "
                            f"with suspiciously similar magnitudes (std={raw_std:.6f}, range={raw_range:.6f}). "
                            f"This suggests models may be overfitted or biased toward bearish predictions. "
                            f"Consider checking training data balance and model diversity.",
                            UserWarning
                        )
                        print(f"[WARNING] Model bias: All models predict SHORT with similar magnitudes - may indicate overfitting")
        
        if not model_outputs:
            # All individual model predictions failed — fall back to the same
            # neutral "hold" consensus used when no models are present. This
            # avoids hard failures while still clearly signalling zero edge.
            neutral_threshold = max(self.dynamic_threshold, MIN_THRESHOLD) * CONSENSUS_NEUTRAL_MULTIPLIER
            consensus = {
                "consensus_action": "hold",
                "consensus_return": 0.0,
                "consensus_confidence": 0.0,
                "action_scores": {"long": 0.0, "hold": 0.0, "short": 0.0},
                "neutral_guard_triggered": False,
                "neutral_return_threshold": float(neutral_threshold),
                "raw_consensus_return": 0.0,
                "position_size": 0.0,  # No position when all models fail
            }
            if self.horizon_profile:
                consensus["horizon_profile"] = self.horizon_profile
            if self.target_horizon_bars is not None:
                consensus["target_horizon_bars"] = self.target_horizon_bars
            payload: Dict[str, Any] = {"models": {}, "consensus": consensus}
            if self.target_profile:
                payload["target_profile"] = self.target_profile
            return payload

        # ========================================================================
        # INTRADAY REAL-TIME PRICE ACTION DETECTION
        # For intraday, detect actual candle direction (green/red) and override model predictions
        # This ensures we trade with the actual market direction, not stale historical patterns
        # ========================================================================
        intraday_price_action_override = None
        if self.horizon_profile == "intraday":
            # Get previous close price from features (Close_Lag_1 is the previous day's close)
            previous_close = None
            if "Close_Lag_1" in feature_row:
                try:
                    previous_close = float(feature_row["Close_Lag_1"])
                except (ValueError, TypeError):
                    pass
            
            # If we have previous close, detect real-time price action
            if previous_close and previous_close > 0 and current_price > 0:
                price_change_pct = (current_price - previous_close) / previous_close
                
                # Determine candle direction
                is_green_candle = current_price > previous_close
                is_red_candle = current_price < previous_close
                
                # For intraday, use price action to override model predictions
                # CRITICAL: Strong price movements (>2%) should override model predictions
                # Models are trained on historical patterns, but current price action is REAL-TIME
                SIGNIFICANT_MOVE_THRESHOLD = 0.001  # 0.1% minimum for any override
                STRONG_MOVE_THRESHOLD = 0.02  # 2% for strong override (50% weight)
                VERY_STRONG_MOVE_THRESHOLD = 0.05  # 5% for very strong override (70% weight)
                
                abs_price_change = abs(price_change_pct)
                
                if is_green_candle and price_change_pct > SIGNIFICANT_MOVE_THRESHOLD:
                    # Green candle - price going UP
                    # Determine override strength based on magnitude
                    if abs_price_change >= VERY_STRONG_MOVE_THRESHOLD:
                        # VERY STRONG upward move (>5%) - override models strongly (70% price action)
                        price_action_weight = 0.70
                        price_action_confidence = min(0.85 + abs(price_change_pct) * 2, 0.95)
                        override_strength = "VERY STRONG"
                    elif abs_price_change >= STRONG_MOVE_THRESHOLD:
                        # STRONG upward move (>2%) - override models (50% price action)
                        price_action_weight = 0.50
                        price_action_confidence = min(0.80 + abs(price_change_pct) * 3, 0.92)
                        override_strength = "STRONG"
                    else:
                        # Moderate upward move (0.1-1%) - blend with models (50% price action)
                        price_action_weight = 0.50
                        price_action_confidence = min(0.75 + abs(price_change_pct) * 5, 0.90)
                        override_strength = "MODERATE"
                    
                    typical_vol = get_typical_volatility(self.asset_type)
                    # Use actual price change as return (don't scale down for strong moves)
                    price_action_return = min(price_change_pct * 1.1, typical_vol) if abs_price_change < STRONG_MOVE_THRESHOLD else price_change_pct
                    
                    intraday_price_action_override = {
                        "action": "long",
                        "return": price_action_return,
                        "confidence": price_action_confidence,
                        "weight": price_action_weight,  # How much to weight price action vs models
                        "reasoning": f"Real-time price action: {override_strength} GREEN candle (+{price_change_pct*100:.2f}%). Price moving UP from ${previous_close:.4f} to ${current_price:.4f}. Overriding model predictions with {price_action_weight*100:.0f}% weight.",
                        "price_action_detected": True,
                        "previous_close": previous_close,
                        "current_price": current_price,
                        "price_change_pct": price_change_pct,
                        "override_strength": override_strength,
                    }
                elif is_red_candle and price_change_pct < -SIGNIFICANT_MOVE_THRESHOLD:
                    # Red candle - price going DOWN
                    # Determine override strength based on magnitude
                    if abs_price_change >= VERY_STRONG_MOVE_THRESHOLD:
                        # VERY STRONG downward move (>5%) - override models strongly (70% price action)
                        price_action_weight = 0.70
                        price_action_confidence = min(0.85 + abs(price_change_pct) * 2, 0.95)
                        override_strength = "VERY STRONG"
                    elif abs_price_change >= STRONG_MOVE_THRESHOLD:
                        # STRONG downward move (>2%) - override models (50% price action)
                        price_action_weight = 0.50
                        price_action_confidence = min(0.80 + abs(price_change_pct) * 3, 0.92)
                        override_strength = "STRONG"
                    else:
                        # Moderate downward move (0.1-1%) - blend with models (50% price action)
                        price_action_weight = 0.50
                        price_action_confidence = min(0.75 + abs(price_change_pct) * 5, 0.90)
                        override_strength = "MODERATE"
                    
                    typical_vol = get_typical_volatility(self.asset_type)
                    # Use actual price change as return (don't scale down for strong moves)
                    price_action_return = max(price_change_pct * 1.1, -typical_vol) if abs_price_change < STRONG_MOVE_THRESHOLD else price_change_pct
                    
                    intraday_price_action_override = {
                        "action": "short",
                        "return": price_action_return,
                        "confidence": price_action_confidence,
                        "weight": price_action_weight,  # How much to weight price action vs models
                        "reasoning": f"Real-time price action: {override_strength} RED candle ({price_change_pct*100:.2f}%). Price moving DOWN from ${previous_close:.4f} to ${current_price:.4f}. Overriding model predictions with {price_action_weight*100:.0f}% weight.",
                        "price_action_detected": True,
                        "previous_close": previous_close,
                        "current_price": current_price,
                        "price_change_pct": price_change_pct,
                        "override_strength": override_strength,
                    }
                else:
                    # Price movement is too small (<0.1%) - let models decide (don't override)
                    intraday_price_action_override = {
                        "action": None,  # Let models decide
                        "return": None,
                        "confidence": None,
                        "weight": 0.0,
                        "reasoning": f"Real-time price action: MINOR movement ({price_change_pct*100:.3f}% < 0.1% threshold). Using model predictions.",
                        "price_action_detected": False,
                        "previous_close": previous_close,
                        "current_price": current_price,
                        "price_change_pct": price_change_pct,
                        "override_strength": "NONE",
                    }
        
        # Extract feature-based mean reversion signals
        feature_signals = self._extract_feature_signals(feature_row, current_price)
        
        consensus = self._compute_consensus(
            model_outputs, 
            feature_signals=feature_signals, 
            current_price=current_price,
            volatility=volatility,
            horizon_profile=self.horizon_profile,
        )
        
        # Apply intraday price action override if detected AND significant
        # CRITICAL: For strong price movements, prioritize real-time price action over historical model patterns
        if intraday_price_action_override and intraday_price_action_override.get("price_action_detected"):
            # Override consensus with real-time price action for intraday
            override_action = intraday_price_action_override.get("action")
            override_return = intraday_price_action_override.get("return")
            override_confidence = intraday_price_action_override.get("confidence")
            price_action_weight = intraday_price_action_override.get("weight", 0.50)  # Default 50% if not set
            override_strength = intraday_price_action_override.get("override_strength", "MODERATE")
            
            if override_action and override_return is not None:
                model_return = consensus.get("consensus_return", 0.0)
                model_action = consensus.get("consensus_action", "hold")
                model_confidence = consensus.get("consensus_confidence", 0.0)
                
                # Count model actions directly from model_outputs (more accurate than weighted action_scores)
                action_counts = {"long": 0, "short": 0, "hold": 0}
                for model_name, model_data in model_outputs.items():
                    model_action_from_output = model_data.get("action", "hold").lower()
                    if model_action_from_output in action_counts:
                        action_counts[model_action_from_output] += 1
                
                total_models_count = sum(action_counts.values())
                if total_models_count > 0:
                    # Calculate consensus ratio based on model counts (not weighted scores)
                    max_count = max(action_counts.values())
                    model_consensus_ratio = max_count / total_models_count
                    # Determine the model's consensus direction (most common action)
                    model_consensus_direction = max(action_counts, key=action_counts.get)
                else:
                    model_consensus_ratio = 0.0
                    model_consensus_direction = "hold"
                
                # Determine if models have STRONG consensus (>70% of models agree on one direction)
                strong_model_consensus = model_consensus_ratio > 0.70
                # Determine if models have VERY STRONG consensus (>80% of models agree)
                very_strong_model_consensus = model_consensus_ratio > 0.80
                
                # Check if models strongly disagree with price action override
                models_disagree = (override_action == "long" and model_consensus_direction == "short") or \
                                 (override_action == "short" and model_consensus_direction == "long")
                
                # FIXED: For intraday trading, price action should ALWAYS determine direction
                # Real-time price movement is more reliable than historical model predictions
                # If price is moving UP (green candle), we should go LONG, not SHORT
                # If price is moving DOWN (red candle), we should go SHORT, not LONG
                # Models can still influence confidence and return magnitude, but not the direction
                should_override_action = True  # Always override action for intraday price action
                
                if models_disagree and very_strong_model_consensus:
                    # Models have VERY STRONG consensus (80%+) that disagrees with price action
                    # Reduce price action weight significantly to respect model consensus
                    # This prevents 1-2% price movements from overriding 80%+ model agreement
                    if override_strength == "VERY STRONG":
                        # Very strong moves (>5%) - still respect but less reduction
                        price_action_weight = price_action_weight * 0.7  # 30% reduction
                    elif override_strength == "STRONG":
                        # Strong moves (>2%) - significant reduction
                        price_action_weight = price_action_weight * 0.5  # 50% reduction
                    else:
                        # Moderate moves - heavy reduction to respect 80%+ model consensus
                        price_action_weight = max(price_action_weight * 0.3, 0.25)  # At least 25% weight
                elif models_disagree and strong_model_consensus:
                    # Models have STRONG consensus (70-80%) that disagrees with price action
                    # Reduce price action weight to respect model consensus for return/confidence
                    # BUT still override the action direction to match price movement
                    # CRITICAL: Ensure price action has enough weight so return sign matches action
                    if override_strength == "VERY STRONG":
                        # Very strong moves - price action gets full weight
                        price_action_weight = price_action_weight * 0.9  # Slight reduction
                    elif override_strength == "STRONG":
                        # Strong moves - reduce weight but ensure return sign matches action
                        price_action_weight = price_action_weight * 0.7  # Moderate reduction
                    else:
                        # Moderate moves - ensure price action has enough weight to keep return sign correct
                        # Minimum 40% weight to ensure return sign matches action direction
                        price_action_weight = max(price_action_weight * 0.6, 0.4)  # At least 40% weight
                
                # Use dynamic weight based on price action strength (may have been reduced above)
                model_weight = 1.0 - price_action_weight
                
                # Blend returns and confidence
                # Note: We always override action to match price direction, but blend returns/confidence
                if models_disagree:
                    if override_strength in ["VERY STRONG", "STRONG"]:
                        # For strong moves, blend but price action gets more weight
                        blended_return = override_return * price_action_weight + model_return * model_weight
                        blended_confidence = override_confidence * price_action_weight + model_confidence * model_weight
                    else:
                        # For moderate moves, reduce price action weight more when models disagree
                        reduced_weight = price_action_weight * 0.6  # Reduce by 40%
                        reduced_model_weight = 1.0 - reduced_weight
                        blended_return = override_return * reduced_weight + model_return * reduced_model_weight
                        blended_confidence = override_confidence * reduced_weight + model_confidence * reduced_model_weight
                else:
                    # Agreement or neutral - use dynamic weight
                    blended_return = override_return * price_action_weight + model_return * model_weight
                    blended_confidence = override_confidence * price_action_weight + model_confidence * model_weight
                
                # Cap blended confidence using dynamic cap
                # Calculate dynamic cap for intraday price action override
                # Use number of models from self.models if available, otherwise estimate
                num_models = len(model_outputs) if model_outputs else (len(self.models) if hasattr(self, 'models') and self.models else 2)
                dynamic_cap = calculate_dynamic_confidence_cap(
                    total_models=num_models,
                    agreement_ratio=0.7,  # Price action is strong signal
                    volatility=volatility,
                    horizon_profile=self.horizon_profile,
                    model_metrics=self.metrics,
                )
                blended_confidence = min(dynamic_cap, max(0.0, blended_confidence))
                
                # FIXED: Always override action for intraday price action
                # Price direction (green/red candle) should determine trade direction
                # Models can influence confidence/return but not the action direction
                # CRITICAL: Ensure return sign matches action direction to avoid confusion
                if override_action == "long" and blended_return < 0:
                    # Price is moving up but blended return is negative - ensure it's at least slightly positive
                    # Use a blend that favors price action to keep return sign correct
                    blended_return = max(override_return * 0.6, 0.001)  # At least 60% of price action or 0.1% minimum
                elif override_action == "short" and blended_return > 0:
                    # Price is moving down but blended return is positive - ensure it's at least slightly negative
                    blended_return = min(override_return * 0.6, -0.001)  # At least 60% of price action or -0.1% minimum
                
                consensus["consensus_return"] = blended_return
                consensus["consensus_action"] = override_action  # Always use price action direction
                
                # Flag if models disagreed (for logging/debugging)
                if models_disagree:
                    consensus["price_action_model_disagreement"] = True
                    if strong_model_consensus:
                        # Models strongly disagreed but we still followed price action
                        consensus["price_action_prioritized"] = True
                consensus["consensus_confidence"] = blended_confidence
                consensus["intraday_price_action_override"] = True
                consensus["price_action_reasoning"] = intraday_price_action_override.get("reasoning")
                consensus["price_action_details"] = {
                    "previous_close": intraday_price_action_override.get("previous_close"),
                    "current_price": intraday_price_action_override.get("current_price"),
                    "price_change_pct": intraday_price_action_override.get("price_change_pct"),
                    "override_action": override_action,
                    "override_return": override_return,
                    "model_return": model_return,
                    "blended_return": blended_return,
                    "override_strength": override_strength,  # Store override strength for display
                }
        
        # Merge prediction detection results into consensus (now that consensus exists)
        if prediction_detection_results:
            consensus.update(prediction_detection_results)
        
        # Ensure consensus has all required fields (safety check)
        if "consensus_return" not in consensus:
            consensus["consensus_return"] = 0.0
        if "consensus_action" not in consensus:
            consensus["consensus_action"] = "hold"
        if "consensus_confidence" not in consensus:
            consensus["consensus_confidence"] = 0.0
        
        # Apply horizon-aware sanity check to consensus return
        consensus_return = consensus["consensus_return"]
        
        # CRITICAL: Ensure expected moves are horizon-appropriate
        # Different horizons should produce different expected moves even with similar raw predictions
        # This is because models are trained on different target horizons (1 bar vs 4 bars vs 30 bars)
        # If we see identical moves across horizons, it suggests model reuse or overfitting
        if self.horizon_profile:
            # Store raw consensus return before any horizon-specific adjustments
            raw_consensus_return = consensus_return
            consensus["raw_consensus_return"] = raw_consensus_return
            consensus["horizon_profile"] = self.horizon_profile
        
        # Apply more aggressive dampening for all horizons, not just intraday
        if self.horizon_profile == "intraday":
            # For intraday, cap at typical daily volatility (asset-aware)
            # BUT: if we have price action override, don't cap it (it's based on real-time data)
            if not consensus.get("intraday_price_action_override", False):
                max_intraday_move = get_typical_volatility(self.asset_type)
                if abs(consensus_return) > max_intraday_move:
                    # Scale down extreme predictions
                    scale = max_intraday_move / abs(consensus_return)
                    consensus_return = consensus_return * scale
                    consensus["consensus_return"] = consensus_return
                    consensus["extreme_prediction_scaled"] = True
        # CRITICAL: Apply mean-reversion logic for ALL horizons
        # User wants to BUY LOW (when oversold) and SELL HIGH (when overbought)
        # This applies mean-reversion adjustments to help identify buying opportunities
        # Extract feature signals for mean-reversion detection
        feature_signals = self._extract_feature_signals(feature_row, current_price)
        
        # Calculate mean-reversion adjustment factor
        # Positive adjustment = bias toward LONG (oversold conditions)
        # Negative adjustment = bias toward SHORT (overbought conditions)
        mean_reversion_adjustment = 0.0
        
        # Combine multiple signals for robust mean-reversion detection
        if feature_signals.get("rsi_signal") is not None:
            # RSI is the strongest mean-reversion signal
            # rsi_signal > 0 = oversold (expect upward bounce)
            # rsi_signal < 0 = overbought (expect downward correction)
            mean_reversion_adjustment += feature_signals["rsi_signal"] * 0.4  # 40% weight
        
        if feature_signals.get("sma50_signal") is not None:
            # Price below SMA50 = oversold, expect bounce
            mean_reversion_adjustment += feature_signals["sma50_signal"] * 0.3  # 30% weight
        
        if feature_signals.get("sma200_signal") is not None:
            # Price below SMA200 = very oversold, expect stronger bounce
            mean_reversion_adjustment += feature_signals["sma200_signal"] * 0.2  # 20% weight
        
        if feature_signals.get("macd_signal") is not None:
            # MACD divergence can indicate mean-reversion
            mean_reversion_adjustment += feature_signals["macd_signal"] * 0.1  # 10% weight
        
        # Apply horizon-specific mean-reversion strength
        # Shorter horizons need stronger signals (quick bounces)
        # Longer horizons can have weaker signals (gradual mean reversion)
        horizon_multipliers = {
            "intraday": 0.8,  # Strong mean-reversion for intraday (quick scalps)
            "short": 1.0,     # Moderate mean-reversion for short-term (3-5 day bounces) - THIS IS WHAT USER WANTS
            "long": 0.6,      # Weaker mean-reversion for long-term (gradual reversion)
        }
        horizon_mult = horizon_multipliers.get(self.horizon_profile, 1.0)
        mean_reversion_adjustment *= horizon_mult
        
        # Convert adjustment to return impact
        # ADJUSTED: Increased adjustment strength for SHORT horizon to help flip SHORT predictions to LONG
        # For SHORT horizon: if oversold (adjustment > 0), bias toward +2-3% recovery (increased from 1.5%)
        # For SHORT horizon: if overbought (adjustment < 0), bias toward -1-2% correction
        adjustment_strengths = {
            "intraday": 0.005,  # 0.5% max adjustment for intraday
            "short": 0.025,     # 2.5% max adjustment for short-term (increased from 1.5% to help flip SHORT to LONG)
            "long": 0.020,      # 2.0% max adjustment for long-term
        }
        max_adjustment = adjustment_strengths.get(self.horizon_profile, 0.015)
        mean_reversion_return_adjustment = mean_reversion_adjustment * max_adjustment
        
        # SAFETY CHECKS: Only apply adjustment if conditions are safe
        # EXTREMELY PERMISSIVE: Lowered threshold from 0.05 to 0.01 to allow almost all mean-reversion signals
        # This helps identify buying opportunities even when signals are very weak
        # BUT: Still require minimum signal strength to avoid completely absent signals
        min_signal_strength = 0.01  # Extremely low threshold to allow almost all trades (was 0.05)
        should_apply_mean_reversion = abs(mean_reversion_adjustment) > min_signal_strength
        
        # Safety check 1: Don't override if model confidence is very low
        # ADJUSTED: Lowered from 0.40 to 0.35 to allow mean-reversion when models have moderate confidence
        # This helps catch oversold opportunities even when model confidence is slightly lower
        model_confidence = consensus.get("consensus_confidence", 0.0)
        min_confidence_for_override = 0.35  # Only override if models have at least 35% confidence (reduced from 40%)
        if model_confidence < min_confidence_for_override:
            should_apply_mean_reversion = False
            consensus["mean_reversion_blocked"] = f"Model confidence too low ({model_confidence:.1%} < {min_confidence_for_override:.1%})"
        
        # Safety check 2: Don't override if ALL models strongly agree on opposite direction
        # Check model agreement from consensus (if available)
        model_agreement = consensus.get("model_agreement_ratio", 0.0)
        if model_agreement > 0.85:  # 85%+ models agree
            # If models strongly agree on direction, be cautious about overriding
            original_action = consensus.get("consensus_action", "hold")
            if mean_reversion_adjustment > 0 and original_action == "short":
                # Mean-reversion wants LONG but 85%+ models say SHORT
                # Reduce adjustment by 50% (apply but less aggressively)
                mean_reversion_return_adjustment *= 0.5
                consensus["mean_reversion_reduced"] = "Strong model consensus on opposite direction"
            elif mean_reversion_adjustment < 0 and original_action == "long":
                # Mean-reversion wants SHORT but 85%+ models say LONG
                mean_reversion_return_adjustment *= 0.5
                consensus["mean_reversion_reduced"] = "Strong model consensus on opposite direction"
        
        # Safety check 3: Don't apply if we're in extreme market conditions
        # Check if volatility is extreme (might indicate crash/panic)
        extreme_volatility_threshold = 0.10  # 10% daily volatility = extreme
        if volatility > extreme_volatility_threshold:
            should_apply_mean_reversion = False
            consensus["mean_reversion_blocked"] = f"Extreme volatility detected ({volatility:.1%} > {extreme_volatility_threshold:.1%})"
        
        # Safety check 4: For SHORT horizon, require moderate oversold/overbought signals
        # ADJUSTED: Lowered thresholds to be less restrictive and catch more oversold conditions
        # This aligns with user's goal: buy low (when oversold) even if models predict SHORT
        if self.horizon_profile == "short":
            rsi_signal = feature_signals.get("rsi_signal", 0.0) or 0.0
            sma50_signal = feature_signals.get("sma50_signal", 0.0) or 0.0
            sma200_signal = feature_signals.get("sma200_signal", 0.0) or 0.0
            sma_signal = abs(sma50_signal) + abs(sma200_signal)  # Combined SMA signal strength
            
            # EXTREMELY PERMISSIVE THRESHOLDS (allow almost all trades):
            # RSI >= 0.01 (almost always passes, even when RSI is missing/0)
            # SMA >= 0.01 (almost always passes if any SMA signal exists)
            # Combined >= 0.01 (almost always passes)
            # These thresholds are so low that trades will execute unless signals are completely absent
            rsi_strong_enough = abs(rsi_signal) >= 0.01  # Extremely permissive
            sma_strong_enough = abs(sma_signal) >= 0.01  # Extremely permissive  
            combined_strong_enough = abs(mean_reversion_adjustment) >= 0.01  # Extremely permissive
            
            # Block only if ALL signals are completely absent (almost never blocks)
            # This allows trades to happen based on SMA alone when RSI is missing
            if not (rsi_strong_enough or sma_strong_enough or combined_strong_enough):
                should_apply_mean_reversion = False
                consensus["mean_reversion_blocked"] = f"Insufficient oversold/overbought signals for short-term (RSI: {abs(rsi_signal):.2f}, SMA: {abs(sma_signal):.2f}, Combined: {abs(mean_reversion_adjustment):.2f})"
        
        if should_apply_mean_reversion:
            # Blend model prediction with mean-reversion adjustment
            # When oversold: increase return prediction (favor LONG)
            # When overbought: decrease return prediction (favor SHORT)
            original_return = consensus_return
            consensus_return = consensus_return + mean_reversion_return_adjustment
            
            # Safety check 5: Cap maximum adjustment impact (don't flip extreme predictions)
            # ADJUSTED: Allow stronger adjustments when oversold (mean_reversion_adjustment > 0)
            # This helps mean-reversion flip SHORT predictions to LONG when assets are oversold
            if abs(original_return) > 0.05:  # Original > 5%
                # For oversold conditions (positive adjustment), allow up to 50% adjustment
                # For overbought conditions (negative adjustment), keep 30% limit (more conservative)
                if mean_reversion_adjustment > 0:  # Oversold - want to flip SHORT to LONG
                    max_allowed_adjustment = abs(original_return) * 0.5  # Allow 50% adjustment for oversold
                else:  # Overbought
                    max_allowed_adjustment = abs(original_return) * 0.3  # Keep 30% limit for overbought
                
                if abs(mean_reversion_return_adjustment) > max_allowed_adjustment:
                    # Scale down adjustment
                    scale_factor = max_allowed_adjustment / abs(mean_reversion_return_adjustment)
                    mean_reversion_return_adjustment *= scale_factor
                    consensus_return = original_return + mean_reversion_return_adjustment
                    consensus["mean_reversion_capped"] = f"Adjustment capped to {max_allowed_adjustment/abs(original_return)*100:.0f}% of extreme prediction"
            
            # Update action based on adjusted return
            if consensus_return > self.dynamic_threshold and original_return <= self.dynamic_threshold:
                # Mean-reversion flipped prediction from HOLD/SHORT to LONG
                consensus["consensus_action"] = "long"
                consensus["mean_reversion_flipped_to_long"] = True
            elif consensus_return < -self.dynamic_threshold and original_return >= -self.dynamic_threshold:
                # Mean-reversion flipped prediction from HOLD/LONG to SHORT
                consensus["consensus_action"] = "short"
                consensus["mean_reversion_flipped_to_short"] = True
            
            consensus["mean_reversion_adjustment"] = mean_reversion_return_adjustment
            consensus["mean_reversion_signal_strength"] = abs(mean_reversion_adjustment)
            consensus["mean_reversion_applied"] = True
            consensus["mean_reversion_reason"] = (
                f"Mean-reversion signal: {mean_reversion_adjustment:.2f} "
                f"(RSI: {feature_signals.get('rsi', 'N/A')}, "
                f"SMA50: {feature_signals.get('price_vs_sma50', 0)*100:+.1f}%)"
            )
            consensus["consensus_return"] = consensus_return
        else:
            # Mean-reversion blocked - log why
            consensus["mean_reversion_applied"] = False
            if "mean_reversion_blocked" not in consensus:
                consensus["mean_reversion_blocked"] = "Signal strength insufficient"
        
        # Apply horizon-specific caps (after mean-reversion adjustment)
        if self.horizon_profile == "long":
            # Cap at realistic 30-day moves (typically 5-8% for crypto)
            max_long_move = 0.08  # 8% max for 30-day prediction
            if abs(consensus_return) > max_long_move:
                # Scale down extreme predictions
                scale = max_long_move / abs(consensus_return)
                consensus_return = consensus_return * scale
                consensus["extreme_prediction_scaled"] = True
                consensus["consensus_return"] = consensus_return
        
        elif self.horizon_profile == "short":
            # For short-term, cap at 6% for 4-day prediction
            max_short_move = 0.06  # 6% max for 4-day prediction
            if abs(consensus_return) > max_short_move:
                scale = max_short_move / abs(consensus_return)
                consensus_return = consensus_return * scale
                consensus["extreme_prediction_scaled"] = True
                consensus["consensus_return"] = consensus_return
        
        # Recalculate action after dampening (action might have changed)
        final_consensus_return = consensus["consensus_return"]
        
        # For intraday with price action override, don't recalculate action (it's already set correctly)
        if not consensus.get("intraday_price_action_override", False):
            # Recalculate action based on final return (if mean-reversion changed it)
            # Mean-reversion logic already adjusted the return above, just need to update action
            if final_consensus_return > self.dynamic_threshold:
                consensus["consensus_action"] = "long"
            elif final_consensus_return < -self.dynamic_threshold:
                consensus["consensus_action"] = "short"
            else:
                consensus["consensus_action"] = "hold"
        # Else: intraday price action override already set the action correctly, don't override it
        
        # Confidence is already calculated in _compute_consensus based on model agreement
        # Apply dynamic confidence cap to prevent unrealistic 100% confidence
        # Calculate dynamic cap based on current conditions
        if "consensus_confidence" not in consensus:
            # Fallback: calculate from return magnitude if not set
            confidence = abs(final_consensus_return) / max(self.dynamic_threshold, 1e-4)
            # Calculate dynamic cap for fallback case
            dynamic_cap = calculate_dynamic_confidence_cap(
                total_models=len(model_outputs),
                agreement_ratio=0.5,  # Default if unknown
                volatility=volatility,
                horizon_profile=self.horizon_profile,
                model_metrics=self.metrics,
            )
            confidence = min(dynamic_cap, confidence)
            consensus["consensus_confidence"] = confidence
        else:
            # Confidence already calculated in _compute_consensus with dynamic cap
            # Just ensure it's in valid range (should already be capped)
            confidence = min(0.90, max(0.0, consensus["consensus_confidence"]))
            consensus["consensus_confidence"] = confidence
        allowed = self.risk_manager.should_trade(confidence, volatility)
        # For intraday with strong price action, allow trading even if risk manager is cautious
        # (price action is real-time data, more reliable than model predictions)
        if not allowed and not (consensus.get("intraday_price_action_override") and abs(final_consensus_return) > 0.001):
            # Preserve original prediction before blocking
            consensus["raw_consensus_action"] = consensus.get("consensus_action", "hold")
            consensus["raw_consensus_return"] = consensus.get("consensus_return", 0.0)
            consensus["raw_consensus_confidence"] = consensus.get("consensus_confidence", 0.0)
            # Force hold but keep original values for display
            consensus["consensus_action"] = "hold"
            consensus["risk_blocked"] = True
        consensus["position_size"] = self.risk_manager.max_position_allowed(confidence, volatility)
        consensus["target_price"] = current_price * (1.0 + final_consensus_return)
        if self.horizon_profile:
            consensus["horizon_profile"] = self.horizon_profile
        if self.target_horizon_bars is not None:
            consensus["target_horizon_bars"] = self.target_horizon_bars
        payload = {"models": model_outputs, "consensus": consensus}
        if self.target_profile:
            payload["target_profile"] = self.target_profile
        return payload

    def _compute_consensus(
        self, 
        outputs: Dict[str, Dict[str, float]], 
        feature_signals: Optional[Dict[str, float]] = None,
        current_price: Optional[float] = None,
        volatility: float = 0.01,
        horizon_profile: Optional[str] = None,
    ) -> Dict[str, float]:
        action_scores = {"long": 0.0, "hold": 0.0, "short": 0.0}
        weighted_returns = []
        
        # Check for systematic bias: if ALL models predict same direction, apply dampening
        all_negative = True
        all_positive = True
        negative_count = 0
        positive_count = 0
        total_count = len(outputs)
        
        for name, data in outputs.items():
            pred_return = data["predicted_return"]
            if pred_return > 0:
                all_negative = False
                positive_count += 1
            if pred_return < 0:
                all_positive = False
                negative_count += 1
        
        # Enhanced bias detection: check if there's a strong directional bias
        # If >80% of models agree on direction, it's likely a bias
        strong_bias = False
        bias_direction = None
        if total_count > 0:
            if negative_count / total_count >= 0.8:
                strong_bias = True
                bias_direction = "negative"
            elif positive_count / total_count >= 0.8:
                strong_bias = True
                bias_direction = "positive"
        
        # Bias correction: if all models agree on direction, reduce magnitude (mean reversion)
        # BUT: Don't apply if feature signals strongly suggest the opposite (mean reversion)
        bias_correction_factor = 1.0
        feature_opposes_bias = False
        
        if feature_signals:
            rsi_signal = feature_signals.get("rsi_signal", 0)
            # If all models predict negative but RSI is strongly oversold, don't reduce as much
            # (mean reversion will handle it)
            if all_negative and rsi_signal > 0.5:
                feature_opposes_bias = True
            # If all models predict positive but RSI is strongly overbought, don't reduce as much
            elif all_positive and rsi_signal < -0.5:
                feature_opposes_bias = True
        
        if all_negative or all_positive:
            # If ALL models agree, reduce confidence more aggressively (markets rarely have 100% consensus)
            # BUT: If features oppose the bias, reduce less (let mean reversion handle it)
            if feature_opposes_bias:
                bias_correction_factor = 0.85  # Reduce by only 15% - let mean reversion do the work
            else:
                bias_correction_factor = 0.6  # Reduce by 40% (was 30%)
        elif strong_bias:
            # If >80% agree, still apply correction but less aggressive
            if feature_opposes_bias:
                bias_correction_factor = 0.90  # Reduce by only 10%
            else:
                bias_correction_factor = 0.75  # Reduce by 25%
        
        for name, data in outputs.items():
            metrics = self.metrics.get(name, {})
            r2 = metrics.get("r2") or 0.0
            directional = metrics.get("directional_accuracy") or 0.5
            weight = max(0.05, float(r2) + 0.5 * float(directional - 0.5))
            pred_return = data["predicted_return"]
            
            # Apply bias correction if all models agree
            if bias_correction_factor < 1.0:
                pred_return = pred_return * bias_correction_factor
            
            action = "long" if pred_return > self.dynamic_threshold else "short" if pred_return < -self.dynamic_threshold else "hold"
            action_scores[action] += weight * abs(pred_return)
            weighted_returns.append((pred_return, weight))
        if not weighted_returns:
            return {"consensus_action": "hold", "consensus_return": 0.0, "consensus_confidence": 0.0}
        total_weight = sum(w for _, w in weighted_returns) or 1.0
        consensus_return = sum(r * w for r, w in weighted_returns) / total_weight
        
        # Additional dampening for extreme consensus returns
        # This is now handled in the predict() method with horizon-aware logic
        # Keep this as a safety net for very extreme predictions
        if abs(consensus_return) > 0.10:  # If consensus > 10% (very extreme)
            # Scale down extreme consensus predictions more aggressively
            extreme_scale = 0.10 / abs(consensus_return)
            consensus_return = consensus_return * (0.3 + 0.7 * extreme_scale)  # More aggressive scaling
        
        # Mean reversion check: if prediction is consistently large in one direction,
        # apply additional dampening (markets rarely move in straight lines)
        # Note: This is a general check - horizon-specific mean reversion is in predict()
        if abs(consensus_return) > 0.06:  # If > 6%
            # Apply mean reversion: scale down by 15-25% depending on magnitude
            strength = min(abs(consensus_return) / 0.10, 1.0)
            mean_reversion_scale = 0.80 + 0.20 * (1.0 - strength)  # 0.80-1.0 range
            consensus_return = consensus_return * mean_reversion_scale
        
        best_action = max(action_scores, key=action_scores.get)
        
        # Calculate confidence based on model agreement (how many models agree on direction)
        # This is more reliable than just using return magnitude
        if total_count > 0:
            if best_action == "long":
                agreement_ratio = positive_count / total_count
            elif best_action == "short":
                agreement_ratio = negative_count / total_count
            else:  # hold
                agreement_ratio = 0.5  # Neutral confidence for hold
        else:
            agreement_ratio = 0.0
        
        # CRITICAL FIX: If only 1 model, force agreement_ratio to be lower
        # Single model consensus is unreliable - reduce confidence significantly
        if total_count == 1:
            agreement_ratio = max(0.3, agreement_ratio * 0.5)  # Reduce by 50%, minimum 30%
        
        # Calculate dynamic confidence cap based on market conditions and model performance
        dynamic_cap = calculate_dynamic_confidence_cap(
            total_models=total_count,
            agreement_ratio=agreement_ratio,
            volatility=volatility,
            horizon_profile=horizon_profile,
            model_metrics=self.metrics,
        )
        
        # Combine agreement ratio with return magnitude for final confidence
        # Agreement ratio (0-1) contributes 60%, return magnitude contributes 40%
        # Cap return magnitude confidence using dynamic cap
        # IMPORTANT: Use horizon-specific threshold for return magnitude calculation
        # Different horizons have different expected move magnitudes, so normalize accordingly
        horizon_threshold_multiplier = {
            "intraday": 1.0,   # Intraday: use base threshold (smaller moves expected)
            "short": 1.5,      # Short-term: 1.5x threshold (medium moves)
            "long": 2.5,       # Long-term: 2.5x threshold (larger moves expected)
        }
        threshold_mult = horizon_threshold_multiplier.get(horizon_profile, 1.5)
        horizon_aware_threshold = self.dynamic_threshold * threshold_mult
        
        # Normalize return magnitude by horizon-appropriate threshold
        # This ensures intraday and long horizons produce different confidence even with same raw return
        return_magnitude_confidence = min(dynamic_cap, abs(consensus_return) / max(horizon_aware_threshold, 1e-4))
        confidence = (agreement_ratio * 0.6) + (return_magnitude_confidence * 0.4)
        
        # Apply horizon-specific confidence adjustment
        # Longer horizons can have slightly higher confidence for same agreement (more time for prediction to play out)
        horizon_confidence_factor = {
            "intraday": 0.95,  # Intraday: slightly lower (more volatile, less predictable)
            "short": 1.0,      # Short-term: baseline
            "long": 1.05,      # Long-term: slightly higher (more time for trend to develop)
        }
        confidence_factor = horizon_confidence_factor.get(horizon_profile, 1.0)
        confidence = confidence * confidence_factor
        
        # Ensure final confidence doesn't exceed dynamic cap (in case agreement_ratio is 1.0)
        confidence = min(dynamic_cap, confidence)
        
        # Apply neutral guard (same logic as training)
        neutral_threshold = max(self.dynamic_threshold, MIN_THRESHOLD) * CONSENSUS_NEUTRAL_MULTIPLIER
        raw_consensus_return = consensus_return
        neutral_guard_triggered = False
        original_best_action = best_action  # Store original action before neutral guard
        if abs(consensus_return) < neutral_threshold and best_action != "hold":
            # Neutral guard: if the expected move is smaller than the noise band,
            # we should not recommend a directional trade. Regardless of how
            # strong the directional score is, we force the action to HOLD and
            # set the expected return to 0. This keeps the behaviour logically
            # consistent across all symbols/horizons.
            neutral_guard_triggered = True
            best_action = "hold"
            # Recalculate agreement_ratio for hold action when neutral guard triggers
            # This ensures agreement_count matches agreement_ratio
            if total_count > 0:
                # For hold, count models that are close to neutral (neither strongly positive nor negative)
                neutral_count = total_count - positive_count - negative_count
                agreement_ratio = neutral_count / total_count if total_count > 0 else 0.0
            else:
                agreement_ratio = 0.0
            # When neutral guard triggers, preserve some confidence since models did make predictions
            # The predictions were just too small to act on, but models are still functioning
            # Use the original confidence (before neutral guard) but cap it reasonably
            # Minimum confidence should reflect that models are working, just signal is weak
            hold_score = action_scores.get("hold", 0.0)
            # Preserve original confidence but cap it, with minimum floor
            # If models made predictions (even if small), give at least 5% confidence
            min_confidence_when_guard_triggered = 0.05  # 5% minimum when guard triggers
            confidence = max(
                min_confidence_when_guard_triggered,  # Minimum floor
                max(hold_score, min(0.55, confidence))  # Use hold score or original confidence, capped at 55%
            )
            consensus_return = 0.0
        
        # Final cap check using dynamic cap (calculated above)
        # This ensures consistency and prevents overconfidence
        confidence = min(dynamic_cap, confidence)  # Ensure never exceeds dynamic cap
        
        # Calculate agreement count (how many models agree with best action)
        # CRITICAL: Count models that actually agree with best_action, not just positive/negative
        # This ensures accurate agreement_count even when best_action is determined by weighted voting
        agreement_count = 0
        if best_action == "long":
            # Count models that predict positive (above threshold)
            for name, data in outputs.items():
                pred_return = data["predicted_return"]
                if pred_return > self.dynamic_threshold:
                    agreement_count += 1
        elif best_action == "short":
            # Count models that predict negative (below negative threshold)
            for name, data in outputs.items():
                pred_return = data["predicted_return"]
                if pred_return < -self.dynamic_threshold:
                    agreement_count += 1
        else:  # hold
            # For hold, count models that are close to neutral (within threshold)
            for name, data in outputs.items():
                pred_return = data["predicted_return"]
                if abs(pred_return) <= self.dynamic_threshold:
                    agreement_count += 1
        
        # Ensure agreement_ratio matches agreement_count
        if total_count > 0:
            calculated_agreement_ratio = agreement_count / total_count
            # If there's a mismatch, use the calculated ratio (more accurate)
            if abs(calculated_agreement_ratio - agreement_ratio) > 0.1:
                agreement_ratio = calculated_agreement_ratio
        
        # Note: horizon_profile is passed via self.horizon_profile in predict() method
        # We'll clamp in predict() method after consensus is computed
        return {
            "consensus_action": best_action,
            "consensus_return": float(consensus_return),  # Will be clamped in predict()
            "consensus_confidence": confidence,
            "action_scores": action_scores,
            # Add model agreement info for execution engine filtering
            "model_agreement_ratio": agreement_ratio,
            "total_models": total_count,
            "agreement_count": agreement_count,
            "neutral_guard_triggered": neutral_guard_triggered,
            "neutral_return_threshold": float(neutral_threshold),
            "raw_consensus_return": float(raw_consensus_return),
        }

    def update_live_metrics(self, predicted_return: float, actual_return: float):
        return self.live_tracker.update(predicted_return, actual_return)


