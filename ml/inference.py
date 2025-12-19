"""
Deployment-ready inference pipeline for classical + RL models.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from ml.risk import RiskManager, RiskManagerConfig
from monitoring.live_metrics import LiveMetricsTracker


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
                    if hasattr(model, "predict_proba") and "directional" in name:
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
                import warnings
                error_msg = f"Model {name} prediction failed: {exc}"
                warnings.warn(error_msg, UserWarning)
                # Also print to stderr for immediate visibility
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
                    print(f"[WARNING] Models produced very similar predictions (std={raw_std:.6f})")
                    prediction_detection_results["similar_predictions_detected"] = True
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
                
                # For intraday, use price action ONLY if movement is SIGNIFICANT (>0.1%)
                # This prevents overriding model predictions with tiny price movements
                # Models should handle small movements, price action override is for STRONG signals only
                SIGNIFICANT_MOVE_THRESHOLD = 0.001  # 0.1% minimum for override
                
                if is_green_candle and price_change_pct > SIGNIFICANT_MOVE_THRESHOLD:
                    # Strong green candle - price going UP significantly
                    # Only override if movement is meaningful (>0.1%)
                    price_action_confidence = min(0.75 + abs(price_change_pct) * 5, 0.90)  # 0.75-0.90 range (reduced from 0.85-0.95)
                    typical_vol = get_typical_volatility(self.asset_type)
                    intraday_price_action_override = {
                        "action": "long",
                        "return": min(price_change_pct * 1.2, typical_vol),  # Reduced scaling (1.2x instead of 1.5x)
                        "confidence": price_action_confidence,
                        "reasoning": f"Real-time price action: STRONG GREEN candle (+{price_change_pct*100:.2f}%). Price moving UP from ${previous_close:.2f} to ${current_price:.2f}",
                        "price_action_detected": True,
                        "previous_close": previous_close,
                        "current_price": current_price,
                        "price_change_pct": price_change_pct,
                    }
                elif is_red_candle and price_change_pct < -SIGNIFICANT_MOVE_THRESHOLD:
                    # Strong red candle - price going DOWN significantly
                    # Only override if movement is meaningful (>0.1%)
                    price_action_confidence = min(0.75 + abs(price_change_pct) * 5, 0.90)  # 0.75-0.90 range
                    typical_vol = get_typical_volatility(self.asset_type)
                    intraday_price_action_override = {
                        "action": "short",
                        "return": max(price_change_pct * 1.2, -typical_vol),  # Reduced scaling
                        "confidence": price_action_confidence,
                        "reasoning": f"Real-time price action: STRONG RED candle ({price_change_pct*100:.2f}%). Price moving DOWN from ${previous_close:.2f} to ${current_price:.2f}",
                        "price_action_detected": True,
                        "previous_close": previous_close,
                        "current_price": current_price,
                        "price_change_pct": price_change_pct,
                    }
                else:
                    # Price movement is too small (<0.1%) - let models decide (don't override)
                    # This allows models to predict SHORT/FLAT even if price is slightly up
                    intraday_price_action_override = {
                        "action": None,  # Let models decide - don't force LONG
                        "return": None,
                        "confidence": None,
                        "reasoning": f"Real-time price action: MINOR movement ({price_change_pct*100:.3f}% < 0.1% threshold). Using model predictions (may be SHORT/FLAT).",
                        "price_action_detected": False,
                        "previous_close": previous_close,
                        "current_price": current_price,
                        "price_change_pct": price_change_pct,
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
        # IMPORTANT: Only override if price action is STRONG (>0.1%) to allow models to predict SHORT/FLAT
        if intraday_price_action_override and intraday_price_action_override.get("price_action_detected"):
            # Override consensus with real-time price action for intraday
            override_action = intraday_price_action_override.get("action")
            override_return = intraday_price_action_override.get("return")
            override_confidence = intraday_price_action_override.get("confidence")
            
            if override_action and override_return is not None:
                # Blend 50% price action + 50% model prediction (reduced from 70/30)
                # This gives models more weight so they can still predict SHORT/FLAT
                model_return = consensus.get("consensus_return", 0.0)
                model_action = consensus.get("consensus_action", "hold")
                
                # If model strongly disagrees (e.g., model says SHORT but price action says LONG),
                # reduce the override weight further
                if (override_action == "long" and model_action == "short") or (override_action == "short" and model_action == "long"):
                    # Strong disagreement - use 30% price action, 70% model
                    blended_return = override_return * 0.3 + model_return * 0.7
                    blended_confidence = override_confidence * 0.3 + consensus.get("consensus_confidence", 0.0) * 0.7
                    consensus["price_action_model_disagreement"] = True
                else:
                    # Agreement or neutral - use 50/50 blend
                    blended_return = override_return * 0.5 + model_return * 0.5
                    blended_confidence = override_confidence * 0.5 + consensus.get("consensus_confidence", 0.0) * 0.5
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
                
                consensus["consensus_return"] = blended_return
                consensus["consensus_action"] = override_action
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
        elif self.horizon_profile == "long":
            # For long-term predictions (30 days), apply mean reversion logic
            # Key insight: price can go down for a few days, but final prediction at 30 days
            # should account for mean reversion. Long-term moves rarely go in straight lines.
            
            # Step 1: Cap at realistic 30-day moves (typically 5-8% for crypto)
            max_long_move = 0.08  # 8% max for 30-day prediction
            if abs(consensus_return) > max_long_move:
                # Scale down extreme predictions
                scale = max_long_move / abs(consensus_return)
                consensus_return = consensus_return * scale
                consensus["extreme_prediction_scaled"] = True
            
            # Step 2: Apply mean reversion for long-term predictions
            # For long-term, we expect mean reversion: if prediction is strongly negative,
            # apply a pull-back factor (markets rarely fall in straight lines for 30 days)
            mean_reversion_factor = 1.0
            if abs(consensus_return) > 0.03:  # If prediction > 3%
                # Apply mean reversion: reduce magnitude by 20-40% depending on strength
                # Stronger predictions get more dampening (they're less likely to persist)
                strength = min(abs(consensus_return) / 0.08, 1.0)  # Normalize to 0-1
                mean_reversion_dampening = 0.6 + 0.4 * (1.0 - strength)  # 0.6-1.0 range
                mean_reversion_factor = mean_reversion_dampening
                consensus["mean_reversion_applied"] = True
                consensus["mean_reversion_factor"] = float(mean_reversion_factor)
            
            # Step 3: Feature-based mean reversion adjustment
            # If technical indicators suggest oversold/overbought, apply STRONG correction
            # This can even flip the direction if indicators strongly suggest mean reversion
            feature_mean_reversion = 0.0
            strong_oversold = False
            strong_overbought = False
            
            if feature_signals.get("rsi_signal") is not None:
                rsi_signal = feature_signals["rsi_signal"]
                rsi_value = feature_signals.get("rsi")
                
                # Strong oversold: RSI < 30 and signal > 0.5
                if rsi_value is not None and rsi_value < 30 and rsi_signal > 0.5:
                    strong_oversold = True
                    # If all models predict negative and we're strongly oversold, apply STRONG mean reversion
                    if consensus_return < -0.02:  # Strong negative prediction
                        # Flip or strongly reduce negative prediction - can flip direction
                        feature_mean_reversion = abs(consensus_return) * 0.8  # Add 80% of negative magnitude (can flip)
                        consensus["strong_oversold_flip"] = True
                    elif consensus_return < 0:
                        feature_mean_reversion = abs(consensus_return) * 0.5  # Add 50% of negative magnitude
                
                # Moderate oversold: RSI 30-40 (signal 0.0-0.5)
                elif rsi_value is not None and 30 <= rsi_value < 40 and rsi_signal > 0:
                    # RSI is oversold (30-40), apply mean reversion
                    if consensus_return < -0.02:  # Strong negative prediction
                        # Apply strong mean reversion for moderate oversold
                        feature_mean_reversion = abs(consensus_return) * 0.6  # Add 60% of negative magnitude
                        consensus["moderate_oversold_detected"] = True
                    elif consensus_return < 0:
                        # Moderate negative prediction
                        feature_mean_reversion = abs(consensus_return) * 0.4  # Add 40% of negative magnitude
                
                # Strong overbought: RSI > 70 and signal < -0.5
                elif rsi_value is not None and rsi_value > 70 and rsi_signal < -0.5:
                    strong_overbought = True
                    # If all models predict positive and we're strongly overbought, apply STRONG mean reversion
                    if consensus_return > 0.02:  # Strong positive prediction
                        feature_mean_reversion = -abs(consensus_return) * 0.8  # Subtract 80% of positive magnitude (can flip)
                        consensus["strong_overbought_flip"] = True
                    elif consensus_return > 0:
                        feature_mean_reversion = -abs(consensus_return) * 0.5  # Subtract 50% of positive magnitude
                
                # Moderate overbought: RSI 60-70 (signal -0.5 to 0.0)
                elif rsi_value is not None and 60 < rsi_value <= 70 and rsi_signal < 0:
                    # RSI is overbought (60-70), apply mean reversion
                    if consensus_return > 0.02:  # Strong positive prediction
                        feature_mean_reversion = -abs(consensus_return) * 0.6  # Subtract 60% of positive magnitude
                        consensus["moderate_overbought_detected"] = True
                    elif consensus_return > 0:
                        feature_mean_reversion = -abs(consensus_return) * 0.4  # Subtract 40% of positive magnitude
            
            if feature_signals.get("sma50_signal") is not None:
                sma50_signal = feature_signals["sma50_signal"]
                price_vs_sma = feature_signals.get("price_vs_sma50", 0)
                
                # If price is VERY far below SMA (>15% below), expect VERY strong mean reversion up
                if price_vs_sma < -0.15 and consensus_return < 0:
                    # Price is >15% below SMA and prediction is negative - VERY strong mean reversion signal
                    # Can flip direction completely
                    feature_mean_reversion += abs(consensus_return) * 0.7  # Add 70% of negative magnitude (can flip)
                    consensus["very_oversold_sma_detected"] = True
                # If price is far below SMA (>10% below), expect strong mean reversion up
                elif price_vs_sma < -0.10 and consensus_return < 0:
                    # Price is >10% below SMA and prediction is negative - strong mean reversion signal
                    feature_mean_reversion += abs(consensus_return) * 0.5  # Add 50% of negative magnitude
                    consensus["oversold_sma_detected"] = True
                # If price is far above SMA (>10% above), expect mean reversion down
                elif price_vs_sma > 0.10 and consensus_return > 0:
                    feature_mean_reversion -= abs(consensus_return) * 0.5  # Subtract 50% of positive magnitude
                    consensus["overbought_sma_detected"] = True
                # Moderate deviation (5-10% away from SMA)
                elif abs(price_vs_sma) > 0.05 and consensus_return * price_vs_sma < 0:  # Opposite directions
                    # Price deviation suggests mean reversion opposite to prediction
                    feature_mean_reversion += abs(consensus_return) * 0.3  # Add 30% adjustment
            
            # Apply feature-based mean reversion
            if abs(feature_mean_reversion) > 0:
                consensus_return_before = consensus_return
                consensus_return = consensus_return + feature_mean_reversion
                consensus["feature_mean_reversion_applied"] = True
                consensus["feature_mean_reversion_adjustment"] = float(feature_mean_reversion)
                consensus["consensus_return_before_mean_reversion"] = float(consensus_return_before)
                if strong_oversold:
                    consensus["strong_oversold_detected"] = True
                if strong_overbought:
                    consensus["strong_overbought_detected"] = True
                
                # Log the adjustment for debugging
                consensus["mean_reversion_debug"] = {
                    "rsi_value": feature_signals.get("rsi"),
                    "rsi_signal": feature_signals.get("rsi_signal"),
                    "price_vs_sma50": feature_signals.get("price_vs_sma50"),
                    "sma50_signal": feature_signals.get("sma50_signal"),
                    "adjustment_applied": float(feature_mean_reversion),
                    "return_before": float(consensus_return_before),
                    "return_after": float(consensus_return),
                }
            
            # Apply mean reversion factor
            consensus_return = consensus_return * mean_reversion_factor
            consensus["consensus_return"] = consensus_return
            
        elif self.horizon_profile == "short":
            # For short-term, cap at 6% for 4-day prediction
            max_short_move = 0.06  # 6% max for 4-day prediction
            if abs(consensus_return) > max_short_move:
                scale = max_short_move / abs(consensus_return)
                consensus_return = consensus_return * scale
                consensus["consensus_return"] = consensus_return
                consensus["extreme_prediction_scaled"] = True
            
            # Apply lighter mean reversion for short-term (less dampening)
            if abs(consensus_return) > 0.04:  # If prediction > 4%
                mean_reversion_factor = 0.85  # Reduce by 15%
                consensus_return = consensus_return * mean_reversion_factor
                consensus["consensus_return"] = consensus_return
                consensus["mean_reversion_applied"] = True
        
        # Recalculate action after dampening (action might have changed)
        final_consensus_return = consensus["consensus_return"]
        
        # For intraday with price action override, don't recalculate action (it's already set correctly)
        if not consensus.get("intraday_price_action_override", False):
            # Final check: if mean reversion strongly suggests opposite direction, respect it
            # Check if we have multiple strong mean reversion signals (RSI + SMA)
            strong_mean_reversion_up = (
                consensus.get("strong_oversold_flip") or 
                consensus.get("moderate_oversold_detected") or
                consensus.get("very_oversold_sma_detected") or
                (consensus.get("oversold_sma_detected") and consensus.get("moderate_oversold_detected"))
            )
            strong_mean_reversion_down = (
                consensus.get("strong_overbought_flip") or
                consensus.get("moderate_overbought_detected") or
                consensus.get("overbought_sma_detected")
            )
            
            if strong_mean_reversion_up and final_consensus_return < -self.dynamic_threshold:
                # Strong oversold detected (RSI + SMA) but still negative - apply strong correction
                # Instead of forcing hold, apply additional correction to flip or neutralize
                if abs(final_consensus_return) > 0.03:  # If still very negative (>3%)
                    # Apply additional correction to bring it closer to neutral or positive
                    additional_correction = abs(final_consensus_return) * 0.5
                    final_consensus_return = final_consensus_return + additional_correction
                    consensus["consensus_return"] = final_consensus_return
                    consensus["mean_reversion_strong_correction"] = True
            elif strong_mean_reversion_down and final_consensus_return > self.dynamic_threshold:
                # Strong overbought detected but still positive - apply strong correction
                if abs(final_consensus_return) > 0.03:  # If still very positive (>3%)
                    additional_correction = abs(final_consensus_return) * 0.5
                    final_consensus_return = final_consensus_return - additional_correction
                    consensus["consensus_return"] = final_consensus_return
                    consensus["mean_reversion_strong_correction"] = True
            elif final_consensus_return > self.dynamic_threshold:
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
        if abs(consensus_return) < neutral_threshold and best_action != "hold":
            # Neutral guard: if the expected move is smaller than the noise band,
            # we should not recommend a directional trade. Regardless of how
            # strong the directional score is, we force the action to HOLD and
            # set the expected return to 0. This keeps the behaviour logically
            # consistent across all symbols/horizons.
            neutral_guard_triggered = True
            best_action = "hold"
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
        agreement_count = 0
        if best_action == "long":
            agreement_count = positive_count
        elif best_action == "short":
            agreement_count = negative_count
        else:  # hold
            # For hold, count models that are close to neutral
            agreement_count = total_count - positive_count - negative_count
        
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


