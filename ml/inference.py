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

# Horizon-aware prediction limits (based on typical volatility)
HORIZON_CLAMPS = {
    "intraday": 0.05,   # ±5% max for 1-day (extreme but possible)
    "short": 0.10,      # ±10% max for 4-day
    "long": 0.20,       # ±20% max for 30-day
}

# Sanity check: typical daily volatility for crypto (conservative)
TYPICAL_DAILY_VOLATILITY = 0.03  # 3% daily volatility for BTC


def _clamp(value: float, clamp: float = PREDICTION_CLAMP, horizon_profile: Optional[str] = None) -> float:
    """Clamp prediction to reasonable range, with horizon-aware limits."""
    # Use horizon-specific clamp if available
    if horizon_profile and horizon_profile in HORIZON_CLAMPS:
        clamp = HORIZON_CLAMPS[horizon_profile]
    # Apply global clamp as well (safety net)
    clamp = min(clamp, PREDICTION_CLAMP)
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
    ):
        self.model_dir = Path(model_dir)
        self.scaler = None
        self.models: Dict[str, any] = {}
        self.metrics: Dict[str, Dict] = {}
        self.summary: Dict[str, any] = {}
        self.metadata: Dict[str, any] = {}
        self.dynamic_threshold = 0.01
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
                        model_outputs[name] = {
                            "predicted_return": pred_return,
                            "probability": prob,
                        }
                    else:
                        pred = float(model.predict(features)[0])
                    # Apply horizon-aware clamping
                    clamped_pred = _clamp(pred, horizon_profile=self.horizon_profile)
                    # Additional sanity check: if prediction is extreme, scale it down
                    if self.horizon_profile == "intraday" and abs(clamped_pred) > TYPICAL_DAILY_VOLATILITY:
                        # For intraday, if prediction exceeds typical daily volatility, scale it
                        scale_factor = TYPICAL_DAILY_VOLATILITY / abs(clamped_pred)
                        clamped_pred = clamped_pred * min(scale_factor, 1.0)
                    base_model_outputs[name] = clamped_pred
                    model_outputs[name] = {"predicted_return": clamped_pred}
            except Exception as exc:
                # Log the error for debugging but continue with other models
                import warnings
                error_msg = f"Model {name} prediction failed: {exc}"
                warnings.warn(error_msg, UserWarning)
                # Also print to stderr for immediate visibility
                print(f"[WARNING] {error_msg}", flush=True)
                continue
        
        # Second pass: if we have stacked_blend and at least 2 base model predictions, use it
        if "stacked_blend" in self.models and len(base_model_outputs) >= 2:
            try:
                stacked_model = self.models["stacked_blend"]
                # Stacked blend expects predictions from base models as input
                base_predictions = np.array([[base_model_outputs[k] for k in sorted(base_model_outputs.keys())]])
                # Suppress sklearn feature name warnings
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="X does not have valid feature names")
                    stack_pred = float(stacked_model.predict(base_predictions)[0])
                clamped_pred = _clamp(stack_pred, horizon_profile=self.horizon_profile)
                model_outputs["stacked_blend"] = {"predicted_return": clamped_pred}
            except Exception as exc:
                import warnings
                error_msg = f"Stacked blend prediction failed: {exc}"
                warnings.warn(error_msg, UserWarning)
                # Also print to stderr for immediate visibility
                print(f"[WARNING] {error_msg}", flush=True)

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

        consensus = self._compute_consensus(model_outputs)
        # Apply horizon-aware sanity check to consensus return
        consensus_return = consensus["consensus_return"]
        if self.horizon_profile == "intraday":
            # For intraday, cap at typical daily volatility (3%)
            max_intraday_move = TYPICAL_DAILY_VOLATILITY
            if abs(consensus_return) > max_intraday_move:
                # Scale down extreme predictions
                scale = max_intraday_move / abs(consensus_return)
                consensus_return = consensus_return * scale
                consensus["consensus_return"] = consensus_return
                consensus["extreme_prediction_scaled"] = True
        
        confidence = abs(consensus["consensus_return"]) / max(self.dynamic_threshold, 1e-4)
        allowed = self.risk_manager.should_trade(confidence, volatility)
        if not allowed:
            consensus["consensus_action"] = "hold"
            consensus["risk_blocked"] = True
        consensus["position_size"] = self.risk_manager.max_position_allowed(confidence, volatility)
        consensus["target_price"] = current_price * (1.0 + consensus["consensus_return"])
        if self.horizon_profile:
            consensus["horizon_profile"] = self.horizon_profile
        if self.target_horizon_bars is not None:
            consensus["target_horizon_bars"] = self.target_horizon_bars
        payload = {"models": model_outputs, "consensus": consensus}
        if self.target_profile:
            payload["target_profile"] = self.target_profile
        return payload

    def _compute_consensus(self, outputs: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        action_scores = {"long": 0.0, "hold": 0.0, "short": 0.0}
        weighted_returns = []
        for name, data in outputs.items():
            metrics = self.metrics.get(name, {})
            r2 = metrics.get("r2") or 0.0
            directional = metrics.get("directional_accuracy") or 0.5
            weight = max(0.05, float(r2) + 0.5 * float(directional - 0.5))
            pred_return = data["predicted_return"]
            action = "long" if pred_return > self.dynamic_threshold else "short" if pred_return < -self.dynamic_threshold else "hold"
            action_scores[action] += weight * abs(pred_return)
            weighted_returns.append((pred_return, weight))
        if not weighted_returns:
            return {"consensus_action": "hold", "consensus_return": 0.0, "consensus_confidence": 0.0}
        total_weight = sum(w for _, w in weighted_returns) or 1.0
        consensus_return = sum(r * w for r, w in weighted_returns) / total_weight
        best_action = max(action_scores, key=action_scores.get)
        confidence = min(1.0, abs(consensus_return) / max(self.dynamic_threshold, 1e-4))
        
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
            confidence = max(action_scores.get("hold", 0.0), min(0.55, confidence))
            consensus_return = 0.0
        
        # Cap confidence to prevent overconfident predictions (anti-overfitting measure)
        # Even well-calibrated models should rarely exceed 0.9 confidence in noisy markets
        MAX_CONFIDENCE_CAP = 0.90
        confidence = min(confidence, MAX_CONFIDENCE_CAP)
        
        # Note: horizon_profile is passed via self.horizon_profile in predict() method
        # We'll clamp in predict() method after consensus is computed
        return {
            "consensus_action": best_action,
            "consensus_return": float(consensus_return),  # Will be clamped in predict()
            "consensus_confidence": confidence,
            "action_scores": action_scores,
            "neutral_guard_triggered": neutral_guard_triggered,
            "neutral_return_threshold": float(neutral_threshold),
            "raw_consensus_return": float(raw_consensus_return),
        }

    def update_live_metrics(self, predicted_return: float, actual_return: float):
        return self.live_tracker.update(predicted_return, actual_return)


