"""
MCP Adapter - Wrappers for agent calls so the agent can be called as a tool by an orchestrator.
Provides request/response logging.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

from .system_bootstrap import ensure_symbol_ready
from ml.horizons import PROFILE_BASES, normalize_profile, DEFAULT_HORIZON_PROFILE
from ml.inference import InferencePipeline
from ml.risk import RiskManagerConfig
from ml.rl_feedback import get_feedback_learner
from ml.bucket_logger import get_bucket_logger
import pandas as pd
from core.model_paths import (
    horizon_dir,
    summary_path as build_summary_path,
    list_horizon_dirs,
    timeframe_dir,
)


class MCPAdapter:
    """
    MCP adapter that wraps agent calls with logging and error handling.
    Allows the agent to be called as a tool by an orchestrator.
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize MCP adapter.
        
        Args:
            log_dir: Directory for request/response logs
        """
        self.log_dir = log_dir or Path("logs/mcp_adapter")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.request_log_file = self.log_dir / "requests.jsonl"
        self.response_log_file = self.log_dir / "responses.jsonl"
        
        self.bucket_logger = get_bucket_logger()
        self.feedback_learner = get_feedback_learner()
        
        # Cache for inference pipelines + feature manifests
        self._pipeline_cache: Dict[str, InferencePipeline] = {}
        self._feature_manifest_cache: Dict[str, List[str]] = {}
        self._meta_min_confidence = 0.55
        self._probability_classes = ("SHORT", "HOLD", "LONG")
        # Confidence calibration: reduce overconfidence
        self._confidence_temperature = 1.5  # Higher = less confident (calibrate down)
        self._confidence_dampening = 0.85  # Multiply confidence by this factor
    
    def _log_request(self, tool_name: str, request_data: Dict[str, Any]):
        """Log incoming request."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "tool": tool_name,
            "request": request_data
        }
        
        try:
            with open(self.request_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[MCP_ADAPTER] Failed to log request: {exc}")
    
    def _log_response(self, tool_name: str, request_data: Dict[str, Any], response_data: Dict[str, Any], success: bool = True):
        """Log response."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "tool": tool_name,
            "request": request_data,
            "response": response_data,
            "success": success
        }
        
        try:
            with open(self.response_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[MCP_ADAPTER] Failed to log response: {exc}")
    
    def _get_pipeline(self, asset_type: str, symbol: str, timeframe: str, horizon: str) -> Optional[InferencePipeline]:
        """Get or load inference pipeline."""
        normalized = normalize_profile(horizon or DEFAULT_HORIZON_PROFILE)
        cache_key = f"{asset_type}/{symbol}/{timeframe}/{normalized}"
        
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]
        
        model_dir = horizon_dir(asset_type, symbol, timeframe, normalized)
        if not model_dir.exists():
            legacy_dir = timeframe_dir(asset_type, symbol, timeframe)
            if legacy_dir.exists():
                model_dir = legacy_dir
            else:
                return None
        
        try:
            pipeline = InferencePipeline(model_dir, risk_config=RiskManagerConfig())
            pipeline.load()
            self._pipeline_cache[cache_key] = pipeline
            return pipeline
        except Exception as exc:
            print(f"[MCP_ADAPTER] Failed to load pipeline for {cache_key}: {exc}")
            return None
    
    def _feature_manifest_path(self, asset_type: str, symbol: str, timeframe: str, horizon: str) -> Path:
        horizon_path = horizon_dir(asset_type, symbol, timeframe, horizon)
        legacy_path = timeframe_dir(asset_type, symbol, timeframe)
        candidate = horizon_path / "feature_manifest.json"
        if candidate.exists() or not (legacy_path / "feature_manifest.json").exists():
            return candidate
        return legacy_path / "feature_manifest.json"

    def _load_feature_manifest(self, asset_type: str, symbol: str, timeframe: str, horizon: str) -> Optional[List[str]]:
        """Discover the exact feature order used during training."""
        normalized = normalize_profile(horizon or DEFAULT_HORIZON_PROFILE)
        cache_key = f"{asset_type}/{symbol}/{timeframe}/{normalized}"
        if cache_key in self._feature_manifest_cache:
            return self._feature_manifest_cache[cache_key]

        manifest_path = self._feature_manifest_path(asset_type, symbol, timeframe, normalized)
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                features = payload.get("selected_features") or payload.get("features")
                if isinstance(features, list) and features:
                    self._feature_manifest_cache[cache_key] = features
                    return features
            except Exception:
                pass

        # Fallback to training log if manifest missing
        training_log = (
            Path("logs")
            / "training"
            / asset_type
            / symbol
            / timeframe
            / "training_log.json"
        )
        if training_log.exists():
            try:
                log_payload = json.loads(training_log.read_text(encoding="utf-8"))
                events = log_payload.get("events") or []
                if isinstance(events, list):
                    for entry in reversed(events):
                        data_block = entry.get("data")
                        if not isinstance(data_block, dict):
                            continue
                        features = data_block.get("selected_features") or data_block.get("sample_features")
                        if isinstance(features, list) and features:
                            normalized = [str(name) for name in features if isinstance(name, str)]
                            if normalized:
                                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                                manifest_payload = {
                                    "selected_features": normalized,
                                    "source": "training_log",
                                    "cached_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                                }
                                try:
                                    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
                                except Exception:
                                    pass
                                self._feature_manifest_cache[cache_key] = normalized
                                return normalized
            except Exception:
                pass
        return None

    def _build_feature_series(
        self,
        asset_type: str,
        symbol: str,
        timeframe: str,
        horizon: str,
        feature_data: Dict[str, Any],
    ):
        features_dict: Dict[str, float] = {}
        if "features" in feature_data:
            for feat_name, feat_data in feature_data["features"].items():
                value: Optional[float] = None
                if isinstance(feat_data, dict):
                    raw = feat_data.get("value")
                    if isinstance(raw, (int, float)):
                        value = float(raw)
                elif isinstance(feat_data, (int, float)):
                    value = float(feat_data)
                if value is not None and not isinstance(value, bool):
                    features_dict[feat_name] = value

        manifest = self._load_feature_manifest(asset_type, symbol, timeframe, horizon)
        if manifest:
            ordered: Dict[str, float] = {}
            for name in manifest:
                value = features_dict.get(name)
                if value is None:
                    # Backfill with neutral value when training feature is unavailable at inference time.
                    ordered[name] = 0.0
                else:
                    ordered[name] = value
            return pd.Series(ordered, dtype=float)
        
        # Fallback: use whatever features are available (previous behaviour)
        return pd.Series(features_dict, dtype=float)

    def _canonical_action(self, action: Optional[str]) -> str:
        mapping = {
            "long": "LONG",
            "short": "SHORT",
            "hold": "HOLD",
            "buy": "LONG",
            "sell": "SHORT",
        }
        if not action:
            return "HOLD"
        return mapping.get(str(action).lower(), str(action).upper())

    def _softmax(self, logits: Dict[str, float]) -> Dict[str, float]:
        max_logit = max(logits.values())
        exp_vals = {k: math.exp(v - max_logit) for k, v in logits.items()}
        total = sum(exp_vals.values()) or 1.0
        return {k: exp_vals[k] / total for k in logits}

    def _return_to_probabilities(self, pred_return: float, threshold: float) -> Dict[str, float]:
        scaled = pred_return / max(threshold, 1e-6)
        scaled = max(-2.5, min(2.5, scaled))
        logits = {
            "LONG": scaled,
            "SHORT": -scaled,
            "HOLD": 1.0 - abs(scaled),
        }
        # Apply temperature scaling to reduce overconfidence
        temp_logits = {k: v / self._confidence_temperature for k, v in logits.items()}
        probs = self._softmax(temp_logits)
        return {cls: float(probs.get(cls, 0.0)) for cls in self._probability_classes}

    def _build_model_actions(self, individual_models: List[Dict[str, Any]]) -> Dict[str, str]:
        actions = {}
        for entry in individual_models:
            name = entry.get("name", "")
            if any(tag in name.lower() for tag in ["random_forest", "lightgbm", "xgboost"]):
                actions[name] = self._canonical_action(entry.get("action"))
        return actions
    
    def _build_price_explanation(
        self,
        main_model_predictions: List[Dict[str, Any]],
        consensus_return: float,
        current_price: float,
        dqn_data: Optional[Dict[str, Any]],
        dqn_action: Optional[str],
    ) -> Dict[str, Any]:
        """Build explanation of how predicted_price is calculated."""
        explanation = {
            "calculation_method": "weighted_average",
            "models_used": len(main_model_predictions),
            "current_price": current_price,
            "consensus_return": consensus_return,
            "predicted_price": float(current_price * (1.0 + consensus_return)),
        }
        
        # Always show calculation_details if we have any models (matching crypto format)
        if len(main_model_predictions) > 0:
            # Calculate weighted average
            total_weight = sum(p["weight"] for p in main_model_predictions)
            weighted_sum = sum(p["predicted_return"] * p["weight"] for p in main_model_predictions)
            calculated_consensus = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            explanation["calculation_details"] = []
            for pred in main_model_predictions:
                explanation["calculation_details"].append({
                    "model": pred["name"],
                    "predicted_return": pred["predicted_return"],
                    "predicted_price": pred["predicted_price"],
                    "weight": pred["weight"],
                    "contribution": pred["predicted_return"] * pred["weight"],
                })
            
            explanation["weighted_average"] = {
                "total_weight": total_weight,
                "weighted_sum": weighted_sum,
                "consensus_return": calculated_consensus,
            }
            
            # Check for disagreement (even with 1 model, we still show agreement metrics)
            prices = [p["predicted_price"] for p in main_model_predictions]
            if len(prices) > 1:
                price_range = max(prices) - min(prices)
                price_variance = sum((p - sum(prices) / len(prices)) ** 2 for p in prices) / len(prices)
                price_std = (price_variance ** 0.5) / current_price if current_price > 0 else 0.0  # Relative std
            else:
                # Single model: no disagreement
                price_range = 0.0
                price_std = 0.0
            
            explanation["model_agreement"] = {
                "price_range": price_range,
                "price_range_pct": (price_range / current_price) * 100 if current_price > 0 else 0.0,
                "price_std_pct": price_std * 100,
                "models_agree": price_std < 0.02,  # Less than 2% std deviation
            }
            
            if len(prices) > 1 and price_std >= 0.02:  # Models disagree significantly
                explanation["disagreement_warning"] = True
                explanation["recommendation"] = (
                    "The price models show significant disagreement (std deviation: {:.2f}%). "
                    "Consider using DQN recommendation as an alternative approach."
                ).format(price_std * 100)
            else:
                explanation["disagreement_warning"] = False
                explanation["recommendation"] = "Models are in good agreement."
        
        # Add DQN comparison if available
        if dqn_data and dqn_action:
            dqn_metrics = dqn_data.get("metrics", {})
            dqn_predicted_return = float(dqn_metrics.get("predicted_return", 0.0))
            if dqn_predicted_return == 0.0:
                # Recalculate DQN return if needed
                test_metrics = dqn_metrics.get("test_policy_metrics", {})
                if test_metrics:
                    base_return = float(test_metrics.get("avg_return", 0.0))
                    if dqn_action.upper() == "SHORT":
                        dqn_predicted_return = -abs(base_return)
                    elif dqn_action.upper() == "LONG":
                        dqn_predicted_return = abs(base_return)
            
            explanation["dqn_comparison"] = {
                "dqn_predicted_return": dqn_predicted_return,
                "dqn_predicted_price": float(current_price * (1.0 + dqn_predicted_return)),
                "difference_from_consensus": dqn_predicted_return - consensus_return,
                "difference_pct": ((dqn_predicted_return - consensus_return) / abs(consensus_return) * 100) if consensus_return != 0 else 0.0,
            }
        
        return explanation
    
    def _build_dqn_recommendation(
        self,
        dqn_action: Optional[str],
        dqn_data: Optional[Dict[str, Any]],
        current_price: float,
        individual_models: List[Dict[str, Any]],
        dqn_calculated_return: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build full DQN recommendation with all details."""
        if not dqn_action and not dqn_data:
            return None
        
        # Build recommendation object
        recommendation = {
            "action": self._canonical_action(dqn_action) if dqn_action else "HOLD",
        }
        
        # Get DQN data from dqn_data
        if dqn_data:
            dqn_metrics = dqn_data.get("metrics", {})
            
            # Use calculated return if provided (from model_outputs), otherwise calculate
            if dqn_calculated_return is not None:
                dqn_predicted_return = dqn_calculated_return
            else:
                dqn_predicted_return = float(dqn_metrics.get("predicted_return", 0.0))
                
                # Recalculate if 0.0 or missing
                if (dqn_predicted_return == 0.0 or dqn_predicted_return is None) and dqn_action:
                    test_metrics = dqn_metrics.get("test_policy_metrics", {})
                    if test_metrics:
                        base_return = float(test_metrics.get("avg_return", 0.0))
                        if dqn_action.upper() == "SHORT":
                            dqn_predicted_return = -abs(base_return)
                        elif dqn_action.upper() == "LONG":
                            dqn_predicted_return = abs(base_return)
                        else:  # HOLD
                            dqn_predicted_return = 0.0
                    else:
                        # If no test metrics, use a small default based on action
                        if dqn_action.upper() == "SHORT":
                            dqn_predicted_return = -0.01  # Small negative default
                        elif dqn_action.upper() == "LONG":
                            dqn_predicted_return = 0.01  # Small positive default
                        else:
                            dqn_predicted_return = 0.0
            
            recommendation["predicted_return"] = float(dqn_predicted_return) if dqn_predicted_return is not None else 0.0
            recommendation["predicted_price"] = float(current_price * (1.0 + recommendation["predicted_return"]))
            
            # Calculate probabilities based on action
            if dqn_action:
                if dqn_action.upper() == "LONG":
                    probabilities = {"LONG": 0.6, "HOLD": 0.25, "SHORT": 0.15}
                elif dqn_action.upper() == "SHORT":
                    probabilities = {"LONG": 0.15, "HOLD": 0.25, "SHORT": 0.6}
                else:  # HOLD
                    probabilities = {"LONG": 0.25, "HOLD": 0.5, "SHORT": 0.25}
            else:
                probabilities = {"LONG": 0.33, "HOLD": 0.34, "SHORT": 0.33}
            
            recommendation["probabilities"] = probabilities
            
            # Use hit_rate as confidence
            test_metrics = dqn_metrics.get("test_policy_metrics", {})
            hit_rate = float(test_metrics.get("hit_rate", 0.5)) if test_metrics else 0.5
            recommendation["confidence"] = hit_rate * self._confidence_dampening
            recommendation["reason"] = dqn_metrics.get("action_reason", "Direct DQN policy decision")
            
            # Add validation/test metrics
            validation_metrics = dqn_metrics.get("validation_policy_metrics", {})
            test_metrics = dqn_metrics.get("test_policy_metrics", {})
            if validation_metrics or test_metrics:
                recommendation["metrics"] = {
                    "validation": {
                        "total_return": validation_metrics.get("total_return"),
                        "sharpe": validation_metrics.get("sharpe"),
                        "hit_rate": validation_metrics.get("hit_rate"),
                        "max_drawdown": validation_metrics.get("max_drawdown"),
                    },
                    "test": {
                        "total_return": test_metrics.get("total_return"),
                        "sharpe": test_metrics.get("sharpe"),
                        "hit_rate": test_metrics.get("hit_rate"),
                        "max_drawdown": test_metrics.get("max_drawdown"),
                    },
                }
        
        return recommendation

    def _compute_meta_probabilities(
        self,
        individual_models: List[Dict[str, Any]],
        consensus: Dict[str, Any],
    ) -> Tuple[Dict[str, float], float, str]:
        weighted_sum = {cls: 0.0 for cls in self._probability_classes}
        total_weight = 0.0
        for model in individual_models:
            probs = model.get("probabilities")
            if not probs:
                continue
            weight = max(float(model.get("confidence", 0.0)), 1e-3)
            for cls in self._probability_classes:
                weighted_sum[cls] += probs.get(cls, 0.0) * weight
            total_weight += weight
        if total_weight == 0.0:
            consensus_action = self._canonical_action(consensus.get("consensus_action", "hold"))
            probs = {cls: (1.0 if cls == consensus_action else 0.0) for cls in self._probability_classes}
        else:
            probs = {cls: weighted_sum[cls] / total_weight for cls in self._probability_classes}
        meta_action = max(probs, key=probs.get)
        meta_confidence = probs[meta_action]
        # Apply confidence dampening to reduce overconfidence
        calibrated_confidence = float(meta_confidence * self._confidence_dampening)
        # Ensure confidence doesn't drop below minimum
        calibrated_confidence = max(calibrated_confidence, 0.1) if meta_confidence > 0.5 else calibrated_confidence
        return probs, calibrated_confidence, meta_action

    def _validate_prediction(self, prediction: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        prediction["predicted_price"] = float(current_price * (1.0 + prediction.get("predicted_return", 0.0)))
        prediction["action"] = self._canonical_action(prediction.get("action"))
        prediction["consensus_action"] = self._canonical_action(prediction.get("consensus_action"))
        prediction["unified_action"] = self._canonical_action(prediction.get("unified_action"))
        prediction["confidence"] = float(min(max(prediction.get("confidence", 0.0), 0.0), 1.0))
        if prediction["confidence"] < self._meta_min_confidence:
            prediction["action"] = "HOLD"
            prediction["unified_action"] = "HOLD"
        model_actions = {}
        for model in prediction.get("individual_models", []):
            if "name" in model:
                model["action"] = self._canonical_action(model.get("action"))
        prediction["model_actions"] = self._build_model_actions(prediction.get("individual_models", []))
        return prediction
    
    def predict(
        self,
        symbols: List[str],
        horizon: Optional[str] = None,  # Changed to string: "long", "short", "intraday"
        risk_profile: Optional[str] = None,
        asset_type: str = "crypto",
        timeframe: str = "1d"
    ) -> Dict[str, Any]:
        """
        Predict for multiple symbols.
        
        Args:
            symbols: List of symbols to predict
            horizon: Prediction horizon profile ("long", "short", "intraday") - defaults to "long"
            risk_profile: Risk profile name (optional)
            asset_type: Asset type (crypto/commodities)
            timeframe: Timeframe (1d, 4h, etc.)
        
        Returns:
            Dictionary with predictions array including unified action and disagreement explanations
        """
        normalized_horizon = normalize_profile(horizon or DEFAULT_HORIZON_PROFILE)
        request_data = {
            "symbols": symbols,
            "horizon": normalized_horizon,
            "risk_profile": risk_profile,
            "asset_type": asset_type,
            "timeframe": timeframe
        }
        
        self._log_request("predict", request_data)
        
        predictions = []
        errors = []
        
        for symbol in symbols:
            try:
                # Normalize requested horizon once per symbol so bootstrap + reporting stay in sync
                horizon_profile = normalized_horizon
                horizon_config = PROFILE_BASES.get(horizon_profile, PROFILE_BASES["long"])
                horizon_bars = horizon_config.horizon

                # Ensure symbol is ready (data + models)
                bootstrap_status = ensure_symbol_ready(
                    asset_type=asset_type,
                    symbol=symbol,
                    timeframe=timeframe,
                    horizon_profile=horizon_profile,
                )
                if not bootstrap_status.get("ready"):
                    reason = bootstrap_status.get("error") or bootstrap_status.get("message") or "Unknown bootstrap error"
                    errors.append(f"Preparation failed for {symbol}: {reason}")
                    continue

                # Load summary.json directly - this is the canonical source of truth
                summary_path = build_summary_path(asset_type, symbol, timeframe, horizon_profile)
                if not summary_path.exists():
                    # Try legacy path
                    legacy_path = timeframe_dir(asset_type, symbol, timeframe) / "summary.json"
                    if legacy_path.exists():
                        summary_path = legacy_path
                    else:
                        errors.append(f"No summary.json found for {symbol}")
                        continue
                
                # Load summary.json
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                
                # Extract data from summary.json
                prediction_data = summary.get("prediction", {})
                consensus_data = summary.get("consensus", {})
                model_predictions = summary.get("model_predictions", {})
                
                # Get current price from summary (most accurate)
                current_price = float(prediction_data.get("current_price", 0))
                if current_price == 0:
                    # Fallback to model_reference_price
                    current_price = float(summary.get("technical", {}).get("model_config", {}).get("model_reference_price", 0))
                
                if current_price == 0:
                    errors.append(f"Could not get current price for {symbol}")
                    continue
                
                # Get predicted return from consensus (decimal form) - needed for model predictions
                predicted_return = float(consensus_data.get("predicted_return", 0))
                # If not available, convert from predicted_return_pct
                if predicted_return == 0:
                    predicted_return_pct = float(consensus_data.get("predicted_return_pct", 0))
                    predicted_return = predicted_return_pct / 100.0
                
                # Load metrics.json to get all trained models (even if not in summary.json)
                # This is used to build logical per-model outputs for ALL symbols.
                metrics_path = horizon_dir(asset_type, symbol, timeframe, horizon_profile) / "metrics.json"
                if not metrics_path.exists():
                    legacy_metrics_path = timeframe_dir(asset_type, symbol, timeframe) / "metrics.json"
                    if legacy_metrics_path.exists():
                        metrics_path = legacy_metrics_path
                
                all_metrics = {}
                if metrics_path.exists():
                    try:
                        with open(metrics_path, "r", encoding="utf-8") as f:
                            all_metrics = json.load(f)
                    except Exception:
                        pass
                
                # Add models from metrics.json that aren't in model_predictions
                # Ensures we surface all trained models for logical outputs
                for model_name, model_metrics in all_metrics.items():
                    if "_quantile" in model_name or "_directional" in model_name:
                        continue  # Skip quantile and directional models
                    if model_name in model_predictions:
                        continue  # Already present from summary.json
                    
                    model_r2 = float(model_metrics.get("r2", 0))
                    model_dir_acc = float(model_metrics.get("directional_accuracy", 0.5))
                    mean_pred_return = model_metrics.get("mean_predicted_return")
                    if mean_pred_return is not None:
                        model_pred_return = float(mean_pred_return)
                    else:
                        model_pred_return = predicted_return  # Fallback to consensus
                    model_pred_price = float(current_price * (1.0 + model_pred_return))
                    
                    # Determine action from predicted return
                    action_threshold = horizon_config.directional_threshold or 0.01
                    if model_pred_return >= action_threshold:
                        model_action = "long"
                    elif model_pred_return <= -action_threshold:
                        model_action = "short"
                    else:
                        model_action = "hold"
                    
                    # Calculate confidence from directional accuracy
                    confidence_pct = max(50.0, float(model_dir_acc * 100))
                    
                    model_predictions[model_name] = {
                        "predicted_price": model_pred_price,
                        "predicted_return_pct": float(model_pred_return * 100),
                        "action": model_action,
                        "confidence": confidence_pct,
                        "r2_score": model_r2,
                        "accuracy": float(model_dir_acc * 100) if model_dir_acc else None,
                    }
                
                # Get predicted price (initial, may be updated later from model-weighted average)
                predicted_price = float(prediction_data.get("predicted_price", current_price * (1.0 + predicted_return)))
                
                # Recalculate consensus action from predicted_return (initial, before model-weighted adjustment)
                action_threshold = horizon_config.directional_threshold or 0.01
                if predicted_return >= action_threshold:
                    consensus_action = "LONG"
                elif predicted_return <= -action_threshold:
                    consensus_action = "SHORT"
                else:
                    consensus_action = "HOLD"
                
                confidence_pct = float(consensus_data.get("confidence_pct", 0))
                confidence = confidence_pct / 100.0 if confidence_pct > 1.0 else confidence_pct
                
                # Build individual_models from model_predictions
                individual_models = []
                main_models = {}
                main_model_predictions = []
                
                for model_name, model_pred in model_predictions.items():
                    if "_quantile" in model_name:
                        continue
                    
                    # Extract model data
                    model_pred_return_pct = float(model_pred.get("predicted_return_pct", 0))
                    model_pred_return = model_pred_return_pct / 100.0
                    model_pred_price = float(model_pred.get("predicted_price", current_price * (1.0 + model_pred_return)))
                    model_confidence_pct = float(model_pred.get("confidence", 0))
                    model_confidence = model_confidence_pct / 100.0 if model_confidence_pct > 1.0 else model_confidence_pct
                    model_r2 = float(model_pred.get("r2_score", 0))
                    
                    # Recalculate action from predicted_return using the same logic as crypto
                    # This ensures action matches the predicted return, not what was saved during training
                    action_threshold = horizon_config.directional_threshold or 0.01
                    if model_pred_return >= action_threshold:
                        model_action = "LONG"
                    elif model_pred_return <= -action_threshold:
                        model_action = "SHORT"
                    else:
                        model_action = "HOLD"
                    
                    # Calculate probabilities from model's own predicted return
                    # Each model should have its own probabilities based on its prediction
                    probabilities = self._return_to_probabilities(model_pred_return, action_threshold)
                    
                    individual_entry = {
                        "name": model_name,
                        "action": model_action,
                        "predicted_return": model_pred_return,
                        "predicted_price": model_pred_price,
                        "confidence": model_confidence,
                        "probabilities": probabilities,
                    }
                    individual_models.append(individual_entry)
                    
                    # Track main models (random_forest, lightgbm, xgboost, stacked_blend)
                    if any(x in model_name.lower() for x in ["random_forest", "lightgbm", "xgboost", "stacked_blend"]):
                        main_models[model_name] = {
                            "predicted_return": model_pred_return,
                            "action": model_action,
                            "confidence": model_confidence,
                            "probabilities": probabilities,
                        }
                        # Calculate weight from R² score
                        weight = max(0.05, float(model_r2) + 0.5 * float(model_confidence - 0.5))
                        main_model_predictions.append({
                            "name": model_name,
                            "predicted_return": model_pred_return,
                            "predicted_price": model_pred_price,
                            "weight": weight,
                        })
                
                # Load DQN if available
                dqn_path = Path("models/dqn") / f"{asset_type}_{symbol}_{timeframe}.json"
                dqn_data = None
                dqn_action = None
                if dqn_path.exists():
                    try:
                        with open(dqn_path, "r", encoding="utf-8") as f:
                            dqn_data = json.load(f)
                            dqn_metrics = dqn_data.get("metrics", {})
                            dqn_action_str = dqn_metrics.get("action", "hold").lower()
                            dqn_action = self._canonical_action(dqn_action_str)
                    except Exception as exc:
                        print(f"[MCP_ADAPTER] Failed to load DQN from {dqn_path}: {exc}")
                
                # Compute meta probabilities from action_scores
                action_scores = consensus_data.get("action_scores", {})
                if action_scores:
                    meta_probs = {
                        "LONG": float(action_scores.get("long", 0)),
                        "HOLD": float(action_scores.get("hold", 0)),
                        "SHORT": float(action_scores.get("short", 0))
                    }
                    meta_action = max(meta_probs, key=meta_probs.get)
                    meta_confidence = float(meta_probs[meta_action])
                else:
                    # Fallback: compute from individual models
                    meta_probs, meta_confidence, meta_action = self._compute_meta_probabilities(individual_models, consensus_data)
                
                # Compute unified action
                unified_action, disagreement_explanation = self._compute_unified_action(main_models, dqn_action, consensus_data, meta_action, meta_confidence)
                
                # Build price calculation explanation using per-model predictions
                price_explanation = self._build_price_explanation(
                    main_model_predictions,
                    predicted_return,
                    current_price,
                    dqn_data,
                    dqn_action
                )
                
                # If we have main model predictions, make the top-level predicted_return
                # follow the weighted-average consensus from those models. This fixes
                # cases where summary.json had a neutral 0.0 return but all models
                # clearly point LONG/SHORT (e.g., commodities like silver/oil).
                if price_explanation.get("models_used", 0) > 0:
                    weighted_info = price_explanation.get("weighted_average")
                    if isinstance(weighted_info, dict):
                        wa_consensus = weighted_info.get("consensus_return")
                        fallback_return = None
                        # Fallback: if consensus_return is missing or numerically ~0
                        # but individual models are directional, use stacked_blend
                        # (or simple average) so predicted_price is not identical
                        # to current_price when models clearly have a view.
                        if not isinstance(wa_consensus, (int, float)) or abs(wa_consensus) < 1e-6:
                            # Prefer stacked_blend if available
                            blend = next((m for m in main_model_predictions if m.get("name") == "stacked_blend"), None)
                            if blend:
                                fallback_return = float(blend.get("predicted_return", 0.0))
                            else:
                                # Otherwise simple average of model returns
                                if main_model_predictions:
                                    fallback_return = float(
                                        sum(m.get("predicted_return", 0.0) for m in main_model_predictions)
                                        / len(main_model_predictions)
                                    )
                            wa_consensus = fallback_return if fallback_return is not None else 0.0
                        if isinstance(wa_consensus, (int, float)):
                            predicted_return = float(wa_consensus)
                            predicted_price = float(current_price * (1.0 + predicted_return))
                            # Recompute consensus_action from model-based consensus
                            action_threshold = horizon_config.directional_threshold or 0.01
                            if predicted_return >= action_threshold:
                                consensus_action = "LONG"
                            elif predicted_return <= -action_threshold:
                                consensus_action = "SHORT"
                            else:
                                consensus_action = "HOLD"
                
                # Format prediction response
                prediction = {
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "timeframe": timeframe,
                    "current_price": current_price,
                    "predicted_return": predicted_return,
                    "predicted_price": predicted_price,
                    "action": unified_action,
                    "consensus_action": consensus_action,
                    "confidence": float(confidence),
                    "horizon": horizon_profile,
                    "horizon_bars": int(consensus_data.get("target_horizon_bars", horizon_bars)),
                    "risk_profile": risk_profile or "conservative",
                    "unified_action": unified_action,
                    "disagreement_explanation": disagreement_explanation,
                    "model_actions": self._build_model_actions(individual_models),
                    "dqn_recommendation": self._build_dqn_recommendation(dqn_action, dqn_data, current_price, individual_models),
                    "individual_models": individual_models,
                    "price_calculation": price_explanation,
                    "meta_learner": {
                        "name": "weighted_vote",
                        "version": "meta_v1",
                        "probabilities": meta_probs,
                    },
                }
                prediction = self._validate_prediction(prediction, current_price)
                predictions.append(prediction)
            
            except Exception as exc:
                import traceback
                errors.append(f"Error predicting {symbol}: {str(exc)}\n{traceback.format_exc()}")
        
        response_data = {
            "predictions": predictions,
            "errors": errors,
            "count": len(predictions)
        }
        
        self._log_response("predict", request_data, response_data, success=len(errors) == 0)
        
        return response_data
    
    def scan_all(
        self,
        asset_type: str = "crypto",
        timeframe: str = "1d",
        min_confidence: float = 0.5,
        limit: int = 50,
        horizon: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Scan all available symbols and return shortlist with scores.
        
        Args:
            asset_type: Asset type to scan
            timeframe: Timeframe to use
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results
        
        Returns:
            Dictionary with shortlist and scores
        """
        normalized_horizon = normalize_profile(horizon or DEFAULT_HORIZON_PROFILE)
        request_data = {
            "asset_type": asset_type,
            "timeframe": timeframe,
            "min_confidence": min_confidence,
            "limit": limit,
            "horizon": normalized_horizon,
        }
        
        self._log_request("scan_all", request_data)
        
        shortlist = []
        
        # Find all symbols with trained models
        models_dir = Path("models") / asset_type
        if not models_dir.exists():
            response_data = {
                "shortlist": [],
                "count": 0,
                "error": f"No models found for {asset_type}"
            }
            self._log_response("scan_all", request_data, response_data, success=False)
            return response_data
        
        for symbol_dir in models_dir.iterdir():
            if not symbol_dir.is_dir():
                continue
            
            symbol = symbol_dir.name
            
            # Check for trained model
            try:
                # Get prediction
                pred_result = self.predict(
                    symbols=[symbol],
                    asset_type=asset_type,
                    timeframe=timeframe,
                    horizon=normalized_horizon,
                )
                
                if pred_result["predictions"]:
                    pred = pred_result["predictions"][0]
                    confidence = pred.get("confidence", 0)
                    
                    if confidence >= min_confidence:
                        shortlist.append({
                            "symbol": symbol,
                            "score": confidence,
                            "action": pred.get("action", "hold"),
                            "predicted_return": pred.get("predicted_return", 0),
                            "current_price": pred.get("current_price", 0)
                        })
            except Exception:
                continue
        
        # Sort by score (confidence) descending
        shortlist.sort(key=lambda x: x["score"], reverse=True)
        shortlist = shortlist[:limit]
        
        response_data = {
            "shortlist": shortlist,
            "count": len(shortlist),
            "asset_type": asset_type,
            "timeframe": timeframe
        }
        
        self._log_response("scan_all", request_data, response_data, success=True)
        
        return response_data
    
    def analyze(
        self,
        tickers: List[str],
        asset_type: str = "crypto",
        timeframe: str = "1d"
    ) -> Dict[str, Any]:
        """
        Analyze custom tickers.
        
        Args:
            tickers: List of ticker symbols to analyze
            asset_type: Asset type
            timeframe: Timeframe
        
        Returns:
            Dictionary with analysis results
        """
        request_data = {
            "tickers": tickers,
            "asset_type": asset_type,
            "timeframe": timeframe
        }
        
        self._log_request("analyze", request_data)
        
        results = []
        
        for ticker in tickers:
            try:
                # Try to get prediction
                pred_result = self.predict(
                    symbols=[ticker],
                    asset_type=asset_type,
                    timeframe=timeframe
                )
                
                if pred_result["predictions"]:
                    pred = pred_result["predictions"][0]
                    results.append({
                        "ticker": ticker,
                        "status": "success",
                        "prediction": pred
                    })
                else:
                    results.append({
                        "ticker": ticker,
                        "status": "no_model",
                        "error": pred_result.get("errors", ["No trained model"])[0] if pred_result.get("errors") else "No trained model"
                    })
            except Exception as exc:
                results.append({
                    "ticker": ticker,
                    "status": "error",
                    "error": str(exc)
                })
        
        response_data = {
            "results": results,
            "count": len(results),
            "asset_type": asset_type,
            "timeframe": timeframe
        }
        
        self._log_response("analyze", request_data, response_data, success=True)
        
        return response_data
    
    def _compute_unified_action(
        self,
        main_models: Dict[str, Dict[str, Any]],
        dqn_action: Optional[str],
        consensus: Dict[str, Any],
        meta_action: str,
        meta_confidence: float,
    ) -> Tuple[str, str]:
        """
        Compute unified action from 3 main models (Random Forest, LightGBM, XGBoost).
        If they disagree, explain why and recommend DQN.
        
        Returns:
            Tuple of (unified_action, disagreement_explanation)
        """
        canonical_consensus = self._canonical_action(consensus.get("consensus_action", "hold"))
        
        action_counts = {"LONG": 0, "SHORT": 0, "HOLD": 0}
        model_actions = {}
        
        for model_name, model_data in main_models.items():
            action = self._canonical_action(model_data.get("action", "hold"))
            action_counts[action] += 1
            model_actions[model_name] = action
        
        max_action = max(action_counts, key=action_counts.get)
        max_count = action_counts[max_action]
        total_models = len(main_models)
        meta_action = self._canonical_action(meta_action)
        
        # Always format explanation in the same way as crypto (matching format)
        explanation = (
            f"Meta action: {meta_action} (confidence {meta_confidence:.2f}). "
            f"Main models -> LONG:{action_counts['LONG']}, HOLD:{action_counts['HOLD']}, SHORT:{action_counts['SHORT']}"
        )
        
        if len(main_models) < 2:
            # Still return the formatted explanation even with fewer models
            return meta_action if meta_confidence >= self._meta_min_confidence else canonical_consensus, explanation
        
        if meta_confidence >= self._meta_min_confidence:
            return meta_action, explanation
        
        if max_count >= 2 and max_action != "HOLD":
            return max_action, f"Meta low confidence; using majority vote: {max_action} ({max_count}/{total_models})"
        
        if max_count == total_models and max_action == "HOLD":
            return "HOLD", f"Meta low confidence; all {total_models} main models recommend HOLD"
        
        if dqn_action:
            canon_dqn = self._canonical_action(dqn_action)
            details = ", ".join([f"{name}={action}" for name, action in model_actions.items()])
            return canon_dqn, (
                f"Meta confidence {meta_confidence:.2f} below threshold; models disagree ({details}). "
                f"Using DQN recommendation: {canon_dqn}."
            )
        
        if max_count > 0:
            details = ", ".join([f"{name}={action}" for name, action in model_actions.items()])
            return max_action, (
                f"Meta confidence {meta_confidence:.2f} below threshold; models disagree ({details}). "
                f"Falling back to majority vote: {max_action} ({max_count}/{total_models})."
            )
        
        return canonical_consensus, f"Meta confidence low and no majority. Using consensus: {canonical_consensus}"
    
    def add_feedback(
        self,
        symbol: str,
        asset_type: str,
        timeframe: str,
        prediction_timestamp: str,
        action_taken: str,
        predicted_return: float,
        predicted_price: float,
        actual_price: Optional[float] = None,
        actual_return: Optional[float] = None,
        features: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Add feedback for RL learning.
        
        Args:
            symbol: Trading symbol
            asset_type: Asset type
            timeframe: Timeframe
            prediction_timestamp: When prediction was made
            action_taken: Action taken (long/short/hold)
            predicted_return: Predicted return
            predicted_price: Predicted price
            actual_price: Actual price (optional)
            actual_return: Actual return (optional)
            features: Feature vector (optional)
        
        Returns:
            Success status
        """
        request_data = {
            "symbol": symbol,
            "asset_type": asset_type,
            "timeframe": timeframe,
            "prediction_timestamp": prediction_timestamp,
            "action_taken": action_taken
        }
        
        self._log_request("add_feedback", request_data)
        
        try:
            self.feedback_learner.add_feedback(
                symbol=symbol,
                asset_type=asset_type,
                timeframe=timeframe,
                prediction_timestamp=prediction_timestamp,
                action_taken=action_taken,
                predicted_return=predicted_return,
                predicted_price=predicted_price,
                actual_price=actual_price,
                actual_return=actual_return,
                features=features
            )
            
            # Check feedback count and retraining status
            feedback_entries, has_enough = self.feedback_learner.get_feedback_data(
                symbol=symbol,
                asset_type=asset_type,
                min_feedback_count=20
            )
            feedback_count = len(feedback_entries)
            
            response_data = {
                "success": True,
                "message": "Feedback added successfully",
                "feedback_count": feedback_count,
                "retrain_suggestion": has_enough,
                "has_enough_for_retrain": has_enough,
                "feedback_entries_for_symbol": feedback_count
            }
            
            if has_enough:
                response_data["message"] = f"Feedback added. {feedback_count} entries collected for {symbol}. Ready for DQN retraining."
            
            self._log_response("add_feedback", request_data, response_data, success=True)
            return response_data
        
        except Exception as exc:
            response_data = {"success": False, "error": str(exc)}
            self._log_response("add_feedback", request_data, response_data, success=False)
            return response_data


# Global adapter instance
_adapter: Optional[MCPAdapter] = None


def get_mcp_adapter() -> MCPAdapter:
    """Get or create global MCP adapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = MCPAdapter()
    return _adapter

