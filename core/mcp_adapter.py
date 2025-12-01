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
        
        if len(main_model_predictions) >= 3:
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
            
            # Check for disagreement
            prices = [p["predicted_price"] for p in main_model_predictions]
            price_range = max(prices) - min(prices)
            price_variance = sum((p - sum(prices) / len(prices)) ** 2 for p in prices) / len(prices)
            price_std = (price_variance ** 0.5) / current_price if current_price > 0 else 0.0  # Relative std
            
            explanation["model_agreement"] = {
                "price_range": price_range,
                "price_range_pct": (price_range / current_price) * 100 if current_price > 0 else 0.0,
                "price_std_pct": price_std * 100,
                "models_agree": price_std < 0.02,  # Less than 2% std deviation
            }
            
            if price_std >= 0.02:  # Models disagree significantly
                explanation["disagreement_warning"] = True
                explanation["recommendation"] = (
                    "The three price models show significant disagreement (std deviation: {:.2f}%). "
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

                # Load features
                feature_path = Path("data/features") / asset_type / symbol / timeframe / "features.json"
                if not feature_path.exists():
                    errors.append(f"No features found for {symbol}")
                    continue
                
                with open(feature_path, "r", encoding="utf-8") as f:
                    feature_data = json.load(f)
                
                feature_series = self._build_feature_series(asset_type, symbol, timeframe, horizon_profile, feature_data)
                if feature_series.empty:
                    errors.append(f"No valid features for {symbol}")
                    continue
                
                # Get pipeline
                pipeline = self._get_pipeline(asset_type, symbol, timeframe, horizon_profile)
                if not pipeline:
                    errors.append(f"No trained model for {symbol}")
                    continue
                
                # Get current price from latest candle
                # Try latest.json first, then data.json (get last entry)
                data_path = Path("data/json/raw") / asset_type / "binance" / symbol / timeframe / "latest.json"
                if not data_path.exists():
                    data_path = Path("data/json/raw") / asset_type / "yahoo_chart" / symbol / timeframe / "latest.json"
                
                current_price = 0.0
                if data_path.exists():
                    with open(data_path, "r", encoding="utf-8") as f:
                        latest_data = json.load(f)
                        if isinstance(latest_data, list) and len(latest_data) > 0:
                            current_price = float(latest_data[0].get("close", 0))
                        elif isinstance(latest_data, dict):
                            current_price = float(latest_data.get("close", 0))
                
                # Fallback to data.json (get last entry)
                if current_price == 0:
                    data_file = Path("data/json/raw") / asset_type / "binance" / symbol / timeframe / "data.json"
                    if not data_file.exists():
                        data_file = Path("data/json/raw") / asset_type / "yahoo_chart" / symbol / timeframe / "data.json"
                    
                    if data_file.exists():
                        with open(data_file, "r", encoding="utf-8") as f:
                            all_data = json.load(f)
                            if isinstance(all_data, list) and len(all_data) > 0:
                                # Get last entry (most recent)
                                latest_entry = all_data[-1]
                                current_price = float(latest_entry.get("close", 0))
                
                if current_price == 0:
                    errors.append(f"Could not get current price for {symbol}")
                    continue
                
                # Calculate volatility
                volatility = abs(feature_data.get("features", {}).get("ATR_14", {}).get("value", 0) / current_price) if current_price > 0 else 0.01
                if volatility == 0:
                    volatility = 0.01
                
                # Run inference
                inference_result = pipeline.predict(feature_series, current_price=current_price, volatility=volatility)
                
                # Get individual model predictions for unified action logic
                model_outputs = inference_result.get("models", {})
                consensus = inference_result.get("consensus", {})
                # Use horizon-specific directional_threshold instead of generic dynamic_threshold
                action_threshold = horizon_config.directional_threshold
                # Fallback to pipeline's dynamic_threshold if horizon config not available
                if action_threshold is None or action_threshold == 0:
                    action_threshold = getattr(pipeline, "dynamic_threshold", 0.01) or 0.01

                # Load DQN prediction from JSON file if available
                dqn_path = Path("models/dqn") / f"{asset_type}_{symbol}_{timeframe}.json"
                dqn_data = None
                dqn_calculated_return = None
                dqn_calculated_price = None
                if dqn_path.exists():
                    try:
                        with open(dqn_path, "r", encoding="utf-8") as f:
                            dqn_data = json.load(f)
                            # Extract DQN metrics
                            dqn_metrics = dqn_data.get("metrics", {})
                            dqn_action_str = dqn_metrics.get("action", "hold").lower()
                            
                            # Calculate predicted_return based on DQN action and policy metrics
                            # Use test policy avg_return as baseline, adjusted by action direction
                            test_metrics = dqn_metrics.get("test_policy_metrics", {})
                            if test_metrics and dqn_action_str != "hold":
                                base_return = float(test_metrics.get("avg_return", 0.0))
                                # Adjust sign based on action (long = positive, short = negative)
                                if dqn_action_str == "long":
                                    dqn_predicted_return = abs(base_return)  # Always positive for long
                                else:  # short
                                    dqn_predicted_return = -abs(base_return)  # Always negative for short
                                # Clamp to reasonable bounds (horizon-aware)
                                if horizon_bars == 1:  # Intraday
                                    dqn_predicted_return = max(-0.03, min(0.03, dqn_predicted_return))
                                elif horizon_bars <= 4:  # Short-term
                                    dqn_predicted_return = max(-0.08, min(0.08, dqn_predicted_return))
                                else:  # Long-term
                                    dqn_predicted_return = max(-0.20, min(0.20, dqn_predicted_return))
                            elif dqn_action_str == "hold":
                                dqn_predicted_return = 0.0
                            else:
                                # Fallback: use dynamic threshold with action direction
                                if dqn_action_str == "long":
                                    dqn_predicted_return = action_threshold
                                elif dqn_action_str == "short":
                                    dqn_predicted_return = -action_threshold
                                else:
                                    dqn_predicted_return = 0.0
                            
                            # Calculate predicted_price from current_price and predicted_return
                            dqn_predicted_price = float(current_price * (1.0 + dqn_predicted_return))
                            
                            # Store calculated DQN return for later use in recommendation
                            # We'll pass this to _build_dqn_recommendation
                            dqn_calculated_return = dqn_predicted_return
                            dqn_calculated_price = dqn_predicted_price
                            
                            # Add DQN to model_outputs so it gets processed like other models (for consensus)
                            model_outputs["dqn"] = {
                                "predicted_return": dqn_predicted_return,
                                "predicted_price": dqn_predicted_price,
                                "action": dqn_action_str,
                            }
                    except Exception as exc:
                        print(f"[MCP_ADAPTER] Failed to load DQN from {dqn_path}: {exc}")
                        dqn_calculated_return = None
                        dqn_calculated_price = None
                else:
                    dqn_calculated_return = None
                    dqn_calculated_price = None

                # Extract predictions from 3 main models (Random Forest, LightGBM, XGBoost)
                main_models = {}
                dqn_action = None
                individual_models = []
                
                # Track main model predictions for price calculation explanation
                main_model_predictions = []
                
                for model_name, model_data in model_outputs.items():
                    if "_quantile" in model_name:
                        continue
                    
                    # Skip DQN in individual_models - it will be shown separately in dqn_recommendation
                    if "dqn" in model_name.lower():
                        # Still extract dqn_action for unified action calculation
                        dqn_action_str = model_data.get("action", "hold").lower()
                        dqn_action = self._canonical_action(dqn_action_str)
                        continue
                    
                    pred_return = float(model_data.get("predicted_return", 0))
                    predicted_price = float(current_price * (1.0 + pred_return))
                    base_confidence = float(model_data.get("confidence") or model_data.get("probability") or 0.0)
                    
                    # Store main model predictions for price calculation explanation
                    if any(x in model_name.lower() for x in ["random_forest", "lightgbm", "xgboost"]):
                        # Get model metrics for weight calculation
                        model_metrics = pipeline.metrics.get(model_name, {})
                        r2 = model_metrics.get("r2", 0.0)
                        directional = model_metrics.get("directional_accuracy", 0.5)
                        weight = max(0.05, float(r2) + 0.5 * float(directional - 0.5))
                        main_model_predictions.append({
                            "name": model_name,
                            "predicted_return": pred_return,
                            "predicted_price": predicted_price,
                            "weight": weight,
                        })
                    
                    # For main models, use return-based probabilities
                    probabilities = self._return_to_probabilities(pred_return, action_threshold)
                    model_action = self._canonical_action(
                        "long"
                        if pred_return > action_threshold
                        else "short"
                        if pred_return < -action_threshold
                        else "hold"
                    )
                    # Use probability-based confidence with calibration
                    prob_confidence = max(probabilities.values())
                    # Combine with model confidence but apply dampening
                    base_confidence = max(base_confidence * self._confidence_dampening, prob_confidence * self._confidence_dampening)
                    # Cap confidence to prevent overconfidence
                    base_confidence = min(base_confidence, 0.95)
                    individual_entry = {
                        "name": model_name,
                        "action": model_action,
                        "predicted_return": pred_return,
                        "predicted_price": predicted_price,
                        "confidence": base_confidence,
                        "probabilities": probabilities,
                    }
                    if "probability" in model_data:
                        individual_entry["probability"] = float(model_data["probability"])
                    individual_models.append(individual_entry)
                    
                    # Main models: random_forest, lightgbm, xgboost
                    if any(x in model_name.lower() for x in ["random_forest", "lightgbm", "xgboost"]):
                        main_models[model_name] = {
                            "predicted_return": pred_return,
                            "action": model_action,
                            "confidence": base_confidence or 0.5,
                            "probabilities": probabilities,
                        }
                
                # Compute unified action from 3 main models
                meta_probs, meta_confidence, meta_action = self._compute_meta_probabilities(individual_models, consensus)
                unified_action, disagreement_explanation = self._compute_unified_action(main_models, dqn_action, consensus, meta_action, meta_confidence)
                
                # Final sanity check: cap extreme predictions based on horizon
                consensus_return = float(consensus.get("consensus_return", 0))
                # Get horizon_bars from pipeline or consensus metadata
                horizon_bars = consensus.get("target_horizon_bars") or getattr(pipeline, "target_horizon_bars", None) or horizon_config.horizon
                
                # Horizon-aware sanity limits (conservative)
                if horizon_bars == 1:  # Intraday
                    max_reasonable_return = 0.03  # 3% max for 1 day
                elif horizon_bars <= 4:  # Short-term
                    max_reasonable_return = 0.08  # 8% max for 4 days
                else:  # Long-term
                    max_reasonable_return = 0.20  # 20% max for 30 days
                
                # If prediction exceeds reasonable limit, scale it down
                if abs(consensus_return) > max_reasonable_return:
                    scale_factor = max_reasonable_return / abs(consensus_return)
                    consensus_return = consensus_return * scale_factor
                    # Also update individual models if they're too extreme
                    for model in individual_models:
                        if abs(model["predicted_return"]) > max_reasonable_return:
                            model["predicted_return"] = model["predicted_return"] * scale_factor
                            model["predicted_price"] = current_price * (1.0 + model["predicted_return"])
                    # Update main_model_predictions for explanation
                    for pred in main_model_predictions:
                        if abs(pred["predicted_return"]) > max_reasonable_return:
                            pred["predicted_return"] = pred["predicted_return"] * scale_factor
                            pred["predicted_price"] = current_price * (1.0 + pred["predicted_return"])
                
                # Calculate price prediction explanation
                price_explanation = self._build_price_explanation(
                    main_model_predictions,
                    consensus_return,
                    current_price,
                    dqn_data,
                    dqn_action
                )
                
                # Format prediction with unified action
                prediction = {
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "timeframe": timeframe,
                    "current_price": current_price,
                    "predicted_return": consensus_return,
                    "predicted_price": float(current_price * (1.0 + consensus_return)),
                    "action": unified_action,
                    "consensus_action": self._canonical_action(consensus.get("consensus_action", "hold")),
                    "confidence": float(meta_confidence),
                    "horizon": horizon_profile,  # Return as string (long/short/intraday)
                    "horizon_bars": horizon_bars,  # Also include number of bars
                    "risk_profile": risk_profile or "default",
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
                
                # Prepare feedback data for RL learning (store features for later feedback submission)
                try:
                    # Store prediction data that can be used for feedback when actual outcomes are known
                    # This will be submitted via /tools/feedback endpoint
                    pass  # Feedback is submitted separately via API
                except Exception:
                    pass
            
            except Exception as exc:
                errors.append(f"Error predicting {symbol}: {str(exc)}")
        
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
        if len(main_models) < 2:
            return canonical_consensus, f"Using consensus action ({canonical_consensus}) - not enough main models available"
        
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
        
        explanation = (
            f"Meta action: {meta_action} (confidence {meta_confidence:.2f}). "
            f"Main models -> LONG:{action_counts['LONG']}, HOLD:{action_counts['HOLD']}, SHORT:{action_counts['SHORT']}"
        )
        
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

