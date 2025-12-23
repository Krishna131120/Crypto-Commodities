"""
Live Trading Readiness Validation Module

This module provides comprehensive validation checks to ensure models are ready
for live trading. It does NOT modify model training or prediction logic - it only
validates existing models and configurations.

All validation functions are non-invasive and can be run independently without
affecting the core model pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class ValidationResult:
    """Container for validation check results."""
    
    def __init__(self, check_name: str, passed: bool, message: str, details: Optional[Dict[str, Any]] = None):
        self.check_name = check_name
        self.passed = passed
        self.message = message
        self.details = details or {}
    
    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"{status}: {self.check_name} - {self.message}"


def validate_model_metrics(
    summary_path: Path,
    min_r2: float = 0.3,
    min_directional_accuracy: float = 0.6,
    max_overfitting_gap: float = 0.2,
) -> List[ValidationResult]:
    """
    Validate model metrics from training summary.
    
    Args:
        summary_path: Path to summary.json
        min_r2: Minimum acceptable R² score
        min_directional_accuracy: Minimum acceptable directional accuracy
        max_overfitting_gap: Maximum acceptable gap between train and test R²
        
    Returns:
        List of ValidationResult objects
    """
    results = []
    
    if not summary_path.exists():
        results.append(ValidationResult(
            "model_metrics_file_exists",
            False,
            f"Summary file not found: {summary_path}",
        ))
        return results
    
    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
    except Exception as e:
        results.append(ValidationResult(
            "model_metrics_parse",
            False,
            f"Failed to parse summary.json: {e}",
        ))
        return results
    
    # Check model predictions
    model_predictions = summary.get("model_predictions", {})
    models = summary.get("models", {})
    
    successful_models = []
    for name, data in models.items():
        if isinstance(data, dict) and data.get("status") != "failed" and name != "dqn":
            successful_models.append(name)
    
    # Check minimum models requirement
    if len(successful_models) < 2:
        results.append(ValidationResult(
            "minimum_models",
            False,
            f"Insufficient models: {len(successful_models)} successful (need at least 2)",
            {"successful_models": successful_models},
        ))
    else:
        results.append(ValidationResult(
            "minimum_models",
            True,
            f"Sufficient models: {len(successful_models)} successful",
            {"successful_models": successful_models},
        ))
    
    # Check each model's metrics
    for model_name in successful_models:
        model_data = model_predictions.get(model_name, {})
        if not model_data:
            model_data = models.get(model_name, {})
        
        r2 = model_data.get("r2_score") or model_data.get("r2", 0.0)
        directional = model_data.get("directional_accuracy", 0.0)
        
        # R² check
        if r2 < min_r2:
            results.append(ValidationResult(
                f"{model_name}_r2",
                False,
                f"{model_name} R² ({r2:.3f}) below minimum ({min_r2:.2f})",
                {"r2": r2, "min_r2": min_r2},
            ))
        else:
            results.append(ValidationResult(
                f"{model_name}_r2",
                True,
                f"{model_name} R² ({r2:.3f}) meets minimum ({min_r2:.2f})",
                {"r2": r2, "min_r2": min_r2},
            ))
        
        # Directional accuracy check
        if directional < min_directional_accuracy:
            results.append(ValidationResult(
                f"{model_name}_directional",
                False,
                f"{model_name} directional accuracy ({directional:.3f}) below minimum ({min_directional_accuracy:.2f})",
                {"directional": directional, "min_directional": min_directional_accuracy},
            ))
        else:
            results.append(ValidationResult(
                f"{model_name}_directional",
                True,
                f"{model_name} directional accuracy ({directional:.3f}) meets minimum ({min_directional_accuracy:.2f})",
                {"directional": directional, "min_directional": min_directional_accuracy},
            ))
    
    # Check overfitting
    overfitting_warnings = summary.get("analysis", {}).get("overfitting_warnings", [])
    if overfitting_warnings:
        severe_warnings = [w for w in overfitting_warnings if "severe" in w.lower() or "generalization" in w.lower()]
        if severe_warnings:
            results.append(ValidationResult(
                "overfitting_check",
                False,
                f"Severe overfitting detected: {len(severe_warnings)} warning(s)",
                {"warnings": severe_warnings},
            ))
        else:
            results.append(ValidationResult(
                "overfitting_check",
                True,
                f"Overfitting warnings present but not severe ({len(overfitting_warnings)} warning(s))",
                {"warnings": overfitting_warnings},
            ))
    else:
        results.append(ValidationResult(
            "overfitting_check",
            True,
            "No overfitting warnings",
        ))
    
    # Check tradability flag
    tradable = summary.get("tradable", False)
    if not tradable:
        reasons = summary.get("tradability_reasons", [])
        results.append(ValidationResult(
            "tradability_flag",
            False,
            f"Model marked as not tradable: {', '.join(reasons)}",
            {"reasons": reasons},
        ))
    else:
        results.append(ValidationResult(
            "tradability_flag",
            True,
            "Model marked as tradable",
        ))
    
    return results


def validate_risk_config(
    risk_config_path: Optional[Path] = None,
    default_stop_loss_pct: float = 0.035,
    max_position_pct: float = 0.10,
) -> List[ValidationResult]:
    """
    Validate risk management configuration.
    
    Args:
        risk_config_path: Optional path to risk config file
        default_stop_loss_pct: Expected default stop-loss percentage
        max_position_pct: Expected maximum position size percentage
        
    Returns:
        List of ValidationResult objects
    """
    results = []
    
    # Check if risk config file exists (if provided)
    if risk_config_path and risk_config_path.exists():
        try:
            with open(risk_config_path, "r") as f:
                config = json.load(f)
            
            stop_loss = config.get("stop_loss_pct", default_stop_loss_pct)
            max_pos = config.get("max_position_pct", max_position_pct)
            
            if stop_loss > 0.10:  # More than 10% stop-loss is risky
                results.append(ValidationResult(
                    "stop_loss_size",
                    False,
                    f"Stop-loss ({stop_loss*100:.1f}%) is too large (>10%)",
                    {"stop_loss_pct": stop_loss},
                ))
            elif stop_loss < 0.01:  # Less than 1% stop-loss is too tight
                results.append(ValidationResult(
                    "stop_loss_size",
                    False,
                    f"Stop-loss ({stop_loss*100:.1f}%) is too tight (<1%)",
                    {"stop_loss_pct": stop_loss},
                ))
            else:
                results.append(ValidationResult(
                    "stop_loss_size",
                    True,
                    f"Stop-loss ({stop_loss*100:.1f}%) is reasonable",
                    {"stop_loss_pct": stop_loss},
                ))
            
            if max_pos > 0.20:  # More than 20% per symbol is risky
                results.append(ValidationResult(
                    "max_position_size",
                    False,
                    f"Max position size ({max_pos*100:.1f}%) is too large (>20%)",
                    {"max_position_pct": max_pos},
                ))
            else:
                results.append(ValidationResult(
                    "max_position_size",
                    True,
                    f"Max position size ({max_pos*100:.1f}%) is reasonable",
                    {"max_position_pct": max_pos},
                ))
        except Exception as e:
            results.append(ValidationResult(
                "risk_config_parse",
                False,
                f"Failed to parse risk config: {e}",
            ))
    else:
        # Use defaults
        results.append(ValidationResult(
            "risk_config_exists",
            True,
            "Using default risk configuration",
            {"default_stop_loss_pct": default_stop_loss_pct, "default_max_position_pct": max_position_pct},
        ))
    
    return results


def validate_data_quality(
    data_path: Path,
    min_candles: int = 500,
    max_age_days: int = 1,
) -> List[ValidationResult]:
    """
    Validate data quality for live trading.
    
    Args:
        data_path: Path to data.json
        min_candles: Minimum number of candles required
        max_age_days: Maximum age of latest candle in days
        
    Returns:
        List of ValidationResult objects
    """
    results = []
    
    if not data_path.exists():
        results.append(ValidationResult(
            "data_file_exists",
            False,
            f"Data file not found: {data_path}",
        ))
        return results
    
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0:
            results.append(ValidationResult(
                "data_format",
                False,
                "Data file is empty or invalid format",
            ))
            return results
        
        # Check data quantity
        candle_count = len(data)
        if candle_count < min_candles:
            results.append(ValidationResult(
                "data_quantity",
                False,
                f"Insufficient data: {candle_count} candles (need at least {min_candles})",
                {"candle_count": candle_count, "min_candles": min_candles},
            ))
        else:
            results.append(ValidationResult(
                "data_quantity",
                True,
                f"Sufficient data: {candle_count} candles",
                {"candle_count": candle_count},
            ))
        
        # Check data freshness
        latest_candle = data[-1]
        timestamp = latest_candle.get("timestamp") or latest_candle.get("time")
        if timestamp:
            from datetime import datetime, timezone
            try:
                if isinstance(timestamp, str):
                    # Parse timestamp
                    if "T" in timestamp:
                        latest_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    else:
                        latest_time = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
                else:
                    latest_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                
                now = datetime.now(timezone.utc)
                age_days = (now - latest_time).total_seconds() / 86400
                
                if age_days > max_age_days:
                    results.append(ValidationResult(
                        "data_freshness",
                        False,
                        f"Data is stale: {age_days:.1f} days old (max: {max_age_days} days)",
                        {"age_days": age_days, "max_age_days": max_age_days},
                    ))
                else:
                    results.append(ValidationResult(
                        "data_freshness",
                        True,
                        f"Data is fresh: {age_days:.1f} days old",
                        {"age_days": age_days},
                    ))
            except Exception as e:
                results.append(ValidationResult(
                    "data_timestamp_parse",
                    False,
                    f"Failed to parse timestamp: {e}",
                ))
        
    except Exception as e:
        results.append(ValidationResult(
            "data_parse",
            False,
            f"Failed to parse data file: {e}",
        ))
    
    return results


def validate_features(
    features_path: Path,
) -> List[ValidationResult]:
    """
    Validate feature file exists and is valid.
    
    Args:
        features_path: Path to features.json
        
    Returns:
        List of ValidationResult objects
    """
    results = []
    
    if not features_path.exists():
        results.append(ValidationResult(
            "features_file_exists",
            False,
            f"Features file not found: {features_path}",
        ))
        return results
    
    try:
        with open(features_path, "r") as f:
            features = json.load(f)
        
        if isinstance(features, dict) and "features" in features:
            feature_count = len(features["features"])
            if feature_count < 10:
                results.append(ValidationResult(
                    "features_quantity",
                    False,
                    f"Insufficient features: {feature_count} (need at least 10)",
                    {"feature_count": feature_count},
                ))
            else:
                results.append(ValidationResult(
                    "features_quantity",
                    True,
                    f"Sufficient features: {feature_count}",
                    {"feature_count": feature_count},
                ))
        else:
            results.append(ValidationResult(
                "features_format",
                False,
                "Invalid features format",
            ))
    except Exception as e:
        results.append(ValidationResult(
            "features_parse",
            False,
            f"Failed to parse features file: {e}",
        ))
    
    return results


def validate_broker_credentials(
    broker: str = "alpaca",
) -> List[ValidationResult]:
    """
    Validate broker API credentials are configured.
    
    Args:
        broker: Broker name ("alpaca" or "dhan")
        
    Returns:
        List of ValidationResult objects
    """
    results = []
    import os
    
    if broker.lower() == "alpaca":
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key:
            results.append(ValidationResult(
                "alpaca_api_key",
                False,
                "ALPACA_API_KEY environment variable not set",
            ))
        else:
            results.append(ValidationResult(
                "alpaca_api_key",
                True,
                "ALPACA_API_KEY is set",
            ))
        
        if not secret_key:
            results.append(ValidationResult(
                "alpaca_secret_key",
                False,
                "ALPACA_SECRET_KEY environment variable not set",
            ))
        else:
            results.append(ValidationResult(
                "alpaca_secret_key",
                True,
                "ALPACA_SECRET_KEY is set",
            ))
    
    elif broker.lower() == "angelone":
        api_key = os.getenv("ANGEL_ONE_API_KEY")
        password = os.getenv("ANGEL_ONE_PASSWORD")
        client_id = os.getenv("ANGEL_ONE_CLIENT_ID")
        totp_secret = os.getenv("ANGEL_ONE_TOTP_SECRET")
        
        if not api_key:
            results.append(ValidationResult(
                "angelone_api_key",
                False,
                "ANGEL_ONE_API_KEY environment variable not set",
            ))
        else:
            results.append(ValidationResult(
                "angelone_api_key",
                True,
                "ANGEL_ONE_API_KEY is set",
            ))
        
        if not password:
            results.append(ValidationResult(
                "angelone_password",
                False,
                "ANGEL_ONE_PASSWORD environment variable not set",
            ))
        else:
            results.append(ValidationResult(
                "angelone_password",
                True,
                "ANGEL_ONE_PASSWORD is set",
            ))
        
        if not client_id:
            results.append(ValidationResult(
                "angelone_client_id",
                False,
                "ANGEL_ONE_CLIENT_ID environment variable not set",
            ))
        else:
            results.append(ValidationResult(
                "angelone_client_id",
                True,
                "ANGEL_ONE_CLIENT_ID is set",
            ))
        
        if not totp_secret:
            results.append(ValidationResult(
                "angelone_totp_secret",
                False,
                "ANGEL_ONE_TOTP_SECRET environment variable not set (optional but recommended)",
            ))
        else:
            results.append(ValidationResult(
                "angelone_totp_secret",
                True,
                "ANGEL_ONE_TOTP_SECRET is set",
            ))
    else:
        results.append(ValidationResult(
            "broker_support",
            False,
            f"Unsupported broker: {broker}",
        ))
    
    return results


def validate_model_for_live_trading(
    model_dir: Path,
    asset_type: str,
    symbol: str,
    timeframe: str,
    broker: Optional[str] = None,
) -> Tuple[bool, List[ValidationResult]]:
    """
    Comprehensive validation for a single model's readiness for live trading.
    
    Args:
        model_dir: Path to model directory (e.g., models/crypto/BTC-USDT/1d/short)
        asset_type: Asset type ("crypto" or "commodities")
        symbol: Symbol (e.g., "BTC-USDT")
        timeframe: Timeframe (e.g., "1d")
        broker: Broker name ("alpaca" or "dhan"). If None, auto-selects:
                - "dhan" for commodities (supports futures)
                - "alpaca" for crypto
        
    Returns:
        Tuple of (is_ready, list_of_validation_results)
    """
    # Auto-determine broker if not specified
    if broker is None:
        if asset_type == "commodities":
            broker = "angelone"  # Angel One supports commodities futures
        else:
            broker = "alpaca"  # Alpaca supports crypto
    
    all_results = []
    
    # Validate summary.json
    summary_path = model_dir / "summary.json"
    all_results.extend(validate_model_metrics(summary_path))
    
    # Validate data
    data_path = Path("data/json/raw") / asset_type / "yahoo_chart" / symbol / timeframe / "data.json"
    if not data_path.exists():
        # Try alternative path for crypto
        data_path = Path("data/json/raw") / asset_type / symbol / timeframe / "data.json"
    all_results.extend(validate_data_quality(data_path))
    
    # Validate features
    features_path = Path("data/features") / asset_type / symbol / timeframe / "features.json"
    all_results.extend(validate_features(features_path))
    
    # Validate broker credentials
    all_results.extend(validate_broker_credentials(broker))
    
    # Determine overall readiness
    failed_checks = [r for r in all_results if not r.passed]
    is_ready = len(failed_checks) == 0
    
    return is_ready, all_results
