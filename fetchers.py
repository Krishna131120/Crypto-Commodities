"""
Complete data ingestion system - all fetchers and utilities merged into one file.
"""
import csv
import json
import math
import time
import threading
import requests
try:
    import websocket
except ImportError:
    websocket = None  # Optional - only needed for Binance WebSocket
try:
    import ccxt
except ImportError:
    ccxt = None  # Optional - only needed for some crypto exchanges
import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum
import logging

import config
from ml.horizons import (
    DEFAULT_HORIZON_PROFILE,
    available_profiles as available_horizon_profiles,
    describe_profile,
)
from train_models import train_symbols
from core.model_paths import (
    list_horizon_dirs,
    summary_path as build_summary_path,
    horizon_dir,
    timeframe_dir,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION: ANGEL ONE MCX API
# ============================================================================
# Set to True to use Angel One for commodities (requires static IP whitelisting)
# Set to False for paper trading with Yahoo Finance only
USE_ANGELONE_MCX = False  # 🔴 DISABLED for paper trading - change to True for real trading
# ============================================================================


# ============================================================================
# UTILITY FUNCTIONS - Normalization
# ============================================================================

def normalize_symbol(symbol: str, source: str) -> str:
    """Normalize symbol names across different sources."""
    symbol = symbol.upper().strip()
    
    if source == "binance":
        if "USDT" in symbol and len(symbol) > 4:
            base = symbol.replace("USDT", "")
            return f"{base}-USDT"
        return symbol
    elif source == "yahoo":
        return symbol
    elif source in ["coinbase", "kucoin", "okx"]:
        return symbol.replace("_", "-").replace("/", "-")
    
    return symbol


def normalize_timestamp(timestamp: Any, source: str) -> str:
    """Convert timestamp to ISO 8601 UTC format."""
    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except:
            try:
                ts = float(timestamp)
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except:
                pass
    
    if isinstance(timestamp, (int, float)):
        if timestamp > 1e10:  # Milliseconds
            timestamp = timestamp / 1000
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    raise ValueError(f"Unable to normalize timestamp: {timestamp}")


def create_canonical_candle(
    symbol: str,
    timeframe: str,
    timestamp: Any,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    source: str,
    fallback_active: bool = False,
    fallback_reason: Optional[str] = None
) -> Dict[str, Any]:
    """Create a canonical JSON candle structure."""
    normalized_symbol = normalize_symbol(symbol, source)
    normalized_timestamp = normalize_timestamp(timestamp, source)
    
    candle = {
        "schema_version": "ohlcv_v1",
        "symbol": normalized_symbol,
        "timeframe": timeframe,
        "timestamp": normalized_timestamp,
        "open": float(open_price),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": float(volume) if volume and volume > 0 else None,
        "source": source,
        "fallback_status": {
            "active": fallback_active,
            "reason": fallback_reason if fallback_reason else None
        }
    }
    
    if not volume or volume <= 0:
        candle["flag"] = "irregular"
    
    return candle


def parse_binance_candle(data: list) -> Dict[str, Any]:
    """Parse Binance candle format [openTime, open, high, low, close, volume, ...]."""
    return {
        "timestamp": int(data[0]),
        "open": float(data[1]),
        "high": float(data[2]),
        "low": float(data[3]),
        "close": float(data[4]),
        "volume": float(data[5])
    }


# ============================================================================
# UTILITY FUNCTIONS - Validation
# ============================================================================

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_candle(candle: Dict[str, Any], previous_timestamp: Optional[int] = None) -> Tuple[bool, Optional[str]]:
    """Validate a single candle according to rules."""
    required_fields = ["open", "high", "low", "close", "timestamp"]
    for field in required_fields:
        if field not in candle:
            return False, f"Missing required field: {field}"
    
    for field in ["open", "high", "low", "close"]:
        try:
            float(candle[field])
        except (ValueError, TypeError):
            return False, f"Field {field} must be numeric"
    
    if float(candle["high"]) < float(candle["low"]):
        return False, "High must be >= Low"
    
    if float(candle["high"]) < float(candle["open"]) or float(candle["high"]) < float(candle["close"]):
        return False, "High must be >= Open and Close"
    
    if float(candle["low"]) > float(candle["open"]) or float(candle["low"]) > float(candle["close"]):
        return False, "Low must be <= Open and Close"
    
    try:
        ts_str = candle["timestamp"]
        if isinstance(ts_str, str):
            dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts = int(dt.timestamp())
        else:
            ts = int(ts_str)
    except:
        return False, "Invalid timestamp format"
    
    if previous_timestamp is not None and ts <= previous_timestamp:
        return False, f"Out-of-order timestamp: {ts} <= {previous_timestamp}"
    
    if "volume" in candle and candle["volume"] is not None:
        try:
            vol = float(candle["volume"])
            if vol < 0:
                return False, "Volume cannot be negative"
        except (ValueError, TypeError):
            return False, "Volume must be numeric or null"
    
    return True, None


def validate_candle_list(candles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate a list of candles and return valid ones with errors."""
    valid_candles = []
    errors = []
    previous_ts = None
    
    for i, candle in enumerate(candles):
        is_valid, error = validate_candle(candle, previous_ts)
        
        if is_valid:
            valid_candles.append(candle)
            try:
                ts_str = candle["timestamp"]
                if isinstance(ts_str, str):
                    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    previous_ts = int(dt.timestamp())
                else:
                    previous_ts = int(ts_str)
            except:
                pass
        else:
            errors.append(f"Candle {i}: {error}")
    
    return valid_candles, errors


def check_duplicate_candles(candles: List[Dict[str, Any]], timeframe: str) -> List[Dict[str, Any]]:
    """Ensure one candle per interval, removing duplicates."""
    if not candles:
        return []
    
    interval_seconds = config.TIMEFRAMES.get(timeframe, 60)
    seen_intervals = {}
    unique_candles = []
    
    for candle in candles:
        try:
            ts_str = candle["timestamp"]
            if isinstance(ts_str, str):
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ts = int(dt.timestamp())
            else:
                ts = int(ts_str)
            
            interval_key = (ts // interval_seconds) * interval_seconds
            
            if interval_key not in seen_intervals:
                seen_intervals[interval_key] = candle
                unique_candles.append(candle)
            else:
                existing = seen_intervals[interval_key]
                if candle.get("volume", 0) > existing.get("volume", 0):
                    idx = unique_candles.index(existing)
                    unique_candles[idx] = candle
                    seen_intervals[interval_key] = candle
        except:
            unique_candles.append(candle)
    
    return unique_candles


# ============================================================================
# UTILITY FUNCTIONS - File Management
# ============================================================================

def get_data_path(
    asset_type: str,
    symbol: str,
    timeframe: str,
    date: Optional[str] = None,
    source_hint: Optional[str] = None
) -> Path:
    """Get the file path for a data file."""
    if date:
        filename = f"{date}.json"
    else:
        filename = "latest.json"
    
    if asset_type == "crypto":
        default_source = "binance"
    elif asset_type == "commodities":
        default_source = "yahoo"
    else:
        default_source = asset_type
    
    source_folder = (source_hint or default_source).replace("/", "_")
    
    path = config.BASE_DATA_DIR / asset_type / source_folder / symbol / timeframe / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_json_file(filepath: Path) -> List[Dict[str, Any]]:
    """Load candles from a JSON file."""
    if not filepath.exists():
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "candles" in data:
                return data["candles"]
            else:
                return []
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading {filepath}: {e}")
        return []


def get_last_timestamp_from_existing_data(
    asset_type: str,
    symbol: str,
    timeframe: str,
    source_hint: Optional[str] = None
) -> Optional[datetime]:
    """
    Get the last timestamp from existing historical data file.
    
    Returns:
        datetime object of the last timestamp, or None if no data exists.
        Returns None if data file doesn't exist or is empty.
    """
    # Try to find data file - check multiple possible source folders
    possible_sources = []
    if source_hint:
        possible_sources.append(source_hint)
    
    # For crypto, check both alpaca and binance folders
    if asset_type == "crypto":
        possible_sources.extend(["alpaca", "binance"])
    elif asset_type == "commodities":
        possible_sources.extend(["yahoo", "yahoo_chart"])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_sources = []
    for src in possible_sources:
        if src not in seen:
            seen.add(src)
            unique_sources.append(src)
    
    # Try each possible source folder
    for source_folder in unique_sources:
        base_path = get_data_path(asset_type, symbol, timeframe, None, source_folder).parent
        data_file = base_path / "data.json"
        
        if data_file.exists():
            existing_candles = load_json_file(data_file)
            if existing_candles:
                # Sort by timestamp and get the last one
                sorted_candles = sorted(existing_candles, key=lambda x: x.get("timestamp", ""))
                last_candle = sorted_candles[-1]
                last_timestamp_str = last_candle.get("timestamp", "")
                
                if last_timestamp_str:
                    try:
                        # Parse ISO 8601 timestamp (format: "2024-01-01T00:00:00Z")
                        if last_timestamp_str.endswith("Z"):
                            last_timestamp_str = last_timestamp_str[:-1] + "+00:00"
                        dt = datetime.fromisoformat(last_timestamp_str)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        else:
                            dt = dt.astimezone(timezone.utc)
                        return dt
                    except (ValueError, AttributeError) as e:
                        print(f"  [WARN] Could not parse last timestamp '{last_timestamp_str}': {e}")
                        continue
    
    return None


def save_json_file(filepath: Path, candles: List[Dict[str, Any]], append: bool = False):
    """Save candles to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if append and filepath.exists():
        existing = load_json_file(filepath)
        all_candles = existing + candles
        seen = {}
        for candle in all_candles:
            ts = candle.get("timestamp", "")
            if ts not in seen or candle.get("volume", 0) > seen[ts].get("volume", 0):
                seen[ts] = candle
        candles = list(seen.values())
        candles.sort(key=lambda x: x.get("timestamp", ""))
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(candles, f, indent=2, ensure_ascii=False)


def _infer_source_from_candles(candles: List[Dict[str, Any]], default: str) -> str:
    """Infer the data source folder from candle metadata."""
    for candle in candles:
        src = candle.get("source")
        if src:
            return src
    return default


def get_current_date_utc() -> str:
    """Get current date in UTC as YYYY-MM-DD."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def update_latest_file(
    asset_type: str,
    symbol: str,
    timeframe: str,
    candle: Dict[str, Any],
    source_hint: Optional[str] = None
):
    """Update the latest.json file with the most recent candle."""
    latest_path = get_data_path(asset_type, symbol, timeframe, None, source_hint)
    with open(latest_path, 'w', encoding='utf-8') as f:
        json.dump([candle], f, indent=2, ensure_ascii=False)


def write_candle_to_daily(
    asset_type: str,
    symbol: str,
    timeframe: str,
    candle: Dict[str, Any]
):
    """Write a candle to the single data.json file (appends to existing data)."""
    if asset_type == "crypto":
        # Use source from candle if it's alpaca, otherwise default to binance
        source_hint = candle.get("source") or "binance"
        if source_hint == "alpaca":
            source_hint = "alpaca"
        else:
            source_hint = "binance"  # Fallback sources go to binance folder
    else:
        source_hint = candle.get("source") or "yahoo"
    # Get path for single data file
    base_path = get_data_path(asset_type, symbol, timeframe, None, source_hint).parent
    data_file = base_path / "data.json"
    
    # Load existing data
    existing = load_json_file(data_file)
    
    # Remove duplicate by timestamp (keep new one)
    candle_ts = candle.get("timestamp", "")
    existing = [c for c in existing if c.get("timestamp", "") != candle_ts]
    
    # Add new candle and sort
    existing.append(candle)
    existing.sort(key=lambda x: x.get("timestamp", ""))
    
    # Save to single file
    save_json_file(data_file, existing, append=False)
    
    # Also update latest.json
    update_latest_file(asset_type, symbol, timeframe, candle, source_hint)

    try:
        update_feature_store(asset_type, symbol, timeframe, base_path)
    except Exception as exc:
        print(f"[FEATURE] Unable to update features for {asset_type}/{symbol}/{timeframe}: {exc}")
    
    # Update summary.json with latest market price if it exists
    try:
        update_summary_with_live_price(asset_type, symbol, timeframe, candle)
    except Exception as exc:
        # Silently fail if summary.json doesn't exist yet (model not trained)
        pass


# Global cache for inference pipelines (loaded once per symbol)
_inference_pipelines: Dict[str, Any] = {}

def update_summary_with_live_price(
    asset_type: str,
    symbol: str,
    timeframe: str,
    candle: Dict[str, Any]
):
    """Update summary.json with NEW predictions from live inference (not just recalculating old predictions)."""
    import pandas as pd
    from ml.inference import InferencePipeline
    from ml.risk import RiskManagerConfig
    
    # Check for any available horizon profiles (prefer default, but use any if available)
    summary_path = build_summary_path(asset_type, symbol, timeframe, DEFAULT_HORIZON_PROFILE)
    if not summary_path.exists():
        # Try to find any available horizon
        horizon_dirs = list_horizon_dirs(asset_type, symbol, timeframe)
        for horizon_dir_path in horizon_dirs:
            candidate = horizon_dir_path / "summary.json"
            if candidate.exists():
                summary_path = candidate
                break
        # Fallback to legacy path
        if not summary_path.exists():
            legacy_path = Path("models") / asset_type / symbol / timeframe / "summary.json"
            if legacy_path.exists():
                summary_path = legacy_path
    
    if not summary_path.exists():
        # Model hasn't been trained yet, skip update
        return
    
    try:
        # Extract price and timestamp from candle
        latest_price = float(candle.get("close", 0))
        timestamp_str = candle.get("timestamp", "")
        
        # Parse timestamp
        if isinstance(timestamp_str, str):
            try:
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                latest_timestamp = dt.isoformat().replace('+00:00', 'Z')
            except:
                latest_timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        elif isinstance(timestamp_str, (int, float)):
            ts = float(timestamp_str)
            if ts > 1e10:  # Milliseconds
                ts = ts / 1000
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            latest_timestamp = dt.isoformat().replace('+00:00', 'Z')
        else:
            latest_timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        
        # Load or create inference pipeline (cached per symbol)
        model_dir = summary_path.parent
        cache_key = f"{asset_type}/{symbol}/{timeframe}"
        
        if cache_key not in _inference_pipelines:
            try:
                pipeline = InferencePipeline(model_dir, risk_config=RiskManagerConfig())
                pipeline.load()  # Load models once
                _inference_pipelines[cache_key] = pipeline
            except Exception as exc:
                print(f"[INFERENCE] Failed to load models for {cache_key}: {exc}")
                # Fallback to old method (just update price, don't run inference)
                _update_summary_price_only(summary_path, latest_price, latest_timestamp)
                return
        
        pipeline = _inference_pipelines[cache_key]
        
        # Load latest features from features.json
        feature_path = Path("data/features") / asset_type / symbol / timeframe / "features.json"
        if not feature_path.exists():
            # Features not ready yet, just update price
            _update_summary_price_only(summary_path, latest_price, latest_timestamp)
            return
        
        try:
            with open(feature_path, "r", encoding="utf-8") as f:
                feature_data = json.load(f)
            
            # Convert features.json format to pandas Series
            features_dict = {}
            if "features" in feature_data:
                for feat_name, feat_data in feature_data["features"].items():
                    if isinstance(feat_data, dict) and "value" in feat_data:
                        features_dict[feat_name] = feat_data["value"]
                    elif isinstance(feat_data, (int, float)):
                        features_dict[feat_name] = feat_data
            
            if not features_dict:
                # No features available, fallback
                _update_summary_price_only(summary_path, latest_price, latest_timestamp)
                return
            
            feature_series = pd.Series(features_dict)
            
            # Calculate volatility proxy (use ATR if available, else default)
            volatility = abs(feature_data.get("features", {}).get("ATR_14", {}).get("value", 0) / latest_price) if latest_price > 0 else 0.01
            if volatility == 0:
                volatility = 0.01
            
            # Run inference to get NEW predictions
            inference_result = pipeline.predict(feature_series, current_price=latest_price, volatility=volatility)
            
            # Load existing summary
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            
            # Remove old format fields if they exist (cleanup)
            old_fields = ["latest_market_price", "latest_market_timestamp", "model_reference_price", "rows", "rows_used"]
            for field in old_fields:
                if field in summary:
                    del summary[field]
            
            # Ensure new format structure exists
            if "prediction" not in summary:
                summary["prediction"] = {
                    "current_price": latest_price,
                    "predicted_price": latest_price,
                    "predicted_return_pct": 0.0,
                    "action": "hold",
                    "confidence": 0.0,
                    "horizon_days": 30,
                    "last_updated": latest_timestamp,
                    "explanation": "No prediction available"
                }
            if "model_predictions" not in summary:
                summary["model_predictions"] = {}
            if "consensus" not in summary:
                summary["consensus"] = {}
            
            # Update models with NEW predictions from inference
            model_outputs = inference_result.get("models", {})
            
            # Update simplified model_predictions section (new format only)
            for model_name, model_data in summary["model_predictions"].items():
                if model_name in model_outputs:
                    new_pred_return = float(model_outputs[model_name].get("predicted_return", 0))
                    model_data["predicted_price"] = float(latest_price * (1.0 + new_pred_return))
                    model_data["predicted_return_pct"] = float(new_pred_return * 100)
                    # Update action based on new prediction
                    model_data["action"] = "long" if new_pred_return > 0.01 else "short" if new_pred_return < -0.01 else "hold"
            
            # Update main prediction section
            consensus_result = inference_result.get("consensus", {})
            if consensus_result:
                new_consensus_return = float(consensus_result.get("consensus_return", 0))
                new_consensus_price = float(latest_price * (1.0 + new_consensus_return))
                consensus_action = consensus_result.get("consensus_action", "hold")
                
                summary["prediction"]["current_price"] = latest_price
                summary["prediction"]["predicted_price"] = new_consensus_price
                summary["prediction"]["predicted_return_pct"] = float(new_consensus_return * 100)
                summary["prediction"]["action"] = consensus_action
                summary["prediction"]["confidence"] = float(consensus_result.get("consensus_confidence", 0) * 100)
                summary["prediction"]["last_updated"] = latest_timestamp
                horizon_days = summary["prediction"].get("horizon_days", 30)
                summary["prediction"]["explanation"] = (
                    f"Price expected to "
                    f"{'rise' if consensus_action == 'long' else 'fall' if consensus_action == 'short' else 'stay flat'} "
                    f"from ${latest_price:,.2f} to ${new_consensus_price:,.2f} "
                    f"({new_consensus_return*100:+.2f}%) over {horizon_days} days"
                )
            
            # Update consensus section
            if consensus_result:
                new_consensus_return = float(consensus_result.get("consensus_return", 0))
                summary["consensus"]["predicted_return_pct"] = float(new_consensus_return * 100)
                summary["consensus"]["action"] = consensus_result.get("consensus_action", summary["consensus"].get("action", "hold"))
                summary["consensus"]["confidence_pct"] = float(consensus_result.get("consensus_confidence", 0) * 100)
            
            # Update top-level last_updated
            summary["last_updated"] = latest_timestamp
            
            # Save updated summary with NEW predictions (new format only)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            
            # Output formatted prediction in requested format
            _output_live_prediction_format(asset_type, symbol, timeframe, latest_timestamp, latest_price, inference_result, summary)
            
            # Log to bucket
            try:
                from ml.bucket_logger import get_bucket_logger
                bucket = get_bucket_logger()
                
                # Extract data for bucket logging
                model_outputs = inference_result.get("models", {})
                consensus_result = inference_result.get("consensus", {})
                
                # Build predictions list
                predictions_list = []
                for model_name, model_data in model_outputs.items():
                    if "_quantile" in model_name:
                        continue
                    pred_return = float(model_data.get("predicted_return", 0))
                    predicted_price = float(latest_price * (1.0 + pred_return))
                    model_metrics = summary.get("model_predictions", {}).get(model_name, {})
                    if not model_metrics:
                        model_metrics = summary.get("models", {}).get(model_name, {})
                    
                    predictions_list.append({
                        "algorithm": model_name,
                        "predicted_return": pred_return,
                        "predicted_price": predicted_price,
                        "current_price": latest_price,
                        "confidence": float(model_metrics.get("confidence", model_metrics.get("vote_weight", 0.5) * 100 if model_metrics.get("vote_weight") else 50)),
                        "r2_score": float(model_metrics.get("r2_score", model_metrics.get("r2", 0.0))) if model_metrics.get("r2_score") or model_metrics.get("r2") else None,
                        "action": "long" if pred_return > 0.01 else "short" if pred_return < -0.01 else "hold"
                    })
                
                # Build consensus dict
                consensus_dict = {
                    "action": consensus_result.get("consensus_action", "hold"),
                    "predicted_return": float(consensus_result.get("consensus_return", 0)),
                    "predicted_price": float(latest_price * (1.0 + consensus_result.get("consensus_return", 0))),
                    "confidence": float(consensus_result.get("consensus_confidence", 0)),
                    "reasoning": summary.get("consensus", {}).get("reasoning", "")
                }
                
                # Build sentiment summary
                action_scores = consensus_result.get("action_scores", {})
                sentiment_summary = {
                    "overall_sentiment": consensus_result.get("consensus_action", "hold"),
                    "confidence": float(consensus_result.get("consensus_confidence", 0)),
                    "action_distribution": {
                        "long": float(action_scores.get("long", 0)),
                        "hold": float(action_scores.get("hold", 0)),
                        "short": float(action_scores.get("short", 0))
                    },
                    "expected_return_pct": float(consensus_result.get("consensus_return", 0)) * 100,
                    "price_target": float(latest_price * (1.0 + consensus_result.get("consensus_return", 0))),
                    "horizon_bars": summary.get("consensus", {}).get("target_horizon_bars", 30)
                }
                
                # Build events
                events_list = []
                consensus_return = float(consensus_result.get("consensus_return", 0))
                if abs(consensus_return) > 0.05:
                    events_list.append({
                        "type": "significant_prediction",
                        "message": f"Strong {consensus_dict['action'].upper()} signal: {consensus_return*100:+.2f}% expected return",
                        "severity": "high" if abs(consensus_return) > 0.10 else "medium"
                    })
                
                # Log to bucket
                bucket.log_prediction(
                    symbol=symbol,
                    asset_type=asset_type,
                    timeframe=timeframe,
                    timestamp=latest_timestamp,
                    current_price=latest_price,
                    predictions=predictions_list,
                    consensus=consensus_dict,
                    sentiment_summary=sentiment_summary,
                    events=events_list,
                    metadata={
                        "inference_cache_key": cache_key,
                        "feature_count": len(features_dict),
                        "volatility": volatility
                    }
                )
            except Exception as bucket_exc:
                # Don't fail if bucket logging fails
                print(f"[BUCKET] Failed to log prediction: {bucket_exc}")
            
            # Also prepare feedback data for RL learning (when actual outcomes are available)
            # This will be used when feedback is submitted via API
            try:
                from ml.rl_feedback import get_feedback_learner
                feedback_learner = get_feedback_learner()
                
                # Store prediction data for potential feedback learning
                # The actual feedback will be submitted via /tools/feedback endpoint
                # when actual outcomes are known
                pass  # Feedback is submitted separately via API
            except Exception:
                pass  # Feedback learning is optional
        
        except Exception as exc:
            # If inference fails, fallback to just updating price
            print(f"[INFERENCE] Error running inference for {cache_key}: {exc}")
            _update_summary_price_only(summary_path, latest_price, latest_timestamp)
        
    except Exception as exc:
        # Log error but don't crash the data ingestion
        print(f"[SUMMARY] Unable to update summary.json for {asset_type}/{symbol}/{timeframe}: {exc}")


def _output_live_prediction_format(
    asset_type: str,
    symbol: str,
    timeframe: str,
    timestamp: str,
    current_price: float,
    inference_result: Dict[str, Any],
    summary: Dict[str, Any]
):
    """Output live predictions in the requested JSON format."""
    model_outputs = inference_result.get("models", {})
    consensus = inference_result.get("consensus", {})
    
    # Build predictions array (individual algorithms)
    predictions = []
    for model_name, model_data in model_outputs.items():
        # Skip quantile models
        if "_quantile" in model_name:
            continue
        
        pred_return = float(model_data.get("predicted_return", 0))
        predicted_price = float(current_price * (1.0 + pred_return))
        
        # Get model metrics from summary (try new format first, then old)
        model_metrics = summary.get("model_predictions", {}).get(model_name, {})
        if not model_metrics:
            model_metrics = summary.get("models", {}).get(model_name, {})
        
        predictions.append({
            "algorithm": model_name,
            "predicted_return": pred_return,
            "predicted_price": predicted_price,
            "current_price": current_price,
            "confidence": float(model_metrics.get("confidence", model_metrics.get("vote_weight", 0.5) * 100 if model_metrics.get("vote_weight") else 50)),
            "r2_score": float(model_metrics.get("r2_score", model_metrics.get("r2", 0.0))) if model_metrics.get("r2_score") or model_metrics.get("r2") else None,
            "action": "long" if pred_return > 0.01 else "short" if pred_return < -0.01 else "hold"
        })
    
    # Unified/Consensus prediction
    consensus_return = float(consensus.get("consensus_return", 0))
    consensus_price = float(current_price * (1.0 + consensus_return))
    consensus_action = consensus.get("consensus_action", "hold")
    consensus_confidence = float(consensus.get("consensus_confidence", 0))
    
    unified_prediction = {
        "algorithm": "consensus",
        "predicted_return": consensus_return,
        "predicted_price": consensus_price,
        "current_price": current_price,
        "confidence": consensus_confidence,
        "action": consensus_action,
        "model_count": len([m for m in model_outputs.keys() if "_quantile" not in m]),
        "reasoning": summary.get("consensus", {}).get("reasoning", "")
    }
    
    # Add unified prediction at the beginning
    predictions.insert(0, unified_prediction)
    
    # Build events (price changes, significant predictions)
    events = []
    if abs(consensus_return) > 0.05:  # Significant prediction (>5%)
        events.append({
            "type": "significant_prediction",
            "message": f"Strong {consensus_action.upper()} signal: {consensus_return*100:+.2f}% expected return",
            "severity": "high" if abs(consensus_return) > 0.10 else "medium"
        })
    
    # Build executed_trades (empty for now - can be populated if trading is enabled)
    executed_trades = []
    
    # Build sentiment_summary
    action_scores = consensus.get("action_scores", {})
    sentiment_summary = {
        "overall_sentiment": consensus_action,
        "confidence": consensus_confidence,
        "action_distribution": {
            "long": float(action_scores.get("long", 0)),
            "hold": float(action_scores.get("hold", 0)),
            "short": float(action_scores.get("short", 0))
        },
        "expected_return_pct": consensus_return * 100,
        "price_target": consensus_price,
        "horizon_bars": summary.get("consensus", {}).get("target_horizon_bars", 30)
    }
    
    # Output formatted JSON
    output = {
        "timestamp": timestamp,
        "symbol": symbol,
        "asset_type": asset_type,
        "timeframe": timeframe,
        "predictions": predictions,
        "events": events,
        "executed_trades": executed_trades,
        "sentiment_summary": sentiment_summary
    }
    
    # Print formatted output
    print("\n" + "="*80)
    print("LIVE PREDICTION UPDATE")
    print("="*80)
    print(json.dumps(output, indent=2))
    print("="*80 + "\n")


def _update_summary_price_only(summary_path: Path, latest_price: float, latest_timestamp: str):
    """Fallback: Just update price without running inference (new format only)."""
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        # Update new simplified format
        if "prediction" in summary:
            summary["prediction"]["current_price"] = latest_price
            summary["prediction"]["last_updated"] = latest_timestamp
            if "predicted_return_pct" in summary["prediction"]:
                pred_return = summary["prediction"]["predicted_return_pct"] / 100
                summary["prediction"]["predicted_price"] = float(latest_price * (1.0 + pred_return))
                action = summary["prediction"].get("action", "hold")
                horizon_days = summary["prediction"].get("horizon_days", 30)
                summary["prediction"]["explanation"] = (
                    f"Price expected to "
                    f"{'rise' if action == 'long' else 'fall' if action == 'short' else 'stay flat'} "
                    f"from ${latest_price:,.2f} to ${summary['prediction']['predicted_price']:,.2f} "
                    f"({summary['prediction']['predicted_return_pct']:+.2f}%) over {horizon_days} days"
                )
        
        summary["last_updated"] = latest_timestamp
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except:
        pass


def generate_manifest(
    asset_type: str,
    symbol: str,
    timeframe: str,
    source_hint: Optional[str] = None
) -> Dict[str, Any]:
    """Generate manifest.json for a symbol/timeframe combination."""
    base_path = get_data_path(asset_type, symbol, timeframe, "dummy", source_hint).parent
    
    manifest = {
        "symbol": symbol,
        "timeframe": timeframe,
        "asset_type": asset_type,
        "generated_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "data_file": "data.json",
        "latest_file": "latest.json"
    }
    
    if not base_path.exists():
        manifest_path = base_path / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return manifest
    
    # Check for data.json file
    data_file = base_path / "data.json"
    if data_file.exists():
        try:
            file_size = data_file.stat().st_size
            candles = load_json_file(data_file)
            
            manifest["data_file_info"] = {
                "filename": "data.json",
                "size_bytes": file_size,
                "candle_count": len(candles),
                "first_timestamp": candles[0]["timestamp"] if candles else None,
                "last_timestamp": candles[-1]["timestamp"] if candles else None
            }
        except Exception as e:
            print(f"Error processing data.json: {e}")
    
    # Check for latest.json
    latest_file = base_path / "latest.json"
    if latest_file.exists():
        try:
            latest_candles = load_json_file(latest_file)
            if latest_candles:
                manifest["latest_candle"] = latest_candles[-1]["timestamp"]
        except Exception as e:
            print(f"Error processing latest.json: {e}")
    
    manifest_path = base_path / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    return manifest


# ============================================================================
# FEATURE CALCULATION
# ============================================================================

FEATURE_ROOT = Path("data/features")


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def calc_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain, index=series.index).ewm(alpha=1 / length, adjust=False).mean()
    roll_down = pd.Series(loss, index=series.index).ewm(alpha=1 / length, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def calc_roc(series: pd.Series, length: int) -> pd.Series:
    return series.pct_change(length, fill_method=None)


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def calc_trix(series: pd.Series, length: int = 15) -> pd.Series:
    e1 = ema(series, length)
    e2 = ema(e1, length)
    e3 = ema(e2, length)
    return e3.pct_change(fill_method=None)


def calc_tsi(series: pd.Series, slow: int = 25, fast: int = 13) -> pd.Series:
    pc = series.diff()
    abs_pc = pc.abs()
    double_smooth = ema(ema(pc, slow), fast)
    double_abs = ema(ema(abs_pc, slow), fast)
    double_abs = double_abs.replace(0, np.nan)
    return 100 * (double_smooth / double_abs)


def calc_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3):
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    range_ = (highest_high - lowest_low).replace(0, np.nan)
    k_percent = 100 * (close - lowest_low) / range_
    d_percent = k_percent.rolling(d).mean()
    return k_percent, d_percent


def calc_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    highest_high = high.rolling(length).max()
    lowest_low = low.rolling(length).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)


def calc_cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    ma = tp.rolling(length).mean()
    md = tp.rolling(length).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - ma) / (0.015 * md)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.rolling(length).mean()


def calc_bbands(series: pd.Series, length: int = 20, std: float = 2.0):
    ma = series.rolling(length).mean()
    dev = series.rolling(length).std()
    upper = ma + std * dev
    lower = ma - std * dev
    return upper, ma, lower


def calc_keltner(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20, mult: float = 2.0):
    mid = ema(close, length)
    atr_val = calc_atr(high, low, close, length)
    upper = mid + mult * atr_val
    lower = mid - mult * atr_val
    return upper, lower


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr = true_range(high, low, close)
    atr_val = calc_atr(high, low, close, length).replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(length).sum() / atr_val
    minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(length).sum() / atr_val
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).abs() * 100
    adx_series = dx.rolling(length).mean()
    return adx_series, plus_di, minus_di


def calc_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical = (high + low + close) / 3
    cum_vol_price = (typical * volume).cumsum()
    cum_volume = volume.cumsum().replace(0, np.nan)
    return cum_vol_price / cum_volume


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def calc_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 20) -> pd.Series:
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0).fillna(0)
    mf_volume = mf_multiplier * volume
    denom = volume.rolling(length).sum().replace(0, np.nan)
    return mf_volume.rolling(length).sum() / denom


def calc_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    typical = (high + low + close) / 3
    money_flow = typical * volume
    positive_flow = np.where(typical > typical.shift(1), money_flow, 0)
    negative_flow = np.where(typical < typical.shift(1), money_flow, 0)
    pos_mf = pd.Series(positive_flow, index=high.index).rolling(length).sum()
    neg_mf = pd.Series(negative_flow, index=high.index).rolling(length).sum()
    ratio = pos_mf / neg_mf.replace(0, np.nan)
    mfi = 100 - (100 / (1 + ratio))
    return mfi


def calc_emv(high: pd.Series, low: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    dm = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
    br = (volume / 1_000_000) / (high - low)
    br = br.replace([np.inf, -np.inf], np.nan).fillna(0)
    emv = dm / br.replace(0, np.nan)
    return emv.rolling(length).mean()


def calc_force_index(close: pd.Series, volume: pd.Series, length: int = 1) -> pd.Series:
    return (close - close.shift(length)) * volume


def calc_adl(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.replace([np.inf, -np.inf], 0).fillna(0)
    return (clv * volume).cumsum()


def calc_chaikin(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    ad_line = calc_adl(high, low, close, volume)
    return ema(ad_line, 3) - ema(ad_line, 10)


def calc_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series):
    conversion = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((conversion + base) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    lagging = close.shift(-26)
    return conversion, base, span_a, span_b, lagging


def calc_psar(high: pd.Series, low: pd.Series, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    sar = pd.Series(index=high.index, dtype=float)
    bull = True
    af = step
    ep = low.iloc[0]
    sar.iloc[0] = low.iloc[0]
    for i in range(1, len(high)):
        prev_sar = sar.iloc[i - 1]
        if bull:
            sar_val = prev_sar + af * (ep - prev_sar)
            sar_val = min(sar_val, low.iloc[i - 1], low.iloc[i])
            if low.iloc[i] < sar_val:
                bull = False
                sar_val = ep
                ep = low.iloc[i]
                af = step
        else:
            sar_val = prev_sar + af * (ep - prev_sar)
            sar_val = max(sar_val, high.iloc[i - 1], high.iloc[i])
            if high.iloc[i] > sar_val:
                bull = True
                sar_val = ep
                ep = high.iloc[i]
                af = step
        if bull and high.iloc[i] > ep:
            ep = high.iloc[i]
            af = min(af + step, max_step)
        elif (not bull) and low.iloc[i] < ep:
            ep = low.iloc[i]
            af = min(af + step, max_step)
        sar.iloc[i] = sar_val
    return sar


def _ensure_feature_dir(asset_type: str, symbol: str, timeframe: str) -> Path:
    path = FEATURE_ROOT / asset_type / symbol / timeframe
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_candles(data_file: Path) -> pd.DataFrame:
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    with open(data_file, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"No candle data in {data_file}")

    df = pd.DataFrame(raw)
    required_cols = {"open", "high", "low", "close", "timestamp"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing candle fields {missing} in {data_file}")

    if "volume" not in df.columns:
        df["volume"] = np.nan

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    cols = ["open", "high", "low", "close", "volume"]
    return df[cols].astype(float)


@dataclass
class FeatureResult:
    value: Optional[float] = None
    timestamp: Optional[pd.Timestamp] = None
    status: Optional[str] = None
    reason: Optional[str] = None


class FeatureCalculator:
    SCHEMA_VERSION = 2
    """Computes a broad catalog of features from OHLCV candles."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["typical_price"] = (self.df["high"] + self.df["low"] + self.df["close"]) / 3.0
        self.df["median_price"] = (self.df["high"] + self.df["low"]) / 2.0
        self.df["price_range"] = self.df["high"] - self.df["low"]
        self.df["intraday_range"] = self.df["close"] - self.df["open"]
        self.df["returns"] = self.df["close"].pct_change(fill_method=None)
        self.df["log_returns"] = np.log(self.df["close"] / self.df["close"].shift(1))
        self.results: Dict[str, FeatureResult] = {}
        self.series_cache: Dict[str, pd.Series] = {}

    def compute_all(self) -> Dict[str, FeatureResult]:
        self._moving_averages()
        self._momentum_indicators()
        self._volatility_indicators()
        self._volume_indicators()
        self._price_structure_indicators()
        self._signals_and_flags()
        self._placeholder_features()
        return self.results

    def _add_series(self, name: str, series: pd.Series, min_points: int = 1):
        clean = series.dropna()
        if len(clean) >= min_points:
            self.series_cache[name] = series
            self.results[name] = FeatureResult(value=float(clean.iloc[-1]), timestamp=clean.index[-1])
        else:
            self.results[name] = FeatureResult(
                value=None,
                timestamp=self.df.index[-1],
                status="insufficient_data",
                reason=f"Not enough data to compute {name}"
            )

    def _add_placeholder(self, name: str, reason: str, status: str = "data_not_available"):
        self.results[name] = FeatureResult(
            value=None,
            timestamp=self.df.index[-1],
            status=status,
            reason=reason
        )

    # --- Feature groups -------------------------------------------------- #

    def _moving_averages(self):
        close = self.df["close"]
        for length in [5, 10, 20, 50, 100, 200]:
            self._add_series(f"SMA_{length}", close.rolling(length).mean())
        for length in [9, 12, 26, 50, 100, 200]:
            self._add_series(f"EMA_{length}", close.ewm(span=length, adjust=False).mean())
        w = 14
        weights = np.arange(1, w + 1)
        wma = close.rolling(w).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        self._add_series("WMA_14", wma)

        def hull_ma(series: pd.Series, length: int = 21):
            half = int(length / 2)
            sqrt_len = int(math.sqrt(length))
            wma1 = series.rolling(length).apply(
                lambda x: np.dot(x, np.arange(1, length + 1)) / np.arange(1, length + 1).sum(),
                raw=True
            )
            wma2 = series.rolling(half).apply(
                lambda x: np.dot(x, np.arange(1, half + 1)) / np.arange(1, half + 1).sum(),
                raw=True
            )
            hull = (2 * wma2) - wma1
            return hull.rolling(sqrt_len).mean()

        self._add_series("Hull_MA", hull_ma(close))
        volume = self.df["volume"]
        vwma = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        self._add_series("VWMA", vwma)
        vwap_series = calc_vwap(self.df["high"], self.df["low"], close, volume)
        self._add_series("VWAP", vwap_series)
        self._add_series("Session_VWAP", vwap_series)

    def _momentum_indicators(self):
        close = self.df["close"]
        high = self.df["high"]
        low = self.df["low"]
        volume = self.df["volume"]

        macd_line, macd_signal, macd_hist = calc_macd(close)
        self._add_series("MACD", macd_line)
        self._add_series("MACD_signal", macd_signal)
        self._add_series("MACD_histogram", macd_hist)

        for length in [7, 14, 21]:
            self._add_series(f"RSI_{length}", calc_rsi(close, length=length))

        for length in [1, 7, 14]:
            self._add_series(f"Momentum_roc_{length}", calc_roc(close, length=length))
        self._add_series("Rate_of_Change", calc_roc(close, length=10))

        self._add_series("TRIX", calc_trix(close))
        self._add_series("TSI", calc_tsi(close))
        k_percent, d_percent = calc_stochastic(high, low, close)
        self._add_series("Stochastic_%K", k_percent)
        self._add_series("Stochastic_%D", d_percent)
        self._add_series("Williams_%R", calc_williams_r(high, low, close))
        self._add_series("CCI_20", calc_cci(high, low, close, length=20))

        atr14 = calc_atr(high, low, close, length=14)
        atr7 = calc_atr(high, low, close, length=7)
        self._add_series("ATR_14", atr14)
        self._add_series("ATR_7", atr7)

        upper, mid, lower = calc_bbands(close, length=20, std=2)
        self._add_series("Bollinger_Bands_upper_20_2", upper)
        self._add_series("Bollinger_Bands_lower_20_2", lower)
        range_band = (upper - lower).replace(0, np.nan)
        bandwidth = range_band / mid
        self._add_series("Bollinger_Bandwidth", bandwidth)
        percent_b = (close - lower) / range_band
        self._add_series("Bollinger_%b", percent_b)

        kc_upper, kc_lower = calc_keltner(high, low, close, length=20, mult=2)
        self._add_series("Keltner_Channel_upper", kc_upper)
        self._add_series("Keltner_Channel_lower", kc_lower)

        donch_hi = high.rolling(20).max()
        donch_lo = low.rolling(20).min()
        self._add_series("Donchian_Channel_high", donch_hi)
        self._add_series("Donchian_Channel_low", donch_lo)

        adx_series, plus_di, minus_di = calc_adx(high, low, close, length=14)
        self._add_series("ADX_14", adx_series)
        self._add_series("DI_plus", plus_di)
        self._add_series("DI_minus", minus_di)

        rsi_vals = calc_rsi(close, length=14)
        self._add_series("RSI_Band_Cross", ((rsi_vals > 70) | (rsi_vals < 30)).astype(int))

        self._add_series("ROC_Signal", calc_roc(close, length=5))

    def _volatility_indicators(self):
        returns = self.df["returns"]
        log_returns = self.df["log_returns"]
        for window in [7, 14, 30]:
            self._add_series(f"Rolling_Volatility_{window}", returns.rolling(window).std())
        realized = log_returns.pow(2).rolling(30).sum().apply(np.sqrt)
        self._add_series("Realized_Volatility", realized)
        hist_vol = returns.rolling(30).std() * np.sqrt(30)
        self._add_series("Historical_Volatility", hist_vol)
        skew = returns.rolling(30).skew()
        kurt = returns.rolling(30).kurt()
        self._add_series("Rolling_Skew", skew)
        self._add_series("Rolling_Kurtosis", kurt)
        vol_skew = (returns.rolling(7).std() - returns.rolling(30).std())
        self._add_series("Volatility_Skew", vol_skew)
        vol_cluster = (returns.rolling(3).std() / returns.rolling(30).std())
        self._add_series("Volatility_Clustering", vol_cluster)

        z7 = (returns - returns.rolling(7).mean()) / returns.rolling(7).std()
        z30 = (returns - returns.rolling(30).mean()) / returns.rolling(30).std()
        self._add_series("ZScore_returns_7", z7)
        self._add_series("ZScore_returns_30", z30)
        self._add_series("Normalized_Returns", z30)
        self._add_series("Log_Returns", log_returns)
        self._add_series("Return_1", self.df["close"].pct_change(1, fill_method=None))
        self._add_series("Return_3", self.df["close"].pct_change(3, fill_method=None))
        self._add_series("Return_7", self.df["close"].pct_change(7, fill_method=None))
        cumulative_30 = (1 + self.df["close"].pct_change(fill_method=None)).rolling(30).apply(np.prod) - 1
        self._add_series("Cumulative_Return_30", cumulative_30)

        rolling_sharpe_30 = returns.rolling(30).mean() / returns.rolling(30).std()
        rolling_sharpe_90 = returns.rolling(90).mean() / returns.rolling(90).std()
        self._add_series("Rolling_Sharpe_30", rolling_sharpe_30)
        self._add_series("Rolling_Sharpe_90", rolling_sharpe_90)

        running_max = self.df["close"].rolling(30, min_periods=1).max()
        drawdown = self.df["close"] / running_max - 1
        self._add_series("Max_Drawdown_30", drawdown.rolling(30).min())

        duration = drawdown.groupby((drawdown == 0).cumsum()).cumcount()
        self._add_series("Drawdown_Duration", duration)

    def _volume_indicators(self):
        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]
        volume = self.df["volume"]
        self._add_series("OBV", calc_obv(close, volume))
        self._add_series("CMF", calc_cmf(high, low, close, volume, length=20))
        self._add_series("MFI", calc_mfi(high, low, close, volume, length=14))
        emv = calc_emv(high, low, volume)
        self._add_series("Ease_of_Movement", emv)
        self._add_series("Force_Index", calc_force_index(close, volume))
        adl = calc_adl(high, low, close, volume)
        self._add_series("Accumulation_Distribution", adl)
        self._add_series("Chaikin_Oscillator", calc_chaikin(high, low, close, volume))

        vol_change_1 = volume.pct_change(1, fill_method=None)
        vol_change_7 = volume.pct_change(7, fill_method=None)
        self._add_series("Volume_Change_1", vol_change_1)
        self._add_series("Volume_Change_7", vol_change_7)
        vol_ratio = volume / volume.rolling(20).mean()
        self._add_series("Volume_Ratio", vol_ratio)
        surge_flag = (volume > volume.rolling(20).mean() * 1.5).astype(int)
        self._add_series("Volume_Surge_Flag", surge_flag)

    def _price_structure_indicators(self):
        close = self.df["close"]
        high = self.df["high"]
        low = self.df["low"]
        open_ = self.df["open"]

        for lag in [1, 2, 3]:
            self._add_series(f"Price_Lag_{lag}", close.shift(lag))
        for lag in range(1, 11):
            self._add_series(f"Close_Lag_{lag}", close.shift(lag))
        self._add_series("Open_Lag_1", open_.shift(1))
        self._add_series("High_Lag_1", high.shift(1))
        self._add_series("Low_Lag_1", low.shift(1))
        self._add_series("Typical_Price", self.df["typical_price"])
        self._add_series("Median_Price", self.df["median_price"])
        self._add_series("Price_Range", self.df["price_range"])
        self._add_series("Intraday_Range", self.df["intraday_range"])
        gap_open = open_ - close.shift(1)
        self._add_series("Gap_Open", gap_open)
        self._add_series("Gap_Percentage", gap_open / close.shift(1))

        donch_hi = high.rolling(20).max()
        donch_lo = low.rolling(20).min()
        breakout = ((close >= donch_hi) | (close <= donch_lo)).astype(int)
        self._add_series("Donchian_Breakout_Flag", breakout)

        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        self._add_series("MA_Crossover_Signal", np.sign(sma50 - sma200).diff())
        macd_series = self.series_cache.get("MACD")
        macd_signal_series = self.series_cache.get("MACD_signal")
        if macd_series is not None and macd_signal_series is not None:
            self._add_series("MACD_Crossover_Signal", np.sign(macd_series - macd_signal_series).diff())

        roc7 = calc_roc(close, length=7)
        self._add_series("ROC_Signal", roc7)

        atr14 = self.series_cache.get("ATR_14")
        if atr14 is None:
            atr14 = calc_atr(high, low, close, length=14)
        vol_expansion = (atr14 > atr14.rolling(20).mean() * 1.25).astype(int)
        self._add_series("Volatility_Expansion_Flag", vol_expansion)

        tenkan, kijun, span_a, span_b, _ = calc_ichimoku(high, low, close)
        self._add_series("Ichimoku_Tenkan", tenkan)
        self._add_series("Ichimoku_Kijun", kijun)
        self._add_series("Ichimoku_Senkou_Span_A", span_a)
        self._add_series("Ichimoku_Senkou_Span_B", span_b)

        psar_series = calc_psar(high, low)
        self._add_series("Parabolic_SAR", psar_series)

        if tenkan is not None and kijun is not None:
            price_channel_slope = (tenkan - kijun).rolling(10).mean()
            self._add_series("Price_Channel_Slope", price_channel_slope)

        # Fractals (using 2-period lookback/forward)
        fractal_high = high[(high.shift(2) < high.shift(1)) &
                            (high.shift(1) < high) &
                            (high.shift(-1) < high) &
                            (high.shift(-2) < high)]
        fractal_low = low[(low.shift(2) > low.shift(1)) &
                          (low.shift(1) > low) &
                          (low.shift(-1) > low) &
                          (low.shift(-2) > low)]
        shifted_fractal_high = fractal_high.shift(2)
        shifted_fractal_low = fractal_low.shift(2)
        self._add_series("Fractal_High", shifted_fractal_high)
        self._add_series("Fractal_Low", shifted_fractal_low)

        # Pivot points (previous period)
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)
        pivot = (prev_high + prev_low + prev_close) / 3.0
        self._add_series("Pivot_Point", pivot)
        self._add_series("Pivot_R1", 2 * pivot - prev_low)
        self._add_series("Pivot_S1", 2 * pivot - prev_high)
        range_ = prev_high - prev_low
        self._add_series("Pivot_R2", pivot + range_)
        self._add_series("Pivot_S2", pivot - range_)

        # Fibonacci retracement levels (rolling 20)
        rolling_high = high.rolling(20).max()
        rolling_low = low.rolling(20).min()
        diff = rolling_high - rolling_low
        for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]:
            fib = rolling_high - diff * ratio
            self._add_series(f"Fibonacci_Level_{ratio}", fib)

        # Seasonality features
        index = self.df.index
        self._add_series("Seasonality_Monthly", pd.Series(index.month, index=index))
        self._add_series("Seasonality_DayOfMonth", pd.Series(index.day, index=index))
        self._add_series("Day_of_Week", pd.Series(index.dayofweek, index=index))
        self._add_series("Hour_of_Day", pd.Series(index.hour, index=index))
        self._add_series("Is_Holiday_Flag", pd.Series(0, index=index))

    def _signals_and_flags(self):
        # Already populated caches handle interactions
        vol14 = self.series_cache.get("Rolling_Volatility_14")
        mom14 = self.series_cache.get("Momentum_roc_14")
        if vol14 is not None and mom14 is not None:
            self._add_series("Feature_Interaction_Terms", vol14 * mom14)

        rsi14 = self.series_cache.get("RSI_14")
        macd_hist = self.series_cache.get("MACD_histogram")
        zscore30 = self.series_cache.get("ZScore_returns_30")
        components = []
        if rsi14 is not None:
            components.append(rsi14 / 100.0)
        if macd_hist is not None:
            components.append(macd_hist)
        if zscore30 is not None:
            components.append(zscore30)
        if components:
            ensemble = pd.concat(components, axis=1).mean(axis=1)
            self._add_series("Ensemble_Feature_Aggregates", ensemble)

    def _placeholder_features(self):
        pass


def update_feature_store(
    asset_type: str,
    symbol: str,
    timeframe: str,
    data_directory: Path
):
    """Load candles and regenerate all feature logs including context features."""
    from ml.context_features import build_context_features, ContextFeatureConfig
    
    data_file = Path(data_directory) / "data.json"
    if not data_file.exists():
        return
    try:
        candles = _load_candles(data_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FEATURE] Skipping {asset_type}/{symbol}/{timeframe}: {exc}")
        return
    
    # Compute basic technical features
    calculator = FeatureCalculator(candles)
    results = calculator.compute_all()
    
    # Add context features (macro, volatility, spreads, etc.) to match training
    context_config = ContextFeatureConfig(
        include_macro=True,
        include_spreads=True,
        include_volatility_indices=True,
        include_regime_features=True,
        include_intraday_aggregates=True,
        intraday_timeframes=("4h", "1h"),
        intraday_lookback=45,
    )
    try:
        context_frame, context_meta = build_context_features(
            base_candles=candles,
            asset_type=asset_type,
            symbol=symbol,
            timeframe=timeframe,
            config=context_config,
        )
        # Add context features to results
        if not context_frame.empty and len(context_frame) > 0:
            last_row = context_frame.iloc[-1]
            for col in context_frame.columns:
                value = last_row[col]
                if pd.notna(value) and isinstance(value, (int, float)):
                    results[col] = FeatureResult(
                        value=float(value),
                        timestamp=candles.index[-1],
                    )
    except Exception as exc:
        # If context features fail, continue with basic features
        print(f"[FEATURE] Context features failed for {symbol}: {exc}")
    
    out_dir = _ensure_feature_dir(asset_type, symbol, timeframe)
    output = {
        "asset_type": asset_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "last_updated": candles.index[-1].isoformat(),
        "feature_count": len(results),
        "features": {}
    }
    for feature_name, result in results.items():
        entry: Dict[str, Optional[float]] = {}
        entry["value"] = result.value
        if result.status:
            entry["status"] = result.status
        if result.reason:
            entry["reason"] = result.reason
        output["features"][feature_name] = entry

    feature_path = out_dir / "features.json"
    with open(feature_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)


# ============================================================================
# UTILITY FUNCTIONS - Fallback Engine
# ============================================================================

class SourceStatus(Enum):
    """Source status enumeration."""
    ACTIVE = "active"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    DISCONNECTED = "disconnected"
    STALE = "stale"


class FallbackEngine:
    """Manages fallback sources with automatic switching and recovery."""
    
    def __init__(self, sources: List[str], primary_source: str):
        self.sources = sources
        self.primary_source = primary_source
        self.current_source = primary_source
        self.source_status: Dict[str, SourceStatus] = {s: SourceStatus.ACTIVE for s in sources}
        self.failure_counts: Dict[str, int] = {s: 0 for s in sources}
        self.last_success: Dict[str, float] = {s: time.time() for s in sources}
        self.fallback_reason: Optional[str] = None
        
    def get_current_source(self) -> str:
        return self.current_source
    
    def is_fallback_active(self) -> bool:
        return self.current_source != self.primary_source
    
    def get_fallback_reason(self) -> Optional[str]:
        return self.fallback_reason
    
    def mark_success(self, source: Optional[str] = None):
        source = source or self.current_source
        self.source_status[source] = SourceStatus.ACTIVE
        self.failure_counts[source] = 0
        self.last_success[source] = time.time()
        
        if source == self.primary_source and self.current_source != self.primary_source:
            logger.info(f"Primary source {self.primary_source} recovered, switching back")
            self.current_source = self.primary_source
            self.fallback_reason = None
    
    def mark_failure(self, source: Optional[str] = None, reason: Optional[str] = None, status_code: Optional[int] = None):
        source = source or self.current_source
        self.failure_counts[source] = self.failure_counts.get(source, 0) + 1
        
        if status_code and status_code == 429:
            self.source_status[source] = SourceStatus.RATE_LIMITED
            reason = reason or "Rate limited"
        elif status_code and status_code >= 500:
            self.source_status[source] = SourceStatus.FAILED
            reason = reason or f"Server error {status_code}"
        else:
            self.source_status[source] = SourceStatus.FAILED
            reason = reason or "Unknown error"
        
        logger.warning(f"Source {source} failed: {reason} (count: {self.failure_counts[source]})")
        self._switch_to_next_source(reason)
    
    def mark_disconnected(self, source: Optional[str] = None):
        source = source or self.current_source
        self.source_status[source] = SourceStatus.DISCONNECTED
        self._switch_to_next_source("WebSocket disconnected")
    
    def mark_stale(self, source: Optional[str] = None, age_seconds: Optional[float] = None):
        source = source or self.current_source
        self.source_status[source] = SourceStatus.STALE
        reason = f"Data stale ({age_seconds:.1f}s old)" if age_seconds else "Data stale"
        self._switch_to_next_source(reason)
    
    def _switch_to_next_source(self, reason: str):
        current_idx = self.sources.index(self.current_source) if self.current_source in self.sources else -1
        
        for i in range(current_idx + 1, len(self.sources)):
            candidate = self.sources[i]
            if self.source_status.get(candidate) == SourceStatus.ACTIVE:
                logger.info(f"Switching from {self.current_source} to {candidate}: {reason}")
                self.current_source = candidate
                self.fallback_reason = reason
                return
        
        if self.current_source != self.primary_source:
            logger.warning(f"No fallback sources available, trying primary {self.primary_source}")
            self.current_source = self.primary_source
            self.fallback_reason = reason


# ============================================================================
# DATA FETCHERS - Alpaca Historical (PRIMARY SOURCE)
# ============================================================================

def _convert_to_alpaca_symbol(data_symbol: str) -> str:
    """
    Convert our data symbol (BTC-USDT) to Alpaca format (BTC/USD).
    
    Alpaca uses format: BASE/QUOTE (e.g., BTC/USD, ETH/USD)
    Our format: BASE-QUOTE (e.g., BTC-USDT, ETH-USDT)
    
    This function is safe and will always return a valid symbol format.
    If conversion fails, returns the original symbol (fallback sources can handle it).
    """
    if not data_symbol or not isinstance(data_symbol, str):
        return data_symbol  # Return as-is if invalid input
    
    try:
        symbol_upper = data_symbol.upper().strip()
        
        # Remove -USDT and replace with /USD
        if "-USDT" in symbol_upper:
            base = symbol_upper.replace("-USDT", "").strip()
            if base:  # Ensure base is not empty
                return f"{base}/USD"
        
        # If already in different format, try to convert
        if "-" in symbol_upper:
            parts = symbol_upper.split("-")
            if len(parts) == 2 and parts[0] and parts[1]:
                return f"{parts[0]}/{parts[1]}"
        
        # If already has /, return as-is (might already be in Alpaca format)
        if "/" in symbol_upper:
            return symbol_upper
        
        # Fallback: return uppercase version (fallback sources will handle conversion)
        return symbol_upper
    except Exception:
        # If anything fails, return original symbol - fallback sources can handle it
        return data_symbol.upper() if isinstance(data_symbol, str) else str(data_symbol)


def fetch_alpaca_historical(
    symbol: str,
    timeframe: str = "1d",
    years: float = 5.0,
    start_date: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch historical crypto data from Alpaca Data API (PRIMARY SOURCE).
    
    NOTE: Alpaca's CryptoHistoricalDataClient does NOT require API keys for crypto data.
    This is free and unlimited for crypto historical data.
    
    Args:
        symbol: Data symbol (e.g., BTC-USDT)
        timeframe: Timeframe (1d, 1h, etc.)
        years: Number of years to fetch (used only if start_date is None)
        start_date: Optional start date for incremental fetching. If provided, 
                   only fetches data from this date to now.
    """
    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from trading.symbol_universe import find_by_data_symbol
    except ImportError:
        raise ImportError(
            "alpaca-py package not installed. Install with: pip install alpaca-py"
        )
    
    # Convert our symbol format to Alpaca format
    # e.g., BTC-USDT -> BTC/USD
    # Use try-except to ensure symbol conversion never fails
    try:
        asset_mapping = find_by_data_symbol(symbol)
        if asset_mapping:
            # Use the trading symbol and convert to Alpaca format
            # BTCUSD -> BTC/USD
            trading_sym = asset_mapping.trading_symbol
            if trading_sym and trading_sym.endswith("USD"):
                alpaca_symbol = f"{trading_sym[:-3]}/USD"
            else:
                alpaca_symbol = _convert_to_alpaca_symbol(symbol)
        else:
            alpaca_symbol = _convert_to_alpaca_symbol(symbol)
    except Exception as sym_exc:
        # If symbol conversion fails, use fallback conversion
        print(f"  [WARN] Symbol mapping failed for {symbol}: {sym_exc}, using fallback conversion")
        alpaca_symbol = _convert_to_alpaca_symbol(symbol)
    
    # Map timeframe to Alpaca TimeFrame
    timeframe_map = {
        "1m": TimeFrame.Minute,
        "5m": TimeFrame(5, TimeFrame.Minute),
        "15m": TimeFrame(15, TimeFrame.Minute),
        "1h": TimeFrame.Hour,
        "4h": TimeFrame(4, TimeFrame.Hour),
        "1d": TimeFrame.Day,
    }
    alpaca_tf = timeframe_map.get(timeframe, TimeFrame.Day)
    
    # Calculate date range
    end_date = datetime.now(timezone.utc)
    
    # If start_date is provided (incremental fetch), use it
    # Otherwise, calculate from years parameter (full historical fetch)
    if start_date is None:
        calculated_start = end_date - timedelta(days=int(years * 365.25))
        fetch_mode = "full historical"
    else:
        # Ensure start_date is timezone-aware
        if start_date.tzinfo is None:
            calculated_start = start_date.replace(tzinfo=timezone.utc)
        else:
            calculated_start = start_date.astimezone(timezone.utc)
        # Subtract 1 day to ensure we get the last candle again (for overlap/verification)
        calculated_start = calculated_start - timedelta(days=1)
        fetch_mode = "incremental"
    
    print(f"Fetching via Alpaca Data API: {alpaca_symbol} ({timeframe}) - {fetch_mode}")
    print(f"  Date range: {calculated_start.date()} to {end_date.date()}")
    
    try:
        # No keys required for crypto data!
        client = CryptoHistoricalDataClient()
        
        # Create request
        request_params = CryptoBarsRequest(
            symbol_or_symbols=[alpaca_symbol],
            timeframe=alpaca_tf,
            start=calculated_start,
            end=end_date,
        )
        
        # Fetch bars
        bars = client.get_crypto_bars(request_params)
        
        if bars is None:
            return []
        
        # Convert to DataFrame if not already
        if hasattr(bars, 'df'):
            df = bars.df
        else:
            # If it's already a DataFrame or list, handle accordingly
            import pandas as pd
            if isinstance(bars, pd.DataFrame):
                df = bars
            else:
                return []
        
        if df is None or len(df) == 0:
            return []
        
        # Convert to our canonical format
        candles = []
        for idx, row in df.iterrows():
            # Alpaca bars have: timestamp (index), open, high, low, close, volume, trade_count, vwap
            timestamp = idx  # pandas Timestamp (index)
            if hasattr(timestamp, 'to_pydatetime'):
                dt = timestamp.to_pydatetime()
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
            elif hasattr(timestamp, 'timestamp'):
                dt = datetime.fromtimestamp(timestamp.timestamp(), tz=timezone.utc)
            else:
                # Fallback: try to parse as datetime
                try:
                    dt = pd.to_datetime(timestamp).to_pydatetime()
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                except:
                    continue  # Skip this row if timestamp parsing fails
            
            canonical = create_canonical_candle(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=dt,
                open_price=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row.get('volume', 0.0)) if 'volume' in row else 0.0,
                source="alpaca",
                fallback_active=False,
            )
            candles.append(canonical)
        
        print(f"  [OK] Successfully fetched {len(candles)} candles from Alpaca")
        return candles
        
    except ImportError as import_exc:
        # If alpaca-py is not installed, raise ImportError to trigger fallback
        error_msg = str(import_exc)
        print(f"  [ERROR] Alpaca import failed: {error_msg}")
        print(f"  [INFO] Original symbol '{symbol}' preserved for fallback sources")
        raise  # Re-raise to trigger fallback (original symbol is preserved)
    except Exception as exc:
        # Any other error (API error, symbol not found, etc.) - trigger fallback
        error_msg = str(exc)
        print(f"  [ERROR] Alpaca API failed: {error_msg}")
        print(f"  [INFO] Falling back to alternative data sources with original symbol '{symbol}'...")
        raise  # Re-raise to trigger fallback (original symbol is preserved)


def fetch_alpaca_live(
    symbol: str,
    timeframe: str = "1d",
) -> Optional[Dict[str, Any]]:
    """
    Fetch latest live candle from Alpaca Data API.
    
    Returns a single canonical candle dict, or None if unavailable.
    """
    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from trading.symbol_universe import find_by_data_symbol
    except ImportError:
        return None
    
    # Convert symbol format (with error handling)
    try:
        asset_mapping = find_by_data_symbol(symbol)
        if asset_mapping and asset_mapping.trading_symbol:
            trading_sym = asset_mapping.trading_symbol
            if trading_sym.endswith("USD"):
                alpaca_symbol = f"{trading_sym[:-3]}/USD"
            else:
                alpaca_symbol = _convert_to_alpaca_symbol(symbol)
        else:
            alpaca_symbol = _convert_to_alpaca_symbol(symbol)
    except Exception:
        # If symbol mapping fails, use fallback conversion
        alpaca_symbol = _convert_to_alpaca_symbol(symbol)
    
    timeframe_map = {
        "1m": TimeFrame.Minute,
        "5m": TimeFrame(5, TimeFrame.Minute),
        "15m": TimeFrame(15, TimeFrame.Minute),
        "1h": TimeFrame.Hour,
        "4h": TimeFrame(4, TimeFrame.Hour),
        "1d": TimeFrame.Day,
    }
    alpaca_tf = timeframe_map.get(timeframe, TimeFrame.Day)
    
    # Fetch last 2 bars (to ensure we get the latest complete one)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=2)
    
    try:
        client = CryptoHistoricalDataClient()
        request_params = CryptoBarsRequest(
            symbol_or_symbols=[alpaca_symbol],
            timeframe=alpaca_tf,
            start=start_date,
            end=end_date,
            limit=2,  # Just get last 2 bars
        )
        
        bars = client.get_crypto_bars(request_params)
        
        if bars is None:
            return None
        
        # Convert to DataFrame if not already
        if hasattr(bars, 'df'):
            df = bars.df
        else:
            import pandas as pd
            if isinstance(bars, pd.DataFrame):
                df = bars
            else:
                return None
        
        if df is None or len(df) == 0:
            return None
        
        # Get the latest bar
        latest_row = df.iloc[-1]
        timestamp = df.index[-1]  # pandas Timestamp from index
        
        if hasattr(timestamp, 'to_pydatetime'):
            dt = timestamp.to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
        elif hasattr(timestamp, 'timestamp'):
            dt = datetime.fromtimestamp(timestamp.timestamp(), tz=timezone.utc)
        else:
            try:
                import pandas as pd
                dt = pd.to_datetime(timestamp).to_pydatetime()
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
            except:
                return None
        
        return create_canonical_candle(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=dt,
            open_price=float(latest_row['open']),
            high=float(latest_row['high']),
            low=float(latest_row['low']),
            close=float(latest_row['close']),
            volume=float(latest_row.get('volume', 0.0)) if 'volume' in latest_row else 0.0,
            source="alpaca",
            fallback_active=False,
        )
        
    except ImportError:
        # alpaca-py not installed - return None (will use fallback with original symbol)
        return None
    except Exception as exc:
        # Any other error - return None (will use fallback with original symbol)
        # Original symbol format is preserved, so fallback sources (Binance, etc.) can handle it
        return None


# ============================================================================
# DATA FETCHERS - Binance Historical (FALLBACK)
# ============================================================================

def fetch_binance_rest_historical(
    symbol: str,
    timeframe: str = "1m",
    years: int = 5,
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """
    Fetch historical data using Binance REST API.
    Raises exception on rate limit to trigger fallback.
    """
    binance_symbol = symbol.replace("-", "")
    
    ccxt_timeframes = {
        "1m": "1m", "5m": "5m", "15m": "15m",
        "1h": "1h", "4h": "4h", "1d": "1d"
    }
    ccxt_tf = ccxt_timeframes.get(timeframe, "1m")
    
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    all_candles = []
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = int((datetime.now(timezone.utc) - timedelta(days=years * 365)).timestamp() * 1000)
    current_time = start_time
    
    print(f"Fetching via REST API: {symbol} ({timeframe})")
    
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    while current_time < end_time:
        try:
            ohlcv = exchange.fetch_ohlcv(binance_symbol, ccxt_tf, since=current_time, limit=limit)
            
            if not ohlcv:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    break
                time.sleep(1)
                continue
            
            consecutive_errors = 0  # Reset on success
            
            batch_candles = []
            for candle_data in ohlcv:
                parsed = parse_binance_candle(candle_data)
                canonical = create_canonical_candle(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=parsed["timestamp"],
                    open_price=parsed["open"],
                    high=parsed["high"],
                    low=parsed["low"],
                    close=parsed["close"],
                    volume=parsed["volume"],
                    source="binance_rest"
                )
                batch_candles.append(canonical)
            
            all_candles.extend(batch_candles)
            last_timestamp = ohlcv[-1][0]
            if last_timestamp <= current_time:
                break
            current_time = last_timestamp + 1
            
            print(f"  Fetched {len(batch_candles)} candles, total: {len(all_candles)}")
            time.sleep(0.1)  # Rate limiting
            
        except ccxt.RateLimitExceeded as e:
            print(f"  Rate limit exceeded: {e}")
            raise Exception(f"429 Rate limit exceeded: {e}")
        except ccxt.ExchangeError as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower():
                print(f"  Rate limit detected: {e}")
                raise Exception(f"429 Rate limit: {e}")
            else:
                print(f"  Exchange error: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    raise
                time.sleep(2)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower():
                raise Exception(f"429 Rate limit: {e}")
            print(f"  Error fetching REST data: {e}")
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                break
            time.sleep(1)
    
    if not all_candles:
        raise Exception("No data fetched from Binance")
    
    valid_candles, errors = validate_candle_list(all_candles)
    if errors:
        print(f"  Validation errors: {len(errors)}")
    
    unique_candles = check_duplicate_candles(valid_candles, timeframe)
    print(f"  Total valid candles: {len(unique_candles)}")
    
    return unique_candles


def get_binance_current_price(symbol: str) -> Optional[float]:
    """
    Get current price for a crypto symbol from Binance REST API.
    
    Args:
        symbol: Data symbol (e.g., "BTC-USDT")
    
    Returns:
        Current price as float, or None if unavailable
    """
    try:
        binance_symbol = symbol.replace("-", "")
        
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        # Fetch ticker to get current price
        ticker = exchange.fetch_ticker(binance_symbol)
        if ticker and 'last' in ticker:
            return float(ticker['last'])
        elif ticker and 'close' in ticker:
            return float(ticker['close'])
        
        return None
    except Exception as e:
        # Silently fail - will fall back to other sources
        return None


def save_historical_data(
    symbol: str,
    timeframe: str,
    candles: List[Dict[str, Any]],
    source_hint: Optional[str] = None
):
    """Save all historical candles to a single JSON file per symbol/timeframe."""
    if not candles:
        return
    # Determine source folder from first candle if not provided
    if source_hint is None and candles:
        source_hint = candles[0].get("source", "binance")
    source_folder = source_hint or "binance"
    # Normalize: if source is "alpaca", use "alpaca" folder, otherwise "binance"
    if source_folder == "alpaca":
        source_folder = "alpaca"
    else:
        source_folder = "binance"  # All fallback sources go to binance folder
    # Sort candles by timestamp
    candles.sort(key=lambda x: x.get("timestamp", ""))
    
    # Get path for single data file (no date)
    base_path = get_data_path("crypto", symbol, timeframe, None, source_folder).parent
    data_file = base_path / "data.json"
    
    # Load existing data if file exists
    existing_candles = load_json_file(data_file) if data_file.exists() else []
    
    # Merge and deduplicate by timestamp
    all_candles = existing_candles + candles
    seen_timestamps = {}
    for candle in all_candles:
        ts = candle.get("timestamp", "")
        # Keep the one with higher volume if duplicate
        new_vol = candle.get("volume", 0) or 0
        prev_vol = seen_timestamps[ts].get("volume", 0) if ts in seen_timestamps else 0
        prev_vol = prev_vol or 0
        if ts not in seen_timestamps or new_vol > prev_vol:
            seen_timestamps[ts] = candle
    
    # Convert back to sorted list
    merged_candles = list(seen_timestamps.values())
    merged_candles.sort(key=lambda x: x.get("timestamp", ""))
    
    # Save to single file
    save_json_file(data_file, merged_candles, append=False)
    print(f"  Saved {len(merged_candles)} total candles to data.json (added {len(candles)} new) [{source_folder}]")

    generate_manifest("crypto", symbol, timeframe, source_folder)
    print(f"  Generated manifest for {symbol}/{timeframe} ({source_folder})")

    try:
        update_feature_store("crypto", symbol, timeframe, base_path)
    except Exception as exc:
        print(f"[FEATURE] Unable to update features for {symbol}/{timeframe}: {exc}")


# ============================================================================
# DATA FETCHERS - Binance Live
# ============================================================================

class BinanceWebSocketClient:
    """WebSocket client for Binance live data with automatic reconnection."""
    
    def __init__(
        self,
        symbol: str,
        timeframe: str = "1m",
        on_candle: Optional[Callable[[Dict[str, Any]], None]] = None,
        fallback_engine: Optional[FallbackEngine] = None
    ):
        self.symbol = symbol.replace("-", "").upper()
        self.timeframe = timeframe
        self.on_candle = on_candle
        self.fallback_engine = fallback_engine
        self.ws = None
        self.running = False
        self.reconnect_delay = config.WS_RECONNECT_DELAY
        self.last_heartbeat = time.time()
        
    def _get_stream_name(self) -> str:
        stream_map = {
            "1m": "1m", "5m": "5m", "15m": "15m",
            "1h": "1h", "4h": "4h", "1d": "1d"
        }
        tf = stream_map.get(self.timeframe, "1m")
        return f"{self.symbol.lower()}@kline_{tf}"
    
    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if "k" in data:
                kline = data["k"]
                if kline.get("x"):
                    self._process_closed_candle(kline)
        except Exception as e:
            print(f"Error processing WebSocket message: {e}")
    
    def _process_closed_candle(self, kline: Dict[str, Any]):
        try:
            open_time = int(kline["t"])
            open_price = float(kline["o"])
            high = float(kline["h"])
            low = float(kline["l"])
            close = float(kline["c"])
            volume = float(kline["v"])
            
            fallback_active = self.fallback_engine.is_fallback_active() if self.fallback_engine else False
            fallback_reason = self.fallback_engine.get_fallback_reason() if self.fallback_engine else None
            
            candle = create_canonical_candle(
                symbol=self.symbol,
                timeframe=self.timeframe,
                timestamp=open_time,
                open_price=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                source="binance_ws",
                fallback_active=fallback_active,
                fallback_reason=fallback_reason
            )
            
            is_valid, error = validate_candle(candle)
            if not is_valid:
                print(f"Invalid candle: {error}")
                return
            
            self.last_heartbeat = time.time()
            
            if self.fallback_engine:
                self.fallback_engine.mark_success("binance_ws")
            
            if self.on_candle:
                self.on_candle(candle)
                
        except Exception as e:
            print(f"Error processing closed candle: {e}")
    
    def _on_error(self, ws, error):
        print(f"WebSocket error: {error}")
        if self.fallback_engine:
            self.fallback_engine.mark_disconnected("binance_ws")
    
    def _on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket closed: {close_status_code} - {close_msg}")
        if self.fallback_engine:
            self.fallback_engine.mark_disconnected("binance_ws")
        
        if self.running:
            time.sleep(self.reconnect_delay)
            self._connect()
    
    def _on_open(self, ws):
        print(f"WebSocket connected for {self.symbol} ({self.timeframe})")
        if self.fallback_engine:
            self.fallback_engine.mark_success("binance_ws")
    
    def _connect(self):
        stream_name = self._get_stream_name()
        url = f"{config.BINANCE_WS_URL}/{stream_name}"
        
        if websocket is None:
            raise ImportError("websocket-client module is required for Binance WebSocket. Install with: pip install websocket-client")
        self.ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
    
    def start(self):
        self.running = True
        self._connect()
        
        def heartbeat_monitor():
            while self.running:
                time.sleep(config.WS_HEARTBEAT_INTERVAL)
                if time.time() - self.last_heartbeat > config.WS_TIMEOUT * 2:
                    print("WebSocket heartbeat timeout, reconnecting...")
                    if self.ws:
                        self.ws.close()
                    if self.fallback_engine:
                        self.fallback_engine.mark_stale("binance_ws", time.time() - self.last_heartbeat)
                    time.sleep(self.reconnect_delay)
                    if self.running:
                        self._connect()
        
        hbt = threading.Thread(target=heartbeat_monitor)
        hbt.daemon = True
        hbt.start()
    
    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()


def start_binance_live_feed(
    symbol: str,
    timeframe: str = "1m",
    fallback_engine: Optional[FallbackEngine] = None
):
    """Start live data feed for a symbol."""
    def on_candle(candle: Dict[str, Any]):
        write_candle_to_daily("crypto", symbol, timeframe, candle)
        print(f"  [{candle['timestamp']}] {symbol} {timeframe}: {candle['close']}")
    
    client = BinanceWebSocketClient(symbol, timeframe, on_candle, fallback_engine)
    client.start()
    return client


# ============================================================================
# DATA FETCHERS - Yahoo Commodities
# ============================================================================

def _download_yahoo_with_yfinance(
    yahoo_symbol: str,
    canonical_symbol: str,
    timeframe: str,
    years: int,
    session: requests.Session
) -> List[Dict[str, Any]]:
    period_map = {
        "1d": ("1y", "1d"),
        "1h": ("60d", "1h"),
        "1m": ("7d", "1m")
    }
    
    _, interval = period_map.get(timeframe, ("1y", "1d"))
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=years * 365)
    current_start = start_date
    all_candles: List[Dict[str, Any]] = []
    
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=365), end_date)
        chunk_fetched = False
        last_error: Optional[Exception] = None
        
        for attempt in range(3):
            try:
                # Suppress yfinance errors - these are optional fallback data sources
                import warnings
                import logging
                yf_logger = logging.getLogger("yfinance")
                old_level = yf_logger.level
                yf_logger.setLevel(logging.CRITICAL)  # Suppress ERROR and WARNING
                
                try:
                    hist = yf.download(
                        tickers=yahoo_symbol,
                        start=current_start.strftime("%Y-%m-%d"),
                        end=current_end.strftime("%Y-%m-%d"),
                        interval=interval,
                        auto_adjust=False,
                        progress=False,
                        group_by="ticker",
                        threads=False
                        # session removed - let yfinance handle it
                    )
                finally:
                    yf_logger.setLevel(old_level)  # Restore original log level

                if isinstance(hist, pd.DataFrame) and isinstance(hist.columns, pd.MultiIndex):
                    try:
                        hist = hist.xs(yahoo_symbol, level=0, axis=1)
                    except Exception:
                        pass

                if hist.empty:
                    raise ValueError("Empty response from Yahoo Finance")
                
                for idx, row in hist.iterrows():
                    if hasattr(idx, 'tz_localize'):
                        if idx.tz is None:
                            idx = idx.tz_localize('UTC')
                        else:
                            idx = idx.tz_convert('UTC')
                    
                    timestamp = int(idx.timestamp() * 1000) if hasattr(idx, 'timestamp') else int(time.mktime(idx.timetuple()) * 1000)
                    
                    candle = create_canonical_candle(
                        symbol=canonical_symbol,
                        timeframe=timeframe,
                        timestamp=timestamp,
                        open_price=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        volume=float(row['Volume']) if 'Volume' in row else 0.0,
                        source="yahoo"
                    )
                    all_candles.append(candle)
                
                print(f"  [Yahoo] {yahoo_symbol}: {len(hist)} candles from {current_start.date()} to {current_end.date()}, total: {len(all_candles)}")
                chunk_fetched = True
                break
            except Exception as e:
                last_error = e
                wait_time = min(2 ** attempt, 5)
                print(f"  Error fetching chunk {current_start.date()} (attempt {attempt + 1}) for {yahoo_symbol}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        if not chunk_fetched:
            raise last_error or Exception(f"Failed to fetch data for {yahoo_symbol}")
        
        current_start = current_end
        time.sleep(0.5)
    
    return all_candles


def _download_yahoo_with_chart_api(
    yahoo_symbol: str,
    canonical_symbol: str,
    timeframe: str,
    years: int,
    session: requests.Session
) -> List[Dict[str, Any]]:
    interval_map = {
        "1d": "1d",
        "1h": "1h",
        "1m": "1m"
    }
    interval = interval_map.get(timeframe, "1d")
    
    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=years * 365)).timestamp())
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": interval,
        "includeAdjustedClose": "true"
    }
    response = session.get(url, params=params, timeout=30)
    if response.status_code != 200:
        raise Exception(f"Yahoo chart API HTTP {response.status_code}")
    
    payload = response.json()
    result = payload.get("chart", {}).get("result")
    if not result:
        raise Exception("Yahoo chart API returned no result")
    
    data = result[0]
    timestamps = data.get("timestamp")
    quote = (data.get("indicators", {}) or {}).get("quote", [{}])[0]
    if not timestamps or not quote:
        raise Exception("Yahoo chart API missing quote data")
    
    opens = quote.get("open", [])
    highs = quote.get("high", [])
    lows = quote.get("low", [])
    closes = quote.get("close", [])
    volumes = quote.get("volume", [])
    
    candles: List[Dict[str, Any]] = []
    for idx, ts in enumerate(timestamps):
        try:
            open_price = opens[idx]
            high = highs[idx]
            low = lows[idx]
            close = closes[idx]
        except IndexError:
            continue
        
        if None in (open_price, high, low, close):
            continue
        
        volume = volumes[idx] if idx < len(volumes) else None
        candle = create_canonical_candle(
            symbol=canonical_symbol,
            timeframe=timeframe,
            timestamp=int(ts) * 1000,
            open_price=float(open_price),
            high=float(high),
            low=float(low),
            close=float(close),
            volume=float(volume) if volume is not None else 0.0,
            source="yahoo_chart"
        )
        candles.append(candle)
    
    if not candles:
        raise Exception("Yahoo chart API returned no candles")
    
    print(f"  [Yahoo Chart] {yahoo_symbol}: {len(candles)} candles fetched via chart API")
    return candles


def fetch_yahoo_historical(
    symbol: str,
    timeframe: str = "1d",
    years: int = 5
) -> List[Dict[str, Any]]:
    """Fetch historical data from Yahoo Finance."""
    print(f"Fetching Yahoo Finance historical data for {symbol} ({timeframe})")
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/119.0.0.0 Safari/537.36"
    })
    
    candidate_symbols = [symbol]
    if "=" not in symbol and not symbol.endswith("=F"):
        candidate_symbols.append(f"{symbol}=F")
    
    try:
        last_error: Optional[Exception] = None
        selected_source = "yahoo"
        candles: List[Dict[str, Any]] = []
        
        for yahoo_symbol in candidate_symbols:
            print(f"  Using Yahoo symbol: {yahoo_symbol}")
            try:
                candles = _download_yahoo_with_yfinance(
                    yahoo_symbol,
                    symbol,
                    timeframe,
                    years,
                    session
                )
                selected_source = "yahoo"
            except Exception as e:
                last_error = e
                print(f"  Yahoo Finance (yfinance) failed for {yahoo_symbol}: {e}")
                candles = []
            
            if not candles:
                try:
                    candles = _download_yahoo_with_chart_api(
                        yahoo_symbol,
                        symbol,
                        timeframe,
                        years,
                        session
                    )
                    selected_source = "yahoo_chart"
                except Exception as chart_err:
                    last_error = chart_err
                    print(f"  Yahoo chart API failed for {yahoo_symbol}: {chart_err}")
                    candles = []
            
            if candles:
                break
        
        if not candles:
            raise last_error or Exception("No data fetched from Yahoo Finance")
        
        valid_candles, errors = validate_candle_list(candles)
        if errors:
            print(f"  Validation errors: {len(errors)}")
        
        unique_candles = check_duplicate_candles(valid_candles, timeframe)
        print(f"  Total valid candles ({selected_source}): {len(unique_candles)}")
        
        return unique_candles
        
    except Exception as e:
        print(f"Error fetching Yahoo Finance data: {e}")
        try:
            print("Falling back to Stooq data feed...")
            return fetch_stooq_historical(symbol, timeframe=timeframe, years=years)
        except Exception as stooq_err:
            print(f"Stooq fallback also failed: {stooq_err}")
            return []


def save_yahoo_historical(
    symbol: str,
    timeframe: str,
    candles: List[Dict[str, Any]],
    source_hint: Optional[str] = None
):
    """Save all historical candles to a single JSON file per symbol/timeframe."""
    if not candles:
        return
    source_folder = "yahoo"  # Always use yahoo for paper trading (no angelone_mcx)
    # Sort candles by timestamp
    candles.sort(key=lambda x: x.get("timestamp", ""))
    
    # Get path for single data file (no date)
    base_path = get_data_path("commodities", symbol, timeframe, None, source_folder).parent
    data_file = base_path / "data.json"
    
    # Load existing data if file exists
    existing_candles = load_json_file(data_file) if data_file.exists() else []
    
    # Merge and deduplicate by timestamp
    all_candles = existing_candles + candles
    seen_timestamps = {}
    for candle in all_candles:
        ts = candle.get("timestamp", "")
        # Keep the one with higher volume if duplicate
        new_vol = candle.get("volume", 0) or 0
        prev_vol = seen_timestamps[ts].get("volume", 0) if ts in seen_timestamps else 0
        prev_vol = prev_vol or 0
        if ts not in seen_timestamps or new_vol > prev_vol:
            seen_timestamps[ts] = candle
    
    # Convert back to sorted list
    merged_candles = list(seen_timestamps.values())
    merged_candles.sort(key=lambda x: x.get("timestamp", ""))
    
    # Save to single file
    save_json_file(data_file, merged_candles, append=False)
    print(f"  Saved {len(merged_candles)} total candles to data.json (added {len(candles)} new) [{source_folder}]")

    generate_manifest("commodities", symbol, timeframe, source_folder)
    print(f"  Generated manifest for {symbol}/{timeframe} ({source_folder})")

    try:
        update_feature_store("commodities", symbol, timeframe, base_path)
    except Exception as exc:
        print(f"[FEATURE] Unable to update features for {symbol}/{timeframe}: {exc}")


def poll_alpaca_live(
    symbol: str,
    timeframe: str = "1d",
    poll_interval: int = 300,  # 5 minutes default
    fallback_engine: Optional[FallbackEngine] = None
):
    """
    Poll Alpaca Data API for live crypto data updates (PRIMARY SOURCE).
    
    This runs in a background thread and continuously updates data.json
    with the latest candles from Alpaca.
    """
    print(f"Starting Alpaca live polling for {symbol} ({timeframe})")
    last_timestamp = None
    
    while True:
        try:
            # Fetch latest candle from Alpaca
            latest_candle = fetch_alpaca_live(symbol, timeframe)
            
            if latest_candle:
                current_timestamp = latest_candle.get("timestamp")
                
                # Only save if this is a new candle (different timestamp)
                if last_timestamp is None or current_timestamp != last_timestamp:
                    is_valid, error = validate_candle(latest_candle)
                    if is_valid:
                        # Save to data.json
                        write_candle_to_daily("crypto", symbol, timeframe, latest_candle)
                        print(f"  [{current_timestamp}] {symbol} {timeframe}: ${latest_candle['close']:.2f} (Alpaca)")
                        last_timestamp = current_timestamp
                    else:
                        print(f"  [WARN] Invalid candle for {symbol}: {error}")
                else:
                    # Same timestamp, no update needed
                    pass
            else:
                # Alpaca failed, mark fallback if engine provided
                if fallback_engine:
                    fallback_engine.mark_failure("alpaca", "No data returned")
                    print(f"  [WARN] Alpaca returned no data for {symbol}, will try fallback on next cycle")
                    print(f"  [INFO] Original symbol '{symbol}' preserved for fallback sources")
            
        except ImportError as import_exc:
            # alpaca-py not installed
            error_msg = str(import_exc)
            print(f"  [ERROR] Alpaca import failed for {symbol}: {error_msg}")
            print(f"  [INFO] Original symbol '{symbol}' preserved for fallback")
            if fallback_engine:
                fallback_engine.mark_failure("alpaca", error_msg)
        except Exception as exc:
            # Any other error (API error, network, etc.)
            error_msg = str(exc)
            print(f"  [ERROR] Alpaca polling error for {symbol}: {error_msg}")
            print(f"  [INFO] Original symbol '{symbol}' preserved for fallback")
            if fallback_engine:
                fallback_engine.mark_failure("alpaca", error_msg)
        
        # Wait before next poll
        time.sleep(poll_interval)


def poll_yahoo_live(
    symbol: str,
    timeframe: str = "1d",
    poll_interval: int = 60,
    fallback_engine: Optional[FallbackEngine] = None
):
    """Poll Yahoo Finance for live data updates (FALLBACK)."""
    print(f"Starting Yahoo Finance live polling for {symbol} ({timeframe})")
    
    # Suppress yfinance errors - this is a fallback source
    import warnings
    import logging
    yf_logger = logging.getLogger("yfinance")
    old_level = yf_logger.level
    yf_logger.setLevel(logging.CRITICAL)  # Suppress ERROR and WARNING
    
    try:
        ticker = yf.Ticker(symbol)
    finally:
        yf_logger.setLevel(old_level)  # Restore original log level
    last_close = None
    
    while True:
        try:
            hist = ticker.history(period="1d", interval=timeframe)
            
            if not hist.empty:
                latest = hist.iloc[-1]
                idx = hist.index[-1]
                if hasattr(idx, 'tz_localize'):
                    if idx.tz is None:
                        idx = idx.tz_localize('UTC')
                    else:
                        idx = idx.tz_convert('UTC')
                
                timestamp = int(idx.timestamp() * 1000) if hasattr(idx, 'timestamp') else int(time.mktime(idx.timetuple()) * 1000)
                current_close = float(latest['Close'])
                
                if last_close is None or current_close != last_close:
                    fallback_active = fallback_engine.is_fallback_active() if fallback_engine else False
                    fallback_reason = fallback_engine.get_fallback_reason() if fallback_engine else None
                    
                    candle = create_canonical_candle(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=timestamp,
                        open_price=float(latest['Open']),
                        high=float(latest['High']),
                        low=float(latest['Low']),
                        close=current_close,
                        volume=float(latest['Volume']) if 'Volume' in latest else 0.0,
                        source="yahoo",
                        fallback_active=fallback_active,
                        fallback_reason=fallback_reason
                    )
                    
                    is_valid, error = validate_candle(candle)
                    if is_valid:
                        write_candle_to_daily("commodities", symbol, timeframe, candle)
                        print(f"  [{candle['timestamp']}] {symbol} {timeframe}: {candle['close']}")
                        
                        if fallback_engine:
                            fallback_engine.mark_success("yahoo")
                    else:
                        print(f"  Invalid candle: {error}")
                    
                    last_close = current_close
            else:
                if fallback_engine:
                    fallback_engine.mark_failure("yahoo", "No data returned")
            
        except Exception as e:
            print(f"  Error polling Yahoo Finance: {e}")
            if fallback_engine:
                fallback_engine.mark_failure("yahoo", str(e))
        
        time.sleep(poll_interval)


# ============================================================================
# DATA FETCHERS - Stooq Commodities
# ============================================================================

def _map_to_stooq_symbol(symbol: str) -> str:
    if symbol.endswith("=F"):
        symbol = symbol[:-2]
    normalized = symbol.replace("-", "").replace("/", "").lower()
    return f"{normalized}.f"


def fetch_stooq_historical(
    symbol: str,
    timeframe: str = "1d",
    years: int = 5
) -> List[Dict[str, Any]]:
    """Fetch historical commodity data from Stooq (CSV, free)."""
    if timeframe != "1d":
        raise ValueError("Stooq only supports daily timeframe")
    
    stooq_symbol = _map_to_stooq_symbol(symbol)
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code != 200 or not response.text.strip():
        raise Exception(f"Stooq request failed ({response.status_code})")
    
    reader = csv.DictReader(response.text.strip().splitlines())
    candles: List[Dict[str, Any]] = []
    min_date = datetime.now(timezone.utc) - timedelta(days=years * 365)
    
    for row in reader:
        date_str = row.get("Date")
        if not date_str or row.get("Open") in (None, ""):
            continue
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        
        if dt < min_date:
            continue
        
        candle = create_canonical_candle(
            symbol=symbol,
            timeframe="1d",
            timestamp=int(dt.timestamp() * 1000),
            open_price=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]) if row.get("Volume") not in (None, "", "-") else 0.0,
            source="stooq"
        )
        candles.append(candle)
    
    if not candles:
        raise Exception(f"No data returned from Stooq for {stooq_symbol}")
    
    print(f"  [Stooq] {stooq_symbol}: {len(candles)} candles fetched")
    return candles


# ============================================================================
# DATA FETCHERS - Fallback Sources
# ============================================================================

def fetch_coinbase_historical(symbol: str, timeframe: str, years: int = 5) -> List[Dict[str, Any]]:
    """Fetch historical data from Coinbase (FREE source)."""
    try:
        exchange = ccxt.coinbase({'enableRateLimit': True})
        coinbase_symbol = symbol.replace("-", "/")
        
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=years * 365)).timestamp() * 1000)
        
        all_candles = []
        current_time = start_time
        consecutive_errors = 0
        
        while current_time < end_time:
            try:
                ohlcv = exchange.fetch_ohlcv(coinbase_symbol, timeframe, since=current_time, limit=1000)
                if not ohlcv:
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        break
                    time.sleep(1)
                    continue
                
                consecutive_errors = 0
                
                for candle_data in ohlcv:
                    candle = create_canonical_candle(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=candle_data[0],
                        open_price=candle_data[1],
                        high=candle_data[2],
                        low=candle_data[3],
                        close=candle_data[4],
                        volume=candle_data[5],
                        source="coinbase"
                    )
                    all_candles.append(candle)
                
                current_time = ohlcv[-1][0] + 1
                time.sleep(0.1)
            except ccxt.RateLimitExceeded:
                raise Exception("429 Rate limit exceeded on Coinbase")
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    raise Exception(f"429 Rate limit: {e}")
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    break
                time.sleep(1)
        
        if not all_candles:
            raise Exception("No data fetched from Coinbase")
        
        return all_candles
    except Exception as e:
        print(f"Coinbase fetch error: {e}")
        raise  # Re-raise to trigger fallback


def fetch_kucoin_historical(symbol: str, timeframe: str, years: int = 5) -> List[Dict[str, Any]]:
    """Fetch historical data from KuCoin (FREE source)."""
    try:
        exchange = ccxt.kucoin({'enableRateLimit': True})
        kucoin_symbol = symbol.replace("-", "-")
        
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=years * 365)).timestamp() * 1000)
        
        all_candles = []
        current_time = start_time
        consecutive_errors = 0
        
        while current_time < end_time:
            try:
                ohlcv = exchange.fetch_ohlcv(kucoin_symbol, timeframe, since=current_time, limit=1000)
                if not ohlcv:
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        break
                    time.sleep(1)
                    continue
                
                consecutive_errors = 0
                
                for candle_data in ohlcv:
                    candle = create_canonical_candle(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=candle_data[0],
                        open_price=candle_data[1],
                        high=candle_data[2],
                        low=candle_data[3],
                        close=candle_data[4],
                        volume=candle_data[5],
                        source="kucoin"
                    )
                    all_candles.append(candle)
                
                current_time = ohlcv[-1][0] + 1
                time.sleep(0.1)
            except ccxt.RateLimitExceeded:
                raise Exception("429 Rate limit exceeded on KuCoin")
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    raise Exception(f"429 Rate limit: {e}")
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    break
                time.sleep(1)
        
        if not all_candles:
            raise Exception("No data fetched from KuCoin")
        
        return all_candles
    except Exception as e:
        print(f"KuCoin fetch error: {e}")
        raise  # Re-raise to trigger fallback


def fetch_okx_historical(symbol: str, timeframe: str, years: int = 5) -> List[Dict[str, Any]]:
    """Fetch historical data from OKX (FREE source)."""
    try:
        exchange = ccxt.okx({'enableRateLimit': True})
        okx_symbol = symbol.replace("-", "-")
        
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=years * 365)).timestamp() * 1000)
        
        all_candles = []
        current_time = start_time
        consecutive_errors = 0
        
        while current_time < end_time:
            try:
                ohlcv = exchange.fetch_ohlcv(okx_symbol, timeframe, since=current_time, limit=1000)
                if not ohlcv:
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        break
                    time.sleep(1)
                    continue
                
                consecutive_errors = 0
                
                for candle_data in ohlcv:
                    candle = create_canonical_candle(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=candle_data[0],
                        open_price=candle_data[1],
                        high=candle_data[2],
                        low=candle_data[3],
                        close=candle_data[4],
                        volume=candle_data[5],
                        source="okx"
                    )
                    all_candles.append(candle)
                
                current_time = ohlcv[-1][0] + 1
                time.sleep(0.1)
            except ccxt.RateLimitExceeded:
                raise Exception("429 Rate limit exceeded on OKX")
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    raise Exception(f"429 Rate limit: {e}")
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    break
                time.sleep(1)
        
        if not all_candles:
            raise Exception("No data fetched from OKX")
        
        return all_candles
    except Exception as e:
        print(f"OKX fetch error: {e}")
        raise  # Re-raise to trigger fallback


def load_from_local_cache(
    symbol: str,
    timeframe: str,
    asset_type_hint: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Load data from local JSON cache when all sources fail."""
    if asset_type_hint == "crypto":
        search_order = ["crypto"]
    elif asset_type_hint == "commodities":
        search_order = ["commodities"]
    else:
        search_order = ["crypto", "commodities"]
    
    for asset_type in search_order:
        asset_root = config.BASE_DATA_DIR / asset_type
        if not asset_root.exists():
            continue
        for source_dir in asset_root.iterdir():
            if not source_dir.is_dir():
                continue
            data_file = source_dir / symbol / timeframe / "data.json"
            if data_file.exists():
                candles = load_json_file(data_file)
                if candles:
                    return candles
    
    return []


# ============================================================================
# INGESTION ORCHESTRATION
# ============================================================================

def fetch_crypto_historical_with_fallback(
    symbol: str,
    timeframe: str = "1d",
    years: float = 5,
    fallback_engine: Optional[FallbackEngine] = None,
    incremental: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch crypto historical data with automatic fallback between free sources.
    
    PRIMARY: Alpaca (free, no keys required for crypto data)
    FALLBACK: Binance, Coinbase, KuCoin, OKX, local cache
    
    Args:
        symbol: Data symbol (e.g., BTC-USDT)
        timeframe: Timeframe (1d, 1h, etc.)
        years: Number of years to fetch (used only if no existing data found)
        fallback_engine: Optional fallback engine for source switching
        incremental: If True, check for existing data and only fetch new data
    """
    if fallback_engine is None:
        # PRIORITY: Yahoo Finance and Binance have better altcoin coverage than Alpaca
        # Alpaca only supports major coins (BTC, ETH, LTC, BCH)
        sources = ["yahoo", "binance_rest", "alpaca", "coinbase", "kucoin", "okx", "local_cache"]
        fallback_engine = FallbackEngine(sources, "yahoo")
    
    # Check for existing data if incremental mode is enabled
    incremental_start_date = None
    if incremental:
        last_timestamp = get_last_timestamp_from_existing_data("crypto", symbol, timeframe)
        if last_timestamp:
            # Use last timestamp as start date for incremental fetch
            incremental_start_date = last_timestamp
            print(f"[{symbol}] Found existing data, last timestamp: {last_timestamp.isoformat()}")
            print(f"[{symbol}] Will fetch incremental data from {last_timestamp.date()} to now")
        else:
            print(f"[{symbol}] No existing data found, will fetch full {years} years")
    
    all_candles = []
    max_attempts = len(fallback_engine.sources) * 2
    
    for attempt in range(max_attempts):
        source = fallback_engine.get_current_source()
        
        try:
            if incremental_start_date:
                print(f"[{symbol}] Fetching incremental data from {source} (attempt {attempt + 1})...")
            else:
                print(f"[{symbol}] Fetching historical from {source} (attempt {attempt + 1})...")
            
            if source == "alpaca":
                # Pass incremental_start_date if available
                candles = fetch_alpaca_historical(
                    symbol, 
                    timeframe, 
                    years, 
                    start_date=incremental_start_date
                )
                
                # If Alpaca returned 0 candles for daily data in incremental mode,
                # use live price API to update the last candle instead of falling back to Binance
                if len(candles) == 0 and incremental_start_date and timeframe == "1d":
                    print(f"  [INFO] Alpaca returned 0 daily candles (day not complete yet)")
                    print(f"  [INFO] Using Alpaca live price API to update last candle instead of falling back to Binance")
                    
                    try:
                        # Get trading symbol for Alpaca
                        from trading.symbol_universe import find_by_data_symbol
                        from trading.alpaca_client import AlpacaClient
                        from pathlib import Path
                        
                        asset_mapping = find_by_data_symbol(symbol)
                        if asset_mapping:
                            client = AlpacaClient()
                            last_trade = client.get_last_trade(asset_mapping.trading_symbol)
                            
                            if last_trade:
                                live_price = last_trade.get("price") or last_trade.get("p")
                                if live_price:
                                    live_price = float(live_price)
                                    
                                    # Load existing data.json and update last candle
                                    data_paths = [
                                        get_data_path("crypto", symbol, timeframe, None, "alpaca").parent / "data.json",
                                        get_data_path("crypto", symbol, timeframe, None, "binance").parent / "data.json",
                                    ]
                                    
                                    data_file = None
                                    for path in data_paths:
                                        if path.exists():
                                            data_file = path
                                            break
                                    
                                    if data_file and data_file.exists():
                                        existing_candles = load_json_file(data_file)
                                        if existing_candles:
                                            # Update the last candle's close price with live price
                                            last_candle = existing_candles[-1].copy()
                                            last_candle["close"] = live_price
                                            if live_price > last_candle.get("high", 0):
                                                last_candle["high"] = live_price
                                            if live_price < last_candle.get("low", float("inf")) or last_candle.get("low", 0) == 0:
                                                last_candle["low"] = live_price
                                            last_candle["source"] = last_candle.get("source", "alpaca")
                                            last_candle["live_updated"] = True
                                            
                                            # Replace last candle and save
                                            existing_candles[-1] = last_candle
                                            save_json_file(data_file, existing_candles, append=False)
                                            
                                            print(f"  [OK] Updated last candle with live price ${live_price:.2f} from Alpaca")
                                            # Mark as success and return empty (data already updated in place)
                                            # DO NOT fallback to Binance - we've updated with Alpaca live price
                                            fallback_engine.mark_success(source)
                                            all_candles = []
                                            break
                                        else:
                                            print(f"  [WARN] No existing candles found to update")
                                    else:
                                        print(f"  [WARN] No existing data.json found to update")
                                else:
                                    print(f"  [WARN] No price in Alpaca last_trade response")
                            else:
                                print(f"  [WARN] Could not get live price from Alpaca")
                        else:
                            print(f"  [WARN] Could not find asset mapping for {symbol}")
                    except Exception as live_exc:
                        print(f"  [WARN] Failed to update with live price: {live_exc}")
                        # Live update failed: now allowed to fallback to Binance (or other sources)
                        # Mark Alpaca as failed so fallback_engine will rotate to the next source.
                        fallback_engine.mark_failure(source, "No data returned and live price update failed")
            elif source == "binance_rest":
                # Binance doesn't support start_date parameter yet, but save_historical_data will merge correctly
                candles = fetch_binance_rest_historical(symbol, timeframe, years)
                # If we're doing incremental and got candles, filter to only new ones
                if incremental_start_date and candles:
                    # Filter candles to only include those after the last timestamp
                    filtered_candles = []
                    for candle in candles:
                        try:
                            ts_str = candle.get("timestamp", "")
                            if ts_str:
                                if ts_str.endswith("Z"):
                                    ts_str = ts_str[:-1] + "+00:00"
                                candle_dt = datetime.fromisoformat(ts_str)
                                if candle_dt.tzinfo is None:
                                    candle_dt = candle_dt.replace(tzinfo=timezone.utc)
                                else:
                                    candle_dt = candle_dt.astimezone(timezone.utc)
                                # Only include candles after the last timestamp (with 1 day buffer for overlap)
                                if candle_dt > (incremental_start_date - timedelta(days=1)):
                                    filtered_candles.append(candle)
                        except Exception:
                            # If timestamp parsing fails, include the candle (better safe than sorry)
                            filtered_candles.append(candle)
                    candles = filtered_candles
                    if candles:
                        print(f"  [INFO] Filtered to {len(candles)} new candles (after {incremental_start_date.date()})")
            elif source == "coinbase":
                candles = fetch_coinbase_historical(symbol, timeframe, years)
            elif source == "kucoin":
                candles = fetch_kucoin_historical(symbol, timeframe, years)
            elif source == "okx":
                candles = fetch_okx_historical(symbol, timeframe, years)
            elif source == "local_cache":
                candles = load_from_local_cache(symbol, timeframe, "crypto")
            else:
                candles = []
            
            if candles and len(candles) > 0:
                fallback_engine.mark_success(source)
                all_candles = candles
                print(f"  [OK] Successfully fetched {len(candles)} candles from {source}")
                break
            else:
                fallback_engine.mark_failure(source, "No data returned")
                
        except Exception as exc:
            error_msg = str(exc)
            status_code = None
            if "429" in error_msg or "rate limit" in error_msg.lower():
                status_code = 429
            elif any(code in error_msg for code in ["503", "502", "500"]):
                status_code = 500
            
            print(f"  [ERROR] {source}: {error_msg}")
            fallback_engine.mark_failure(source, error_msg, status_code)
        
        if attempt < max_attempts - 1:
            wait_time = min(2 ** (attempt % 3), 10)
            print(f"  Waiting {wait_time}s before trying next source...")
            time.sleep(wait_time)
    
    return all_candles


def fetch_commodities_historical_with_fallback(
    symbol: str,
    timeframe: str = "1d",
    years: float = 7,
    fallback_engine: Optional[FallbackEngine] = None,
    incremental: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch commodities historical data with automatic fallback.
    
    UPDATED: Now uses AngelOne (MCX) for ALL commodities (both MCX_* and Yahoo Finance symbols).
    This ensures consistency - we trade on MCX, so we use MCX data for historical analysis too.
    
    Priority:
    1. AngelOne MCX API (primary - for all commodities)
    2. Yahoo Finance (fallback - only if AngelOne fails and symbol has Yahoo equivalent)
    3. Local cache (final fallback)
    
    Args:
        symbol: Data symbol (e.g., "GC=F" or "MCX_GOLDM")
        timeframe: Timeframe (1d, 1h, etc.)
        years: Number of years to fetch (used only if no existing data found)
        fallback_engine: Optional fallback engine for source switching
        incremental: If True, check for existing data and only fetch if missing or incomplete
    """
    timeframe = "1d"
    
    # Check for existing data if incremental mode is enabled
    if incremental:
        # Check if data.json already exists and has sufficient data
        from pathlib import Path
        
        # Try multiple possible source folders for commodities (ONLY yahoo for paper trading)
        possible_sources = ["yahoo"]  # Removed angelone_mcx, stooq, yahoo_chart
        existing_data_found = False
        existing_candles = []
        
        for source_folder in possible_sources:
            try:
                # data.json is stored in the timeframe directory, not as latest.json
                base_path = get_data_path("commodities", symbol, timeframe, None, source_folder).parent
                data_path = base_path / "data.json"
                if data_path.exists():
                    existing_candles = load_json_file(data_path)
                    if existing_candles and len(existing_candles) > 0:
                        # Check if we have sufficient data (at least 1 year worth)
                        min_required_candles = 365  # At least 1 year of daily data
                        if len(existing_candles) >= min_required_candles:
                            existing_data_found = True
                            print(f"[{symbol}] Found existing data ({len(existing_candles)} candles) - skipping fetch")
                            print(f"[{symbol}] Using existing data from {source_folder}")
                            return existing_candles  # Return existing data, don't fetch again
                        else:
                            print(f"[{symbol}] Existing data found but insufficient ({len(existing_candles)} candles < {min_required_candles}) - will fetch more")
                            break  # Continue to fetch more data
            except Exception:
                continue  # Try next source
        
        if existing_data_found:
            # Data exists and is sufficient - return it
            return existing_candles
    
    # PAPER TRADING MODE: SKIP Angel One MCX API completely
    # Use ONLY Yahoo Finance for all commodities to avoid static IP errors
    # This is for local paper trading - no MCX API needed
    
    # Check if this symbol maps to an MCX trading symbol
    from trading.symbol_universe import find_by_data_symbol
    asset_mapping = find_by_data_symbol(symbol)
    is_mcx_symbol = symbol.startswith("MCX_")
    maps_to_mcx = asset_mapping and asset_mapping.asset_type == "commodities" and asset_mapping.trading_symbol
    
    # Try Angel One MCX API if enabled (requires static IP whitelisting)
    if USE_ANGELONE_MCX and (is_mcx_symbol or maps_to_mcx):
        # Try MCX data FIRST (for both MCX_* symbols and Yahoo Finance symbols that map to MCX)
        try:
            from trading.angelone_client import AngelOneClient
            from trading.mcx_symbol_mapper import get_mcx_contract_symbol
            
            # Get asset mapping to find MCX contract symbol
            if not asset_mapping:
                asset_mapping = find_by_data_symbol(symbol)
            
            if asset_mapping and asset_mapping.asset_type == "commodities":
                # Get MCX contract symbol (e.g., MCX_SILVERM -> SILVERMDEC25)
                mcx_contract = get_mcx_contract_symbol(asset_mapping.data_symbol)
                
                print(f"[{symbol}] Using AngelOne MCX API for historical data (MCX contract: {mcx_contract})")
                
                # Initialize AngelOne client
                client = AngelOneClient()
                
                # Fetch historical candles
                candles = client.get_historical_candles(
                    symbol=mcx_contract,
                    timeframe=timeframe,
                    years=years
                )
                
                if candles and len(candles) > 0:
                    # Convert to canonical format with symbol name
                    canonical_candles = []
                    for candle in candles:
                        canonical = create_canonical_candle(
                            symbol=symbol,  # Use original data_symbol (MCX_SILVERM)
                            timeframe=timeframe,
                            timestamp=candle["timestamp"],
                            open_price=candle["open"],
                            high=candle["high"],
                            low=candle["low"],
                            close=candle["close"],
                            volume=candle.get("volume", 0),
                            source="yahoo"  # Changed from angelone_mcx - paper trading uses Yahoo only
                        )
                        canonical_candles.append(canonical)
                    
                    print(f"  [OK] Successfully fetched {len(canonical_candles)} candles from AngelOne MCX")
                    return canonical_candles
                else:
                    print(f"  [WARNING] AngelOne returned no data for {mcx_contract}")
                    print(f"  [INFO] This might mean: (1) Contract expired, (2) Symbol token not found, or (3) No data for date range")
            else:
                print(f"  [WARNING] Asset mapping not found or not a commodity: {symbol}")
                
        except ImportError:
            print(f"  [WARNING] AngelOneClient not available")
        except Exception as angel_exc:
            print(f"  [WARNING] AngelOne fetch failed: {angel_exc}")
        
        # Fallback: Try local cache, then Yahoo Finance (if symbol has Yahoo equivalent)
        try:
            candles = load_from_local_cache(symbol, timeframe, "commodities")
            if candles and len(candles) > 0:
                print(f"  [OK] Found {len(candles)} candles in local cache")
                return candles
        except Exception as cache_exc:
            print(f"  [WARNING] Local cache check failed: {cache_exc}")
        
        # If MCX data failed and this is a Yahoo Finance symbol, fallback to Yahoo Finance
        if not is_mcx_symbol:
            print(f"  [INFO] MCX data unavailable, falling back to Yahoo Finance (COMEX/NYMEX) for {symbol}")
            # Continue to Yahoo Finance fallback below
        else:
            # MCX_* symbols: No Yahoo Finance fallback (they don't exist there)
            print(f"  [WARNING] No data available for {symbol} from AngelOne or local cache")
            return []
    
    # Yahoo Finance fallback (only for Yahoo Finance symbols that failed MCX fetch)
    if not is_mcx_symbol:
        # Yahoo Finance symbols (GC=F, CL=F, etc.): Use Yahoo Finance as fallback
        print(f"[{symbol}] Using Yahoo Finance for historical data (fallback - MCX data unavailable)")
        
        if fallback_engine is None:
            sources = ["yahoo", "stooq", "local_cache"]
            fallback_engine = FallbackEngine(sources, "yahoo")
        
        all_candles = []
        max_attempts = len(fallback_engine.sources) * 2
        
        for attempt in range(max_attempts):
            source = fallback_engine.get_current_source()
            
            try:
                print(f"[{symbol}] Fetching historical from {source} (attempt {attempt + 1})...")
                
                if source == "yahoo":
                    candles = fetch_yahoo_historical(symbol, timeframe, years)
                elif source == "stooq":
                    candles = fetch_stooq_historical(symbol, timeframe, years)
                elif source == "local_cache":
                    candles = load_from_local_cache(symbol, timeframe, "commodities")
                else:
                    candles = []
                
                if candles and len(candles) > 0:
                    fallback_engine.mark_success(source)
                    all_candles = candles
                    print(f"  [OK] Successfully fetched {len(candles)} candles from {source}")
                    break
                else:
                    fallback_engine.mark_failure(source, "No data returned")
                    
            except Exception as exc:
                error_msg = str(exc)
                status_code = None
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    status_code = 429
                
                print(f"  [ERROR] {source}: {error_msg}")
                fallback_engine.mark_failure(source, error_msg, status_code)
            
            if attempt < max_attempts - 1:
                wait_time = min(2 ** (attempt % 3), 10)
                print(f"  Waiting {wait_time}s before trying next source...")
                time.sleep(wait_time)
        
        return all_candles


def ingest_all_historical(
    crypto_symbols: Optional[List[str]] = None,
    commodities_symbols: Optional[List[str]] = None,
    timeframe: str = "1d",
    years: float = 7
):
    """
    Ingest historical data for all configured symbols with automatic fallback.
    """
    # Only use defaults if None, but respect empty list []
    crypto_symbols = config.CRYPTO_SYMBOLS if crypto_symbols is None else crypto_symbols
    commodities_symbols = config.COMMODITIES_SYMBOLS if commodities_symbols is None else commodities_symbols
    
    print("=" * 80)
    print("PHASE 1: HISTORICAL DATA INGESTION")
    print("=" * 80)
    print(f"Fetching {years} years of DAILY (1d) price data for all symbols")
    print("All sources are FREE. Automatic fallback ensures uninterrupted data flow.")
    print("=" * 80)
    
    crypto_fallback = FallbackEngine(
        ["binance_rest", "alpaca", "coinbase", "kucoin", "okx", "local_cache"],
        "binance_rest"  # PRIMARY: Binance (excellent altcoin coverage, already implemented)
    )
    commodities_fallback = FallbackEngine(
        ["yahoo", "local_cache"],
        "yahoo"
    )
    
    print(f"\n[CRYPTO] Ingesting {len(crypto_symbols)} symbols...")
    print("  Using incremental fetching: will only fetch new data if existing data found")
    for idx, symbol in enumerate(crypto_symbols, 1):
        try:
            print(f"\n[{idx}/{len(crypto_symbols)}] {symbol} ({timeframe})")
            candles = fetch_crypto_historical_with_fallback(
                symbol, 
                timeframe, 
                years, 
                crypto_fallback,
                incremental=True,  # Enable incremental fetching
            )
            if candles:
                # Determine source from first candle
                source_hint = candles[0].get("source", "binance") if candles else "binance"
                save_historical_data(symbol, timeframe, candles, source_hint=source_hint)
                print(f"  [SUCCESS] {symbol} complete: {len(candles)} candles saved (source: {source_hint})")
            else:
                print(f"  [WARNING] No data fetched for {symbol}")
        except Exception as exc:
            print(f"  [ERROR] Failed to ingest {symbol}: {exc}")
            import traceback
            traceback.print_exc()
    
    print(f"\n[COMMODITIES] Ingesting {len(commodities_symbols)} symbols...")
    print("  Using incremental fetching: will only fetch new data if existing data found")
    for idx, symbol in enumerate(commodities_symbols, 1):
        try:
            print(f"\n[{idx}/{len(commodities_symbols)}] {symbol} ({timeframe})")
            candles = fetch_commodities_historical_with_fallback(
                symbol, timeframe, years, commodities_fallback, incremental=True
            )
            if candles:
                save_yahoo_historical(symbol, timeframe, candles)
                print(f"  [SUCCESS] {symbol} complete: {len(candles)} candles saved")
            else:
                print(f"  [WARNING] No data fetched for {symbol}")
        except Exception as exc:
            print(f"  [ERROR] Failed to ingest {symbol}: {exc}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("HISTORICAL DATA INGESTION COMPLETE")
    print("=" * 80)


def start_live_feeds_with_fallback(
    crypto_symbols: Optional[List[str]] = None,
    commodities_symbols: Optional[List[str]] = None,
    crypto_timeframe: str = "1d",
    commodities_timeframe: str = "1d"
):
    """
    Start live data feeds with automatic fallback.
    """
    # Only use defaults if None, but respect empty list []
    crypto_symbols = config.CRYPTO_SYMBOLS if crypto_symbols is None else crypto_symbols
    commodities_symbols = config.COMMODITIES_SYMBOLS if commodities_symbols is None else commodities_symbols
    
    print("\n" + "=" * 80)
    print("PHASE 2: LIVE DATA FEEDS")
    print("=" * 80)
    print("All sources are FREE. Automatic fallback ensures uninterrupted data flow.")
    print("=" * 80)
    
    crypto_clients = []
    crypto_fallbacks: Dict[str, FallbackEngine] = {}
    
    for symbol in crypto_symbols:
        try:
            print(f"\nStarting live feed for {symbol} ({crypto_timeframe})...")
            # PRIMARY: Alpaca polling (free, no keys required for crypto data)
            # FALLBACK: Binance websocket, Coinbase, KuCoin, OKX
            sources = ["alpaca", "binance_ws", "coinbase", "kucoin", "okx"]
            fallback = FallbackEngine(sources, "alpaca")
            crypto_fallbacks[symbol] = fallback
            
            # Try Alpaca first (polling-based, simpler and more reliable)
            try:
                thread = threading.Thread(
                    target=poll_alpaca_live,
                    args=(symbol, crypto_timeframe, 300, fallback),  # Poll every 5 minutes
                    daemon=True
                )
                thread.start()
                crypto_clients.append((symbol, thread))
                print(f"  [OK] {symbol} Alpaca live polling started (PRIMARY)")
            except Exception as alpaca_exc:
                print(f"  [WARN] Alpaca polling failed: {alpaca_exc}, trying Binance fallback...")
                # Fallback to Binance websocket
                client = start_binance_live_feed(symbol, crypto_timeframe, fallback)
                crypto_clients.append((symbol, client))
                print(f"  [OK] {symbol} Binance live feed started (FALLBACK)")
        except Exception as exc:
            print(f"  [ERROR] Failed to start live feed for {symbol}: {exc}")
    
    commodity_threads = []
    
    for symbol in commodities_symbols:
        try:
            print(f"\nStarting live polling for {symbol} ({commodities_timeframe})...")
            sources = ["yahoo"]
            fallback = FallbackEngine(sources, "yahoo")
            
            thread = threading.Thread(
                target=poll_yahoo_live,
                args=(symbol, commodities_timeframe, 60, fallback),
                daemon=True
            )
            thread.start()
            commodity_threads.append((symbol, thread))
            print(f"  [OK] {symbol} live polling started")
        except Exception as exc:
            print(f"  [ERROR] Failed to start polling for {symbol}: {exc}")
    
    print("\n" + "=" * 80)
    print("LIVE DATA FEEDS RUNNING")
    print("=" * 80)
    print("All feeds are active with automatic fallback protection.")
    print("Data is being saved continuously to ensure uninterrupted model training.")
    
    # Show monitoring info for single symbol
    if len(crypto_symbols) == 1 and not commodities_symbols:
        symbol = list(crypto_symbols)[0]
        # Check for any available horizon profiles
        horizon_dirs = list_horizon_dirs("crypto", symbol, crypto_timeframe)
        found_summaries = []
        for horizon_dir_path in horizon_dirs:
            summary_file = horizon_dir_path / "summary.json"
            if summary_file.exists():
                horizon_name = horizon_dir_path.name
                found_summaries.append((horizon_name, summary_file))
        
        if found_summaries:
            print(f"\n📊 Monitoring {symbol}:")
            print(f"   - Live candles: data/json/raw/crypto/binance/{symbol}/{crypto_timeframe}/")
            for horizon_name, summary_path in found_summaries:
                print(f"   - ✓ Model trained ({horizon_name}): {summary_path}")
            print(f"   - Summary.json will update automatically with new predictions")
        else:
            print(f"\n📊 Monitoring {symbol}:")
            print(f"   - Live candles: data/json/raw/crypto/binance/{symbol}/{crypto_timeframe}/")
            print(f"   - ⚠️  Model not trained yet. Train model to see summary.json updates.")
    
    print("\nPress Ctrl+C to stop")
    print("=" * 80)
    
    # Monitor summary.json updates for single symbol
    last_summary_state = {}
    if len(crypto_symbols) == 1 and not commodities_symbols:
        symbol = list(crypto_symbols)[0]
        # Check for any available horizon profiles (prefer default, but use any if available)
        horizon_dirs = list_horizon_dirs("crypto", symbol, crypto_timeframe)
        summary_path = None
        # Try default first
        default_path = build_summary_path("crypto", symbol, crypto_timeframe, DEFAULT_HORIZON_PROFILE)
        if default_path.exists():
            summary_path = default_path
        else:
            # Use first available horizon
            for horizon_dir_path in horizon_dirs:
                candidate = horizon_dir_path / "summary.json"
                if candidate.exists():
                    summary_path = candidate
                    break
        
        if summary_path and summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                    # Use new format if available
                    if "prediction" in summary:
                        pred = summary["prediction"]
                        last_summary_state[symbol] = (
                            pred.get("current_price"),
                            pred.get("last_updated")
                        )
                    # Backward compatibility
                    elif "latest_market_price" in summary:
                        last_summary_state[symbol] = (
                            summary.get("latest_market_price"),
                            summary.get("latest_market_timestamp")
                        )
            except:
                pass
    
    try:
        update_count = 0
        while True:
            time.sleep(5)
            
            # Check for summary.json updates
            if len(crypto_symbols) == 1 and not commodities_symbols:
                symbol = list(crypto_symbols)[0]
                summary_path = build_summary_path("crypto", symbol, crypto_timeframe, DEFAULT_HORIZON_PROFILE)
                if summary_path.exists():
                    try:
                        with open(summary_path, "r", encoding="utf-8") as f:
                            summary = json.load(f)
                        
                        # Check new format first
                        if "prediction" in summary:
                            pred = summary["prediction"]
                            current_state = (
                                pred.get("current_price"),
                                pred.get("last_updated")
                            )
                            if current_state != last_summary_state.get(symbol):
                                update_count += 1
                                print(f"\n🔄 [{update_count}] Summary.json updated for {symbol}:")
                                print(f"   Current Price: ${pred.get('current_price', 0):,.2f}")
                                print(f"   Predicted Price: ${pred.get('predicted_price', 0):,.2f}")
                                print(f"   Predicted Return: {pred.get('predicted_return_pct', 0):+.2f}%")
                                print(f"   Action: {pred.get('action', 'N/A').upper()}")
                                print(f"   Explanation: {pred.get('explanation', 'N/A')}")
                                last_summary_state[symbol] = current_state
                        # Backward compatibility with old format
                        elif "latest_market_price" in summary:
                            current_state = (
                                summary.get("latest_market_price"),
                                summary.get("latest_market_timestamp")
                            )
                            if current_state != last_summary_state.get(symbol):
                                update_count += 1
                                print(f"\n🔄 [{update_count}] Summary.json updated for {symbol}:")
                                print(f"   Latest Price: ${summary.get('latest_market_price', 0):,.2f}")
                                if "consensus" in summary:
                                    consensus = summary["consensus"]
                                    if "predicted_price_live" in consensus:
                                        print(f"   Predicted Price: ${consensus.get('predicted_price_live', 0):,.2f}")
                                    if "predicted_return" in consensus:
                                        print(f"   Predicted Return: {consensus.get('predicted_return', 0)*100:+.2f}%")
                                last_summary_state[symbol] = current_state
                    except Exception:
                        pass
            
            # Check fallback status
            for symbol, fallback in crypto_fallbacks.items():
                if fallback.is_fallback_active():
                    print(f"  [INFO] {symbol} using fallback: {fallback.get_fallback_reason()}")
    except KeyboardInterrupt:
        print("\n\nStopping live feeds...")
        for symbol, client in crypto_clients:
            if client:
                try:
                    client.stop()
                    print(f"  [OK] Stopped {symbol}")
                except Exception:
                    pass
        print("Live feeds stopped.")


def run_complete_ingestion(
    crypto_symbols: Optional[List[str]] = None,
    commodities_symbols: Optional[List[str]] = None,
    timeframe: str = "1d",
    years: float = 5
):
    """
    Run complete ingestion: Historical first, then live feeds.
    """
    print("\n" + "=" * 80)
    print("CRYPTO + COMMODITIES DATA INGESTION SYSTEM")
    print("=" * 80)
    print("All sources: FREE")
    print("Fallback: Automatic and seamless")
    print("Goal: Uninterrupted data flow for model training")
    print("=" * 80)
    
    ingest_all_historical(crypto_symbols, commodities_symbols, timeframe, years)
    
    print("\n" + "=" * 80)
    print("Transitioning to live feeds in 5 seconds...")
    print("=" * 80)
    time.sleep(5)
    
    start_live_feeds_with_fallback(
        crypto_symbols,
        commodities_symbols,
        timeframe,
        "1d"
    )


def _ordered_horizon_profiles() -> List[str]:
    preferred = ["intraday", "short", "hold"]
    available = available_horizon_profiles()
    ordered = [p for p in preferred if p in available]
    ordered.extend([p for p in available if p not in ordered])
    return ordered


def _prompt_horizon_choice(asset_label: str) -> str:
    ordered_profiles = _ordered_horizon_profiles()
    print("\n" + "-" * 80)
    print(f"{asset_label.upper()} HORIZON PREFERENCE")
    print("-" * 80)
    print("Choose how far ahead predictions should look for this asset class:")
    for idx, profile in enumerate(ordered_profiles, 1):
        description = describe_profile(profile)
        label = profile.title()
        default_marker = " (default)" if profile == DEFAULT_HORIZON_PROFILE else ""
        print(f"  {idx}. {label}{default_marker}")
        if description:
            print(f"     - {description}")
    while True:
        choice = input(f"Select horizon for {asset_label} (1-{len(ordered_profiles)}): ").strip()
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(ordered_profiles):
                return ordered_profiles[index - 1]
        print("Invalid choice. Please enter a valid number from the list.")


def run_training_pipeline(
    crypto_symbols: Optional[List[str]],
    commodities_symbols: Optional[List[str]],
    timeframe: str,
    horizon_profiles: Optional[Dict[str, str]] = None,
):
    crypto_symbols = crypto_symbols or []
    commodities_symbols = commodities_symbols or []
    if not crypto_symbols and not commodities_symbols:
        print("[TRAIN] No symbols provided; skipping training.")
        return
    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)
    train_symbols(
        crypto_symbols,
        commodities_symbols,
        timeframe,
        horizon_profiles=horizon_profiles,
    )
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


def get_user_input():
    """Interactive function to get user input for symbols and options."""
    print("=" * 80)
    print("CRYPTO + COMMODITIES DATA INGESTION SYSTEM")
    print("=" * 80)
    print("Enter the currencies and commodities you want to fetch data for.")
    print("=" * 80)
    
    print("\nWhat type of data would you like to fetch?")
    print("  1. Crypto only")
    print("  2. Commodities only")
    print("  3. Both crypto and commodities (historical + live + training)")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        if choice in ["1", "2", "3"]:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    crypto_symbols = None
    commodities_symbols = None
    
    if choice in ["1", "3"]:
        print("\n" + "-" * 80)
        print("CRYPTO SYMBOLS")
        print("-" * 80)
        print("Enter crypto symbols (e.g., BTC-USDT, ETH-USDT, SOL-USDT)")
        print("Format: SYMBOL-USDT (e.g., BTC-USDT)")
        print("Enter multiple symbols separated by commas or spaces")
        print("Example: BTC-USDT ETH-USDT SOL-USDT")
        
        crypto_input = input("\nEnter crypto symbols: ").strip()
        if crypto_input:
            crypto_symbols = [s.strip().upper() for s in crypto_input.replace(",", " ").split() if s.strip()]
            crypto_symbols = [s if "-USDT" in s else f"{s}-USDT" for s in crypto_symbols]
            print(f"  Selected crypto symbols: {', '.join(crypto_symbols)}")
        else:
            print("  No crypto symbols entered.")
    
    if choice in ["2", "3"]:
        print("\n" + "-" * 80)
        print("COMMODITIES SYMBOLS")
        print("-" * 80)
        print("Enter commodity symbols (e.g., GC=F for Gold, CL=F for Crude Oil)")
        print("Enter multiple symbols separated by commas or spaces")
        print("Example: GC=F SI=F CL=F")
        
        commodities_input = input("\nEnter commodity symbols: ").strip()
        if commodities_input:
            commodities_symbols = [s.strip().upper() for s in commodities_input.replace(",", " ").split() if s.strip()]
            print(f"  Selected commodity symbols: {', '.join(commodities_symbols)}")
        else:
            print("  No commodity symbols entered.")
    
    if not crypto_symbols and not commodities_symbols:
        print("\n[ERROR] No symbols entered. Please enter at least one crypto or commodity symbol.")
        return None, None, None, None, None, False

    horizon_preferences: Dict[str, str] = {}
    if crypto_symbols:
        horizon_preferences["crypto"] = _prompt_horizon_choice("crypto")
    if commodities_symbols:
        horizon_preferences["commodities"] = _prompt_horizon_choice("commodities")
    
    print("\n" + "-" * 80)
    print("INGESTION MODE")
    print("-" * 80)
    print("  1. Historical data only (fetch past data)")
    print("  2. Live data only (real-time updates)")
    print("  3. Historical + Live + Train models")
    
    auto_train = False
    while True:
        mode_choice = input("\nEnter mode (1/2/3): ").strip()
        if mode_choice == "1":
            mode = "historical"
            break
        if mode_choice == "2":
            mode = "live"
            break
        if mode_choice == "3":
            mode = "both"
            auto_train = True
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    years = 7.0
    if mode in ["historical", "both"]:
        print("\n" + "-" * 80)
        print("HISTORICAL DATA PERIOD")
        print("-" * 80)
        while True:
            years_input = input("Enter number of years of historical data (default 7): ").strip()
            if not years_input:
                years = 7.0
                break
            try:
                years = float(years_input)
                if years <= 0:
                    print("  Please enter a positive number of years.")
                else:
                    break
            except ValueError:
                print("  Invalid input. Please enter a number.")
    
    return crypto_symbols, commodities_symbols, years, mode, horizon_preferences, auto_train


def main():
    """Entry point for running ingestion directly from this module."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Crypto + Commodities Data Ingestion System (Free Sources Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetchers.py --mode both --crypto-symbols BTC-USDT ETH-USDT --commodities-symbols GC=F SI=F
  python fetchers.py --mode historical --crypto-symbols BTC-USDT
  python fetchers.py --mode live --crypto-symbols BTC-USDT
        """
    )
    parser.add_argument(
        "--mode",
        choices=["historical", "live", "both"],
        help="Ingestion mode (if not provided, interactive mode will ask)"
    )
    parser.add_argument(
        "--crypto-symbols",
        nargs="+",
        help="Crypto symbols to ingest"
    )
    parser.add_argument(
        "--commodities-symbols",
        nargs="+",
        help="Commodity symbols to ingest"
    )
    parser.add_argument(
        "--timeframe",
        default="1d",
        help="Timeframe for historical data (default: 1d for daily prices)"
    )
    parser.add_argument(
        "--years",
        type=float,
        help="Years of historical data to fetch (default: 5)"
    )
    horizon_choices = available_horizon_profiles()
    parser.add_argument(
        "--crypto-horizon",
        choices=horizon_choices,
        help="Horizon profile for crypto models when training.",
    )
    parser.add_argument(
        "--commodities-horizon",
        choices=horizon_choices,
        help="Horizon profile for commodity models when training.",
    )
    
    args = parser.parse_args()
    
    horizon_preferences: Dict[str, str] = {}
    auto_train = False
    if not args.crypto_symbols and not args.commodities_symbols and not args.mode and args.years is None:
        crypto_symbols, commodities_symbols, years, mode, horizon_preferences, auto_train = get_user_input()
        if crypto_symbols is None and commodities_symbols is None:
            print("\nExiting. No symbols provided.")
            return
    else:
        crypto_symbols = args.crypto_symbols
        commodities_symbols = args.commodities_symbols
        years = args.years if args.years is not None else 7.0
        mode = args.mode if args.mode else "both"
        horizon_preferences = {
            asset: profile
            for asset, profile in (
                ("crypto", args.crypto_horizon),
                ("commodities", args.commodities_horizon),
            )
            if profile
        }
    
    if years <= 0:
        print("\n[ERROR] Years of historical data must be a positive number.")
        return
    
    historical_timeframe = args.timeframe if args.timeframe == "1d" else "1d"
    if args.timeframe != "1d":
        print(f"INFO: Using daily (1d) timeframe for historical data instead of {args.timeframe}")
    
    if not crypto_symbols and not commodities_symbols:
        print("\n[ERROR] No symbols provided. Please specify at least one crypto or commodity symbol.")
        return
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if crypto_symbols:
        print(f"Crypto symbols: {', '.join(crypto_symbols)}")
        if horizon_preferences.get("crypto"):
            print(f"  Horizon: {horizon_preferences['crypto'].title()}")
    if commodities_symbols:
        print(f"Commodity symbols: {', '.join(commodities_symbols)}")
        if horizon_preferences.get("commodities"):
            print(f"  Horizon: {horizon_preferences['commodities'].title()}")
    print(f"Years: {years}")
    print(f"Mode: {mode}")
    print(f"Timeframe: {historical_timeframe}")
    print("=" * 80)
    
    if mode == "both" and not args.train:
        auto_train = True

    run_historical = mode in {"historical", "both"}
    run_live = mode in {"live", "both"}

    if run_historical:
        ingest_all_historical(
            crypto_symbols,
            commodities_symbols,
            historical_timeframe,
            years
        )
    
    if auto_train or args.train or mode == "both":
        run_training_pipeline(
            crypto_symbols,
            commodities_symbols,
            historical_timeframe,
            horizon_preferences or None,
        )
    
    if run_live:
        start_live_feeds_with_fallback(
            crypto_symbols,
            commodities_symbols,
            historical_timeframe,
            "1d"
        )


if __name__ == "__main__":
    main()

