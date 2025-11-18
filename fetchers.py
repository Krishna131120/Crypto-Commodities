"""
Complete data ingestion system - all fetchers and utilities merged into one file.
"""
import csv
import json
import time
import threading
import requests
import websocket
import ccxt
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum
import logging

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        source_hint = "binance"
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
# DATA FETCHERS - Binance Historical
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


def save_historical_data(
    symbol: str,
    timeframe: str,
    candles: List[Dict[str, Any]],
    source_hint: Optional[str] = None
):
    """Save all historical candles to a single JSON file per symbol/timeframe."""
    if not candles:
        return
    source_folder = source_hint or "binance"
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
        if ts not in seen_timestamps or candle.get("volume", 0) > seen_timestamps[ts].get("volume", 0):
            seen_timestamps[ts] = candle
    
    # Convert back to sorted list
    merged_candles = list(seen_timestamps.values())
    merged_candles.sort(key=lambda x: x.get("timestamp", ""))
    
    # Save to single file
    save_json_file(data_file, merged_candles, append=False)
    print(f"  Saved {len(merged_candles)} total candles to data.json (added {len(candles)} new) [{source_folder}]")

    generate_manifest("crypto", symbol, timeframe, source_folder)
    print(f"  Generated manifest for {symbol}/{timeframe} ({source_folder})")


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
                hist = yf.download(
                    tickers=yahoo_symbol,
                    start=current_start.strftime("%Y-%m-%d"),
                    end=current_end.strftime("%Y-%m-%d"),
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    group_by="ticker",
                    threads=False,
                    session=session
                )

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
    source_folder = source_hint or _infer_source_from_candles(candles, "yahoo")
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
        if ts not in seen_timestamps or candle.get("volume", 0) > seen_timestamps[ts].get("volume", 0):
            seen_timestamps[ts] = candle
    
    # Convert back to sorted list
    merged_candles = list(seen_timestamps.values())
    merged_candles.sort(key=lambda x: x.get("timestamp", ""))
    
    # Save to single file
    save_json_file(data_file, merged_candles, append=False)
    print(f"  Saved {len(merged_candles)} total candles to data.json (added {len(candles)} new) [{source_folder}]")

    generate_manifest("commodities", symbol, timeframe, source_folder)
    print(f"  Generated manifest for {symbol}/{timeframe} ({source_folder})")


def poll_yahoo_live(
    symbol: str,
    timeframe: str = "1d",
    poll_interval: int = 60,
    fallback_engine: Optional[FallbackEngine] = None
):
    """Poll Yahoo Finance for live data updates."""
    print(f"Starting Yahoo Finance live polling for {symbol} ({timeframe})")
    
    ticker = yf.Ticker(symbol)
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

