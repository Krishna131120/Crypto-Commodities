"""
Bucket logging system for storing all predictions, trades, and feedback.
Every prediction, trade, and feedback is logged with timestamp, confidence, decision, and sentiment snapshot.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config


class BucketLogger:
    """
    Logs all predictions, trades, and feedback to a bucket storage system.
    Each entry includes timestamp, confidence, decision, and sentiment snapshot.
    """
    
    def __init__(self, bucket_dir: Optional[Path] = None):
        """
        Initialize bucket logger.
        
        Args:
            bucket_dir: Directory for bucket storage. Defaults to data/buckets/
        """
        self.bucket_dir = bucket_dir or Path("data/buckets")
        self.bucket_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate buckets for different event types
        self.prediction_bucket = self.bucket_dir / "predictions.jsonl"
        self.trade_bucket = self.bucket_dir / "trades.jsonl"
        self.feedback_bucket = self.bucket_dir / "feedback.jsonl"
        
        # Index file for quick lookups
        self.index_file = self.bucket_dir / "index.json"
        self._load_index()
    
    def _load_index(self):
        """Load or create index for fast lookups."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    self.index = json.load(f)
            except:
                self.index = {
                    "predictions": {"count": 0, "last_timestamp": None},
                    "trades": {"count": 0, "last_timestamp": None},
                    "feedback": {"count": 0, "last_timestamp": None}
                }
        else:
            self.index = {
                "predictions": {"count": 0, "last_timestamp": None},
                "trades": {"count": 0, "last_timestamp": None},
                "feedback": {"count": 0, "last_timestamp": None}
            }
    
    def _update_index(self, bucket_type: str, timestamp: str):
        """Update index with new entry."""
        if bucket_type in self.index:
            self.index[bucket_type]["count"] += 1
            self.index[bucket_type]["last_timestamp"] = timestamp
        
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.index, f, indent=2)
        except Exception as exc:
            print(f"[BUCKET] Failed to update index: {exc}")
    
    def _write_jsonl(self, file_path: Path, entry: Dict[str, Any]):
        """Append entry to JSONL file."""
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[BUCKET] Failed to write to {file_path}: {exc}")
    
    def log_prediction(
        self,
        symbol: str,
        asset_type: str,
        timeframe: str,
        timestamp: str,
        current_price: float,
        predictions: List[Dict[str, Any]],
        consensus: Dict[str, Any],
        sentiment_summary: Dict[str, Any],
        events: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a prediction event to the bucket.
        
        Args:
            symbol: Trading symbol (e.g., BTC-USDT)
            asset_type: Asset type (crypto/commodities)
            timeframe: Timeframe (e.g., 1d)
            timestamp: ISO timestamp of prediction
            current_price: Current market price
            predictions: List of individual model predictions
            consensus: Consensus prediction data
            sentiment_summary: Sentiment summary data
            events: List of events
            metadata: Additional metadata
        """
        entry = {
            "event_type": "prediction",
            "timestamp": timestamp,
            "symbol": symbol,
            "asset_type": asset_type,
            "timeframe": timeframe,
            "current_price": float(current_price),
            "predictions": predictions,
            "consensus": {
                "action": consensus.get("action", "hold"),
                "predicted_return": float(consensus.get("predicted_return", 0)),
                "predicted_price": float(consensus.get("predicted_price", current_price)),
                "confidence": float(consensus.get("confidence", 0)),
                "reasoning": consensus.get("reasoning", "")
            },
            "sentiment_snapshot": sentiment_summary,
            "events": events,
            "metadata": metadata or {}
        }
        
        self._write_jsonl(self.prediction_bucket, entry)
        self._update_index("predictions", timestamp)
    
    def log_trade(
        self,
        symbol: str,
        asset_type: str,
        timeframe: str,
        timestamp: str,
        action: str,
        entry_price: float,
        predicted_price: float,
        confidence: float,
        quantity: Optional[float] = None,
        trade_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a trade execution to the bucket.
        
        Args:
            symbol: Trading symbol
            asset_type: Asset type
            timeframe: Timeframe
            timestamp: ISO timestamp of trade
            action: Trade action (long/short/hold)
            entry_price: Entry price
            predicted_price: Predicted price at time of trade
            confidence: Confidence level (0-1)
            quantity: Trade quantity (optional)
            trade_id: Unique trade identifier (optional)
            metadata: Additional metadata
        """
        entry = {
            "event_type": "trade",
            "timestamp": timestamp,
            "symbol": symbol,
            "asset_type": asset_type,
            "timeframe": timeframe,
            "action": action,
            "entry_price": float(entry_price),
            "predicted_price": float(predicted_price),
            "confidence": float(confidence),
            "quantity": quantity,
            "trade_id": trade_id,
            "metadata": metadata or {}
        }
        
        self._write_jsonl(self.trade_bucket, entry)
        self._update_index("trades", timestamp)
    
    def log_feedback(
        self,
        symbol: str,
        asset_type: str,
        timeframe: str,
        timestamp: str,
        prediction_timestamp: str,
        actual_price: Optional[float] = None,
        actual_return: Optional[float] = None,
        feedback_type: str = "outcome",
        rating: Optional[float] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log feedback on a prediction (e.g., actual outcome, user rating).
        
        Args:
            symbol: Trading symbol
            asset_type: Asset type
            timeframe: Timeframe
            timestamp: ISO timestamp of feedback
            prediction_timestamp: ISO timestamp of original prediction
            actual_price: Actual price at prediction horizon (optional)
            actual_return: Actual return at prediction horizon (optional)
            feedback_type: Type of feedback (outcome, rating, correction, etc.)
            rating: User rating (optional, 0-1 scale)
            notes: Additional notes (optional)
            metadata: Additional metadata
        """
        entry = {
            "event_type": "feedback",
            "timestamp": timestamp,
            "symbol": symbol,
            "asset_type": asset_type,
            "timeframe": timeframe,
            "prediction_timestamp": prediction_timestamp,
            "actual_price": float(actual_price) if actual_price is not None else None,
            "actual_return": float(actual_return) if actual_return is not None else None,
            "feedback_type": feedback_type,
            "rating": float(rating) if rating is not None else None,
            "notes": notes,
            "metadata": metadata or {}
        }
        
        self._write_jsonl(self.feedback_bucket, entry)
        self._update_index("feedback", timestamp)
    
    def get_summary(
        self,
        symbol: Optional[str] = None,
        asset_type: Optional[str] = None,
        start_timestamp: Optional[str] = None,
        end_timestamp: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get summarized history from buckets for dashboard view.
        
        Args:
            symbol: Filter by symbol (optional)
            asset_type: Filter by asset type (optional)
            start_timestamp: Start timestamp filter (optional)
            end_timestamp: End timestamp filter (optional)
            limit: Maximum number of entries per type (optional)
        
        Returns:
            Dictionary with summarized data
        """
        summary = {
            "index": self.index,
            "predictions": [],
            "trades": [],
            "feedback": []
        }
        
        # Read predictions
        if self.prediction_bucket.exists():
            summary["predictions"] = self._read_jsonl(
                self.prediction_bucket,
                symbol=symbol,
                asset_type=asset_type,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                limit=limit
            )
        
        # Read trades
        if self.trade_bucket.exists():
            summary["trades"] = self._read_jsonl(
                self.trade_bucket,
                symbol=symbol,
                asset_type=asset_type,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                limit=limit
            )
        
        # Read feedback
        if self.feedback_bucket.exists():
            summary["feedback"] = self._read_jsonl(
                self.feedback_bucket,
                symbol=symbol,
                asset_type=asset_type,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                limit=limit
            )
        
        return summary
    
    def _read_jsonl(
        self,
        file_path: Path,
        symbol: Optional[str] = None,
        asset_type: Optional[str] = None,
        start_timestamp: Optional[str] = None,
        end_timestamp: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Read and filter JSONL file."""
        entries = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        
                        # Apply filters
                        if symbol and entry.get("symbol") != symbol:
                            continue
                        if asset_type and entry.get("asset_type") != asset_type:
                            continue
                        if start_timestamp and entry.get("timestamp", "") < start_timestamp:
                            continue
                        if end_timestamp and entry.get("timestamp", "") > end_timestamp:
                            continue
                        
                        entries.append(entry)
                        
                        if limit and len(entries) >= limit:
                            break
                    except json.JSONDecodeError:
                        continue
            
            # Return most recent first
            entries.reverse()
        except FileNotFoundError:
            pass
        
        return entries


# Global bucket logger instance
_bucket_logger: Optional[BucketLogger] = None


def get_bucket_logger() -> BucketLogger:
    """Get or create global bucket logger instance."""
    global _bucket_logger
    if _bucket_logger is None:
        _bucket_logger = BucketLogger()
    return _bucket_logger

