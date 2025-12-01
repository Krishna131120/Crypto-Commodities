"""
JSON logging utility for structured log output.
All logs are written to JSON format for easy parsing and viewing.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class JSONLogger:
    """Logger that writes all events to a JSON log file."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.events: List[Dict[str, Any]] = []
        
    def log(self, 
            level: str,
            message: str,
            category: Optional[str] = None,
            data: Optional[Dict[str, Any]] = None,
            symbol: Optional[str] = None,
            asset_type: Optional[str] = None):
        """
        Log an event to JSON.
        
        Args:
            level: Log level (INFO, WARNING, ERROR, SUCCESS)
            message: Human-readable message
            category: Category of event (DATA, TRAIN, MODEL, etc.)
            data: Additional structured data
            symbol: Symbol name if applicable
            asset_type: Asset type if applicable
        """
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "message": message,
        }
        if category:
            event["category"] = category
        if symbol:
            event["symbol"] = symbol
        if asset_type:
            event["asset_type"] = asset_type
        if data:
            event["data"] = data
        
        self.events.append(event)
        self._write_log()
    
    def info(self, message: str, **kwargs):
        """Log an info message."""
        self.log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log a warning message."""
        self.log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log an error message."""
        self.log("ERROR", message, **kwargs)
    
    def success(self, message: str, **kwargs):
        """Log a success message."""
        self.log("SUCCESS", message, **kwargs)
    
    def _write_log(self):
        """Write all events to the log file."""
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump({
                    "log_file": str(self.log_file),
                    "total_events": len(self.events),
                    "events": self.events
                }, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            # Fallback: don't crash if logging fails
            print(f"[LOG ERROR] Failed to write log: {exc}")
    
    def flush(self):
        """Force write all events to disk."""
        self._write_log()
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get a summary of the log."""
        return {
            "log_file": str(self.log_file),
            "total_events": len(self.events),
            "events_by_level": self._count_by_level(),
            "events_by_category": self._count_by_category(),
        }
    
    def _count_by_level(self) -> Dict[str, int]:
        """Count events by level."""
        counts = {}
        for event in self.events:
            level = event.get("level", "UNKNOWN")
            counts[level] = counts.get(level, 0) + 1
        return counts
    
    def _count_by_category(self) -> Dict[str, int]:
        """Count events by category."""
        counts = {}
        for event in self.events:
            category = event.get("category", "UNCATEGORIZED")
            counts[category] = counts.get(category, 0) + 1
        return counts


def get_training_logger(asset_type: str, symbol: str, timeframe: str, base_dir: Path = Path("logs")) -> JSONLogger:
    """Get a logger for a specific training session."""
    log_file = base_dir / "training" / asset_type / symbol / timeframe / "training_log.json"
    return JSONLogger(log_file)


def get_pipeline_logger(base_dir: Path = Path("logs")) -> JSONLogger:
    """Get a logger for the entire pipeline."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_file = base_dir / f"pipeline_{timestamp}.json"
    return JSONLogger(log_file)

