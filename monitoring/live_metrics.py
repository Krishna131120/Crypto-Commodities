"""
Live monitoring utilities for drift detection.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np


class LiveMetricsTracker:
    """
    Tracks live prediction quality vs training benchmarks.
    """

    def __init__(self, window: int = 250, log_path: Optional[Path] = None):
        self.window = window
        self.records: Deque[Dict[str, float]] = deque(maxlen=window)
        self.log_path = log_path or Path("logs/live_metrics.json")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def update(self, predicted: float, actual: float, directional: Optional[float] = None) -> Dict[str, float]:
        self.records.append(
            {
                "predicted": float(predicted),
                "actual": float(actual),
                "directional": float(directional) if directional is not None else None,
            }
        )
        metrics = self.compute_metrics()
        self._write(metrics)
        return metrics

    def compute_metrics(self) -> Dict[str, float]:
        if not self.records:
            return {}
        preds = np.array([r["predicted"] for r in self.records])
        actuals = np.array([r["actual"] for r in self.records])
        with np.errstate(divide="ignore", invalid="ignore"):
            directional = np.mean(np.sign(preds) == np.sign(actuals))
        mae = np.mean(np.abs(actuals - preds))
        rmse = np.sqrt(np.mean((actuals - preds) ** 2))
        corr = np.corrcoef(preds, actuals)[0, 1] if len(self.records) > 5 else 0.0
        return {
            "window": len(self.records),
            "rolling_mae": float(mae),
            "rolling_rmse": float(rmse),
            "directional_accuracy": float(directional),
            "correlation": float(corr if not np.isnan(corr) else 0.0),
        }

    def _write(self, metrics: Dict[str, float]):
        payload = {
            "metrics": metrics,
            "records": list(self.records),
        }
        with open(self.log_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


