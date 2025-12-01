"""
Command-line entry point for running the inference pipeline on fresh data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml.inference import InferencePipeline
from ml.risk import RiskManagerConfig


def load_feature_row(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix in {".json", ".jsonl"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            payload = payload[-1]
        return pd.Series(payload)
    if path.suffix in {".csv"}:
        df = pd.read_csv(path)
        return df.iloc[-1]
    if path.suffix in {".parquet"}:
        df = pd.read_parquet(path)
        return df.iloc[-1]
    raise ValueError(f"Unsupported feature file format: {path.suffix}")


def main():
    parser = argparse.ArgumentParser(description="Run inference using trained models.")
    parser.add_argument("--model-dir", required=True, help="Directory containing trained models.")
    parser.add_argument("--feature-path", required=True, help="Path to feature row (json/csv/parquet).")
    parser.add_argument("--current-price", type=float, required=True, help="Latest close price.")
    parser.add_argument("--volatility", type=float, default=0.01, help="Recent volatility proxy.")
    parser.add_argument("--paper-trade", action="store_true", help="Enable paper trading mode.")
    args = parser.parse_args()

    feature_row = load_feature_row(Path(args.feature_path))
    risk_config = RiskManagerConfig(paper_trade=args.paper_trade)
    pipeline = InferencePipeline(Path(args.model_dir), risk_config=risk_config)
    result = pipeline.predict(feature_row, current_price=args.current_price, volatility=args.volatility)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


