"""
View training logs and summaries in a sequential, readable JSON format.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any


def fix_nan_in_dict(obj: Any) -> Any:
    """Recursively replace NaN values with null in dictionaries."""
    if isinstance(obj, dict):
        return {k: fix_nan_in_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_nan_in_dict(item) for item in obj]
    elif isinstance(obj, float) and (obj != obj):  # NaN check: NaN != NaN
        return None
    elif isinstance(obj, str) and obj.lower() == "nan":
        return None
    return obj


def view_training_log(log_file: Path):
    """View training log in sequential format."""
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return
    
    with open(log_file, "r", encoding="utf-8") as f:
        log_data = json.load(f)
    
    print("=" * 80)
    print("TRAINING LOG - SEQUENTIAL EVENTS")
    print("=" * 80)
    print(f"Log File: {log_data.get('log_file', 'N/A')}")
    print(f"Total Events: {log_data.get('total_events', 0)}")
    print("=" * 80)
    print()
    
    events = log_data.get("events", [])
    for i, event in enumerate(events, 1):
        print(f"[{i}/{len(events)}] {event.get('timestamp', 'N/A')}")
        print(f"  Level: {event.get('level', 'N/A')}")
        print(f"  Category: {event.get('category', 'N/A')}")
        print(f"  Message: {event.get('message', 'N/A')}")
        if event.get('symbol'):
            print(f"  Symbol: {event.get('symbol')}")
        if event.get('asset_type'):
            print(f"  Asset Type: {event.get('asset_type')}")
        if event.get('data'):
            print(f"  Data: {json.dumps(event.get('data'), indent=4, ensure_ascii=False)}")
        print()
    
    print("=" * 80)
    print("FULL JSON OUTPUT")
    print("=" * 80)
    print(json.dumps(log_data, indent=2, ensure_ascii=False))


def view_summary(summary_file: Path):
    """View summary in readable format."""
    if not summary_file.exists():
        print(f"Summary file not found: {summary_file}")
        return
    
    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)
    
    # Fix NaN values
    summary = fix_nan_in_dict(summary)
    
    print("=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Symbol: {summary.get('symbol', 'N/A')}")
    print(f"Asset Type: {summary.get('asset_type', 'N/A')}")
    print(f"Timeframe: {summary.get('timeframe', 'N/A')}")
    target_profile = summary.get("target_profile")
    if target_profile:
        label = target_profile.get("label") or target_profile.get("name", "N/A")
        horizon_bars = target_profile.get("horizon_bars")
        desc = target_profile.get("description")
        print(f"Horizon Profile: {label} ({horizon_bars} bars ahead)" if horizon_bars is not None else f"Horizon Profile: {label}")
        if desc:
            print(f"  {desc}")
    print(f"Total Rows: {summary.get('rows', 0)}")
    print(f"Train Rows: {summary.get('train_rows', 0)}")
    print(f"Val Rows: {summary.get('val_rows', 0)}")
    print(f"Test Rows: {summary.get('test_rows', 0)}")
    print(f"Current Price: ${summary.get('current_price', 0):,.2f}")
    print()
    
    # Show consensus action prominently
    consensus = summary.get('consensus', {})
    if consensus:
        print("=" * 80)
        print("CONSENSUS ACTION (FINAL RECOMMENDATION)")
        print("=" * 80)
        consensus_action = consensus.get('action', 'hold').upper()
        confidence = consensus.get('confidence', 0) * 100
        consensus_price = consensus.get('predicted_price', 0)
        consensus_return = consensus.get('predicted_return', 0) * 100
        current_price = summary.get('current_price', 0)
        horizon_label = consensus.get('horizon_profile') or summary.get('target_profile', {}).get('label')
        horizon_bars = consensus.get('target_horizon_bars') or summary.get('target_profile', {}).get('horizon_bars')
        
        print(f"  ACTION: {consensus_action}")
        if horizon_label:
            if horizon_bars is not None:
                print(f"  Horizon: {horizon_label} ({horizon_bars} bars)")
            else:
                print(f"  Horizon: {horizon_label}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Current Price: ${current_price:,.2f}")
        print(f"  Consensus Predicted Price: ${consensus_price:,.2f}")
        print(f"  Expected Return: {consensus_return:+.2f}%")
        print(f"  Reasoning: {consensus.get('reasoning', 'N/A')}")
        print()
        
        # Show action scores
        action_scores = consensus.get('action_scores', {})
        if action_scores:
            print("  Action Scores:")
            for action, score in action_scores.items():
                print(f"    {action.upper()}: {score*100:.1f}%")
        print()
    
    print("=" * 80)
    print("INDIVIDUAL MODEL PREDICTIONS")
    print("=" * 80)
    models = summary.get('models', {})
    current_price = summary.get('current_price', 0)
    print(f"  Current Price: ${current_price:,.2f}")
    print()
    for model_name, model_data in models.items():
        if model_data.get('status') == 'failed':
            print(f"  {model_name.upper()}: FAILED - {model_data.get('reason', 'Unknown error')}")
        else:
            predicted_price = model_data.get('predicted_price', 0)
            predicted_return = model_data.get('predicted_return', 0)
            price_change = predicted_price - current_price
            print(f"  {model_name.upper()}:")
            print(f"    Current Price: ${current_price:,.2f}")
            print(f"    Predicted Price: ${predicted_price:,.2f}")
            print(f"    Price Change: ${price_change:+,.2f} ({predicted_return*100:+.2f}%)")
            print(f"    R² Score: {model_data.get('r2', 'N/A')}")
            print(f"    MAE: {model_data.get('mae', 'N/A')}")
            print(f"    RMSE: {model_data.get('rmse', 'N/A')}")
            print(f"    Action: {model_data.get('action', 'N/A').upper()}")
            print(f"    Reason: {model_data.get('action_reason', 'N/A')}")
        print()
    
    print("ANALYSIS:")
    print("-" * 80)
    analysis = summary.get('analysis', {})
    print(f"  Dynamic Threshold: {analysis.get('dynamic_threshold', 0)*100:.2f}%")
    print()
    print("  Context Signals:")
    signals = analysis.get('context_signals', {})
    for signal, value in signals.items():
        print(f"    {signal}: {value}")
    print()
    
    warnings = analysis.get('overfitting_warnings')
    if warnings:
        print("  Overfitting Warnings:")
        for warning in warnings:
            print(f"    ⚠ {warning}")
    else:
        print("  Overfitting Warnings: None")
    
    print()
    print("=" * 80)
    print("FULL JSON OUTPUT")
    print("=" * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python view_logs.py <log_file_or_summary_file>")
        print("\nExamples:")
        print("  python view_logs.py logs/training/crypto/BTC-USDT/1d/training_log.json")
        print("  python view_logs.py models/crypto/BTC-USDT/1d/summary.json")
        return
    
    file_path = Path(sys.argv[1])
    
    if "training_log.json" in file_path.name:
        view_training_log(file_path)
    elif "summary.json" in file_path.name:
        view_summary(file_path)
    else:
        print(f"Unknown file type: {file_path.name}")
        print("Expected: training_log.json or summary.json")


if __name__ == "__main__":
    main()

