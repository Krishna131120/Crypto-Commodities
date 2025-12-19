"""
Comprehensive training quality test.
Trains models and verifies:
1. No overfitting (train/val/test gaps)
2. All models learning (positive R², good variance)
3. Model agreement
4. Prediction quality suitable for live trading
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Import training functions
from train_commodities import train_commodity_symbol
from ml.json_logger import get_training_logger

def analyze_training_results(symbol: str, model_dir: Path) -> dict:
    """Analyze training results for quality metrics."""
    results = {
        "symbol": symbol,
        "models_trained": [],
        "overfitting_detected": False,
        "all_models_learning": True,
        "model_agreement": None,
        "ready_for_trading": False,
        "issues": [],
        "metrics": {}
    }
    
    # Check for model files
    model_files = list(model_dir.glob("*.joblib"))
    if not model_files:
        results["issues"].append("No model files found")
        return results
    
    # Load metrics files
    metrics_file = model_dir / "metrics.json"
    summary_file = model_dir / "summary.json"
    
    if not metrics_file.exists():
        results["issues"].append("No metrics.json found")
        return results
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    except Exception as e:
        results["issues"].append(f"Error loading metrics: {e}")
        return results
    
    # Analyze each model
    model_names = ["random_forest", "lightgbm", "xgboost"]
    model_metrics = {}
    valid_models = []
    
    for model_name in model_names:
        if model_name not in metrics:
            continue
        
        model_data = metrics[model_name]
        model_metrics[model_name] = {}
        
        # Extract R² scores
        train_r2 = model_data.get("train_r2")
        val_r2 = model_data.get("val_r2")
        test_r2 = model_data.get("test_r2")
        
        # Extract directional accuracy
        val_dir = model_data.get("val_directional_accuracy")
        test_dir = model_data.get("test_directional_accuracy")
        
        # Extract MAE
        val_mae = model_data.get("val_mae")
        test_mae = model_data.get("test_mae")
        
        # Check if model is learning
        is_learning = True
        issues = []
        
        if val_r2 is not None and val_r2 < 0:
            is_learning = False
            issues.append(f"Negative validation R²: {val_r2:.4f}")
        
        if test_r2 is not None and test_r2 < 0:
            is_learning = False
            issues.append(f"Negative test R²: {test_r2:.4f}")
        
        if val_r2 is not None and val_r2 < 0.05:
            issues.append(f"Low validation R²: {val_r2:.4f} (< 0.05)")
        
        if test_r2 is not None and test_r2 < 0.05:
            issues.append(f"Low test R²: {test_r2:.4f} (< 0.05)")
        
        # Check for overfitting
        overfitting = False
        if train_r2 is not None and val_r2 is not None:
            train_val_gap = train_r2 - val_r2
            if train_val_gap > 0.15:  # 15% gap is concerning
                overfitting = True
                issues.append(f"Large train-val gap: {train_val_gap:.4f} (> 0.15)")
        
        if val_r2 is not None and test_r2 is not None:
            val_test_gap = val_r2 - test_r2
            if val_test_gap > 0.10:  # 10% gap is concerning
                overfitting = True
                issues.append(f"Large val-test gap: {val_test_gap:.4f} (> 0.10)")
        
        if train_r2 is not None and test_r2 is not None:
            train_test_gap = train_r2 - test_r2
            if train_test_gap > 0.20:  # 20% gap is concerning
                overfitting = True
                issues.append(f"Large train-test gap: {train_test_gap:.4f} (> 0.20)")
        
        model_metrics[model_name] = {
            "train_r2": train_r2,
            "val_r2": val_r2,
            "test_r2": test_r2,
            "val_dir": val_dir,
            "test_dir": test_dir,
            "val_mae": val_mae,
            "test_mae": test_mae,
            "is_learning": is_learning,
            "overfitting": overfitting,
            "issues": issues
        }
        
        if is_learning and not overfitting:
            valid_models.append(model_name)
        
        if overfitting:
            results["overfitting_detected"] = True
        
        if not is_learning:
            results["all_models_learning"] = False
    
    results["models_trained"] = list(model_metrics.keys())
    results["metrics"] = model_metrics
    
    # Check model agreement (if we have consensus data)
    if "consensus" in summary:
        consensus = summary["consensus"]
        num_models = len(valid_models)
        if num_models >= 2:
            results["model_agreement"] = "good" if num_models >= 2 else "poor"
        else:
            results["model_agreement"] = "poor"
            results["issues"].append(f"Only {num_models} valid model(s), need at least 2")
    
    # Determine if ready for trading
    ready = True
    if results["overfitting_detected"]:
        ready = False
        results["issues"].append("Overfitting detected in one or more models")
    
    if not results["all_models_learning"]:
        ready = False
        results["issues"].append("Some models are not learning (negative or very low R²)")
    
    if len(valid_models) < 2:
        ready = False
        results["issues"].append(f"Only {len(valid_models)} valid model(s), need at least 2 for consensus")
    
    # Check minimum quality thresholds
    min_test_r2 = 0.05
    min_dir_acc = 0.50
    
    for model_name, model_data in model_metrics.items():
        test_r2 = model_data.get("test_r2")
        test_dir = model_data.get("test_dir")
        
        if test_r2 is not None and test_r2 < min_test_r2:
            ready = False
            results["issues"].append(f"{model_name}: Test R² {test_r2:.4f} < {min_test_r2}")
        
        if test_dir is not None and test_dir < min_dir_acc:
            ready = False
            results["issues"].append(f"{model_name}: Test directional accuracy {test_dir:.4f} < {min_dir_acc}")
    
    results["ready_for_trading"] = ready
    
    return results


def print_analysis(results: dict):
    """Print detailed analysis results."""
    print("\n" + "=" * 80)
    print("TRAINING QUALITY ANALYSIS")
    print("=" * 80)
    print(f"Symbol: {results['symbol']}")
    print(f"Models trained: {', '.join(results['models_trained']) if results['models_trained'] else 'None'}")
    print()
    
    # Model-by-model breakdown
    print("MODEL METRICS:")
    print("-" * 80)
    for model_name, metrics in results['metrics'].items():
        print(f"\n{model_name.upper()}:")
        print(f"  Train R²: {metrics.get('train_r2', 'N/A'):.4f}" if metrics.get('train_r2') is not None else f"  Train R²: N/A")
        print(f"  Val R²:   {metrics.get('val_r2', 'N/A'):.4f}" if metrics.get('val_r2') is not None else f"  Val R²:   N/A")
        print(f"  Test R²:  {metrics.get('test_r2', 'N/A'):.4f}" if metrics.get('test_r2') is not None else f"  Test R²:  N/A")
        print(f"  Val Dir:  {metrics.get('val_dir', 'N/A'):.4f}" if metrics.get('val_dir') is not None else f"  Val Dir:  N/A")
        print(f"  Test Dir: {metrics.get('test_dir', 'N/A'):.4f}" if metrics.get('test_dir') is not None else f"  Test Dir: N/A")
        
        if metrics.get('overfitting'):
            print(f"  STATUS: OVERFITTING DETECTED")
        elif not metrics.get('is_learning'):
            print(f"  STATUS: NOT LEARNING")
        else:
            print(f"  STATUS: OK")
        
        if metrics.get('issues'):
            for issue in metrics['issues']:
                print(f"    - {issue}")
    
    print("\n" + "-" * 80)
    print("OVERALL ASSESSMENT:")
    print("-" * 80)
    
    if results['overfitting_detected']:
        print("X OVERFITTING DETECTED - Models may not generalize well")
    else:
        print("OK No significant overfitting detected")
    
    if results['all_models_learning']:
        print("OK All models are learning (positive R²)")
    else:
        print("X Some models are not learning properly")
    
    if results['model_agreement'] == "good":
        print("OK Good model agreement (multiple valid models)")
    else:
        print("X Poor model agreement (insufficient valid models)")
    
    if results['ready_for_trading']:
        print("\n" + "=" * 80)
        print("READY FOR LIVE TRADING")
        print("=" * 80)
        print("All quality checks passed:")
        print("  - No overfitting detected")
        print("  - All models learning")
        print("  - Sufficient model agreement")
        print("  - Minimum quality thresholds met")
    else:
        print("\n" + "=" * 80)
        print("NOT READY FOR LIVE TRADING")
        print("=" * 80)
        print("Issues found:")
        for issue in results['issues']:
            print(f"  - {issue}")
    
    print("=" * 80)


def main():
    """Run comprehensive training quality test."""
    symbol = "GC=F"
    timeframe = "1d"
    horizon = "short"  # Use short horizon for commodities
    
    print("=" * 80)
    print("COMPREHENSIVE TRAINING QUALITY TEST")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Horizon: {horizon}")
    print("\nTraining models...")
    print("=" * 80)
    
    try:
        # Train the model
        summary = train_commodity_symbol(
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
        )
        
        # Get model directory
        model_dir = Path("models") / "commodities" / symbol / timeframe / horizon
        
        # Analyze results
        print("\nAnalyzing training results...")
        results = analyze_training_results(symbol, model_dir)
        
        # Print analysis
        print_analysis(results)
        
        # Return exit code
        return 0 if results['ready_for_trading'] else 1
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
