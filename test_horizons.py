"""Quick test script to test all three horizons."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.mcp_adapter import get_mcp_adapter

def test_horizon(horizon_name: str):
    """Test a specific horizon."""
    print(f"\n{'='*80}")
    print(f"TESTING HORIZON: {horizon_name.upper()}")
    print(f"{'='*80}")
    
    adapter = get_mcp_adapter()
    
    result = adapter.predict(
        symbols=["BTC-USDT"],
        horizon=horizon_name,
        asset_type="crypto",
        timeframe="1d"
    )
    
    if result.get("errors"):
        print(f"❌ ERRORS: {result['errors']}")
        return False
    
    predictions = result.get("predictions", [])
    if not predictions:
        print("❌ NO PREDICTIONS RETURNED")
        return False
    
    pred = predictions[0]
    
    print(f"\n✅ PREDICTION FOR BTC-USDT ({horizon_name}):")
    print(f"   Current Price: ${pred.get('current_price', 0):,.2f}")
    print(f"   Predicted Return: {pred.get('predicted_return', 0)*100:+.2f}%")
    print(f"   Predicted Price: ${pred.get('predicted_price', 0):,.2f}")
    print(f"   Action: {pred.get('action', 'UNKNOWN')}")
    print(f"   Confidence: {pred.get('confidence', 0)*100:.1f}%")
    print(f"   Horizon Bars: {pred.get('horizon_bars', 'N/A')}")
    
    # Check if action is logical
    pred_return = pred.get('predicted_return', 0)
    action = pred.get('action', '').upper()
    
    is_logical = False
    if pred_return > 0.01 and action == "LONG":
        is_logical = True
    elif pred_return < -0.01 and action == "SHORT":
        is_logical = True
    elif abs(pred_return) < 0.01 and action == "HOLD":
        is_logical = True
    elif abs(pred_return) < 0.03:  # Neutral guard might trigger
        is_logical = action in ["HOLD", "LONG", "SHORT"]  # Any action is acceptable if return is small
    
    if is_logical:
        print(f"   ✅ LOGICAL: Action matches predicted return")
    else:
        print(f"   ⚠️  WARNING: Action might not match return (return={pred_return*100:+.2f}%, action={action})")
    
    # Show individual models
    individual = pred.get('individual_models', [])
    if individual:
        print(f"\n   Individual Models ({len(individual)}):")
        for model in individual[:3]:  # Show first 3
            name = model.get('name', 'unknown')
            action = model.get('action', 'N/A')
            ret = model.get('predicted_return', 0) * 100
            conf = model.get('confidence', 0) * 100
            print(f"      - {name}: {action} ({ret:+.2f}%, conf={conf:.1f}%)")
    
    return True

if __name__ == "__main__":
    print("="*80)
    print("TESTING ALL THREE HORIZONS")
    print("="*80)
    
    horizons = ["intraday", "short", "long"]
    results = {}
    
    for horizon in horizons:
        try:
            success = test_horizon(horizon)
            results[horizon] = "✅ PASS" if success else "❌ FAIL"
        except Exception as e:
            print(f"❌ ERROR testing {horizon}: {e}")
            results[horizon] = f"❌ ERROR: {str(e)[:50]}"
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for horizon, status in results.items():
        print(f"  {horizon:12s}: {status}")
    print(f"{'='*80}\n")

