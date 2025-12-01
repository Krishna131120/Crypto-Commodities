"""
Watch live price and prediction updates in real-time.
Shows what's being updated automatically vs what requires retraining.
"""
import json
import time
from pathlib import Path
from datetime import datetime

from ml.horizons import normalize_profile


def watch_predictions(symbol: str = "ETH-USDT", timeframe: str = "1d", horizon: str = "short"):
    """Monitor summary.json for live updates."""
    normalized_horizon = normalize_profile(horizon)
    summary_path = Path("models") / "crypto" / symbol / timeframe / normalized_horizon / "summary.json"
    
    if not summary_path.exists():
        print(f"❌ Model not trained! Run training first:")
        print(
            f"   python pipeline_runner.py --mode both --crypto-symbols {symbol} "
            f"--timeframe {timeframe} --train --crypto-horizon {normalized_horizon}"
        )
        return
    
    print(f"\n{'='*80}")
    print(f"WATCHING LIVE PREDICTIONS: {symbol} ({timeframe})")
    print(f"{'='*80}")
    print(f"File: {summary_path}")
    print(f"\n🔄 Updates automatically:")
    print(f"   ✓ latest_market_price (from WebSocket)")
    print(f"   ✓ predicted_price_live (recalculated)")
    print(f"\n⚠️  Does NOT update automatically:")
    print(f"   ✗ predicted_return (requires model retraining)")
    print(f"   ✗ Model training (must run manually)")
    print(f"\n{'='*80}\n")
    
    last_price = None
    last_timestamp = None
    last_predicted_live = None
    
    try:
        while True:
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                
                current_price = summary.get("latest_market_price")
                current_timestamp = summary.get("latest_market_timestamp")
                consensus = summary.get("consensus", {})
                predicted_live = consensus.get("predicted_price_live")
                predicted_return = consensus.get("predicted_return", 0) * 100
                
                # Check if anything changed
                if (current_price != last_price or 
                    current_timestamp != last_timestamp or
                    predicted_live != last_predicted_live):
                    
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] UPDATE DETECTED:")
                    print(f"  📊 Latest Market Price: ${current_price:,.2f}")
                    print(f"  📅 Timestamp: {current_timestamp}")
                    print(f"  🎯 Predicted Price (Live): ${predicted_live:,.2f}")
                    print(f"  📈 Predicted Return: {predicted_return:+.2f}% (FIXED - needs retraining to change)")
                    print(f"  💡 Action: {consensus.get('action', 'N/A').upper()}")
                    
                    if last_price:
                        price_change = current_price - last_price
                        print(f"  📉 Price Change: ${price_change:+,.2f}")
                    
                    last_price = current_price
                    last_timestamp = current_timestamp
                    last_predicted_live = predicted_live
            
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped.")

if __name__ == "__main__":
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "ETH-USDT"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "1d"
    horizon = sys.argv[3] if len(sys.argv) > 3 else "short"
    watch_predictions(symbol, timeframe, horizon)

