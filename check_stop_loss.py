"""Check stop-loss settings for commodities trading."""

from ml.horizons import PROFILE_METADATA, get_horizon_risk_config
from trading.execution_engine import TradingRiskConfig

print("=" * 80)
print("COMMODITY STOP-LOSS SETTINGS")
print("=" * 80)
print()

print("By Horizon Profile:")
print()
for profile in ['intraday', 'short', 'long']:
    meta = PROFILE_METADATA[profile]
    print(f"{profile.upper()}:")
    print(f"  Stop-Loss: {meta['stop_loss_pct']*100:.1f}%")
    print(f"  Position Size: {meta['position_size_pct']*100:.0f}% of equity")
    print(f"  Min Confidence: {meta['min_confidence']*100:.0f}%")
    print()

print("=" * 80)
print("DEFAULT STOP-LOSS FOR COMMODITIES")
print("=" * 80)
print()
print("From ExecutionEngine.get_effective_stop_loss_pct():")
print("  Default Stop-Loss: 2.0% (tighter for real money trading)")
print("  Crypto Stop-Loss: 3.5% (wider to avoid volatility triggers)")
print()

print("=" * 80)
print("HOW STOP-LOSS IS DETERMINED")
print("=" * 80)
print()
print("Priority order (highest to lowest):")
print("  1. User override (--user-stop-loss-pct flag) - takes precedence")
print("  2. Horizon-specific stop-loss (from PROFILE_METADATA)")
print("  3. Default commodity stop-loss (2.0%)")
print()

print("=" * 80)
print("EXAMPLE VALUES")
print("=" * 80)
print()
print("For 'short' horizon commodity trading:")
risk_config = get_horizon_risk_config("short")
print(f"  Stop-Loss: {risk_config['default_stop_loss_pct']*100:.1f}%")
print(f"  Position Size: {risk_config['max_notional_per_symbol_pct']*100:.0f}%")
print(f"  Min Confidence: {risk_config['min_confidence']*100:.0f}%")
print()

print("Note: The ExecutionEngine uses 2.0% as the base for commodities,")
print("      but horizon-specific values (3.5% for short) may override it.")
print("      Check the actual execution code to see which takes precedence.")

