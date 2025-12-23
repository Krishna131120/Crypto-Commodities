"""
Pre-Live Trading Verification Checklist Script

This script runs comprehensive validation checks to ensure models are ready
for live trading. It does NOT modify any models or configurations - it only
validates and reports.

Usage:
    python verify_live_trading_readiness.py --symbol GC=F --asset-type commodities
    python verify_live_trading_readiness.py --symbol BTC-USDT --asset-type crypto --broker dhan
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from ml.live_trading_readiness import (
    ValidationResult,
    validate_model_for_live_trading,
)


def print_checklist_header():
    """Print formatted checklist header."""
    print("=" * 80)
    print("PRE-LIVE TRADING VERIFICATION CHECKLIST")
    print("=" * 80)
    print()


def print_validation_results(results: List[ValidationResult], verbose: bool = True):
    """Print validation results in a formatted way."""
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    
    print(f"Validation Results: {len(passed)} passed, {len(failed)} failed")
    print("-" * 80)
    
    if failed:
        print("\n[FAIL] FAILED CHECKS:")
        for result in failed:
            print(f"  [X] {result.check_name}")
            print(f"     {result.message}")
            if verbose and result.details:
                for key, value in result.details.items():
                    print(f"     {key}: {value}")
        print()
    
    if passed:
        print("\n[PASS] PASSED CHECKS:")
        for result in passed:
            print(f"  [OK] {result.check_name}: {result.message}")
        print()
    
    print("-" * 80)


def print_checklist_summary(is_ready: bool, symbol: str, broker: str):
    """Print final checklist summary."""
    print("\n" + "=" * 80)
    if is_ready:
        print(f"[PASS] READY FOR LIVE TRADING")
        print(f"   Symbol: {symbol}")
        print(f"   Broker: {broker.upper()}")
        print("\n   All validation checks passed. Model is ready for live trading.")
        print("   RECOMMENDATION: Start with small position sizes and monitor closely.")
    else:
        print(f"[FAIL] NOT READY FOR LIVE TRADING")
        print(f"   Symbol: {symbol}")
        print(f"   Broker: {broker.upper()}")
        print("\n   Some validation checks failed. Please address the issues above")
        print("   before attempting live trading.")
    print("=" * 80)


def print_detailed_checklist():
    """Print detailed pre-live trading checklist."""
    print("\n" + "=" * 80)
    print("DETAILED PRE-LIVE TRADING CHECKLIST")
    print("=" * 80)
    print()
    
    checklist = [
        ("Model Quality", [
            "[*] At least 2 models trained successfully",
            "[*] R² score > 0.3 for all models",
            "[*] Directional accuracy > 60% for all models",
            "[*] No severe overfitting warnings",
            "[*] Models marked as 'tradable' in summary",
        ]),
        ("Data Quality", [
            "[*] At least 500 candles of historical data",
            "[*] Latest data is less than 1 day old",
            "[*] Features file exists and is valid",
            "[*] Feature count >= 10",
        ]),
        ("Risk Management", [
            "[*] Stop-loss configured (1-10% recommended)",
            "[*] Maximum position size configured (<20% recommended)",
            "[*] Risk limits are reasonable for your capital",
        ]),
        ("Broker Configuration", [
            "[*] API credentials configured (environment variables)",
            "[*] Broker account is funded (for live trading)",
            "[*] Paper trading tested successfully",
            "[*] Order execution tested",
        ]),
        ("Infrastructure", [
            "[*] Error handling and logging configured",
            "[*] Network connectivity stable",
            "[*] API rate limits understood",
            "[*] Monitoring/alerting set up",
        ]),
        ("Testing", [
            "[*] Paper trading tested for at least 1-2 weeks",
            "[*] Models show consistent performance",
            "[*] No unexpected errors in logs",
            "[*] Position management works correctly",
        ]),
    ]
    
    for category, items in checklist:
        print(f"{category}:")
        for item in items:
            print(f"  {item}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Verify model readiness for live trading"
    )
    parser.add_argument(
        "--symbol",
        help="Symbol to validate (e.g., BTC-USDT, GC=F). Required unless --checklist-only is used.",
    )
    parser.add_argument(
        "--asset-type",
        choices=["crypto", "commodities"],
        help="Asset type. Required unless --checklist-only is used.",
    )
    parser.add_argument(
        "--checklist-only",
        action="store_true",
        help="Show only the detailed checklist without validation",
    )
    parser.add_argument(
        "--timeframe",
        default="1d",
        help="Timeframe (default: 1d)",
    )
    parser.add_argument(
        "--horizon",
        default="short",
        help="Horizon profile (default: short)",
    )
    parser.add_argument(
        "--broker",
        default=None,  # Will be auto-determined based on asset_type
        choices=["alpaca", "angelone"],
        help="Broker to validate (default: auto-determined - alpaca for crypto, angelone for commodities)",
    )
    parser.add_argument(
        "--model-dir",
        help="Override model directory path (auto-detected if not provided)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed validation information",
    )
    parser.add_argument(
        "--checklist",
        action="store_true",
        help="Show detailed pre-live trading checklist",
    )
    
    args = parser.parse_args()
    
    # Print header
    print_checklist_header()
    
    # Show detailed checklist only if requested
    if args.checklist_only:
        print_detailed_checklist()
        sys.exit(0)
    
    # If no symbol provided, show usage
    if not args.symbol or not args.asset_type:
        print("Usage: python verify_live_trading_readiness.py --symbol <SYMBOL> --asset-type <TYPE>")
        print("\nOr show checklist: python verify_live_trading_readiness.py --checklist-only")
        print("\nExample:")
        print("  python verify_live_trading_readiness.py --symbol GC=F --asset-type commodities")
        sys.exit(1)
    
    # Auto-determine broker if not specified
    if args.broker is None:
        if args.asset_type == "commodities":
            args.broker = "angelone"  # Angel One supports commodities futures
            print(f"[INFO] Auto-selected broker: Angel One (commodities require futures trading)")
        else:
            args.broker = "alpaca"  # Alpaca supports crypto
            print(f"[INFO] Auto-selected broker: Alpaca (crypto trading)")
    
    # Determine model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = Path("models") / args.asset_type / args.symbol / args.timeframe / args.horizon
    
    if not model_dir.exists():
        print(f"[ERROR] Model directory not found: {model_dir}")
        print(f"   Please train models first or specify --model-dir")
        print(f"\n   To train a model, run:")
        if args.asset_type == "commodities":
            print(f"   python end_to_end_commodities.py --commodities-symbols {args.symbol} --timeframe {args.timeframe} --horizon {args.horizon}")
        else:
            print(f"   python end_to_end_crypto.py --crypto-symbols {args.symbol} --timeframe {args.timeframe} --horizon {args.horizon}")
        sys.exit(1)
    
    # Show detailed checklist if requested
    if args.checklist:
        print_detailed_checklist()
    
    # Run validation
    print(f"Validating: {args.symbol} ({args.asset_type.upper()})")
    print(f"Model Directory: {model_dir}")
    print(f"Broker: {args.broker.upper()}")
    if args.asset_type == "commodities" and args.broker == "alpaca":
        print(f"[WARNING] Alpaca doesn't support commodity futures directly.")
        print(f"         Current setup uses ETF proxies (GC=F -> GLD).")
        print(f"         For actual futures trading, use Angel One broker.")
    print()
    
    is_ready, results = validate_model_for_live_trading(
        model_dir=model_dir,
        asset_type=args.asset_type,
        symbol=args.symbol,
        timeframe=args.timeframe,
        broker=args.broker,
    )
    
    # Print results
    print_validation_results(results, verbose=args.verbose)
    
    # Print summary
    print_checklist_summary(is_ready, args.symbol, args.broker)
    
    # Exit with appropriate code
    sys.exit(0 if is_ready else 1)


if __name__ == "__main__":
    main()
