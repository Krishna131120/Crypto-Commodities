"""
End-to-end commodities pipeline:

Runs the full flow for selected commodity symbols:
1) Historical ingestion  (raw candles)
2) Feature generation    (features.json)
3) Model training        (models/commodities/...)
4) Show predictions      (display predictions without trading)

This script is a single entry point so you can:
- Choose symbols yourself via CLI.
- Let it do all steps in sequence.
- See predictions for validation before live trading.

IMPORTANT:
- This script only shows predictions, it does NOT execute trades.
- For live trading with DHAN API, use a separate trading script (to be created).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from pipeline_runner import run_ingestion, regenerate_features
from train_commodities import train_commodity_symbols  # Use commodity-specific training
from trading.symbol_universe import find_by_data_symbol
from live_trader import load_feature_row, get_current_price_from_features, discover_tradable_symbols
from ml.inference import InferencePipeline
from ml.risk import RiskManagerConfig
from ml.horizons import print_horizon_summary, normalize_profile, DEFAULT_HORIZON_PROFILE


def show_prediction(
    symbol: str,
    asset_type: str,
    timeframe: str,
    horizon: str,
    model_dir: Path,
    verbose: bool = True
) -> Optional[Dict[str, Any]]:
    """Show prediction for a single commodity symbol."""
    
    # Load features
    feature_row = load_feature_row(asset_type, symbol, timeframe)
    if feature_row is None:
        if verbose:
            print(f"❌ {symbol}: No features found")
            print(f"   Expected path: data/features/{asset_type}/{symbol}/{timeframe}/features.json")
        return None
    
    # Get current price
    current_price = get_current_price_from_features(asset_type, symbol, timeframe, force_live=False, verbose=verbose)
    if current_price is None or current_price <= 0:
        if verbose:
            print(f"❌ {symbol}: Could not determine current price")
        return None
    
    # Calculate volatility proxy
    volatility = 0.01  # Default
    if "ATR_14" in feature_row and current_price > 0:
        atr = feature_row.get("ATR_14")
        if atr and isinstance(atr, (int, float)) and atr > 0:
            volatility = abs(float(atr) / current_price)
    if volatility == 0:
        volatility = 0.01
    
    # Load model and predict
    try:
        # Create risk config with asset-aware thresholds (commodities use lower min_confidence)
        risk_config = RiskManagerConfig(paper_trade=True, min_confidence=0.45)  # Lower threshold for commodities
        pipeline = InferencePipeline(model_dir, risk_config=risk_config, asset_type=asset_type)
        result = pipeline.predict(feature_row, current_price=current_price, volatility=volatility)
        
        # Load summary for additional info
        summary_path = model_dir / "summary.json"
        summary = {}
        if summary_path.exists():
            with open(summary_path, "r") as f:
                summary = json.load(f)
        
        # Format output
        # NOTE: result structure is {"models": {}, "consensus": {...}}
        consensus_data = result.get("consensus", {})
        action = consensus_data.get("consensus_action", "hold").upper()
        confidence = consensus_data.get("consensus_confidence", 0.0) * 100
        
        # If risk blocked, show original prediction; otherwise show current consensus
        if consensus_data.get("risk_blocked", False):
            # Risk manager blocked the trade - show original prediction
            expected_return = consensus_data.get("raw_consensus_return", consensus_data.get("consensus_return", 0.0)) * 100
            predicted_price = current_price * (1.0 + consensus_data.get("raw_consensus_return", consensus_data.get("consensus_return", 0.0)))
        else:
            # Normal case - show consensus prediction
            expected_return = consensus_data.get("consensus_return", 0.0) * 100
            predicted_price = current_price * (1.0 + consensus_data.get("consensus_return", 0.0))
        
        # Calculate support/resistance from features
        support_level = None
        resistance_level = None
        if "Pivot_S1" in feature_row:
            support_level = feature_row.get("Pivot_S1")
        elif "Bollinger_Bands_lower_20_2" in feature_row:
            support_level = feature_row.get("Bollinger_Bands_lower_20_2")
        if "Pivot_R1" in feature_row:
            resistance_level = feature_row.get("Pivot_R1")
        elif "Bollinger_Bands_upper_20_2" in feature_row:
            resistance_level = feature_row.get("Bollinger_Bands_upper_20_2")
        
        # Check for model disagreement
        # individual_models might be in consensus or models dict
        individual = consensus_data.get("individual_models", [])
        if not individual:
            # Try to extract from models dict and add R² from summary
            models_dict = result.get("models", {})
            model_predictions = summary.get("model_predictions", {})
            individual = []
            for k, v in models_dict.items():
                if isinstance(v, dict):
                    # Get R² from summary (training metrics)
                    model_summary = model_predictions.get(k, {})
                    r2_score = model_summary.get("r2_score", 0.0) or 0.0
                    individual.append({
                        "model_name": k,
                        "action": v.get("action", "hold"),
                        "predicted_return_pct": v.get("predicted_return", 0.0) * 100,
                        "confidence": v.get("confidence", 0.0) * 100,
                        "r2_score": r2_score  # Add R² from training summary
                    })
        model_actions = [m.get("action", "hold").upper() for m in individual if m.get("action")]
        models_disagree = len(set(model_actions)) > 1 if model_actions else False
        
        # Always load DQN output (for live predictions section)
        dqn_action = None
        dqn_return = None
        # Try to load DQN from summary (always, not just when models disagree)
            dqn_data = summary.get("models", {}).get("dqn", {})
            if not dqn_data:
                dqn_data = summary.get("model_predictions", {}).get("dqn", {})
            if dqn_data:
                dqn_action = dqn_data.get("action", "hold").upper()
                dqn_return_pct = dqn_data.get("predicted_return_pct")
                if dqn_return_pct is not None:
                    dqn_return = dqn_return_pct
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"📊 {symbol} ({asset_type.upper()}) - {horizon.upper()} TERM")
            print(f"{'='*80}")
            print(f"Current Price:     ${current_price:,.2f}")
            print(f"Predicted Price:   ${predicted_price:,.2f} ({expected_return:+.2f}%)")
            if support_level:
                print(f"Support Level:     ${support_level:,.2f} ({((support_level/current_price - 1)*100):+.2f}%)")
            if resistance_level:
                print(f"Resistance Level:  ${resistance_level:,.2f} ({((resistance_level/current_price - 1)*100):+.2f}%)")
            print(f"Action:            {action}")
            print(f"Confidence:        {confidence:.1f}%")
            
            # Show model quality and all models (including failed ones)
            if summary.get("tradable"):
                print(f"Model Status:      ✅ TRADABLE (passed robustness checks)")
            else:
                reasons = summary.get("tradability_reasons", [])
                print(f"Model Status:      ⚠️  NOT TRADABLE")
                if reasons:
                    for reason in reasons:
                        print(f"  ⚠️  {reason}")
            
            # Show ALL models from summary (including failed ones)
            # This section shows TRAINING/TEST SET performance (historical evaluation)
            model_predictions = summary.get("model_predictions", {})
            all_models = summary.get("models", {})  # This contains status info
            
            print(f"\n{'='*80}")
            print(f"📊 TRAINING PERFORMANCE (Test Set Evaluation)")
            print(f"{'='*80}")
            print(f"  This shows how models performed on historical test data during training.")
            print(f"  Use this to assess model quality and reliability.\n")
            successful_count = 0
            failed_count = 0
            
            # Check each expected model
            expected_models = ["random_forest", "lightgbm", "xgboost", "stacked_blend", "dqn"]
            for model_name in expected_models:
                model_data = all_models.get(model_name, {})
                model_pred = model_predictions.get(model_name, {})
                
                if model_data.get("status") == "failed":
                    failed_count += 1
                    reason = model_data.get("reason", "Unknown")
                    print(f"  ❌ {model_name:15s}: FAILED - {reason[:60]}")
                elif model_name in model_predictions:
                    successful_count += 1
                    pred_data = model_predictions[model_name]
                    model_action = pred_data.get("action", "hold").upper()
                    model_return = pred_data.get("predicted_return_pct", 0.0)
                    model_r2 = pred_data.get("r2_score", 0.0) or 0.0
                    model_conf = pred_data.get("confidence", 0.0) or 0.0
                    model_price = pred_data.get("predicted_price", current_price)
                    action_emoji = "🟢" if model_action == "LONG" else "🔴" if model_action == "SHORT" else "⚪"
                    print(f"  ✅ {model_name:15s}: {action_emoji} {model_action:5s} | "
                          f"Price: ${model_price:,.2f} ({model_return:+.2f}%) | "
                          f"R²={model_r2:.3f} | Conf={model_conf:.1f}%")
                elif model_name == "dqn" and model_data:
                    # DQN might be in models but not in model_predictions
                    successful_count += 1
                    dqn_action = model_data.get("action", "hold").upper()
                    print(f"  ✅ {model_name:15s}: {dqn_action:5s} (DQN - policy-based)")
                else:
                    print(f"  ⚪ {model_name:15s}: NOT TRAINED")
            
            print(f"\n  Summary: {successful_count} successful, {failed_count} failed, {len(expected_models) - successful_count - failed_count} not trained")
            
            # Show individual model predictions from inference (LIVE DATA)
            # This is the section that matters for trading decisions
            if individual:
                print(f"\n{'='*80}")
                print(f"📈 LIVE TRADING PREDICTIONS (Current Market Data)")
                print(f"{'='*80}")
                print(f"  ⚠️  USE THESE PREDICTIONS FOR TRADING DECISIONS")
                print(f"  These are fresh predictions using current live market features.\n")
                
                # Analyze model agreement
                live_actions = {}
                for model_info in individual:
                    model_name = model_info.get("model_name", "unknown")
                    model_action = model_info.get("action", "hold").upper()
                    live_actions[model_name] = model_action
                
                # Count actions
                action_counts = {}
                for action in live_actions.values():
                    action_counts[action] = action_counts.get(action, 0) + 1
                
                # Determine if models agree (check non-HOLD actions)
                non_hold_actions = [a for a in live_actions.values() if a != "HOLD"]
                unique_non_hold = set(non_hold_actions)
                
                if len(unique_non_hold) <= 1:
                    if len(non_hold_actions) > 0:
                        agreement_status = "✅ MODELS AGREE"
                        agreement_emoji = "✅"
                    else:
                        agreement_status = "⚪ ALL MODELS HOLD"
                        agreement_emoji = "⚪"
                else:
                    agreement_status = "⚠️  MODELS DISAGREE"
                    agreement_emoji = "⚠️"
                
                active_count = len(non_hold_actions)
                total_count = len(live_actions)
                print(f"  {agreement_status}: {active_count}/{total_count} models with active signals")
                if len(action_counts) > 1:
                    print(f"     Action breakdown: {', '.join([f'{k}: {v}' for k, v in sorted(action_counts.items())])}")
                print()
                
                # Show each model's live prediction
                for model_info in individual:
                    model_name = model_info.get("model_name", "unknown")
                    model_action = model_info.get("action", "hold").upper()
                    model_return = model_info.get("predicted_return_pct", 0.0)
                    model_r2 = model_info.get("r2_score", 0.0)
                    model_conf = model_info.get("confidence", 0.0)
                    model_price = current_price * (1.0 + model_return / 100.0)
                    action_emoji = "🟢" if model_action == "LONG" else "🔴" if model_action == "SHORT" else "⚪"
                    print(f"  {action_emoji} {model_name:15s}: {model_action:5s} | "
                          f"Price: ${model_price:,.2f} ({model_return:+.2f}%) | "
                          f"R²={model_r2:.3f} | Conf={model_conf:.1f}%")
            
                # Always show DQN decision in live predictions section
                print()
                if dqn_action:
                dqn_price = current_price * (1.0 + dqn_return / 100.0) if dqn_return else None
                dqn_emoji = "🟢" if dqn_action == "LONG" else "🔴" if dqn_action == "SHORT" else "⚪"
                    if models_disagree:
                        print(f"  🤖 DQN DECISION (Models Disagree - DQN Recommendation):")
                    else:
                        print(f"  🤖 DQN DECISION (Policy-Based Recommendation):")
                if dqn_price:
                        print(f"     {dqn_emoji} Action: {dqn_action:5s} | "
                          f"Price: ${dqn_price:,.2f} ({dqn_return:+.2f}%)")
                else:
                        print(f"     {dqn_emoji} Action: {dqn_action}")
                else:
                    print(f"  🤖 DQN: Not available")
            
            # Show reasoning (from consensus)
            reasoning = consensus_data.get("reasoning", "")
            if reasoning:
                print(f"\n💭 Reasoning: {reasoning}")
            
            # Show risk blocking info if trade was blocked
            if consensus_data.get("risk_blocked", False):
                raw_action = consensus_data.get("raw_consensus_action", "hold").upper()
                raw_return = consensus_data.get("raw_consensus_return", 0.0) * 100
                raw_confidence = consensus_data.get("raw_consensus_confidence", 0.0) * 100
                print(f"\n⚠️  Risk Manager: Trade BLOCKED")
                print(f"   Confidence: {confidence:.1f}% (minimum: {risk_config.min_confidence*100:.0f}%)")
                print(f"   Original prediction: {raw_action} ({raw_return:+.2f}%, conf: {raw_confidence:.1f}%)")
                # Also show what the prediction would have been
                if raw_return != 0:
                    raw_predicted_price = current_price * (1.0 + raw_return / 100.0)
                    print(f"   Would have predicted: ${raw_predicted_price:,.2f} (from ${current_price:,.2f})")
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "expected_return_pct": expected_return,
            "action": action.lower(),
            "confidence": confidence,
            "tradable": summary.get("tradable", False),
            "result": result
        }
        
    except Exception as e:
        if verbose:
            print(f"❌ {symbol}: Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end commodities pipeline: ingest -> features -> train -> show predictions."
    )
    parser.add_argument(
        "--commodities-symbols",
        nargs="+",
        required=True,
        help="Commodity symbols (e.g., GC=F CL=F SI=F PL=F).",
    )
    parser.add_argument(
        "--timeframe",
        default="1d",
        help="Timeframe to use for ingestion and models (default: 1d).",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=10.0,
        help="Approximate years of historical data to request (default: 10).",
    )
    parser.add_argument(
        "--horizon",
        default="short",
        choices=["intraday", "short", "long"],
        help="Horizon profile for training (intraday/short/long). Default: short",
    )

    args = parser.parse_args()
    
    # Normalize symbols
    raw_symbols = [s.strip().upper() for s in args.commodities_symbols if s.strip()]
    commodities_symbols = raw_symbols  # Commodity symbols are usually already in correct format (GC=F, CL=F, etc.)
    
    timeframe = args.timeframe
    years = max(args.years, 0.5)
    horizon = args.horizon  # Single horizon parameter for commodities-only script

    print("=" * 80)
    print("END-TO-END COMMODITIES PIPELINE")
    print("=" * 80)
    print(f"Symbols:   {', '.join(commodities_symbols)}")
    print(f"Timeframe: {timeframe}")
    print(f"Horizon:   {horizon}")
    print(f"Mode:      PREDICTIONS ONLY (no trading)")
    print("=" * 80)
    print()
    
    # Show available horizons and their trading behavior
    print_horizon_summary()

    # ------------------------------------------------------------------
    # Stage 1: Historical ingestion (COMMODITIES ONLY)
    # ------------------------------------------------------------------
    print("[1/4] Ingesting historical data...")
    run_ingestion(
        mode="historical",
        crypto_symbols=None,  # Commodities-only script - no crypto
        commodities_symbols=commodities_symbols,
        timeframe=timeframe,
        years=years,
    )
    print("    ✓ Historical data ingestion complete.")

    # ------------------------------------------------------------------
    # Stage 2: Feature generation (COMMODITIES ONLY)
    # ------------------------------------------------------------------
    print("[2/4] Regenerating features...")
    regenerate_features("commodities", set(commodities_symbols), timeframe)
    print("    ✓ Feature generation complete.")

    # ------------------------------------------------------------------
    # Stage 3: Model training (COMMODITIES ONLY)
    # ------------------------------------------------------------------
    print("[3/4] Training models...")
    print("    NOTE: Each model training starts with 'Trial 0' - this is normal.")
    print("    Optuna creates a fresh study for each model, so trials always start at 0.")
    print()
    # Use commodity-specific training function (separate from crypto)
    horizon_map = {symbol: horizon for symbol in commodities_symbols}  # Map each symbol to horizon
    train_commodity_symbols(
        commodities_symbols=commodities_symbols,
        timeframe=timeframe,
        output_dir="models",
        horizon_profiles=horizon_map,
    )
    print("    ✓ Model training complete.")

    # ------------------------------------------------------------------
    # Stage 4: Show predictions
    # ------------------------------------------------------------------
    print("[4/4] Generating predictions...")
    print()

    # Discover which of the requested symbols actually have trained models.
    # CRITICAL: Pass the trained horizon as override_horizon so discover_tradable_symbols
    # finds the correct models (e.g., intraday models instead of default short models)
    all_tradable = discover_tradable_symbols(
        asset_type="commodities", 
        timeframe=timeframe,
        override_horizon=horizon  # Override asset's default horizon_profile with the trained horizon
    )
    
    # Restrict to the user-selected symbols only.
    requested_set = {s.upper() for s in commodities_symbols}
    tradable = [
        info
        for info in all_tradable
        if info["asset"].data_symbol.upper() in requested_set
    ]

    if not tradable:
        print("    ✗ No tradable symbols found after training. Exiting.")
        print("    This usually means:")
        print("      - Models failed robustness checks")
        print("      - Models were not trained successfully")
        print("      - Check the training output above for errors")
        return

    print(f"    ✓ Found {len(tradable)} tradable symbol(s):")
    for info in tradable:
        asset = info["asset"]
        print(
            f"      - {asset.data_symbol} - horizon: {info['horizon']}"
        )
    print()

    # Show predictions for each tradable symbol
    results = []
    for info in tradable:
        asset = info["asset"]
        symbol = asset.data_symbol
        model_dir = info["model_dir"]
        
        result = show_prediction(
            symbol=symbol,
            asset_type="commodities",
            timeframe=timeframe,
            horizon=horizon,
            model_dir=model_dir,
            verbose=True
        )
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total Symbols:     {len(commodities_symbols)}")
    print(f"Successful:        {len(results)}")
    print(f"Failed:            {len(commodities_symbols) - len(results)}")
    
    if results:
        print(f"\nPredictions:")
        for r in results:
            action_emoji = "🟢" if r["action"] == "long" else "🔴" if r["action"] == "short" else "⚪"
            tradable_mark = "✅" if r["tradable"] else "⚠️"
            print(f"  {action_emoji} {r['symbol']:10s}: {r['action']:5s} "
                  f"({r['expected_return_pct']:+.2f}%, conf: {r['confidence']:.1f}%) {tradable_mark}")
    
    print(f"\n{'='*80}")
    print("NOTE: These are predictions only. No trades were executed.")
    print("      For live trading with DHAN API, use a separate trading script.")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Fix Windows console encoding for emojis
    import sys
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    main()

