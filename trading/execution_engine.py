"""
Execution engine: turn model consensus into Alpaca paper trades (crypto only).

This module focuses on:
- Reading consensus outputs from the inference pipeline
  (e.g. consensus_action, consensus_return, position_size).
- Comparing desired exposure with current Alpaca positions.
- Submitting market/bracket orders for crypto on the paper account.

Supported states for each symbol:
- FLAT  : no position
- LONG  : positive quantity
- SHORT : negative quantity (if Alpaca/account permits shorting the asset)

NOTE:
- Whether SHORT is actually allowed for a given symbol depends on Alpaca
  and your account configuration. The engine exposes the logic, but the
  API may reject short orders for unsupported assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import json

from .alpaca_client import AlpacaClient
from .symbol_universe import AssetMapping
from .position_manager import PositionManager
from ml.horizons import get_horizon_risk_config, normalize_profile


@dataclass
class TradingRiskConfig:
    """
    Basic risk limits for live paper trading.

    All values are conservative defaults and can be tuned later.
    """

    max_notional_per_symbol_pct: float = 0.10  # max 10% of equity per symbol
    max_total_equity_pct: float = 0.50        # cap total deployed capital at 50% equity (not enforced yet)
    default_stop_loss_pct: float = 0.02       # 2% stop-loss distance from entry
    take_profit_risk_multiple: float = 2.0    # TP distance = multiple * stop-loss distance
    min_confidence: float = 0.10              # ignore very low-confidence signals
    profit_target_pct: Optional[float] = None  # User's desired profit percentage (e.g., 10.0 for 10%)
    # IMPORTANT: Shorting support depends on your Alpaca account and asset type.
    # Paper trading may not support crypto shorts, but live trading accounts might.
    # This engine is designed to work correctly in both environments.
    allow_short: bool = True                  # enable shorting (ready for live trading)


class ExecutionEngine:
    """
    Bridge between model consensus and Alpaca orders (crypto only, long/flat).
    """

    def __init__(
        self,
        client: Optional[AlpacaClient] = None,
        risk_config: Optional[TradingRiskConfig] = None,
        log_path: Path = Path("logs") / "trading" / "crypto_trades.jsonl",
        default_horizon: Optional[str] = None,
        position_manager: Optional[PositionManager] = None,
    ):
        self.client = client or AlpacaClient()
        self.risk = risk_config or TradingRiskConfig()
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_horizon = normalize_profile(default_horizon) if default_horizon else None
        self.position_manager = position_manager or PositionManager()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def execute_from_consensus(
        self,
        asset: AssetMapping,
        consensus: Dict[str, Any],
        current_price: float,
        dry_run: bool = False,
        horizon_profile: Optional[str] = None,
        profit_target_pct: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Given a consensus dict and current price, align Alpaca position.

        asset:      AssetMapping (crypto or commodities)
        consensus:  InferencePipeline consensus payload, e.g.:
                    {
                        "consensus_action": "long" | "short" | "hold",
                        "consensus_return": float,
                        "position_size": float,   # fraction of equity (0..1)
                        "consensus_confidence": float,
                        ...
                    }
        current_price: latest market price for the asset
        dry_run:   if True, only logs intended action without sending orders
        horizon_profile: Optional horizon profile override
        profit_target_pct: User's desired profit percentage (e.g., 10.0 for 10%). If provided,
                          position sizing will be adjusted to achieve this target.
        """
        action = str(consensus.get("consensus_action", "hold")).lower()
        confidence = float(consensus.get("consensus_confidence", 0.0) or 0.0)
        position_size = float(consensus.get("position_size", 0.0) or 0.0)
        
        # Use profit_target_pct from parameter or risk config
        effective_profit_target = profit_target_pct or self.risk.profit_target_pct

        # Determine which horizon profile to use (from parameter, asset mapping, or default)
        effective_horizon = normalize_profile(horizon_profile) if horizon_profile else (
            normalize_profile(getattr(asset, "horizon_profile", None)) if hasattr(asset, "horizon_profile") else self.default_horizon
        ) or "short"  # Fallback to short if nothing specified
        
        # Get horizon-specific risk parameters and create a temporary risk config override
        horizon_risk = get_horizon_risk_config(effective_horizon)
        # Create a risk config that uses horizon-specific values, falling back to base config
        effective_risk = TradingRiskConfig(
            max_notional_per_symbol_pct=horizon_risk.get("max_notional_per_symbol_pct", self.risk.max_notional_per_symbol_pct),
            max_total_equity_pct=self.risk.max_total_equity_pct,
            default_stop_loss_pct=horizon_risk.get("default_stop_loss_pct", self.risk.default_stop_loss_pct),
            take_profit_risk_multiple=self.risk.take_profit_risk_multiple,
            min_confidence=horizon_risk.get("min_confidence", self.risk.min_confidence),
            allow_short=self.risk.allow_short,
        )

        trading_symbol = asset.trading_symbol.upper()
        is_crypto = getattr(asset, "asset_type", "").lower() == "crypto"
        account = self.client.get_account()
        equity = float(account.get("equity", 0.0) or 0.0)
        buying_power = float(account.get("buying_power", 0.0) or 0.0)

        if equity <= 0 or buying_power <= 0:
            return None

        existing_position = self.client.get_position(trading_symbol)
        existing_qty = float(existing_position["qty"]) if existing_position else 0.0
        if existing_qty > 0:
            side_in_market = "long"
        elif existing_qty < 0:
            side_in_market = "short"
        else:
            side_in_market = "flat"

        # Determine target side from model action.
        if action == "short" and self.risk.allow_short:
            target_side = "short"
        elif action == "long":
            target_side = "long"
        else:
            target_side = "flat"
        
        # NEW STRATEGY: Exit IMMEDIATELY when profit target is hit, regardless of prediction
        # Check if we have a tracked position with profit target
        tracked_position = self.position_manager.get_position(trading_symbol)
        profit_target_hit = False
        
        if tracked_position and tracked_position.status == "open":
            # Check if profit target is hit
            if tracked_position.side == "long":
                profit_target_hit = current_price >= tracked_position.profit_target_price
            elif tracked_position.side == "short":
                profit_target_hit = current_price <= tracked_position.profit_target_price
        
        # PRIORITY 1: Exit IMMEDIATELY if profit target is hit (regardless of prediction)
        must_exit_position = profit_target_hit
        
        # PRIORITY 2: Also exit on stop-loss (safety mechanism)
        stop_loss_hit = False
        if tracked_position and tracked_position.status == "open":
            if tracked_position.side == "long":
                stop_loss_hit = current_price <= tracked_position.stop_loss_price
            elif tracked_position.side == "short":
                stop_loss_hit = current_price >= tracked_position.stop_loss_price
        
        if stop_loss_hit:
            must_exit_position = True
        
        # NEW TRADING LOGIC: Buy on ANY UP prediction (even small), regardless of confidence
        # Only apply confidence threshold for SHORT positions or when exiting
        # If we need to exit a position, do it immediately regardless of confidence
        if must_exit_position:
            # Exit logic will be handled below, but we skip confidence check for exits
            pass
        elif side_in_market == "flat":
            # No existing position - check if we should enter
            # NEW: For LONG positions, buy on ANY positive prediction (even small)
            # Only require confidence threshold for SHORT positions
            if target_side == "long" and action == "long":
                # Buy on any UP prediction, even if confidence is low
                # The profit target will ensure we only hold until target is hit
                pass  # Continue to enter position
            elif target_side == "short" and action == "short" and confidence < effective_risk.min_confidence:
                # For SHORT positions, still require confidence threshold (more risky)
                return None
            elif action == "hold" or target_side == "flat":
                # Model says hold or flat - don't enter
                return None
            elif confidence < effective_risk.min_confidence:
                # Fallback: if action is not long/short/hold, require confidence
                return None
        else:
            # We have an existing position - continue to exit logic below
            pass

        # Determine target notional for exposure, *clamped by buying power*.
        # If profit_target_pct is provided, adjust position sizing to achieve the target.
        max_symbol_notional = equity * effective_risk.max_notional_per_symbol_pct
        
        if effective_profit_target is not None and effective_profit_target > 0:
            # Profit-target-based position sizing
            # Get model's predicted return
            predicted_return = abs(consensus.get("consensus_return", 0.0) or 0.0)
            
            if predicted_return > 0:
                # Calculate position size needed to achieve profit target
                # If model predicts 5% return and user wants 10%, we need 2x position
                # But we cap it to avoid over-leveraging
                profit_multiplier = min(effective_profit_target / predicted_return, 2.0)  # Cap at 2x
                adjusted_position_size = position_size * profit_multiplier
            else:
                # Model predicts no return or very small return - use base position size
                adjusted_position_size = position_size
            
            # Still respect max position limits
            raw_desired_notional = max(0.0, min(adjusted_position_size * equity, max_symbol_notional))
        else:
            # Standard position sizing (no profit target)
            raw_desired_notional = max(0.0, min(position_size * equity, max_symbol_notional))
        
        desired_notional = min(raw_desired_notional, buying_power)

        # Decide order(s) needed to move from side_in_market -> target_side.
        orders: Dict[str, Any] = {
            "asset": asset.logical_name,
            "data_symbol": asset.data_symbol,
            "trading_symbol": trading_symbol,
            "current_price": current_price,
            "model_action": action,
            "target_side": target_side,
            "existing_side": side_in_market,
            "existing_qty": existing_qty,
            "desired_notional": desired_notional,
            "equity": equity,
            "buying_power": buying_power,
            "horizon_profile": effective_horizon,
            "horizon_max_notional_pct": effective_risk.max_notional_per_symbol_pct,
            "horizon_stop_loss_pct": effective_risk.default_stop_loss_pct,
            "horizon_min_confidence": effective_risk.min_confidence,
            "profit_target_pct": effective_profit_target,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "dry_run": dry_run,
        }

        if side_in_market == target_side:
            # Already aligned (long/short/flat) - NO PYRAMIDING, just hold the position
            # Check position status and provide detailed documentation
            tracked_position = self.position_manager.get_position(trading_symbol)
            
            if tracked_position and tracked_position.status == "open":
                # Calculate current profit/loss status
                if tracked_position.side == "long":
                    unrealized_pl_pct = ((current_price - tracked_position.entry_price) / tracked_position.entry_price) * 100
                    profit_target_hit = current_price >= tracked_position.profit_target_price
                    stop_loss_hit = current_price <= tracked_position.stop_loss_price
                else:  # short
                    unrealized_pl_pct = ((tracked_position.entry_price - current_price) / tracked_position.entry_price) * 100
                    profit_target_hit = current_price <= tracked_position.profit_target_price
                    stop_loss_hit = current_price >= tracked_position.stop_loss_price
                
                unrealized_pl = (current_price - tracked_position.entry_price) * tracked_position.quantity if tracked_position.side == "long" else (tracked_position.entry_price - current_price) * tracked_position.quantity
                progress_to_target = (unrealized_pl_pct / tracked_position.profit_target_pct) * 100 if tracked_position.profit_target_pct > 0 else 0
                
                # Detailed position documentation
                orders["decision"] = "hold_position"
                orders["position_status"] = {
                    "symbol": trading_symbol,
                    "side": tracked_position.side,
                    "entry_price": tracked_position.entry_price,
                    "entry_time": tracked_position.entry_time,
                    "quantity": tracked_position.quantity,
                    "current_price": current_price,
                    "profit_target_pct": tracked_position.profit_target_pct,
                    "profit_target_price": tracked_position.profit_target_price,
                    "stop_loss_pct": tracked_position.stop_loss_pct,
                    "stop_loss_price": tracked_position.stop_loss_price,
                    "unrealized_pl": unrealized_pl,
                    "unrealized_pl_pct": unrealized_pl_pct,
                    "progress_to_target_pct": progress_to_target,
                    "profit_target_hit": profit_target_hit,
                    "stop_loss_hit": stop_loss_hit,
                    "target_status": "HIT" if profit_target_hit else ("AT_RISK" if stop_loss_hit else "IN_PROGRESS"),
                    "why_target_not_hit": None if profit_target_hit else (
                        f"Price ${current_price:.2f} below target ${tracked_position.profit_target_price:.2f} (need +{tracked_position.profit_target_pct - unrealized_pl_pct:.2f}% more)" if tracked_position.side == "long" else
                        f"Price ${current_price:.2f} above target ${tracked_position.profit_target_price:.2f} (need -{abs(tracked_position.profit_target_pct - unrealized_pl_pct):.2f}% more)"
                    ),
                }
                
                # Calculate investment details
                initial_investment = tracked_position.entry_price * tracked_position.quantity
                current_value = current_price * tracked_position.quantity
                
                # Calculate expected profit/loss at target and stop-loss
                if tracked_position.side == "long":
                    expected_profit_at_target = (tracked_position.profit_target_price - tracked_position.entry_price) * tracked_position.quantity
                    expected_loss_at_stop = (tracked_position.stop_loss_price - tracked_position.entry_price) * tracked_position.quantity
                else:  # short
                    expected_profit_at_target = (tracked_position.entry_price - tracked_position.profit_target_price) * tracked_position.quantity
                    expected_loss_at_stop = (tracked_position.stop_loss_price - tracked_position.entry_price) * tracked_position.quantity
                
                # Print detailed status
                print(f"\n{'='*80}")
                print(f"POSITION STATUS: {trading_symbol}")
                print(f"{'='*80}")
                print(f"\n💰 INVESTMENT DETAILS:")
                print(f"  Initial Investment: ${initial_investment:.2f}")
                print(f"    └─ Entry Price:   ${tracked_position.entry_price:.2f}")
                print(f"    └─ Quantity:      {tracked_position.quantity:.6f}")
                print(f"    └─ Entry Time:     {tracked_position.entry_time}")
                print(f"    └─ Side:           {tracked_position.side.upper()}")
                
                print(f"\n📊 CURRENT STATUS:")
                print(f"  Current Price:     ${current_price:.2f}")
                print(f"  Current Value:     ${current_value:.2f}")
                print(f"  Current P/L:       ${unrealized_pl:+.2f} ({unrealized_pl_pct:+.2f}%)")
                print(f"  Progress to Target: {progress_to_target:.1f}%")
                
                print(f"\n🎯 TARGET SCENARIOS:")
                print(f"  Profit Target:")
                print(f"    └─ Target Price:  ${tracked_position.profit_target_price:.2f} ({tracked_position.profit_target_pct:+.2f}%)")
                print(f"    └─ Expected Profit: ${expected_profit_at_target:+.2f}")
                print(f"    └─ Total Value at Target: ${initial_investment + expected_profit_at_target:.2f}")
                print(f"  Stop-Loss:")
                print(f"    └─ Stop Price:    ${tracked_position.stop_loss_price:.2f} ({tracked_position.stop_loss_pct:.2f}%)")
                print(f"    └─ Expected Loss: ${expected_loss_at_stop:.2f}")
                print(f"    └─ Total Value at Stop: ${initial_investment + expected_loss_at_stop:.2f}")
                
                print(f"\nStatus:           {'✅ TARGET HIT' if profit_target_hit else ('⚠️  AT RISK (Stop-Loss)' if stop_loss_hit else '⏳ IN PROGRESS')}")
                if not profit_target_hit and not stop_loss_hit:
                    print(f"Reason Not Hit:  {orders['position_status']['why_target_not_hit']}")
                print(f"{'='*80}\n")
            else:
                orders["decision"] = "no_action_needed"
                orders["reason"] = "position_aligned_no_tracking"
            
            self._log(orders)
            return orders
        
        # Handle exit logic (when must_exit_position is True)
        if must_exit_position:
            # Exit position immediately (profit target hit or stop-loss hit)
            # Get position details for documentation
            tracked_position = self.position_manager.get_position(trading_symbol)
            avg_entry_price = float(existing_position.get("avg_entry_price", 0) or 0) if existing_position else current_price
            market_value = abs(existing_qty) * current_price
            
            # Calculate realized P/L
            if side_in_market == "long":
                realized_pl = (current_price - avg_entry_price) * abs(existing_qty)
                realized_pl_pct = ((current_price - avg_entry_price) / avg_entry_price) * 100 if avg_entry_price > 0 else 0
            else:  # short
                realized_pl = (avg_entry_price - current_price) * abs(existing_qty)
                realized_pl_pct = ((avg_entry_price - current_price) / avg_entry_price) * 100 if avg_entry_price > 0 else 0
            
            # Determine exit reason
            exit_reason = "profit_target_hit" if profit_target_hit else "stop_loss_hit"
            
            if dry_run:
                orders["decision"] = "would_exit_position"
                orders["trade_qty"] = abs(existing_qty)
                orders["trade_side"] = "sell" if existing_qty > 0 else "buy"
                orders["final_side"] = "flat"
                orders["final_qty"] = 0.0
                orders["exit_reason"] = exit_reason
                orders["entry_price"] = avg_entry_price
                orders["exit_price"] = current_price
                orders["realized_pl"] = realized_pl
                orders["realized_pl_pct"] = realized_pl_pct
                self._log(orders)
                return orders
            
            # Execute exit order
            close_resp = self.client.submit_order(
                symbol=trading_symbol,
                qty=abs(existing_qty) if asset.asset_type == "commodities" else None,
                notional=round(abs(existing_qty) * current_price, 2) if asset.asset_type == "crypto" else None,
                side="sell" if existing_qty > 0 else "buy",
                order_type="market",
                time_in_force="gtc",
            )
            
            # Cancel any pending stop-loss or take-profit orders before closing position
            if tracked_position:
                # Cancel stop-loss order if it exists
                if tracked_position.stop_loss_order_id:
                    try:
                        self.client.cancel_order(tracked_position.stop_loss_order_id)
                        print(f"  ✅ Cancelled stop-loss order: {tracked_position.stop_loss_order_id}")
                    except Exception as cancel_exc:
                        print(f"  ⚠️  Could not cancel stop-loss order {tracked_position.stop_loss_order_id}: {cancel_exc}")
                
                # Cancel take-profit order if it exists
                if tracked_position.take_profit_order_id:
                    try:
                        self.client.cancel_order(tracked_position.take_profit_order_id)
                        print(f"  ✅ Cancelled take-profit order: {tracked_position.take_profit_order_id}")
                    except Exception as cancel_exc:
                        print(f"  ⚠️  Could not cancel take-profit order {tracked_position.take_profit_order_id}: {cancel_exc}")
                
                # Close position in position manager
                self.position_manager.close_position(
                    trading_symbol,
                    current_price,
                    exit_reason,
                    realized_pl,
                    realized_pl_pct,
                )
            
            orders["decision"] = "exit_position"
            orders["trade_qty"] = abs(existing_qty)
            orders["trade_side"] = "sell" if existing_qty > 0 else "buy"
            orders["final_side"] = "flat"
            orders["final_qty"] = 0.0
            orders["close_order"] = close_resp
            orders["exit_reason"] = exit_reason
            orders["entry_price"] = avg_entry_price
            orders["exit_price"] = current_price
            orders["realized_pl"] = realized_pl
            orders["realized_pl_pct"] = realized_pl_pct
            orders["market_value_at_exit"] = market_value
            
            # Calculate investment details for exit
            initial_investment = avg_entry_price * abs(existing_qty)
            exit_value = current_price * abs(existing_qty)
            
            # Print exit documentation
            print(f"\n{'='*80}")
            print(f"POSITION EXITED: {trading_symbol}")
            print(f"{'='*80}")
            print(f"\n💰 INVESTMENT SUMMARY:")
            print(f"  Initial Investment: ${initial_investment:.2f}")
            print(f"    └─ Entry Price:   ${avg_entry_price:.2f}")
            print(f"    └─ Quantity:      {abs(existing_qty):.6f}")
            print(f"  Exit Value:         ${exit_value:.2f}")
            print(f"    └─ Exit Price:    ${current_price:.2f}")
            print(f"    └─ Quantity:      {abs(existing_qty):.6f}")
            print(f"\n📊 FINAL RESULTS:")
            print(f"  Realized P/L:       ${realized_pl:+.2f} ({realized_pl_pct:+.2f}%)")
            print(f"  Return on Investment: {(realized_pl / initial_investment * 100):+.2f}%")
            print(f"  Exit Reason:         {exit_reason.upper().replace('_', ' ')}")
            print(f"  Order ID:            {close_resp.get('id', 'N/A')}")
            print(f"{'='*80}\n")
            
            self._log(orders)
            return orders

        # First, exit any existing position if target is flat (model says "hold").
        # This happens when model changes from "long" to "hold" or "short" to "hold".
        # We ALWAYS exit in this case, regardless of confidence, to lock in profits or cut losses.
        if side_in_market in {"long", "short"} and target_side == "flat":
            if existing_qty == 0:
                orders["decision"] = "exit_position_skipped_no_qty"
                orders["trade_qty"] = 0.0
                orders["final_side"] = "flat"
                orders["final_qty"] = 0.0
                self._log(orders)
                return orders
            if dry_run:
                orders["decision"] = "would_exit_position"
                orders["trade_qty"] = abs(existing_qty)
                orders["trade_side"] = "sell" if existing_qty > 0 else "buy"
                orders["final_side"] = "flat"
                orders["final_qty"] = 0.0
                orders["exit_reason"] = "model_changed_to_hold"
                self._log(orders)
                return orders
            
            # Get position details for profit calculation
            avg_entry_price = float(existing_position.get("avg_entry_price", 0) or 0) if existing_position else current_price
            market_value = float(existing_position.get("market_value", 0) or 0) if existing_position else (abs(existing_qty) * current_price)
            unrealized_pl = float(existing_position.get("unrealized_pl", 0) or 0) if existing_position else 0
            unrealized_pl_pct = float(existing_position.get("unrealized_plpc", 0) or 0) * 100 if existing_position else 0
            
            close_resp = self.client.submit_order(
                symbol=trading_symbol,
                qty=abs(existing_qty),
                side="sell" if existing_qty > 0 else "buy",
                order_type="market",
                time_in_force="gtc",
            )
            orders["decision"] = "exit_position"
            orders["trade_qty"] = abs(existing_qty)
            orders["trade_side"] = "sell" if existing_qty > 0 else "buy"
            orders["final_side"] = "flat"
            orders["final_qty"] = 0.0
            orders["close_order"] = close_resp
            orders["exit_reason"] = "model_changed_to_hold"
            orders["entry_price"] = avg_entry_price
            orders["exit_price"] = current_price
            orders["realized_pl"] = unrealized_pl
            orders["realized_pl_pct"] = unrealized_pl_pct
            orders["market_value_at_exit"] = market_value
            self._log(orders)
            return orders

        # Flipping positions: long -> short or short -> long
        # This happens when model changes from "long" to "short" or vice versa.
        # We ALWAYS close the existing position first, regardless of confidence.
        # This ensures we exit immediately when model changes its mind.
        if side_in_market in {"long", "short"} and target_side in {"long", "short"} and side_in_market != target_side:
            # Need to close existing position first, then open new one
            if existing_qty == 0:
                orders["decision"] = "flip_position_skipped_no_qty"
                orders["final_side"] = side_in_market
                orders["final_qty"] = existing_qty
                self._log(orders)
                return orders
            
            # Get position details for profit calculation before closing
            avg_entry_price = float(existing_position.get("avg_entry_price", 0) or 0) if existing_position else current_price
            market_value = float(existing_position.get("market_value", 0) or 0) if existing_position else (abs(existing_qty) * current_price)
            unrealized_pl = float(existing_position.get("unrealized_pl", 0) or 0) if existing_position else 0
            unrealized_pl_pct = float(existing_position.get("unrealized_plpc", 0) or 0) * 100 if existing_position else 0
            
            if dry_run:
                orders["decision"] = "would_flip_position"
                orders["trade_qty"] = abs(existing_qty)
                orders["trade_side"] = "sell" if existing_qty > 0 else "buy"
                orders["final_side"] = target_side
                orders["exit_reason"] = f"model_changed_from_{side_in_market}_to_{target_side}"
                orders["entry_price"] = avg_entry_price
                orders["exit_price"] = current_price
                orders["realized_pl"] = unrealized_pl
                orders["realized_pl_pct"] = unrealized_pl_pct
                # Will open new position after closing
                new_qty = max(desired_notional / current_price, 0.0) if desired_notional > 0 else 0.0
                orders["final_qty"] = new_qty if target_side == "long" else -new_qty
                self._log(orders)
                return orders
            
            # Close existing position IMMEDIATELY when model changes direction
            # This locks in profits or cuts losses when prediction changes
            close_resp = self.client.submit_order(
                symbol=trading_symbol,
                qty=abs(existing_qty),
                side="sell" if existing_qty > 0 else "buy",
                order_type="market",
                time_in_force="gtc",
            )
            
            # Now open new position in opposite direction
            if desired_notional > 0:
                new_qty = max(desired_notional / current_price, 0.0)
                if new_qty > 0:
                    if target_side == "long":
                        stop_loss_price = current_price * (1.0 - effective_risk.default_stop_loss_pct)
                        side = "buy"
                    else:  # short
                        stop_loss_price = current_price * (1.0 + effective_risk.default_stop_loss_pct)
                        side = "sell"
                    
                    # Open new position in opposite direction with proper risk management.
                    # For crypto shorts, use notional (rounded to 2 decimals as required).
                    # For longs or non-crypto, use qty with optional bracket orders.
                    if is_crypto and target_side == "short":
                        # Crypto shorts: use USD notional, rounded to 2 decimals with safety checks
                        notional_rounded = round(desired_notional, 2)
                        
                        # Safety check: ensure rounded notional is positive and meaningful
                        if notional_rounded <= 0 or notional_rounded < 0.01:
                            # Notional too small after rounding - stay flat after closing
                            orders["decision"] = "flip_to_flat_after_close"
                            orders["close_order"] = close_resp
                            orders["trade_qty"] = abs(existing_qty)
                            orders["trade_side"] = "sell" if existing_qty > 0 else "buy"
                            orders["final_side"] = "flat"
                            orders["final_qty"] = 0.0
                            orders["reason"] = f"Rounded notional too small: {notional_rounded} (original: {desired_notional})"
                            self._log(orders)
                            return orders
                        
                        # Re-fetch buying power after closing position to ensure accuracy
                        account_after_close = self.client.get_account()
                        buying_power_after = float(account_after_close.get("buying_power", 0.0) or 0.0)
                        
                        # Ensure we don't exceed buying power after rounding
                        if notional_rounded > buying_power_after:
                            notional_rounded = round(buying_power_after, 2)
                            if notional_rounded <= 0:
                                orders["decision"] = "flip_to_flat_after_close"
                                orders["close_order"] = close_resp
                                orders["trade_qty"] = abs(existing_qty)
                                orders["trade_side"] = "sell" if existing_qty > 0 else "buy"
                                orders["final_side"] = "flat"
                                orders["final_qty"] = 0.0
                                orders["reason"] = f"Insufficient buying power after closing: {buying_power_after}"
                                self._log(orders)
                                return orders
                        
                        try:
                            entry_resp = self.client.submit_order(
                                symbol=trading_symbol,
                                notional=notional_rounded,
                                side=side,
                                order_type="market",
                                time_in_force="gtc",
                            )
                            # Recalculate implied qty from rounded notional for accuracy
                            implied_qty_from_rounded = notional_rounded / current_price
                            new_qty = implied_qty_from_rounded
                        except RuntimeError as exc:
                            error_msg = str(exc)
                            # Handle crypto short rejections gracefully (paper trading limitation)
                            if target_side == "short" and is_crypto and ("insufficient balance" in error_msg.lower() or "403" in error_msg or "422" in error_msg or "wash trade" in error_msg.lower()):
                                # Crypto short rejected by Alpaca - this is about ASSET availability, not USD buying power
                                # For crypto shorts, you need the actual crypto asset (ETH) available to borrow, not just USD
                                # Alpaca paper trading has limited crypto inventory for shorting
                                # Log as execution attempt but stay flat after closing
                                orders["decision"] = "flip_to_short_rejected"
                                orders["close_order"] = close_resp
                                orders["entry_notional"] = notional_rounded
                                orders["entry_qty"] = implied_qty_from_rounded if 'implied_qty_from_rounded' in locals() else notional_rounded / current_price
                                orders["trade_side"] = side
                                orders["final_side"] = "flat"  # Stay flat since short was rejected
                                orders["final_qty"] = 0.0
                                orders["alpaca_error"] = error_msg
                                orders["execution_status"] = "rejected_by_alpaca"
                                orders["note"] = f"SHORT order rejected: Alpaca doesn't have enough {trading_symbol.replace('USD', '')} available to borrow for shorting (paper trading limitation). This is about ASSET availability, not USD buying power. You have ${buying_power_after:,.2f} buying power, but Alpaca only has limited crypto inventory. In live trading, this may execute successfully."
                                self._log(orders)
                                return orders  # Return the order record so it's counted as "traded" (attempted)
                            else:
                                # For other errors, re-raise to let caller handle
                                raise
                    else:
                        # For longs or non-crypto, use qty with optional stop loss
                        entry_kwargs = {
                            "symbol": trading_symbol,
                            "qty": new_qty,
                            "side": side,
                            "order_type": "market",
                            "time_in_force": "gtc",
                        }
                        if not is_crypto:
                            # Non-crypto can use bracket orders with stop-loss
                            entry_kwargs["stop_loss_price"] = stop_loss_price
                        entry_resp = self.client.submit_order(**entry_kwargs)
                    
                    # Calculate stop-loss and take-profit for risk management logging.
                    # These levels are calculated for both longs and shorts to track risk exposure.
                    # Use horizon-specific stop-loss.
                    stop_pct = effective_risk.default_stop_loss_pct
                    tp_mult = effective_risk.take_profit_risk_multiple
                    
                    # Use horizon-specific stop-loss
                    stop_pct = effective_risk.default_stop_loss_pct
                    tp_mult = effective_risk.take_profit_risk_multiple
                    
                    if target_side == "long":
                        stop_loss_price = current_price * (1.0 - stop_pct)
                        take_profit_price = current_price * (1.0 + stop_pct * tp_mult)
                    else:  # short
                        stop_loss_price = current_price * (1.0 + stop_pct)
                        take_profit_price = current_price * (1.0 - stop_pct * tp_mult)
                    
                    orders["decision"] = "flip_to_" + target_side
                    orders["close_order"] = close_resp
                    orders["entry_order"] = entry_resp
                    orders["entry_qty"] = new_qty
                    orders["entry_notional"] = desired_notional if is_crypto and target_side == "short" else (new_qty * current_price)
                    orders["trade_qty"] = abs(existing_qty)
                    orders["trade_side"] = "sell" if existing_qty > 0 else "buy"
                    orders["stop_loss_price"] = stop_loss_price
                    orders["take_profit_price"] = take_profit_price
                    orders["stop_loss_pct"] = effective_risk.default_stop_loss_pct
                    orders["take_profit_pct"] = stop_pct * tp_mult
                    orders["final_side"] = target_side
                    orders["final_qty"] = new_qty if target_side == "long" else -new_qty
                    orders["exit_reason"] = f"model_changed_from_{side_in_market}_to_{target_side}"
                    orders["entry_price"] = avg_entry_price
                    orders["exit_price"] = current_price
                    orders["realized_pl"] = unrealized_pl
                    orders["realized_pl_pct"] = unrealized_pl_pct
                    orders["market_value_at_exit"] = market_value
                    
                    if is_crypto:
                        orders["stop_loss_note"] = "Crypto positions: stop-loss must be managed separately (bracket orders not supported)"
                else:
                    orders["decision"] = "flip_to_flat"
                    orders["close_order"] = close_resp
                    orders["trade_qty"] = abs(existing_qty)
                    orders["trade_side"] = "sell" if existing_qty > 0 else "buy"
                    orders["final_side"] = "flat"
                    orders["final_qty"] = 0.0
                    orders["exit_reason"] = f"model_changed_from_{side_in_market}_to_flat"
                    orders["entry_price"] = avg_entry_price
                    orders["exit_price"] = current_price
                    orders["realized_pl"] = unrealized_pl
                    orders["realized_pl_pct"] = unrealized_pl_pct
                    orders["market_value_at_exit"] = market_value
            else:
                orders["decision"] = "flip_to_flat"
                orders["close_order"] = close_resp
                orders["trade_qty"] = abs(existing_qty)
                orders["trade_side"] = "sell" if existing_qty > 0 else "buy"
                orders["final_side"] = "flat"
                orders["final_qty"] = 0.0
                orders["exit_reason"] = f"model_changed_from_{side_in_market}_to_flat"
                orders["entry_price"] = avg_entry_price
                orders["exit_price"] = current_price
                orders["realized_pl"] = unrealized_pl
                orders["realized_pl_pct"] = unrealized_pl_pct
                orders["market_value_at_exit"] = market_value
            
            self._log(orders)
            return orders

        # Entering new long/short from flat.
        # IMPORTANT: Only enter new positions if confidence meets threshold.
        # We already checked confidence above, but if we're here and confidence is low,
        # it means we had a position that we just exited, so we shouldn't enter a new one.
        if side_in_market == "flat" and target_side in {"long", "short"} and desired_notional > 0:
            # Apply confidence threshold for ENTERING new positions
            if confidence < effective_risk.min_confidence:
                orders["decision"] = "enter_position_skipped_low_confidence"
                orders["entry_notional"] = 0.0
                orders["entry_qty"] = 0.0
                orders["final_side"] = "flat"
                orders["final_qty"] = existing_qty
                orders["reason"] = f"Confidence {confidence*100:.1f}% below threshold {effective_risk.min_confidence*100:.1f}%"
                self._log(orders)
                return orders
            # For crypto, Alpaca expects USD notional rather than coin qty.
            # We still compute an implied quantity for logging.
            trade_notional = desired_notional
            implied_qty = max(trade_notional / current_price, 0.0)

            if trade_notional <= 0 or implied_qty <= 0:
                orders["decision"] = "enter_position_skipped_zero_notional"
                orders["entry_notional"] = 0.0
                orders["entry_qty"] = 0.0
                orders["final_side"] = "flat"
                orders["final_qty"] = existing_qty
                self._log(orders)
                return orders

            # Compute stop-loss and take-profit levels with proper risk management.
            # For LONGS: stop-loss below entry (price drops), take-profit above entry (price rises).
            # For SHORTS: stop-loss above entry (price rises), take-profit below entry (price drops).
            # Use horizon-specific stop-loss percentage.
            stop_pct = effective_risk.default_stop_loss_pct
            tp_mult = effective_risk.take_profit_risk_multiple

            if target_side == "long":
                # Long: lose if price drops, win if price rises
                stop_loss_price = current_price * (1.0 - stop_pct)  # e.g., $100 * 0.98 = $98 (2% down)
                take_profit_price = current_price * (1.0 + stop_pct * tp_mult)  # e.g., $100 * 1.04 = $104 (4% up)
                side = "buy"
            else:  # short
                # Short: lose if price rises, win if price drops
                stop_loss_price = current_price * (1.0 + stop_pct)  # e.g., $100 * 1.02 = $102 (2% up = loss)
                take_profit_price = current_price * (1.0 - stop_pct * tp_mult)  # e.g., $100 * 0.96 = $96 (4% down = profit)
                side = "sell"

            if dry_run:
                orders["decision"] = "would_enter_position"
                orders["entry_notional"] = trade_notional
                orders["entry_qty"] = implied_qty
                orders["trade_side"] = side
                orders["stop_loss_price"] = stop_loss_price
                orders["take_profit_price"] = take_profit_price
                orders["final_side"] = target_side
                orders["final_qty"] = implied_qty if target_side == "long" else -implied_qty
                self._log(orders)
                return orders

            # Build order kwargs.
            # For crypto, send USD notional (rounded to 2 decimals as required
            # by Alpaca) and avoid advanced order classes.
            order_kwargs = {
                "symbol": trading_symbol,
                "side": side,
                "order_type": "market",
                "time_in_force": "gtc",
            }

            # Calculate profit target price based on user's desired profit percentage
            if effective_profit_target is not None and effective_profit_target > 0:
                if target_side == "long":
                    profit_target_price = current_price * (1.0 + effective_profit_target / 100.0)
                else:  # short
                    profit_target_price = current_price * (1.0 - effective_profit_target / 100.0)
            else:
                # Use default take-profit from risk config
                profit_target_price = take_profit_price
            
            if is_crypto:
                # Alpaca requires notional to have at most 2 decimal places.
                # Round down slightly to ensure we never exceed buying power due to rounding.
                notional_rounded = round(trade_notional, 2)
                
                # Safety check: ensure rounded notional is positive and meaningful
                if notional_rounded <= 0 or notional_rounded < 0.01:
                    orders["decision"] = "enter_position_skipped_zero_notional_after_round"
                    orders["entry_notional"] = 0.0
                    orders["entry_qty"] = 0.0
                    orders["final_side"] = "flat"
                    orders["final_qty"] = existing_qty
                    orders["reason"] = f"Rounded notional too small: {notional_rounded} (original: {trade_notional})"
                    self._log(orders)
                    return orders
                
                # Ensure we don't exceed buying power after rounding
                if notional_rounded > buying_power:
                    notional_rounded = round(buying_power, 2)
                    if notional_rounded <= 0:
                        orders["decision"] = "enter_position_skipped_insufficient_buying_power"
                        orders["entry_notional"] = 0.0
                        orders["entry_qty"] = 0.0
                        orders["final_side"] = "flat"
                        orders["final_qty"] = existing_qty
                        orders["reason"] = f"Buying power too low: {buying_power}"
                        self._log(orders)
                        return orders
                
                order_kwargs["notional"] = notional_rounded
                # Recalculate implied qty from rounded notional for accuracy
                implied_qty = notional_rounded / current_price
                
                # NOTE: Alpaca does NOT support bracket orders (OCO) for crypto
                # For crypto, we rely on system-level monitoring for stop-loss and profit targets
                # Bracket orders are only used for non-crypto assets (commodities)
            else:
                # For non-crypto assets (commodities), use qty + Alpaca bracket orders
                # This ensures stop-loss executes at broker level even if system is down
                order_kwargs["qty"] = implied_qty
                order_kwargs["stop_loss_price"] = stop_loss_price
                order_kwargs["take_profit_limit_price"] = profit_target_price

            # Submit the order with detailed documentation
            # In paper trading, crypto shorts may be rejected.
            # In live trading with proper account permissions, shorts will execute.
            try:
                entry_resp = self.client.submit_order(**order_kwargs)
                
                # CRITICAL FIX: For crypto, submit separate stop-loss and take-profit orders
                # Alpaca doesn't support bracket orders (OCO) for crypto, but we can submit
                # standalone stop-loss and take-profit orders that execute at broker level
                stop_loss_order_id = None
                take_profit_order_id = None
                
                if is_crypto and stop_loss_price:
                    try:
                        # Submit stop-loss order for crypto (broker-level protection)
                        stop_side = "sell" if target_side == "long" else "buy"
                        stop_order_resp = self.client.submit_stop_order(
                            symbol=trading_symbol,
                            qty=implied_qty,
                            stop_price=stop_loss_price,
                            side=stop_side,
                            time_in_force="gtc",
                            client_order_id=f"{trading_symbol}_stop_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                        )
                        stop_loss_order_id = stop_order_resp.get("id")
                        print(f"  ✅ Stop-loss order placed at broker level: ${stop_loss_price:.2f} (Order ID: {stop_loss_order_id})")
                    except Exception as stop_exc:
                        print(f"  ⚠️  WARNING: Failed to place broker-level stop-loss order: {stop_exc}")
                        print(f"     Stop-loss will only work while monitoring script is running")
                
                if is_crypto and profit_target_price and effective_profit_target:
                    try:
                        # Submit take-profit limit order for crypto (broker-level protection)
                        tp_side = "sell" if target_side == "long" else "buy"
                        tp_order_resp = self.client.submit_take_profit_order(
                            symbol=trading_symbol,
                            qty=implied_qty,
                            limit_price=profit_target_price,
                            side=tp_side,
                            time_in_force="gtc",
                            client_order_id=f"{trading_symbol}_tp_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                        )
                        take_profit_order_id = tp_order_resp.get("id")
                        print(f"  ✅ Take-profit order placed at broker level: ${profit_target_price:.2f} (Order ID: {take_profit_order_id})")
                    except Exception as tp_exc:
                        print(f"  ⚠️  WARNING: Failed to place broker-level take-profit order: {tp_exc}")
                        print(f"     Take-profit will only work while monitoring script is running")
                
                # Calculate all trade details for comprehensive documentation
                entry_cost = notional_rounded if is_crypto else (implied_qty * current_price)
                expected_profit_at_target = (profit_target_price - current_price) * implied_qty if target_side == "long" else (current_price - profit_target_price) * implied_qty
                expected_profit_pct = effective_profit_target if effective_profit_target else (stop_pct * tp_mult * 100)
                max_loss_at_stop = (current_price - stop_loss_price) * implied_qty if target_side == "long" else (stop_loss_price - current_price) * implied_qty
                max_loss_pct = effective_risk.default_stop_loss_pct * 100
                risk_reward_ratio = abs(expected_profit_at_target / max_loss_at_stop) if max_loss_at_stop > 0 else 0
                
                orders["decision"] = "enter_long" if target_side == "long" else "enter_short"
                orders["entry_notional"] = entry_cost
                orders["entry_qty"] = implied_qty
                orders["trade_side"] = side
                orders["entry_price"] = current_price
                orders["stop_loss_price"] = stop_loss_price
                orders["stop_loss_pct"] = effective_risk.default_stop_loss_pct
                orders["profit_target_price"] = profit_target_price
                orders["profit_target_pct"] = effective_profit_target if effective_profit_target else (stop_pct * tp_mult * 100)
                orders["take_profit_price"] = take_profit_price
                orders["take_profit_pct"] = stop_pct * tp_mult
                orders["final_side"] = target_side
                orders["final_qty"] = implied_qty if target_side == "long" else -implied_qty
                orders["entry_order"] = entry_resp
                orders["execution_status"] = "success"
                
                # For crypto, we now use separate stop-loss orders (broker-level)
                # For non-crypto, we use bracket orders
                orders["bracket_order_used"] = not is_crypto and ("stop_loss_price" in order_kwargs or "take_profit_limit_price" in order_kwargs)
                orders["broker_level_stop_loss_status"] = "enabled" if (orders["bracket_order_used"] or stop_loss_order_id) else "system_monitoring"
                orders["stop_loss_order_id"] = stop_loss_order_id
                orders["take_profit_order_id"] = take_profit_order_id
                
                # Comprehensive trade documentation
                orders["trade_documentation"] = {
                    "entry_details": {
                        "symbol": trading_symbol,
                        "data_symbol": asset.data_symbol,
                        "asset_type": asset.asset_type,
                        "side": target_side,
                        "entry_price": current_price,
                        "quantity": implied_qty,
                        "entry_cost": entry_cost,
                        "entry_time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "order_id": entry_resp.get("id"),
                        "order_status": entry_resp.get("status"),
                    },
                    "risk_management": {
                        "stop_loss_price": stop_loss_price,
                        "stop_loss_pct": effective_risk.default_stop_loss_pct * 100,
                        "max_loss_amount": max_loss_at_stop,
                        "max_loss_pct": max_loss_pct,
                        "stop_loss_at_broker": orders["bracket_order_used"],  # True if bracket order used
                    },
                    "profit_target": {
                        "profit_target_pct": effective_profit_target if effective_profit_target else None,
                        "profit_target_price": profit_target_price,
                        "expected_profit_amount": expected_profit_at_target,
                        "expected_profit_pct": expected_profit_pct,
                    },
                    "risk_reward": {
                        "risk_reward_ratio": risk_reward_ratio,
                        "risk_amount": max_loss_at_stop,
                        "reward_amount": expected_profit_at_target,
                    },
                    "account_status": {
                        "equity": equity,
                        "buying_power_before": buying_power,
                        "buying_power_after": buying_power - entry_cost,
                    },
                }
                
                # Calculate expected values at target and stop-loss
                expected_value_at_target = entry_cost + expected_profit_at_target
                expected_value_at_stop = entry_cost + max_loss_at_stop
                
                # Print comprehensive trade documentation
                print(f"\n{'='*80}")
                print(f"NEW POSITION ENTERED: {trading_symbol}")
                print(f"{'='*80}")
                print(f"\n💰 INVESTMENT DETAILS:")
                print(f"  Initial Investment: ${entry_cost:.2f}")
                print(f"    └─ Symbol:         {trading_symbol} ({asset.data_symbol})")
                print(f"    └─ Side:           {target_side.upper()}")
                print(f"    └─ Entry Price:    ${current_price:.2f}")
                print(f"    └─ Quantity:       {implied_qty:.6f}")
                print(f"    └─ Order ID:       {entry_resp.get('id', 'N/A')}")
                
                print(f"\n🎯 TARGET SCENARIOS:")
                if effective_profit_target:
                    print(f"  Profit Target:")
                    print(f"    └─ Target Price:  ${profit_target_price:.2f} ({effective_profit_target:+.2f}%)")
                    print(f"    └─ Expected Profit: ${expected_profit_at_target:+.2f}")
                    print(f"    └─ Total Value at Target: ${expected_value_at_target:.2f}")
                else:
                    print(f"  Profit Target:")
                    print(f"    └─ Target Price:  ${take_profit_price:.2f} ({stop_pct * tp_mult * 100:+.2f}%)")
                    print(f"    └─ Expected Profit: ${expected_profit_at_target:+.2f}")
                    print(f"    └─ Total Value at Target: ${expected_value_at_target:.2f}")
                
                print(f"\n⚠️  RISK MANAGEMENT:")
                print(f"  Stop-Loss:")
                print(f"    └─ Stop Price:    ${stop_loss_price:.2f} ({effective_risk.default_stop_loss_pct*100:.2f}%)")
                print(f"    └─ Max Loss:      ${max_loss_at_stop:.2f} ({max_loss_pct:.2f}%)")
                print(f"    └─ Total Value at Stop: ${expected_value_at_stop:.2f}")
                print(f"  Broker-Level Protection: {'✅ YES (Bracket Order)' if orders['bracket_order_used'] else '❌ NO (Manual Monitoring)'}")
                
                print(f"\n📊 RISK/REWARD ANALYSIS:")
                print(f"  Risk/Reward Ratio: {risk_reward_ratio:.2f}:1")
                print(f"  Risk Amount:      ${max_loss_at_stop:.2f}")
                print(f"  Reward Amount:   ${expected_profit_at_target:.2f}")
                print(f"{'='*80}\n")
                
                # Save position to position manager if profit target is set
                if effective_profit_target is not None and effective_profit_target > 0:
                    try:
                        self.position_manager.save_position(
                            symbol=trading_symbol,
                            data_symbol=asset.data_symbol,
                            asset_type=asset.asset_type,
                            side=target_side,
                            entry_price=current_price,
                            quantity=implied_qty,
                            profit_target_pct=effective_profit_target,
                            stop_loss_pct=effective_risk.default_stop_loss_pct,
                            stop_loss_order_id=stop_loss_order_id,
                            take_profit_order_id=take_profit_order_id,
                        )
                        orders["position_tracked"] = True
                    except Exception as pos_exc:
                        # Log error but don't fail the trade
                        orders["position_tracked"] = False
                        orders["position_tracking_error"] = str(pos_exc)
                else:
                    orders["position_tracked"] = False
                    orders["position_tracking_reason"] = "no_profit_target_provided"
                
                # Note about stop-loss execution
                if is_crypto and not orders["bracket_order_used"]:
                    orders["stop_loss_note"] = "Crypto positions: stop-loss managed by system (bracket orders may not be supported). Monitor actively."
                    print(f"[WARNING] {trading_symbol}: Stop-loss is NOT at broker level. System must be running for stop-loss to execute.")
                elif orders["bracket_order_used"]:
                    print(f"[INFO] {trading_symbol}: Stop-loss and take-profit are at broker level (bracket order). Will execute even if system is down.")
                
                self._log(orders)
                return orders
            except RuntimeError as exc:
                error_msg = str(exc)
                # Handle crypto short rejections gracefully (paper trading limitation)
                # In live trading, shorts may work, so we attempt them but handle rejections
                if target_side == "short" and is_crypto and ("insufficient balance" in error_msg.lower() or "403" in error_msg or "422" in error_msg or "wash trade" in error_msg.lower()):
                    # Crypto short rejected by Alpaca (paper trading doesn't support it)
                    # Log as execution attempt but stay flat
                    orders["decision"] = "enter_short_rejected"
                    orders["entry_notional"] = trade_notional
                    orders["entry_qty"] = implied_qty
                    orders["trade_side"] = side
                    orders["stop_loss_price"] = stop_loss_price
                    orders["take_profit_price"] = take_profit_price
                    orders["stop_loss_pct"] = self.risk.default_stop_loss_pct
                    orders["take_profit_pct"] = stop_pct * tp_mult
                    orders["final_side"] = "flat"  # Stay flat since short was rejected
                    orders["final_qty"] = existing_qty  # Keep existing position (should be 0 if entering from flat)
                    orders["alpaca_error"] = error_msg
                    orders["execution_status"] = "rejected_by_alpaca"
                    orders["note"] = f"SHORT order attempted but rejected by Alpaca (paper trading limitation). Model signal: SHORT. In live trading, this may execute successfully."
                    if is_crypto:
                        orders["stop_loss_note"] = "Crypto positions: stop-loss must be managed separately (bracket orders not supported)"
                    self._log(orders)
                    return orders  # Return the order record so it's counted as "traded" (attempted)
                else:
                    # For other errors (long orders, non-crypto), re-raise to let caller handle
                    raise

        # Any other combination is currently unsupported; just log it.
        orders["decision"] = "unsupported_transition"
        orders["final_side"] = side_in_market
        orders["final_qty"] = existing_qty
        self._log(orders)
        return orders

    # ------------------------------------------------------------------
    # Explicit helpers
    # ------------------------------------------------------------------
    def explicit_short(
        self,
        asset: AssetMapping,
        target_notional: float,
        current_price: float,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Explicitly move the symbol to a SHORT state with given notional exposure.

        This bypasses consensus and:
        - Closes any existing LONG position.
        - Enters a SHORT position sized by target_notional (if allow_short is True).
        """
        if not self.risk.allow_short:
            record = {
                "asset": asset.logical_name,
                "trading_symbol": asset.trading_symbol,
                "decision": "short_disabled",
                "reason": "allow_short=False",
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
            self._log(record)
            return record

        account = self.client.get_account()
        equity = float(account.get("equity", 0.0) or 0.0)
        if equity <= 0 or target_notional <= 0:
            record = {
                "asset": asset.logical_name,
                "trading_symbol": asset.trading_symbol,
                "decision": "short_skipped_invalid_notional",
                "equity": equity,
                "target_notional": target_notional,
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
            self._log(record)
            return record

        # Use base risk config for explicit_short (not horizon-specific)
        pos_fraction = min(
            target_notional / equity,
            self.risk.max_notional_per_symbol_pct,
        )
        consensus = {
            "consensus_action": "short",
            "consensus_confidence": 1.0,
            "position_size": pos_fraction,
        }
        return self.execute_from_consensus(
            asset=asset,
            consensus=consensus,
            current_price=current_price,
            dry_run=dry_run,
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(self, record: Dict[str, Any]) -> None:
        """Append a JSON line to the trading log."""
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")




