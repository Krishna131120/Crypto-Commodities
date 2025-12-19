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
from .broker_interface import BrokerClient
from .symbol_universe import AssetMapping
from .position_manager import PositionManager
from .mcx_symbol_mapper import round_to_lot_size, get_mcx_lot_size
from ml.horizons import get_horizon_risk_config, normalize_profile


@dataclass
class TradingRiskConfig:
    """
    Basic risk limits for live paper trading.

    All values are conservative defaults and can be tuned later.
    """

    max_notional_per_symbol_pct: float = 0.10  # max 10% of equity per symbol
    max_total_equity_pct: float = 0.50        # cap total deployed capital at 50% equity (not enforced yet)
    default_stop_loss_pct: float = 0.035      # 3.5% stop-loss (for crypto - wider to avoid volatility triggers)
    take_profit_risk_multiple: float = 2.0    # TP distance = multiple * stop-loss distance
    min_confidence: float = 0.10              # ignore very low-confidence signals
    profit_target_pct: Optional[float] = None  # User's desired profit percentage (e.g., 10.0 for 10%)
    user_stop_loss_pct: Optional[float] = None  # User-defined stop-loss override (for real money trading)
    manual_stop_loss: bool = False            # If True, user manages stop-losses manually (system won't submit or execute stop-loss orders)
    # IMPORTANT: Shorting support depends on your broker account and asset type.
    # MCX supports shorting for commodity futures.
    # This engine is designed to work correctly in both environments.
    allow_short: bool = True                  # enable shorting (ready for live trading)
    
    def get_effective_stop_loss_pct(self, asset_type: str = "crypto") -> float:
        """
        Get effective stop-loss percentage based on asset type and user override.
        
        For commodities (real money), uses tighter stop-loss (2.0% default).
        For crypto, uses wider stop-loss (3.5% default).
        User override takes precedence.
        
        Args:
            asset_type: "crypto" or "commodities"
            
        Returns:
            Effective stop-loss percentage
        """
        if self.user_stop_loss_pct is not None:
            # User override takes precedence
            return max(0.005, min(0.10, self.user_stop_loss_pct))  # Clamp between 0.5% and 10%
        
        if asset_type == "commodities":
            # Real money trading - tighter stop-loss
            return 0.020  # 2.0% for commodities
        else:
            # Crypto - wider stop-loss
            return self.default_stop_loss_pct  # 3.5% for crypto


class ExecutionEngine:
    """
    Bridge between model consensus and broker orders (supports Alpaca for crypto, DHAN for commodities).
    
    IMPORTANT:
    - Crypto: Uses AlpacaClient (default)
    - Commodities: MUST use DhanClient (MCX exchange) - will raise error if AlpacaClient is used
    
    The ExecutionEngine is broker-agnostic but enforces broker selection based on asset type.
    """

    def __init__(
        self,
        client: Optional[BrokerClient] = None,
        risk_config: Optional[TradingRiskConfig] = None,
        log_path: Path = Path("logs") / "trading" / "crypto_trades.jsonl",
        default_horizon: Optional[str] = None,
        position_manager: Optional[PositionManager] = None,
    ):
        # Backward compatible: if no client provided, default to AlpacaClient (for crypto)
        # NOTE: For commodities, you MUST provide DhanClient explicitly
        if client is None:
            client = AlpacaClient()
        elif not isinstance(client, BrokerClient):
            # If old code passes AlpacaClient directly, it's still valid (AlpacaClient implements BrokerClient)
            pass
        
        self.client: BrokerClient = client
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
        user_stop_loss_pct: Optional[float] = None,
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
        
        # REQUIRE profit target - user must specify it
        # Profit target is REQUIRED for trading - no default allowed
        effective_profit_target = profit_target_pct or self.risk.profit_target_pct
        if effective_profit_target is None or effective_profit_target <= 0:
            raise ValueError(
                f"Profit target is REQUIRED but not provided. "
                f"You must specify --profit-target (crypto) or --profit-target-pct (commodities) before trading. "
                f"Example: --profit-target 10.0 for 10% profit target."
            )
        
        # Use user_stop_loss_pct from parameter or risk config
        if user_stop_loss_pct is not None:
            # Validate user stop-loss (0.5% to 10%)
            user_stop_loss_pct = max(0.005, min(0.10, user_stop_loss_pct))
            self.risk.user_stop_loss_pct = user_stop_loss_pct

        # Determine which horizon profile to use (from parameter, asset mapping, or default)
        effective_horizon = normalize_profile(horizon_profile) if horizon_profile else (
            normalize_profile(getattr(asset, "horizon_profile", None)) if hasattr(asset, "horizon_profile") else self.default_horizon
        ) or "short"  # Fallback to short if nothing specified
        
        # Get horizon-specific risk parameters and create a temporary risk config override
        horizon_risk = get_horizon_risk_config(effective_horizon)
        
        # Get effective stop-loss (commodities use tighter stop-loss for real money)
        asset_type = getattr(asset, "asset_type", "crypto").lower()
        effective_stop_loss = self.risk.get_effective_stop_loss_pct(asset_type)
        
        # Create a risk config that uses horizon-specific values, falling back to base config
        effective_risk = TradingRiskConfig(
            max_notional_per_symbol_pct=horizon_risk.get("max_notional_per_symbol_pct", self.risk.max_notional_per_symbol_pct),
            max_total_equity_pct=self.risk.max_total_equity_pct,
            default_stop_loss_pct=horizon_risk.get("default_stop_loss_pct", effective_stop_loss),
            take_profit_risk_multiple=self.risk.take_profit_risk_multiple,
            min_confidence=horizon_risk.get("min_confidence", self.risk.min_confidence),
            user_stop_loss_pct=self.risk.user_stop_loss_pct,  # Pass through user override
            manual_stop_loss=self.risk.manual_stop_loss,
            allow_short=self.risk.allow_short,
        )

        # Get trading symbol - commodities MUST use MCX with DHAN (no Alpaca fallback)
        is_crypto = getattr(asset, "asset_type", "").lower() == "crypto"
        is_commodities = getattr(asset, "asset_type", "").lower() == "commodities"
        
        # For commodities, ALWAYS use MCX contract symbol with DHAN (enforced)
        if is_commodities:
            if self.client.broker_name != "dhan":
                raise RuntimeError(
                    f"Commodities ({asset.data_symbol}) require DHAN broker for MCX trading. "
                    f"Current broker: {self.client.broker_name}. "
                    f"Please use DhanClient() instead of AlpacaClient() for commodities."
                )
            trading_symbol = asset.get_mcx_symbol(effective_horizon).upper()
        else:
            trading_symbol = asset.trading_symbol.upper()
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
        
        # PRIORITY 2: Also exit on stop-loss (safety mechanism) - but only if not in manual mode
        stop_loss_hit = False
        if not effective_risk.manual_stop_loss and tracked_position and tracked_position.status == "open":
            if tracked_position.side == "long":
                stop_loss_hit = current_price <= tracked_position.stop_loss_price
            elif tracked_position.side == "short":
                stop_loss_hit = current_price >= tracked_position.stop_loss_price
        
        if stop_loss_hit:
            must_exit_position = True
        
        # ENHANCED TRADING LOGIC: Stricter filtering for commodities (real money)
        # For commodities, require higher confidence and model agreement
        # For crypto, allow more lenient entry (paper trading)
        
        # Get model agreement info from consensus (if available)
        model_agreement = consensus.get("model_agreement_ratio", None)
        total_models = consensus.get("total_models", None)
        agreement_count = consensus.get("agreement_count", None)
        
        # Calculate minimum agreement requirement
        # For commodities (real money): require at least 66% model agreement
        # For crypto (paper trading): more lenient (50% agreement)
        if is_commodities:
            min_agreement_ratio = 0.66  # 66% of models must agree
            min_confidence_for_commodities = max(effective_risk.min_confidence, 0.15)  # At least 15% confidence for commodities
        else:
            min_agreement_ratio = 0.50  # 50% agreement for crypto
            min_confidence_for_commodities = effective_risk.min_confidence  # Use horizon-specific threshold
        
        # Check model agreement if available
        agreement_met = True
        if model_agreement is not None and total_models is not None and total_models > 1:
            # We have agreement data - enforce agreement requirement
            agreement_met = model_agreement >= min_agreement_ratio
            if not agreement_met:
                print(f"  [FILTER] Model agreement {model_agreement*100:.1f}% below required {min_agreement_ratio*100:.0f}% ({agreement_count}/{total_models} models agree)")
        elif total_models == 1:
            # Single model - reduce confidence requirement but still need minimum
            min_confidence_for_commodities = max(min_confidence_for_commodities, 0.20)  # 20% minimum for single model
            print(f"  [WARNING] Single model prediction - using stricter confidence threshold")
        
        # If we need to exit a position, do it immediately regardless of confidence
        if must_exit_position:
            # Exit logic will be handled below, but we skip confidence check for exits
            pass
        elif side_in_market == "flat":
            # No existing position - check if we should enter
            # For commodities (real money): STRICT filtering
            if is_commodities:
                # Commodities require:
                # 1. Confidence >= threshold (15% minimum)
                # 2. Model agreement >= 66% (if multiple models)
                # 3. Action must match target side
                if confidence < min_confidence_for_commodities:
                    print(f"  [FILTER] Confidence {confidence*100:.1f}% below required {min_confidence_for_commodities*100:.0f}% for commodities (real money)")
                    return None
                if not agreement_met:
                    print(f"  [FILTER] Model agreement insufficient for commodities (real money)")
                    return None
                if action != target_side or target_side == "flat":
                    # Action doesn't match or target is flat - don't enter
                    if action == "hold" or action == "flat":
                        print(f"  [FILTER] Model prediction is {action.upper()} - no trade signal")
                    elif action != target_side:
                        print(f"  [FILTER] Action mismatch: model says {action.upper()}, target is {target_side.upper()}")
                    return None
            else:
                # Crypto (paper trading): More lenient
                # For LONG positions, buy on positive prediction (even small)
                # Only require confidence threshold for SHORT positions
                if target_side == "long" and action == "long":
                    # Buy on any UP prediction for crypto (paper trading)
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
                    stop_loss_hit = False if effective_risk.manual_stop_loss else (current_price <= tracked_position.stop_loss_price)
                else:  # short
                    unrealized_pl_pct = ((tracked_position.entry_price - current_price) / tracked_position.entry_price) * 100
                    profit_target_hit = current_price <= tracked_position.profit_target_price
                    stop_loss_hit = False if effective_risk.manual_stop_loss else (current_price >= tracked_position.stop_loss_price)
                
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
        # IMPORTANT: Only exit if we actually have a position in the broker
        # If tracked_position exists but broker has no position, clear the tracked position and proceed to entry
        if must_exit_position:
            # Check if we actually have a position in the broker
            if abs(existing_qty) <= 0:
                # No actual position in broker, but tracked_position thinks there is
                # This can happen if position was closed manually or broker reset
                # Clear the tracked position and proceed to entry logic
                if tracked_position:
                    print(f"  [INFO] Tracked position exists but no broker position found. Clearing tracked position.")
                    self.position_manager.close_position(
                        trading_symbol,
                        current_price,
                        "position_not_found_in_broker",
                        0.0,
                        0.0,
                    )
                # Reset must_exit_position and proceed to entry logic
                must_exit_position = False
            else:
                # We have an actual position - proceed with exit
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
                # Safety check: ensure we have a meaningful quantity to close
                abs_qty = abs(existing_qty)
                if abs_qty <= 0 or (asset.asset_type == "crypto" and round(abs_qty * current_price, 2) <= 0):
                    # Position is effectively zero - skip exit order
                    orders["decision"] = "exit_position_skipped_zero_qty"
                    orders["trade_qty"] = 0.0
                    orders["final_side"] = "flat"
                    orders["final_qty"] = 0.0
                    orders["exit_reason"] = exit_reason
                    orders["entry_price"] = avg_entry_price
                    orders["exit_price"] = current_price
                    orders["realized_pl"] = realized_pl
                    orders["realized_pl_pct"] = realized_pl_pct
                    orders["reason"] = f"Position quantity too small to exit: {existing_qty}"
                    self._log(orders)
                    return orders
                
                close_resp = self.client.submit_order(
                    symbol=trading_symbol,
                    qty=abs_qty if asset.asset_type == "commodities" else None,
                    notional=round(abs_qty * current_price, 2) if asset.asset_type == "crypto" else None,
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
                # For MCX commodities, round to lot size
                if is_commodities and self.client.broker_name == "dhan":
                    raw_qty = max(desired_notional / current_price, 0.0)
                    new_qty = round_to_lot_size(raw_qty, asset.data_symbol)
                    print(f"[MCX] Rounded quantity to lot size: {raw_qty:.2f} -> {new_qty} (lot size: {get_mcx_lot_size(asset.data_symbol)})")
                else:
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
                        if not is_crypto and not effective_risk.manual_stop_loss:
                            # Non-crypto can use bracket orders with stop-loss (unless manual mode)
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
            # For MCX commodities, round to lot size. For crypto, use notional.
            if is_commodities and self.client.broker_name == "dhan":
                # MCX requires lot-based trading - round quantity to nearest lot
                raw_qty = max(desired_notional / current_price, 0.0)
                trade_qty = round_to_lot_size(raw_qty, asset.data_symbol)
                trade_notional = trade_qty * current_price
                implied_qty = trade_qty
                print(f"\n[MCX] Rounded quantity to lot size: {raw_qty:.2f} -> {trade_qty} (lot size: {get_mcx_lot_size(asset.data_symbol)})")
                
                # CRITICAL: Validate buying power for commodities (real money)
                if trade_notional > buying_power:
                    orders["decision"] = "enter_position_skipped_insufficient_buying_power"
                    orders["entry_notional"] = 0.0
                    orders["entry_qty"] = 0.0
                    orders["final_side"] = "flat"
                    orders["final_qty"] = existing_qty
                    orders["reason"] = f"Insufficient buying power: need ₹{trade_notional:,.2f}, have ₹{buying_power:,.2f}"
                    print(f"  [ERROR] {orders['reason']}")
                    self._log(orders)
                    return orders
            else:
                # For crypto, Alpaca expects USD notional rather than coin qty.
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
            # Profit target is REQUIRED - effective_profit_target is guaranteed to be set (validated above)
            if target_side == "long":
                profit_target_price = current_price * (1.0 + effective_profit_target / 100.0)
            else:  # short
                profit_target_price = current_price * (1.0 - effective_profit_target / 100.0)
            
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
                # For commodities (MCX), use qty (lot-based)
                # MCX lot sizes are already handled above
                order_kwargs["qty"] = implied_qty
                
                # For MCX with DHAN, stop-loss is handled via DHAN API
                # Commodities MUST use DHAN (enforced above), so this is always DHAN
                if self.client.broker_name == "dhan":
                    # DHAN MCX: stop-loss is submitted as separate order type
                    if not effective_risk.manual_stop_loss:
                        order_kwargs["stop_loss_price"] = stop_loss_price
                else:
                    # This should never happen for commodities (enforced above), but handle gracefully
                    if not effective_risk.manual_stop_loss:
                        order_kwargs["stop_loss_price"] = stop_loss_price
                        order_kwargs["take_profit_limit_price"] = profit_target_price
                    else:
                        print(f"  [INFO] MANUAL STOP-LOSS MODE: Bracket order stop-loss NOT included (you manage it manually)")

            # Submit the order with detailed documentation
            # In paper trading, crypto shorts may be rejected.
            # In live trading with proper account permissions, shorts will execute.
            try:
                entry_resp = self.client.submit_order(**order_kwargs)
                
                # CRITICAL FIX: For crypto, submit separate stop-loss and take-profit orders
                # Alpaca doesn't support bracket orders (OCO) for crypto, but we can submit
                # standalone stop-loss and take-profit orders that execute at broker level
                # UNLESS manual_stop_loss is enabled - then user manages stop-losses manually
                stop_loss_order_id = None
                take_profit_order_id = None
                
                if is_crypto and stop_loss_price and not effective_risk.manual_stop_loss:
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
                elif effective_risk.manual_stop_loss and stop_loss_price:
                    print(f"  📝 MANUAL STOP-LOSS MODE: Stop-loss calculated at ${stop_loss_price:.2f} but NOT submitted (you manage it manually)")
                
                # Submit take-profit limit order for crypto (broker-level protection)
                # Profit target is REQUIRED, so always try to place take-profit order
                if is_crypto and profit_target_price:
                    try:
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
                        print(f"  ✅ Take-profit order placed at broker level: ${profit_target_price:.2f} ({effective_profit_target:.2f}% target, Order ID: {take_profit_order_id})")
                    except Exception as tp_exc:
                        print(f"  ⚠️  WARNING: Failed to place broker-level take-profit order: {tp_exc}")
                        print(f"     Take-profit will only work while monitoring script is running")
                
                # Calculate all trade details for comprehensive documentation
                entry_cost = notional_rounded if is_crypto else (implied_qty * current_price)
                expected_profit_at_target = (profit_target_price - current_price) * implied_qty if target_side == "long" else (current_price - profit_target_price) * implied_qty
                expected_profit_pct = effective_profit_target  # Always use user-specified profit target (required)
                max_loss_at_stop = abs((current_price - stop_loss_price) * implied_qty) if target_side == "long" else abs((stop_loss_price - current_price) * implied_qty)
                max_loss_pct = effective_risk.default_stop_loss_pct * 100
                risk_reward_ratio = abs(expected_profit_at_target / max_loss_at_stop) if max_loss_at_stop > 0 else 0
                expected_value_at_target = entry_cost + expected_profit_at_target
                expected_value_at_stop = entry_cost - max_loss_at_stop
                
                orders["decision"] = "enter_long" if target_side == "long" else "enter_short"
                orders["entry_notional"] = entry_cost
                orders["entry_qty"] = implied_qty
                orders["trade_side"] = side
                orders["entry_price"] = current_price
                orders["stop_loss_price"] = stop_loss_price
                orders["stop_loss_pct"] = effective_risk.default_stop_loss_pct
                orders["profit_target_price"] = profit_target_price
                orders["profit_target_pct"] = effective_profit_target  # Always user-specified (required)
                orders["take_profit_price"] = take_profit_price
                orders["take_profit_pct"] = stop_pct * tp_mult  # For reference only (profit_target_pct is what's used)
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
                        "profit_target_pct": effective_profit_target,  # Always set (required)
                        "profit_target_price": profit_target_price,
                        "expected_profit_amount": expected_profit_at_target,
                        "expected_profit_pct": effective_profit_target,  # Use user-specified value
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
                expected_value_at_stop = entry_cost - max_loss_at_stop  # Loss reduces value
                
                # Enhanced user-friendly display for real money trading
                print(f"\n{'='*80}")
                if is_commodities:
                    print(f"[REAL MONEY TRADING] NEW POSITION ENTERED: {trading_symbol}")
                    print(f"{'='*80}")
                    print(f"\n[WARNING] REAL MONEY IS AT RISK - This is a live trade with actual capital")
                else:
                    print(f"NEW POSITION ENTERED: {trading_symbol}")
                    print(f"{'='*80}")
                
                print(f"\n[INVESTMENT] YOUR INVESTMENT DETAILS:")
                print(f"  Initial Investment: ${entry_cost:,.2f}")
                print(f"    Symbol:            {trading_symbol} ({asset.data_symbol})")
                if is_commodities and self.client.broker_name == "dhan":
                    print(f"    Exchange:          MCX (Multi Commodity Exchange)")
                    print(f"    Contract:          {trading_symbol} (MCX Futures Contract)")
                print(f"    Side:              {target_side.upper()}")
                print(f"    Entry Price:       ${current_price:,.2f}")
                print(f"    Quantity:          {implied_qty:,.2f}")
                if is_commodities and self.client.broker_name == "dhan":
                    lot_size = get_mcx_lot_size(asset.data_symbol)
                    print(f"    Lot Size:          {lot_size} (MCX minimum tradable unit)")
                print(f"    Order ID:          {entry_resp.get('id', 'N/A')}")
                
                print(f"\n[PROFIT TARGET] WHERE YOU WILL EXIT WITH PROFIT (USER SPECIFIED):")
                print(f"  Target Price:       ${profit_target_price:,.2f} ({effective_profit_target:+.2f}% from entry - YOUR TARGET)")
                print(f"  Expected Profit:    ${expected_profit_at_target:+,.2f}")
                print(f"  Total Value:        ${expected_value_at_target:,.2f}")
                print(f"  Return on Investment: {((expected_profit_at_target / entry_cost) * 100):+.2f}%")
                
                print(f"\n[STOP-LOSS] YOUR MAXIMUM RISK (REAL MONEY):")
                if user_stop_loss_pct is not None:
                    print(f"  Stop Price:         ${stop_loss_price:,.2f} ({user_stop_loss_pct*100:.2f}% from entry - USER SPECIFIED)")
                else:
                    print(f"  Stop Price:         ${stop_loss_price:,.2f} ({effective_risk.default_stop_loss_pct*100:.2f}% from entry - DEFAULT)")
                print(f"  Maximum Loss:        ${max_loss_at_stop:,.2f} ({max_loss_pct:.2f}% of investment)")
                print(f"  Total Value at Stop: ${expected_value_at_stop:,.2f}")
                print(f"  Protection Level:    {'[ENABLED] Broker-level (executes even if system is down)' if orders['bracket_order_used'] or stop_loss_order_id else '[MONITORING] System-level (requires script running)'}")
                if is_commodities:
                    print(f"  [IMPORTANT]         Stop-loss is CRITICAL for real money trading")
                    print(f"                      Position will auto-exit if price hits stop-loss")
                
                print(f"\n[ANALYSIS] RISK/REWARD BREAKDOWN:")
                print(f"  Risk/Reward Ratio:   {risk_reward_ratio:.2f}:1")
                print(f"  Risk Amount:      ${max_loss_at_stop:.2f}")
                print(f"  Reward Amount:   ${expected_profit_at_target:.2f}")
                print(f"{'='*80}\n")
                
                # Save position to position manager (profit target is REQUIRED, so always track)
                try:
                    self.position_manager.save_position(
                        symbol=trading_symbol,
                        data_symbol=asset.data_symbol,
                        asset_type=asset.asset_type,
                        side=target_side,
                        entry_price=current_price,
                        quantity=implied_qty,
                        profit_target_pct=effective_profit_target,  # Always set (required)
                        stop_loss_pct=effective_risk.default_stop_loss_pct,
                        stop_loss_order_id=stop_loss_order_id,
                        take_profit_order_id=take_profit_order_id,
                    )
                    orders["position_tracked"] = True
                except Exception as pos_exc:
                    # Log error but don't fail the trade
                    orders["position_tracked"] = False
                    orders["position_tracking_error"] = str(pos_exc)
                
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




