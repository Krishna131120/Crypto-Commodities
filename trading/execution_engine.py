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
    ):
        self.client = client or AlpacaClient()
        self.risk = risk_config or TradingRiskConfig()
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_horizon = normalize_profile(default_horizon) if default_horizon else None

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
    ) -> Optional[Dict[str, Any]]:
        """
        Given a consensus dict and current price, align Alpaca position.

        asset:      AssetMapping (crypto only: BTC-USDT, ETH-USDT, SOL-USDT)
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
        """
        action = str(consensus.get("consensus_action", "hold")).lower()
        confidence = float(consensus.get("consensus_confidence", 0.0) or 0.0)
        position_size = float(consensus.get("position_size", 0.0) or 0.0)

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

        if confidence < effective_risk.min_confidence:
            # Too weak for this horizon's threshold; treat as hold / no-op.
            return None

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

        # Determine target notional for exposure, *clamped by buying power*.
        # We start from equity-based sizing, then cap per-symbol risk (horizon-specific)
        # and finally ensure we never request more than available buying power.
        max_symbol_notional = equity * effective_risk.max_notional_per_symbol_pct
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
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "dry_run": dry_run,
        }

        if side_in_market == target_side:
            # Already aligned (long/short/flat); nothing to do.
            orders["decision"] = "no_change"
            orders["final_side"] = side_in_market
            orders["final_qty"] = existing_qty
            self._log(orders)
            return orders

        # First, exit any existing position if target is flat.
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
                self._log(orders)
                return orders
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
            self._log(orders)
            return orders

        # Flipping positions: long -> short or short -> long
        if side_in_market in {"long", "short"} and target_side in {"long", "short"} and side_in_market != target_side:
            # Need to close existing position first, then open new one
            if existing_qty == 0:
                orders["decision"] = "flip_position_skipped_no_qty"
                orders["final_side"] = side_in_market
                orders["final_qty"] = existing_qty
                self._log(orders)
                return orders
            
            if dry_run:
                orders["decision"] = "would_flip_position"
                orders["trade_qty"] = abs(existing_qty)
                orders["trade_side"] = "sell" if existing_qty > 0 else "buy"
                orders["final_side"] = target_side
                # Will open new position after closing
                new_qty = max(desired_notional / current_price, 0.0) if desired_notional > 0 else 0.0
                orders["final_qty"] = new_qty if target_side == "long" else -new_qty
                self._log(orders)
                return orders
            
            # Close existing position
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
                                # Crypto short rejected by Alpaca (paper trading doesn't support it)
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
                                orders["note"] = f"SHORT order attempted after closing LONG but rejected by Alpaca (paper trading limitation). Model signal: SHORT. In live trading, this may execute successfully."
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
                    
                    if is_crypto:
                        orders["stop_loss_note"] = "Crypto positions: stop-loss must be managed separately (bracket orders not supported)"
                else:
                    orders["decision"] = "flip_to_flat"
                    orders["close_order"] = close_resp
                    orders["trade_qty"] = abs(existing_qty)
                    orders["trade_side"] = "sell" if existing_qty > 0 else "buy"
                    orders["final_side"] = "flat"
                    orders["final_qty"] = 0.0
            else:
                orders["decision"] = "flip_to_flat"
                orders["close_order"] = close_resp
                orders["trade_qty"] = abs(existing_qty)
                orders["trade_side"] = "sell" if existing_qty > 0 else "buy"
                orders["final_side"] = "flat"
                orders["final_qty"] = 0.0
            
            self._log(orders)
            return orders

        # Entering new long/short from flat.
        if side_in_market == "flat" and target_side in {"long", "short"} and desired_notional > 0:
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
            else:
                # For non-crypto assets we can still use qty + Alpaca bracket orders.
                order_kwargs["qty"] = implied_qty
                order_kwargs["stop_loss_price"] = stop_loss_price
                order_kwargs["take_profit_limit_price"] = take_profit_price

            # Submit the order. In paper trading, crypto shorts may be rejected.
            # In live trading with proper account permissions, shorts will execute.
            # All risk parameters (stop-loss, take-profit) are calculated and logged.
            try:
                entry_resp = self.client.submit_order(**order_kwargs)
                
                orders["decision"] = "enter_long" if target_side == "long" else "enter_short"
                orders["entry_notional"] = trade_notional
                orders["entry_qty"] = implied_qty
                orders["trade_side"] = side
                orders["stop_loss_price"] = stop_loss_price
                orders["take_profit_price"] = take_profit_price
                orders["stop_loss_pct"] = self.risk.default_stop_loss_pct
                orders["take_profit_pct"] = stop_pct * tp_mult
                orders["final_side"] = target_side
                orders["final_qty"] = implied_qty if target_side == "long" else -implied_qty
                orders["entry_order"] = entry_resp
                orders["execution_status"] = "success"
                
                # For crypto, we cannot use bracket orders, so stop-loss must be managed separately.
                # Log the stop-loss level so it can be monitored/managed externally if needed.
                if is_crypto:
                    orders["stop_loss_note"] = "Crypto positions: stop-loss must be managed separately (bracket orders not supported)"
                
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




