"""
Position tracking and management for profit-target-based trading.

This module handles:
- Saving/loading active positions with profit targets
- Tracking entry prices, profit targets, and stop-loss levels
- Monitoring positions for profit target achievement
- Position state persistence
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Position:
    """Represents an active trading position with profit target."""
    
    symbol: str  # Trading symbol (e.g., BTCUSD, GLD)
    data_symbol: str  # Data symbol (e.g., BTC-USDT, GC=F)
    asset_type: str  # "crypto" or "commodities"
    side: str  # "long" or "short"
    entry_price: float  # Average entry price (updated when adding to position)
    entry_time: str  # ISO timestamp
    quantity: float  # Total quantity (accumulated as we add more)
    profit_target_pct: float  # User's desired profit percentage (e.g., 10.0 for 10%)
    profit_target_price: float  # Calculated target price (based on average entry)
    stop_loss_pct: float  # Stop-loss percentage
    stop_loss_price: float  # Calculated stop-loss price (based on average entry)
    status: str = "open"  # "open", "profit_target_hit", "stop_loss_hit", "closed"
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None
    realized_pl: Optional[float] = None
    realized_pl_pct: Optional[float] = None
    initial_entry_price: Optional[float] = None  # First entry price (for tracking)
    total_cost_basis: Optional[float] = None  # Total cost of all purchases
    stop_loss_order_id: Optional[str] = None  # Alpaca order ID for stop-loss (broker-level protection)
    take_profit_order_id: Optional[str] = None  # Alpaca order ID for take-profit (broker-level protection)
    short_prediction_cycles: int = 0  # Track cycles with SHORT prediction while holding LONG (for commodities when shorting disabled)
    highest_price: Optional[float] = None  # Track highest price for trailing stop (LONG positions)
    lowest_price: Optional[float] = None  # Track lowest price for trailing stop (SHORT positions)
    trailing_stop_triggered: bool = False  # Flag if trailing stop has been triggered


class PositionManager:
    """Manages active positions with profit targets."""
    
    def __init__(self, positions_file: Optional[Path] = None):
        """
        Initialize position manager.
        
        Args:
            positions_file: Path to JSON file storing positions. Defaults to data/positions/active_positions.json
        """
        if positions_file is None:
            positions_file = Path("data/positions/active_positions.json")
        self.positions_file = Path(positions_file)
        self.positions_file.parent.mkdir(parents=True, exist_ok=True)
        self._positions: Dict[str, Position] = {}
        self._load_positions()
    
    def _load_positions(self) -> None:
        """Load positions from JSON file."""
        if not self.positions_file.exists():
            self._positions = {}
            return
        
        try:
            with open(self.positions_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self._positions = {}
            for symbol, pos_data in data.items():
                # Convert dict to Position dataclass
                self._positions[symbol] = Position(**pos_data)
        except Exception as exc:
            print(f"[WARNING] Failed to load positions from {self.positions_file}: {exc}")
            self._positions = {}
    
    def _save_positions(self) -> None:
        """Save positions to JSON file."""
        try:
            # Convert Position dataclasses to dicts
            data = {}
            for symbol, position in self._positions.items():
                data[symbol] = asdict(position)
            
            # Write to file atomically (write to temp, then rename)
            temp_file = self.positions_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            temp_file.replace(self.positions_file)
        except Exception as exc:
            print(f"[ERROR] Failed to save positions to {self.positions_file}: {exc}")
    
    def save_position(
        self,
        symbol: str,
        data_symbol: str,
        asset_type: str,
        side: str,
        entry_price: float,
        quantity: float,
        profit_target_pct: float,
        stop_loss_pct: float,
        stop_loss_order_id: Optional[str] = None,
        take_profit_order_id: Optional[str] = None,
    ) -> Position:
        """
        Save a new position or update existing one (for pyramiding).
        
        Args:
            symbol: Trading symbol (e.g., BTCUSD)
            data_symbol: Data symbol (e.g., BTC-USDT)
            asset_type: "crypto" or "commodities"
            side: "long" or "short"
            entry_price: Entry price for this purchase
            quantity: Quantity for this purchase (will be added to existing if position exists)
            profit_target_pct: User's desired profit percentage (e.g., 10.0)
            stop_loss_pct: Stop-loss percentage (e.g., 2.0)
        
        Returns:
            Position object (with updated average entry price if adding to existing position)
        """
        existing_position = self.get_position(symbol)
        
        if existing_position and existing_position.status == "open" and existing_position.side == side:
            # Adding to existing position - calculate new average entry price
            existing_qty = abs(existing_position.quantity)
            existing_cost = existing_position.total_cost_basis or (existing_position.entry_price * existing_qty)
            new_cost = entry_price * abs(quantity)
            total_cost = existing_cost + new_cost
            total_qty = existing_qty + abs(quantity)
            
            # Calculate new average entry price
            new_avg_entry = total_cost / total_qty if total_qty > 0 else entry_price
            
            # Update profit target and stop-loss based on new average entry
            # FIX 6: Add entry buffer (0.75% default) to stop-loss to account for entry slippage and initial volatility
            entry_buffer_pct = 0.75  # 0.75% buffer (between 0.5-1% as recommended)
            if side == "long":
                profit_target_price = new_avg_entry * (1.0 + profit_target_pct / 100.0)
                # Add buffer to stop-loss: stop_loss_pct + entry_buffer_pct
                stop_loss_price = new_avg_entry * (1.0 - (stop_loss_pct + entry_buffer_pct) / 100.0)
            else:  # short
                profit_target_price = new_avg_entry * (1.0 - profit_target_pct / 100.0)
                # Add buffer to stop-loss: stop_loss_pct + entry_buffer_pct
                stop_loss_price = new_avg_entry * (1.0 + (stop_loss_pct + entry_buffer_pct) / 100.0)
            
            # Update position
            existing_position.entry_price = new_avg_entry
            existing_position.quantity = total_qty if side == "long" else -total_qty
            existing_position.total_cost_basis = total_cost
            existing_position.profit_target_price = profit_target_price
            existing_position.stop_loss_price = stop_loss_price
            if existing_position.initial_entry_price is None:
                existing_position.initial_entry_price = existing_position.entry_price
            
            self._save_positions()
            return existing_position
        else:
            # New position
            # FIX 6: Add entry buffer (0.75% default) to stop-loss to account for entry slippage and initial volatility
            # This prevents stop-loss from triggering immediately after entry
            entry_buffer_pct = 0.75  # 0.75% buffer (between 0.5-1% as recommended)
            if side == "long":
                profit_target_price = entry_price * (1.0 + profit_target_pct / 100.0)
                # Add buffer to stop-loss: stop_loss_pct + entry_buffer_pct
                stop_loss_price = entry_price * (1.0 - (stop_loss_pct + entry_buffer_pct) / 100.0)
            else:  # short
                profit_target_price = entry_price * (1.0 - profit_target_pct / 100.0)
                # Add buffer to stop-loss: stop_loss_pct + entry_buffer_pct
                stop_loss_price = entry_price * (1.0 + (stop_loss_pct + entry_buffer_pct) / 100.0)
            
            position = Position(
                symbol=symbol,
                data_symbol=data_symbol,
                asset_type=asset_type,
                side=side,
                entry_price=entry_price,
                entry_time=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                quantity=quantity,
                profit_target_pct=profit_target_pct,
                profit_target_price=profit_target_price,
                stop_loss_pct=stop_loss_pct,
                stop_loss_price=stop_loss_price,
                status="open",
                initial_entry_price=entry_price,
                total_cost_basis=entry_price * abs(quantity),
                stop_loss_order_id=stop_loss_order_id,
                take_profit_order_id=take_profit_order_id,
                highest_price=entry_price if side == "long" else None,  # Initialize for trailing stop
                lowest_price=entry_price if side == "short" else None,  # Initialize for trailing stop
            )
            
            self._positions[symbol] = position
            self._save_positions()
            return position
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol, or None if not found."""
        return self._positions.get(symbol.upper())
    
    def get_all_positions(self) -> List[Position]:
        """Get all active positions."""
        return [pos for pos in self._positions.values() if pos.status == "open"]
    
    def get_positions_by_asset_type(self, asset_type: str) -> List[Position]:
        """Get all active positions for a specific asset type."""
        return [
            pos for pos in self._positions.values()
            if pos.status == "open" and pos.asset_type == asset_type
        ]
    
    def update_position(
        self,
        symbol: str,
        current_price: float,
        current_time: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update position with current price and check if profit target or stop-loss is hit.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            current_time: Current timestamp (ISO format). If None, uses current time.
        
        Returns:
            Dict with exit information if target/stop-loss hit, None otherwise.
            Format: {
                "should_exit": bool,
                "exit_reason": str,  # "profit_target_hit" or "stop_loss_hit"
                "exit_price": float,
                "position": Position,
            }
        """
        position = self.get_position(symbol)
        if not position or position.status != "open":
            return None
        
        if current_time is None:
            current_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        # Update highest/lowest prices for trailing stop
        price_updated = False
        if position.side == "long":
            if position.highest_price is None or current_price > position.highest_price:
                position.highest_price = current_price
                price_updated = True
            # Check if price has dropped from peak (trailing stop) - SELL AT PEAK
            if position.highest_price is not None:
                drawdown_from_peak = ((current_price - position.highest_price) / position.highest_price) * 100
                # CRITICAL: Exit when price drops from peak (trailing stop) - SELL HIGH
                # This captures profit near highs instead of waiting for prediction to turn
                # SAME for crypto and commodities: 2.5% trailing stop (sell at peak)
                trailing_stop_pct = 2.5  # Exit when drops 2.5% from peak - SELL AT PEAK
                if drawdown_from_peak < -trailing_stop_pct and not position.trailing_stop_triggered:
                    position.trailing_stop_triggered = True
                    self._save_positions()  # Save before exiting
                    # Return exit signal due to trailing stop
                    return {
                        "should_exit": True,
                        "exit_reason": f"trailing_stop_from_peak",
                        "exit_price": current_price,
                        "position": position,
                    }
        else:  # short
            if position.lowest_price is None or current_price < position.lowest_price:
                position.lowest_price = current_price
                price_updated = True
            # Check if price has risen from bottom (trailing stop for shorts)
            if position.lowest_price is not None:
                rise_from_bottom = ((current_price - position.lowest_price) / position.lowest_price) * 100
                # SAME for crypto and commodities: 2.5% trailing stop (sell at peak)
                trailing_stop_pct = 2.5  # Same for both crypto and commodities - exit when rises 2.5% from bottom
                if rise_from_bottom > trailing_stop_pct and not position.trailing_stop_triggered:
                    position.trailing_stop_triggered = True
                    self._save_positions()  # Save before exiting
                    return {
                        "should_exit": True,
                        "exit_reason": f"trailing_stop_from_bottom",
                        "exit_price": current_price,
                        "position": position,
                    }
        
        # Save position if peak/valley was updated
        if price_updated:
            self._save_positions()
        
        # Check profit target
        profit_target_hit = False
        stop_loss_hit = False
        
        if position.side == "long":
            profit_target_hit = current_price >= position.profit_target_price
            stop_loss_hit = current_price <= position.stop_loss_price
        else:  # short
            profit_target_hit = current_price <= position.profit_target_price
            stop_loss_hit = current_price >= position.stop_loss_price
        
        if profit_target_hit:
            # Calculate realized P/L
            if position.side == "long":
                realized_pl = (current_price - position.entry_price) * position.quantity
                realized_pl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            else:  # short
                realized_pl = (position.entry_price - current_price) * position.quantity
                realized_pl_pct = ((position.entry_price - current_price) / position.entry_price) * 100
            
            # Update position status
            position.status = "profit_target_hit"
            position.exit_price = current_price
            position.exit_time = current_time
            position.exit_reason = "profit_target_hit"
            position.realized_pl = realized_pl
            position.realized_pl_pct = realized_pl_pct
            
            self._save_positions()
            
            return {
                "should_exit": True,
                "exit_reason": "profit_target_hit",
                "exit_price": current_price,
                "position": position,
                "realized_pl": realized_pl,
                "realized_pl_pct": realized_pl_pct,
            }
        
        if stop_loss_hit:
            # Calculate realized P/L (will be negative)
            if position.side == "long":
                realized_pl = (current_price - position.entry_price) * position.quantity
                realized_pl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            else:  # short
                realized_pl = (position.entry_price - current_price) * position.quantity
                realized_pl_pct = ((position.entry_price - current_price) / position.entry_price) * 100
            
            # Update position status
            position.status = "stop_loss_hit"
            position.exit_price = current_price
            position.exit_time = current_time
            position.exit_reason = "stop_loss_hit"
            position.realized_pl = realized_pl
            position.realized_pl_pct = realized_pl_pct
            
            self._save_positions()
            
            return {
                "should_exit": True,
                "exit_reason": "stop_loss_hit",
                "exit_price": current_price,
                "position": position,
                "realized_pl": realized_pl,
                "realized_pl_pct": realized_pl_pct,
            }
        
        # Position still active
        return None
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        realized_pl: Optional[float] = None,
        realized_pl_pct: Optional[float] = None,
    ) -> Optional[Position]:
        """
        Close a position (mark as closed) with clear entry/exit prices and P/L.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price (actual filled price)
            exit_reason: Reason for exit (e.g., "profit_target_hit", "stop_loss_hit")
            realized_pl: Realized profit/loss (calculated if not provided)
            realized_pl_pct: Realized profit/loss percentage (calculated if not provided)
        
        Returns:
            Updated Position object with exit_price, realized_pl, and realized_pl_pct, or None if position not found
        """
        position = self.get_position(symbol)
        if not position:
            return None
        
        # Calculate P/L if not provided
        if realized_pl is None or realized_pl_pct is None:
            if position.side == "long":
                realized_pl = (exit_price - position.entry_price) * abs(position.quantity)
                realized_pl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100 if position.entry_price > 0 else 0
            else:  # short
                realized_pl = (position.entry_price - exit_price) * abs(position.quantity)
                realized_pl_pct = ((position.entry_price - exit_price) / position.entry_price) * 100 if position.entry_price > 0 else 0
        
        # Update position with exit details
        position.status = "closed"
        position.exit_price = exit_price  # Clear exit price for easy P/L calculation
        position.exit_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        position.exit_reason = exit_reason
        position.realized_pl = realized_pl  # Clear P/L in currency
        position.realized_pl_pct = realized_pl_pct  # Clear P/L percentage
        
        # ENHANCED: Add P/L summary for easy reference
        # This makes it easy to calculate: entry_price, exit_price, quantity, realized_pl are all available
        
        self._save_positions()
        return position
    
    def update_profit_target(
        self,
        symbol: str,
        new_profit_target_pct: float,
        stop_loss_pct: Optional[float] = None,
    ) -> Optional[Position]:
        """
        Update the profit target and stop-loss for an existing position.
        
        Args:
            symbol: Trading symbol
            new_profit_target_pct: New profit target percentage (e.g., 1.5 for 1.5%)
            stop_loss_pct: New stop-loss percentage (optional, uses existing if not provided)
        
        Returns:
            Updated Position object, or None if position not found
        """
        position = self.get_position(symbol)
        if not position or position.status != "open":
            return None
        
        # Use existing stop-loss if not provided
        if stop_loss_pct is None:
            stop_loss_pct = position.stop_loss_pct
        
        # Recalculate profit target and stop-loss prices based on current entry price
        # FIX 6: Add entry buffer (0.75% default) to stop-loss to account for entry slippage and initial volatility
        entry_buffer_pct = 0.75  # 0.75% buffer (between 0.5-1% as recommended)
        if position.side == "long":
            profit_target_price = position.entry_price * (1.0 + new_profit_target_pct / 100.0)
            # Add buffer to stop-loss: stop_loss_pct + entry_buffer_pct
            stop_loss_price = position.entry_price * (1.0 - (stop_loss_pct + entry_buffer_pct) / 100.0)
        else:  # short
            profit_target_price = position.entry_price * (1.0 - new_profit_target_pct / 100.0)
            # Add buffer to stop-loss: stop_loss_pct + entry_buffer_pct
            stop_loss_price = position.entry_price * (1.0 + (stop_loss_pct + entry_buffer_pct) / 100.0)
        
        # Update position
        position.profit_target_pct = new_profit_target_pct
        position.profit_target_price = profit_target_price
        position.stop_loss_pct = stop_loss_pct
        position.stop_loss_price = stop_loss_price
        
        self._save_positions()
        return position
    
    def remove_position(self, symbol: str) -> bool:
        """
        Remove a position from tracking (e.g., if it was closed externally).
        
        Returns:
            True if position was removed, False if not found
        """
        if symbol.upper() in self._positions:
            del self._positions[symbol.upper()]
            self._save_positions()
            return True
        return False

