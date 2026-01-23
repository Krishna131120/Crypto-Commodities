"""
Paper Trading Simulator Client

This client simulates broker behavior without actually placing orders.
Perfect for testing your trading logic without needing TradeTron, Angel One, or any external service.

Features:
- Simulates order execution at current market price
- Tracks virtual positions locally
- Simulates account balance and buying power
- No external API calls needed
- Free - no subscriptions required
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .broker_interface import BrokerClient


@dataclass
class PaperTradingConfig:
    """Configuration for paper trading simulator."""
    
    initial_equity: float = 1000000.0  # Starting capital (Rs. 10,00,000 / 10 Lakhs default)
    initial_cash: float = 1000000.0
    commission_rate: float = 0.001  # 0.1% commission per trade (simulated)


class PaperTradingClient(BrokerClient):
    """
    Paper trading simulator that implements BrokerClient interface.
    
    This client simulates broker behavior without actually placing orders.
    All positions and account balances are tracked locally.
    """
    
    def __init__(self, config: Optional[PaperTradingConfig] = None):
        """
        Initialize paper trading client.
        
        Args:
            config: Optional PaperTradingConfig. If None, uses defaults.
        """
        self.config = config or PaperTradingConfig()
        
        # Track virtual positions
        self._positions: Dict[str, Dict[str, Any]] = {}
        
        # Track account state
        self._equity = self.config.initial_equity
        self._cash = self.config.initial_cash
        self._order_counter = 0  # For generating order IDs
        
        # Track order history
        self._orders: Dict[str, Dict[str, Any]] = {}
    
    @property
    def broker_name(self) -> str:
        return "paper_trading"
    
    def get_account(self) -> Dict[str, Any]:
        """
        Return simulated account details.
        
        Calculates equity based on current positions and cash.
        """
        # Calculate total position value
        total_position_value = 0.0
        for pos in self._positions.values():
            # Use current price if available, otherwise entry price
            current_price = pos.get("ltp", pos.get("avg_entry_price", 0))
            qty = pos.get("qty", 0)
            total_position_value += abs(qty * current_price)
        
        # Equity = cash + position values
        equity = self._cash + total_position_value
        
        return {
            "equity": equity,
            "buying_power": self._cash,  # Can only use cash for new positions
            "cash": self._cash,
            "portfolio_value": equity,
            "margin_used": total_position_value,
            "margin_available": self._cash,
            "note": "Paper trading simulator - all values are virtual",
        }
    
    def list_positions(self) -> List[Dict[str, Any]]:
        """
        Return all simulated positions.
        
        Automatically updates position prices from market data sources.
        """
        # Update prices for all positions before returning
        for symbol in list(self._positions.keys()):
            try:
                # Try to get current price and update position
                price_data = self.get_last_trade(symbol)
                if price_data and price_data.get("price"):
                    self.set_position_price(symbol, price_data["price"])
            except Exception:
                pass  # If price update fails, continue with existing price
        
        return list(self._positions.values())
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Return simulated position for a symbol, or None.
        
        Automatically updates position price from market data if available.
        """
        symbol_upper = symbol.upper()
        pos = self._positions.get(symbol_upper)
        
        if pos:
            # Try to update price from market data
            try:
                price_data = self.get_last_trade(symbol)
                if price_data and price_data.get("price"):
                    self.set_position_price(symbol, price_data["price"])
                    pos = self._positions.get(symbol_upper)  # Get updated position
            except Exception:
                pass  # If price update fails, return position with existing price
        
        return pos
    
    def submit_order(
        self,
        *,
        symbol: str,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        side: str,
        order_type: str = "market",
        time_in_force: str = "gtc",
        take_profit_limit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Simulate order submission.
        
        For market orders, immediately fills at current price (you need to provide price via get_last_trade).
        For limit orders, stores the order but doesn't fill until price is reached (simplified - always fills for now).
        """
        symbol_upper = symbol.upper()
        side_lower = side.lower()
        
        if side_lower not in {"buy", "sell"}:
            raise ValueError(f"side must be 'buy' or 'sell', got: {side}")
        
        # Generate order ID
        self._order_counter += 1
        order_id = client_order_id or f"paper_order_{self._order_counter}_{int(time.time())}"
        
        # For paper trading, we need current price to simulate execution
        # In real usage, get_last_trade() should be called first to get current price
        # For now, we'll use a placeholder - the actual price should come from market data
        
        # Check if we have existing position
        existing_pos = self._positions.get(symbol_upper)
        
        # Get current market price for order execution
        # Try multiple sources in priority order
        fill_price = None
        
        # 1. Try existing position's LTP
        if existing_pos and existing_pos.get("ltp"):
            fill_price = existing_pos.get("ltp")
        
        # 2. Try get_last_trade() method (uses local data.json or features.json)
        if not fill_price:
            price_data = self.get_last_trade(symbol)
            if price_data and price_data.get("price"):
                fill_price = price_data["price"]
        
        # 3. Try get_current_price_from_features (if available)
        if not fill_price:
            try:
                from live_trader import get_current_price_from_features
                # Determine asset type from symbol (commodities for MCX symbols)
                asset_type = "commodities"  # Default for this script
                price = get_current_price_from_features(asset_type, symbol, "1d", force_live=False, verbose=False)
                if price and price > 0:
                    fill_price = price
            except Exception:
                pass
        
        # 4. If still no price, raise error
        if not fill_price or fill_price <= 0:
            raise ValueError(
                f"Cannot simulate order for {symbol_upper} without current price. "
                f"Price sources tried: (1) existing position, (2) get_last_trade(), (3) get_current_price_from_features(). "
                f"Please ensure data exists in data/json/raw/commodities/{symbol}/1d/data.json or "
                f"data/features/commodities/{symbol}/1d/features.json"
            )
        
        # Calculate quantity
        if qty is not None:
            qty_to_trade = qty
        elif notional is not None:
            qty_to_trade = notional / fill_price
        else:
            raise ValueError("Either qty or notional must be provided")
        
        # Calculate commission
        order_value = qty_to_trade * fill_price
        commission = order_value * self.config.commission_rate
        
        # Update position
        if side_lower == "buy":
            if existing_pos:
                # Add to existing position
                old_qty = existing_pos.get("qty", 0)
                old_avg = existing_pos.get("avg_entry_price", 0)
                old_cost = old_qty * old_avg
                new_cost = qty_to_trade * fill_price
                
                new_qty = old_qty + qty_to_trade
                new_avg = (old_cost + new_cost) / new_qty if new_qty > 0 else fill_price
                
                existing_pos["qty"] = new_qty
                existing_pos["avg_entry_price"] = new_avg
                existing_pos["ltp"] = fill_price  # Update last traded price
            else:
                # New long position
                self._positions[symbol_upper] = {
                    "symbol": symbol_upper,
                    "qty": qty_to_trade,
                    "avg_entry_price": fill_price,
                    "ltp": fill_price,
                    "market_value": qty_to_trade * fill_price,
                    "unrealized_pl": 0.0,
                }
            
            # Deduct cash (including commission)
            self._cash -= (order_value + commission)
        
        else:  # sell
            if existing_pos:
                old_qty = existing_pos.get("qty", 0)
                if old_qty >= qty_to_trade:
                    # Reduce position
                    existing_pos["qty"] = old_qty - qty_to_trade
                    if existing_pos["qty"] <= 0:
                        # Position closed
                        del self._positions[symbol_upper]
                else:
                    raise ValueError(f"Insufficient position: have {old_qty}, trying to sell {qty_to_trade}")
            else:
                # Short selling (if allowed)
                # For simplicity, we'll allow it in paper trading
                self._positions[symbol_upper] = {
                    "symbol": symbol_upper,
                    "qty": -qty_to_trade,  # Negative for short
                    "avg_entry_price": fill_price,
                    "ltp": fill_price,
                    "market_value": abs(qty_to_trade * fill_price),
                    "unrealized_pl": 0.0,
                }
            
            # Add cash (minus commission)
            self._cash += (order_value - commission)
        
        # Store order
        self._orders[order_id] = {
            "id": order_id,
            "symbol": symbol_upper,
            "qty": qty_to_trade,
            "side": side_lower,
            "order_type": order_type,
            "filled_price": fill_price,
            "status": "filled",
            "filled_at": datetime.now(timezone.utc).isoformat(),
            "commission": commission,
        }
        
        return {
            "id": order_id,
            "status": "filled",
            "symbol": symbol_upper,
            "qty": qty_to_trade,
            "side": side.upper(),
            "order_type": order_type,
            "filled_avg_price": fill_price,
            "filled_qty": qty_to_trade,
            "commission": commission,
        }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order (for paper trading, just marks it as cancelled)."""
        if order_id in self._orders:
            self._orders[order_id]["status"] = "cancelled"
            return {
                "order_id": order_id,
                "status": "cancelled",
            }
        else:
            return {
                "order_id": order_id,
                "status": "not_found",
            }
    
    def get_last_trade(
        self,
        symbol: str,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        force_retry: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Get last trade price for a symbol.
        
        For paper trading, this tries to get price from:
        1. Existing position (if we have one)
        2. Local data.json files (Yahoo Finance data)
        3. Features.json (last candle close price)
        
        NOTE: The execution engine typically uses get_current_price_from_features() 
        which handles price fetching. This method is a fallback.
        """
        # Check if we have a position with LTP
        pos = self._positions.get(symbol.upper())
        if pos and pos.get("ltp"):
            return {
                "price": pos["ltp"],
                "ltp": pos["ltp"],
                "p": pos["ltp"],
            }
        
        # Try to get price from local data.json (Yahoo Finance for commodities)
        try:
            from pathlib import Path
            import json
            
            # Try commodities data path
            data_path = Path("data/json/raw/commodities/yahoo_chart") / symbol / "1d" / "data.json"
            if not data_path.exists():
                # Try alternative path
                data_path = Path("data/json/raw/commodities") / symbol / "1d" / "data.json"
            
            if data_path.exists():
                with open(data_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                    if isinstance(payload, list) and payload:
                        latest = payload[-1]
                        price = float(latest.get("close", 0))
                        if price > 0:
                            return {"price": price, "ltp": price, "p": price}
                    elif isinstance(payload, dict) and "close" in payload:
                        price = float(payload["close"])
                        if price > 0:
                            return {"price": price, "ltp": price, "p": price}
        except Exception:
            pass
        
        # Try to get from features.json (last candle close)
        try:
            from pathlib import Path
            import json
            
            feature_path = Path("data/features/commodities") / symbol / "1d" / "features.json"
            if feature_path.exists():
                with open(feature_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                    if isinstance(payload, list) and payload:
                        latest = payload[-1]
                        price = float(latest.get("close", latest.get("price", 0)))
                        if price > 0:
                            return {"price": price, "ltp": price, "p": price}
        except Exception:
            pass
        
        # Return None - caller should use get_current_price_from_features() instead
        return None
    
    def set_position_price(self, symbol: str, price: float) -> None:
        """
        Update the last traded price for a position.
        
        This is useful when you get price updates from your market data source.
        Call this before submit_order() to ensure orders execute at current market price.
        """
        symbol_upper = symbol.upper()
        if symbol_upper in self._positions:
            self._positions[symbol_upper]["ltp"] = price
            # Update market value and unrealized P/L
            pos = self._positions[symbol_upper]
            qty = pos.get("qty", 0)
            avg_entry = pos.get("avg_entry_price", 0)
            
            market_value = abs(qty * price)
            if qty > 0:  # Long position
                unrealized_pl = (price - avg_entry) * qty
            else:  # Short position
                unrealized_pl = (avg_entry - price) * abs(qty)
            
            pos["market_value"] = market_value
            pos["unrealized_pl"] = unrealized_pl
            pos["ltp"] = price
    
    def get_order_history(self) -> List[Dict[str, Any]]:
        """Get all order history."""
        return list(self._orders.values())
    
    def reset_account(self, new_equity: Optional[float] = None) -> None:
        """
        Reset paper trading account to initial state.
        
        Args:
            new_equity: Optional new starting equity. If None, uses initial_equity from config.
        """
        self._positions = {}
        self._orders = {}
        self._order_counter = 0
        
        if new_equity is not None:
            self._equity = new_equity
            self._cash = new_equity
        else:
            self._equity = self.config.initial_equity
            self._cash = self.config.initial_cash
