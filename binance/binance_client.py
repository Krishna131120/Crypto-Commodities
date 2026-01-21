"""
Binance paper trading REST client.

This client implements the BrokerClient interface for Binance spot trading.
Supports paper trading via Binance Testnet (or mainnet if configured).

Environment variables expected:
- BINANCE_API_KEY: Your Binance API key
- BINANCE_SECRET_KEY: Your Binance secret key
- BINANCE_TESTNET: Optional, set to "true" to use testnet (default: false for mainnet)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
import hmac
import hashlib
from urllib.parse import urlencode

# Import from parent trading package
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading.broker_interface import BrokerClient


# Binance API endpoints
BINANCE_MAINNET_BASE_URL = "https://api.binance.com/api/v3"
BINANCE_TESTNET_BASE_URL = "https://testnet.binance.vision/api/v3"


class BinanceAuthError(RuntimeError):
    """Raised when Binance credentials are missing or invalid."""


@dataclass
class BinanceConfig:
    """Configuration wrapper for Binance API settings."""

    api_key: str
    secret_key: str
    base_url: str = BINANCE_MAINNET_BASE_URL
    testnet: bool = False

    @classmethod
    def from_env(cls) -> "BinanceConfig":
        """Load Binance config from environment variables."""
        api_key = os.getenv("BINANCE_API_KEY") or os.getenv("binance_api_key")
        secret_key = os.getenv("BINANCE_SECRET_KEY") or os.getenv("binance_secret_key")
        testnet_str = os.getenv("BINANCE_TESTNET", "false").lower()
        testnet = testnet_str == "true" or testnet_str == "1"
        
        # Fallback: try .env file if env vars not set
        if (not api_key or not secret_key) and os.path.exists(".env"):
            try:
                with open(".env", "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        key = key.strip().strip('"').strip("'")
                        val = val.strip().strip('"').strip("'")
                        
                        if key.upper() == "BINANCE_API_KEY" and not api_key:
                            api_key = val
                        elif key.upper() == "BINANCE_SECRET_KEY" and not secret_key:
                            secret_key = val
                        elif key.upper() == "BINANCE_TESTNET":
                            testnet = val.lower() == "true" or val.lower() == "1"
            except Exception:
                pass

        if not api_key or not secret_key:
            raise BinanceAuthError(
                "Missing Binance credentials. Please set BINANCE_API_KEY and "
                "BINANCE_SECRET_KEY in your environment or .env file."
            )
        
        base_url = BINANCE_TESTNET_BASE_URL if testnet else BINANCE_MAINNET_BASE_URL
        return cls(api_key=api_key, secret_key=secret_key, base_url=base_url, testnet=testnet)


class BinanceClient(BrokerClient):
    """Binance spot trading client implementing BrokerClient interface."""

    def __init__(self, config: Optional[BinanceConfig] = None):
        self.config = config or BinanceConfig.from_env()
        self._session = requests.Session()
    
    @property
    def broker_name(self) -> str:
        return "binance"
    
    def _sign_request(self, params: Dict[str, Any]) -> str:
        """Sign request parameters for authenticated endpoints."""
        query_string = urlencode(params)
        signature = hmac.new(
            self.config.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        """Make HTTP request to Binance API."""
        url = f"{self.config.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.config.api_key} if signed else {}
        
        if params is None:
            params = {}
        
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._sign_request(params)
        
        resp = self._session.request(method, url, params=params, headers=headers, timeout=10)
        
        if resp.status_code == 401:
            raise BinanceAuthError("Binance authentication failed (401). Check your keys.")
        
        try:
            data = resp.json()
        except ValueError:
            resp.raise_for_status()
            return resp.text
        
        if not resp.ok:
            error_msg = data.get("msg") if isinstance(data, dict) else str(data)
            raise RuntimeError(f"Binance API error {resp.status_code}: {error_msg}")
        
        return data
    
    # ------------------------------------------------------------------
    # Account and positions
    # ------------------------------------------------------------------
    def get_account(self) -> Dict[str, Any]:
        """Return account details (buying power, equity, etc.)."""
        account = self._request("GET", "/account", signed=True)
        
        # Convert Binance format to standard format
        equity = 0.0
        buying_power = 0.0
        cash = 0.0
        
        # Binance account info includes total asset value in USDT
        # Get balances and calculate total value
        for balance in account.get("balances", []):
            asset = balance.get("asset", "")
            free = float(balance.get("free", 0) or 0)
            locked = float(balance.get("locked", 0) or 0)
            total = free + locked
            
            if total <= 0:
                continue
            
            # USDT is the quote currency - this is our cash
            if asset == "USDT":
                cash = free
                buying_power = free  # Available for trading
                equity += free
            else:
                # For other assets, get current price and calculate value
                try:
                    symbol = f"{asset}USDT"
                    ticker = self._request("GET", "/ticker/price", params={"symbol": symbol})
                    price = float(ticker.get("price", 0) or 0)
                    if price > 0:
                        asset_value = total * price
                        equity += asset_value
                        # Free balance can be used for trading
                        if free > 0:
                            buying_power += free * price
                except Exception:
                    # Skip if price unavailable or symbol doesn't exist
                    pass
        
        # If equity calculation failed, use account balance if available
        if equity <= 0:
            # Try to get total balance from account if available
            total_asset = account.get("totalWalletBalance")
            if total_asset:
                try:
                    equity = float(total_asset)
                    buying_power = equity  # Simplified
                except (ValueError, TypeError):
                    pass
        
        return {
            "equity": equity,
            "buying_power": buying_power if buying_power > 0 else equity,  # Fallback to equity
            "cash": cash,
            "raw_account": account,  # Include raw response for debugging
        }
    
    def list_positions(self) -> List[Dict[str, Any]]:
        """Return a list of all open positions (non-zero balances)."""
        account = self._request("GET", "/account", signed=True)
        positions = []
        
        for balance in account.get("balances", []):
            free = float(balance.get("free", 0) or 0)
            locked = float(balance.get("locked", 0) or 0)
            total = free + locked
            
            if total > 0:
                asset = balance.get("asset", "")
                
                # Skip USDT as it's the quote currency
                if asset == "USDT":
                    continue
                
                # Get current price and symbol
                symbol = f"{asset}USDT"
                try:
                    ticker = self._request("GET", f"/ticker/price", params={"symbol": symbol})
                    current_price = float(ticker.get("price", 0) or 0)
                    
                    # Get average entry price from recent trades (simplified)
                    # In production, track entry prices properly
                    avg_entry_price = current_price  # Simplified - should track actual entry
                    
                    positions.append({
                        "symbol": symbol,
                        "qty": total,
                        "market_value": total * current_price,
                        "avg_entry_price": avg_entry_price,
                        "current_price": current_price,
                        "asset": asset,
                    })
                except:
                    # Skip if symbol doesn't exist or price unavailable
                    continue
        
        return positions
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return open position for a symbol, or None if no position."""
        positions = self.list_positions()
        for pos in positions:
            if pos.get("symbol", "").upper() == symbol.upper():
                return pos
        return None
    
    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def submit_order(
        self,
        *,
        symbol: str,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        side: str,
        order_type: str = "market",
        time_in_force: str = "GTC",
        take_profit_limit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit an order to Binance.
        
        Binance uses different parameters:
        - timeInForce: "GTC", "IOC", "FOK"
        - type: "MARKET", "LIMIT", "STOP_LOSS", "TAKE_PROFIT", etc.
        """
        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be 'buy' or 'sell'")
        
        # Convert symbol format: BTCUSD -> BTCUSDT (Binance uses USDT, not USD)
        # If symbol already ends with USDT, keep it
        symbol = symbol.upper()
        if symbol.endswith("USD") and not symbol.endswith("USDT"):
            symbol = symbol[:-3] + "USDT"
        
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET" if order_type == "market" else order_type.upper(),
        }
        
        if qty is not None:
            params["quantity"] = qty
        elif notional is not None:
            # For market orders, Binance supports quoteOrderQty (USDT amount)
            if order_type == "market":
                params["quoteOrderQty"] = notional
            else:
                raise ValueError("notional only supported for market orders")
        else:
            raise ValueError("Exactly one of qty or notional must be provided")
        
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        
        # Binance doesn't support bracket orders directly like Alpaca
        # Take-profit and stop-loss need to be separate orders
        # For now, just submit the main order
        # TODO: Implement bracket orders using OCO (One-Cancels-Other) orders
        
        response = self._request("POST", "/order", params=params, signed=True)
        
        # Convert response to standard format
        return {
            "order_id": str(response.get("orderId", "")),
            "symbol": response.get("symbol", ""),
            "side": response.get("side", ""),
            "status": self._convert_order_status(response.get("status", "")),
            "qty": float(response.get("executedQty", 0) or 0),
            "filled_qty": float(response.get("executedQty", 0) or 0),
            "filled_avg_price": float(response.get("price", 0) or 0) if response.get("price") else None,
            "raw_response": response,
        }
    
    def _convert_order_status(self, binance_status: str) -> str:
        """Convert Binance order status to standard format."""
        status_map = {
            "NEW": "pending",
            "PARTIALLY_FILLED": "partially_filled",
            "FILLED": "filled",
            "CANCELED": "canceled",
            "PENDING_CANCEL": "pending_cancel",
            "REJECTED": "rejected",
            "EXPIRED": "expired",
        }
        return status_map.get(binance_status.upper(), "unknown")
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order by ID."""
        # Binance requires symbol to cancel order
        # For now, return error - in production, track symbol per order
        raise NotImplementedError("Binance cancel_order requires symbol. Use cancel_order_with_symbol instead.")
    
    def cancel_order_with_symbol(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order by ID and symbol."""
        symbol = symbol.upper()
        if symbol.endswith("USD") and not symbol.endswith("USDT"):
            symbol = symbol[:-3] + "USDT"
        
        params = {
            "symbol": symbol,
            "orderId": order_id,
        }
        return self._request("DELETE", "/order", params=params, signed=True)
    
    def get_last_trade(
        self,
        symbol: str,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        force_retry: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Get the last trade price for a symbol."""
        symbol = symbol.upper()
        if symbol.endswith("USD") and not symbol.endswith("USDT"):
            symbol = symbol[:-3] + "USDT"
        
        for attempt in range(max_retries):
            try:
                ticker = self._request("GET", "/ticker/price", params={"symbol": symbol})
                price = float(ticker.get("price", 0) or 0)
                if price > 0:
                    return {
                        "price": price,
                        "symbol": symbol,
                    }
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
        
        return None
