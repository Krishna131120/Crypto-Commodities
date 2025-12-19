"""
DHAN API client for live MCX commodity futures trading.

This module provides a DHAN API client that implements the BrokerClient interface,
allowing seamless integration with the ExecutionEngine for MCX commodity trading.

Authentication:
- Uses access token (JWT) with Bearer authentication
- Access token is valid for 24 hours
- Client ID is required for API calls

Environment variables expected:
- DHAN_ACCESS_TOKEN : your DHAN access token (JWT)
- DHAN_CLIENT_ID    : your DHAN client ID
- DHAN_BASE_URL     : optional override for the base URL
                       (defaults to DHAN production URL: https://api.dhan.co)

IMPORTANT:
- Access tokens expire after 24 hours - you may need to refresh daily
- Static IP is required for order APIs (place, modify, cancel orders)
- Test thoroughly in paper trading mode before live trading
- Real money is involved - use with caution
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from .broker_interface import BrokerClient

# DHAN API endpoints (based on DHAN API v2 documentation)
DEFAULT_DHAN_BASE_URL = "https://api.dhan.co"  # DHAN production API
DHAN_PAPER_URL = "https://api-sandbox.dhan.co"  # Paper trading (if available)

# MCX Exchange code (for commodity futures)
MCX_EXCHANGE_CODE = "MCX"  # Multi Commodity Exchange

# DHAN API v2 Endpoints (all require /v2 prefix):
# - GET /v2/fundlimit - Account/fund details
# - GET /v2/positions - List all positions
# - POST /v2/orders - Place order
# - GET /v2/orders/{order_id} - Get order details
# - PUT /v2/orders/{order_id} - Modify order
# - DELETE /v2/orders/{order_id} - Cancel order
# - POST /v2/marketfeed/ltp - Last traded price
# - POST /v2/marketfeed/ohlc - OHLC data
# - POST /v2/marketfeed/quote - Full quote data
# Authentication: Use 'access-token' header (not Authorization: Bearer)


class DhanAuthError(RuntimeError):
    """Raised when DHAN credentials are missing or invalid."""


@dataclass
class DhanConfig:
    """Configuration wrapper for DHAN API settings."""

    access_token: str  # JWT access token (valid for 24 hours)
    client_id: str      # DHAN client ID
    base_url: str = DEFAULT_DHAN_BASE_URL

    @classmethod
    def from_env(cls) -> "DhanConfig":
        """
        Load DHAN config from .env file FIRST (user preference), then fallback to environment variables.
        This ensures .env file always takes precedence when it exists.
        """
        access_token = None
        client_id = None
        base_url = DEFAULT_DHAN_BASE_URL

        # PRIORITY 1: Read from .env file FIRST (user wants to use .env only)
        if os.path.exists(".env"):
            try:
                with open(".env", "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("$env:"):
                            # PowerShell style: $env:DHAN_ACCESS_TOKEN="..."
                            parts = line.split("=", 1)
                            if len(parts) != 2:
                                continue
                            key_part = parts[0].replace("$env:", "").strip().strip('"').strip("'")
                            val_part = parts[1].strip().strip('"').strip("'")
                            if key_part == "DHAN_ACCESS_TOKEN":
                                access_token = val_part
                            elif key_part == "DHAN_CLIENT_ID":
                                client_id = val_part
                            elif key_part == "DHAN_BASE_URL":
                                base_url = val_part
                        else:
                            # Standard .env style: KEY=VALUE
                            if "=" not in line:
                                continue
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip().strip('"').strip("'")
                            if key == "DHAN_ACCESS_TOKEN":
                                access_token = val
                            elif key == "DHAN_CLIENT_ID":
                                client_id = val
                            elif key == "DHAN_BASE_URL":
                                base_url = val
            except Exception:
                pass  # If .env parsing fails, fallback to env vars
        
        # PRIORITY 2: Fallback to environment variables only if .env didn't provide values
        if not access_token:
            access_token = os.getenv("DHAN_ACCESS_TOKEN")
        if not client_id:
            client_id = os.getenv("DHAN_CLIENT_ID")
        if base_url == DEFAULT_DHAN_BASE_URL:
            base_url = os.getenv("DHAN_BASE_URL", DEFAULT_DHAN_BASE_URL)

        if not access_token or not client_id:
            raise DhanAuthError(
                "DHAN credentials not found. Set DHAN_ACCESS_TOKEN and DHAN_CLIENT_ID "
                "in .env file or as environment variables."
            )

        return cls(access_token=access_token, client_id=client_id, base_url=base_url)


class DhanClient(BrokerClient):
    """
    DHAN API client implementing the BrokerClient interface.
    
    NOTE: This is a template implementation. You need to update the API endpoints,
    request/response formats, and authentication method based on actual DHAN API documentation.
    """

    def __init__(self, config: Optional[DhanConfig] = None, access_token: Optional[str] = None, client_id: Optional[str] = None):
        """
        Initialize DHAN client.
        
        Args:
            config: Optional DhanConfig object
            access_token: Optional access token (overrides config)
            client_id: Optional client ID (overrides config)
        """
        if config:
            self.config = config
        elif access_token and client_id:
            # Direct initialization with token and client ID
            self.config = DhanConfig(access_token=access_token, client_id=client_id)
        else:
            # Try to load from environment
            self.config = DhanConfig.from_env()
        
        self._session = requests.Session()
        
        # DHAN API v2 uses 'access-token' header (not Bearer)
        # Access token is a JWT that's valid for 24 hours
        self._session.headers.update(
            {
                "access-token": self.config.access_token,  # DHAN uses 'access-token' header
                "Content-Type": "application/json",
                "dhanClientId": self.config.client_id,  # DHAN uses 'dhanClientId' header
            }
        )
        
        # Store token expiry info (extract from JWT if needed)
        self._token_expiry = None
        self._validate_token()

    def _validate_token(self):
        """Validate access token and extract expiry if possible."""
        try:
            import jwt
            # Decode JWT to check expiry (without verification - just to read)
            decoded = jwt.decode(self.config.access_token, options={"verify_signature": False})
            self._token_expiry = decoded.get("exp")
            if self._token_expiry:
                import time
                if time.time() >= self._token_expiry:
                    print("[WARNING] DHAN access token appears to be expired. You may need to refresh it.")
        except Exception:
            # If JWT decode fails, continue anyway - token might still be valid
            pass

    def refresh_token(self) -> str:
        """
        Refresh the access token using DHAN RenewToken endpoint.
        
        Returns:
            New access token (valid for 24 hours)
            
        Raises:
            DhanAuthError: If refresh fails (token expired or invalid)
            RuntimeError: If API call fails
        """
        # DHAN API v2 RenewToken endpoint: GET /v2/RenewToken
        # Requires: access-token header (current token) and dhanClientId header
        try:
            response = self._request("GET", "/v2/RenewToken")
            
            # Extract new token from response
            new_token = response.get("accessToken") or response.get("access_token") or response.get("token")
            if not new_token:
                raise RuntimeError("RenewToken response did not contain access token")
            
            # Update client configuration with new token
            self.config.access_token = new_token
            self._session.headers["access-token"] = new_token
            
            # Validate new token
            self._validate_token()
            
            return new_token
        except DhanAuthError:
            raise DhanAuthError(
                "Failed to refresh token. Your current token may be expired. "
                "Please generate a new token from web.dhan.co"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to refresh DHAN token: {e}")

    def _reload_credentials_from_env(self) -> bool:
        """
        Reload credentials from .env file (user preference).
        Returns True if credentials were reloaded, False otherwise.
        """
        try:
            # Read from .env file directly
            if os.path.exists(".env"):
                access_token = None
                client_id = None
                with open(".env", "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key == "DHAN_ACCESS_TOKEN":
                            access_token = val
                        elif key == "DHAN_CLIENT_ID":
                            client_id = val
                
                if access_token and client_id:
                    # Update config and session headers
                    self.config.access_token = access_token
                    self.config.client_id = client_id
                    self._session.headers["access-token"] = access_token
                    self._session.headers["dhanClientId"] = client_id
                    self._validate_token()
                    return True
        except Exception:
            pass
        return False

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Low-level HTTP request helper."""
        url = f"{self.config.base_url}{path}"
        resp = self._session.request(method, url, params=params, json=json_body, timeout=15)
        
        if resp.status_code == 401:
            # Try to reload credentials from .env file (user may have updated it)
            if self._reload_credentials_from_env():
                # Retry once with new credentials
                resp = self._session.request(method, url, params=params, json=json_body, timeout=15)
                if resp.status_code == 401:
                    raise DhanAuthError(
                        "DHAN authentication failed (401) even after reloading from .env. "
                        "Your access token may be expired. Access tokens are valid for 24 hours - "
                        "please update .env file with a fresh token."
                    )
            else:
                raise DhanAuthError(
                    "DHAN authentication failed (401). Your access token may be expired. "
                    "Access tokens are valid for 24 hours - please update .env file with a fresh token."
                )
        
        try:
            data = resp.json()
        except ValueError:
            resp.raise_for_status()
            return resp.text
        
        if not resp.ok:
            msg = data.get("message") if isinstance(data, dict) else str(data)
            error_msg = data.get("error", "") if isinstance(data, dict) else ""
            full_msg = f"{msg} {error_msg}".strip() if error_msg else msg
            raise RuntimeError(f"DHAN API error {resp.status_code}: {full_msg}")
        
        return data

    # ------------------------------------------------------------------
    # BrokerClient interface implementation
    # ------------------------------------------------------------------
    
    @property
    def broker_name(self) -> str:
        return "dhan"

    def get_account(self) -> Dict[str, Any]:
        """Return account details."""
        # DHAN API v2 endpoint for fund details: GET /v2/fundlimit
        response = self._request("GET", "/v2/fundlimit")
        
        # Handle DHAN response format
        if isinstance(response, list) and len(response) > 0:
            account = response[0]  # Get first account
        elif isinstance(response, dict):
            account = response
        else:
            account = {}
        
        # Map DHAN account fields to standard format
        equity = float(account.get("equity", account.get("net", 0)) or 0)
        cash = float(account.get("cash", account.get("availableCash", 0)) or 0)
        buying_power = float(account.get("buyingPower", account.get("availableMargin", equity)) or equity)
        
        return {
            "equity": equity,
            "buying_power": buying_power,
            "cash": cash,
            "portfolio_value": equity,
            "margin_used": float(account.get("marginUsed", 0) or 0),
            "margin_available": float(account.get("marginAvailable", buying_power) or buying_power),
        }

    def list_positions(self) -> list[Dict[str, Any]]:
        """Return all open positions."""
        # DHAN API v2 endpoint for positions: GET /v2/positions
        response = self._request("GET", "/v2/positions")
        
        # Handle DHAN response format
        positions = response.get("data", []) if isinstance(response, dict) else response
        if not isinstance(positions, list):
            positions = []
        
        # Normalize to common format, PRESERVING exchange segment for filtering
        normalized = []
        for pos in positions:
            qty = float(pos.get("quantity", pos.get("qty", 0)) or 0)
            avg_price = float(pos.get("averagePrice", pos.get("avg_entry_price", pos.get("entryPrice", 0))) or 0)
            ltp = float(pos.get("ltp", pos.get("lastPrice", 0)) or 0)
            market_value = abs(qty * ltp) if ltp > 0 else abs(qty * avg_price)
            
            # PRESERVE exchange segment information (critical for filtering MCX vs other exchanges)
            exchange_segment = pos.get("exchangeSegment", pos.get("exchange", pos.get("segment", ""))).upper()
            
            normalized.append({
                "symbol": pos.get("tradingSymbol", pos.get("symbol", "")),
                "qty": qty,  # Positive for long, negative for short
                "market_value": market_value,
                "avg_entry_price": avg_price,
                "ltp": ltp,
                "unrealized_pl": float(pos.get("unrealizedPL", 0) or 0),
                "exchange_segment": exchange_segment,  # CRITICAL: Preserve exchange info (MCX, NSE, BSE, etc.)
                "_raw_exchange": exchange_segment,  # Backup field for filtering
            })
        return normalized

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return position for a symbol, or None."""
        # Get all positions and find the one matching the symbol
        all_positions = self.list_positions()
        for pos in all_positions:
            if pos["symbol"].upper() == symbol.upper():
                return pos
        return None

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
        Submit an order to DHAN MCX with comprehensive validation.
        
        For MCX commodities, qty is required (not notional).
        Lot sizes are fixed - qty must be in multiples of lot size.
        
        Args:
            symbol: MCX trading symbol
            qty: Quantity (must be positive integer, multiple of lot size)
            notional: Not supported for MCX (raises error)
            side: "buy" or "sell"
            order_type: "market" or "limit"
            time_in_force: "gtc" (good till cancelled) or "day"
            take_profit_limit_price: Optional take-profit price
            stop_loss_price: Optional stop-loss price
            client_order_id: Optional client order ID
            
        Returns:
            Order response dict with order_id, status, etc.
            
        Raises:
            ValueError: If validation fails
            RuntimeError: If API call fails
        """
        # Validation: qty is required for MCX
        if qty is None and notional is None:
            raise ValueError("For MCX commodities, qty must be provided (not notional).")
        
        if notional is not None:
            # For MCX, we need qty, not notional. This is a limitation.
            raise ValueError("MCX commodities require qty (lot-based), not notional amount.")

        # Validation: side must be valid
        side = side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError(f"side must be 'buy' or 'sell', got: {side}")

        # Validation: qty must be positive integer
        qty_int = int(qty) if qty is not None else 0
        if qty_int <= 0:
            raise ValueError(f"Quantity must be positive integer (lot-based), got: {qty}")
        
        # Validation: order_type must be valid
        order_type_lower = order_type.lower()
        if order_type_lower not in {"market", "limit"}:
            raise ValueError(f"order_type must be 'market' or 'limit', got: {order_type}")
        
        # Validation: stop-loss and take-profit prices must be positive if provided
        if stop_loss_price is not None and stop_loss_price <= 0:
            raise ValueError(f"stop_loss_price must be positive, got: {stop_loss_price}")
        if take_profit_limit_price is not None and take_profit_limit_price <= 0:
            raise ValueError(f"take_profit_limit_price must be positive, got: {take_profit_limit_price}")
        
        # Validation: Check account balance (basic check)
        try:
            account = self.get_account()
            buying_power = float(account.get("buying_power", 0) or 0)
            if buying_power <= 0:
                raise RuntimeError(f"Insufficient buying power: {buying_power}")
        except Exception as e:
            # If account check fails, log warning but continue (might be API issue)
            print(f"[WARNING] Could not validate account balance: {e}")

        # DHAN API order format
        # For MCX, we need: tradingSymbol, exchangeSegment, transactionType, quantity, orderType, productType, priceType
        body: Dict[str, Any] = {
            "tradingSymbol": symbol.upper(),
            "exchangeSegment": MCX_EXCHANGE_CODE,  # MCX for commodities
            "transactionType": "BUY" if side == "buy" else "SELL",
            "quantity": qty_int,  # MCX requires integer quantities (lot-based)
            "orderType": "MARKET" if order_type_lower == "market" else "LIMIT",
            "productType": "INTRADAY",  # Can be INTRADAY, MARGIN, or DELIVERY
            "priceType": "MARKET" if order_type_lower == "market" else "LIMIT",
        }
        
        # Add limit price if order type is limit
        if order_type.lower() == "limit" and notional:
            # If limit order, price might be needed - but for market orders, omit
            pass  # Price will be set separately if needed
        
        if client_order_id:
            body["clientId"] = client_order_id

        # Handle stop-loss and take-profit (DHAN supports SL and SL-M orders)
        if stop_loss_price:
            # For MCX, stop-loss is typically a separate order type
            body["stopLoss"] = float(stop_loss_price)
            body["orderType"] = "SL"  # Stop-loss order type
        
        if take_profit_limit_price:
            # Take-profit as limit order
            body["price"] = float(take_profit_limit_price)
            if order_type.lower() == "market":
                body["orderType"] = "LIMIT"

        # DHAN API v2 endpoint for order placement: POST /v2/orders
        try:
            response = self._request("POST", "/v2/orders", json_body=body)
        except RuntimeError as e:
            # Provide more helpful error messages for common failures
            error_msg = str(e)
            if "insufficient" in error_msg.lower() or "margin" in error_msg.lower():
                raise RuntimeError(
                    f"Insufficient margin/buying power for order. "
                    f"Symbol: {symbol}, Quantity: {qty_int}, Side: {side}. "
                    f"Check your account balance and margin requirements."
                ) from e
            elif "invalid" in error_msg.lower() or "symbol" in error_msg.lower():
                raise RuntimeError(
                    f"Invalid symbol or order parameters. "
                    f"Symbol: {symbol}, Quantity: {qty_int}. "
                    f"Ensure symbol is valid MCX contract and quantity is multiple of lot size."
                ) from e
            elif "market closed" in error_msg.lower() or "trading" in error_msg.lower():
                raise RuntimeError(
                    f"Market is closed or trading is not allowed. "
                    f"MCX trading hours: Typically 9:00 AM - 11:30 PM IST (Monday-Friday)."
                ) from e
            else:
                # Re-raise with original message
                raise
        
        # Validate response
        order_id = response.get("orderId", response.get("id", ""))
        if not order_id:
            raise RuntimeError(f"DHAN API did not return order ID. Response: {response}")
        
        # Return standardized response
        return {
            "id": order_id,
            "status": response.get("status", "PENDING"),
            "symbol": symbol.upper(),
            "side": side,
            "qty": qty_int,
            "order_type": order_type,
        }

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        # DHAN API v2 endpoint for order cancellation: DELETE /v2/orders/{order_id}
        response = self._request("DELETE", f"/v2/orders/{order_id}")
        return {
            "id": order_id,
            "status": "CANCELLED",
            "message": response.get("message", "Order cancelled"),
        }

    def submit_stop_order(
        self,
        *,
        symbol: str,
        qty: float,
        stop_price: float,
        side: str = "sell",  # "sell" for long positions, "buy" for short positions
        time_in_force: str = "gtc",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a standalone stop-loss order for MCX commodity positions.
        
        This creates a broker-level stop-loss order that will execute even if the monitoring
        script is not running. Critical for real money trading protection.
        
        DHAN supports SL (Stop-Loss) and SL-M (Stop-Loss Market) order types.
        For MCX, we use SL order type which triggers a limit order when stop price is hit.
        
        Args:
            symbol: MCX trading symbol (e.g., "GOLDFEB25FUT")
            qty: Quantity (must be multiple of lot size)
            stop_price: Price at which to trigger the stop order
            side: "sell" for long positions (stop-loss), "buy" for short positions (stop-loss)
            time_in_force: Order time in force (default: "gtc" - good till cancelled)
            client_order_id: Optional client order ID for tracking
            
        Returns:
            Order response dict with order_id, status, etc.
        """
        # Validate quantity is integer (MCX requires lot-based quantities)
        qty_int = int(qty)
        if qty_int <= 0:
            raise ValueError(f"Quantity must be positive integer (lot-based), got: {qty}")
        
        side = side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        
        # DHAN API v2 order format for stop-loss
        # SL order: triggers when price hits stop_price, then executes as limit order
        body: Dict[str, Any] = {
            "tradingSymbol": symbol.upper(),
            "exchangeSegment": MCX_EXCHANGE_CODE,
            "transactionType": "BUY" if side == "buy" else "SELL",
            "quantity": qty_int,
            "orderType": "SL",  # Stop-Loss order type
            "productType": "INTRADAY",  # Can be INTRADAY, MARGIN, or DELIVERY
            "priceType": "SL",  # Stop-Loss price type
            "stopLoss": float(stop_price),  # Stop price trigger
        }
        
        if client_order_id:
            body["clientId"] = client_order_id
        
        # DHAN API v2 endpoint for order placement: POST /v2/orders
        response = self._request("POST", "/v2/orders", json_body=body)
        
        # Return standardized response
        return {
            "id": response.get("orderId", response.get("id", "")),
            "status": response.get("status", "PENDING"),
            "symbol": symbol.upper(),
            "side": side,
            "qty": qty_int,
            "stop_price": stop_price,
            "order_type": "stop",
        }
    
    def submit_take_profit_order(
        self,
        *,
        symbol: str,
        qty: float,
        limit_price: float,
        side: str = "sell",  # "sell" for long positions, "buy" for short positions
        time_in_force: str = "gtc",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a standalone take-profit limit order for MCX commodity positions.
        
        This creates a broker-level take-profit order that will execute even if the monitoring
        script is not running. Ensures profits are locked in automatically.
        
        Args:
            symbol: MCX trading symbol (e.g., "GOLDFEB25FUT")
            qty: Quantity (must be multiple of lot size)
            limit_price: Price at which to execute the limit order
            side: "sell" for long positions (take-profit), "buy" for short positions (take-profit)
            time_in_force: Order time in force (default: "gtc" - good till cancelled)
            client_order_id: Optional client order ID for tracking
            
        Returns:
            Order response dict with order_id, status, etc.
        """
        # Validate quantity is integer (MCX requires lot-based quantities)
        qty_int = int(qty)
        if qty_int <= 0:
            raise ValueError(f"Quantity must be positive integer (lot-based), got: {qty}")
        
        side = side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        
        # DHAN API v2 order format for take-profit (limit order)
        body: Dict[str, Any] = {
            "tradingSymbol": symbol.upper(),
            "exchangeSegment": MCX_EXCHANGE_CODE,
            "transactionType": "BUY" if side == "buy" else "SELL",
            "quantity": qty_int,
            "orderType": "LIMIT",  # Limit order for take-profit
            "productType": "INTRADAY",  # Can be INTRADAY, MARGIN, or DELIVERY
            "priceType": "LIMIT",  # Limit price type
            "price": float(limit_price),  # Limit price for take-profit
        }
        
        if client_order_id:
            body["clientId"] = client_order_id
        
        # DHAN API v2 endpoint for order placement: POST /v2/orders
        response = self._request("POST", "/v2/orders", json_body=body)
        
        # Return standardized response
        return {
            "id": response.get("orderId", response.get("id", "")),
            "status": response.get("status", "PENDING"),
            "symbol": symbol.upper(),
            "side": side,
            "qty": qty_int,
            "limit_price": limit_price,
            "order_type": "limit",
        }

    def get_last_trade(
        self,
        symbol: str,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        force_retry: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Get last trade price for MCX symbol with improved error handling.
        
        This is critical for real money trading - accurate prices are essential.
        """
        # Increase retries if force_retry is True (for active positions)
        if force_retry:
            max_retries = max(max_retries * 2, 10)
            retry_delay = max(retry_delay * 1.5, 2.0)
        
        for attempt in range(max_retries):
            try:
                # DHAN API v2 endpoint for market data: POST /v2/marketfeed/ltp
                # Request body format: {"securityId": "SYMBOL", "exchangeSegment": "MCX"}
                body = {
                    "securityId": symbol.upper(),
                    "exchangeSegment": MCX_EXCHANGE_CODE,
                }
                response = self._request("POST", "/v2/marketfeed/ltp", json_body=body)
                
                # Handle DHAN API v2 response format
                # Response format: {"data": [{"ltp": price, ...}]} or {"ltp": price, ...}
                if isinstance(response, dict):
                    # Check if response has "data" array
                    if "data" in response and isinstance(response["data"], list) and len(response["data"]) > 0:
                        data = response["data"][0]  # Get first item from data array
                    else:
                        data = response  # Response is the data itself
                else:
                    data = response
                
                # Extract LTP (Last Traded Price) from response
                price = data.get("ltp", data.get("lastPrice", data.get("price", 0)))
                if price and float(price) > 0:
                    price_float = float(price)
                    # Sanity check: price should be reasonable (not zero, not negative, not extremely large)
                    if price_float > 0 and price_float < 1e10:  # Reasonable upper bound
                        return {"price": price_float}
                    else:
                        # Invalid price - log and retry
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        return None
                else:
                    # No price found - retry
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    # Log error on final attempt
                    print(f"[WARNING] Failed to fetch price for {symbol} after {max_retries} attempts: {e}")
                    return None
        return None
