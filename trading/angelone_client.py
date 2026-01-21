"""
Angel One API client for live MCX commodity futures trading.

This module provides an Angel One API client that implements the BrokerClient interface,
allowing seamless integration with the ExecutionEngine for MCX commodity trading.

Authentication:
- Uses API Key + Client ID + TOTP (Time-Based One-Time Password)
- Access token valid for session duration
- Requires trading password/MPIN

Environment variables expected:
- ANGEL_ONE_API_KEY : your Angel One API key
- ANGEL_ONE_CLIENT_ID : your Angel One client ID
- ANGEL_ONE_PASSWORD : your trading password/MPIN
- ANGEL_ONE_TOTP_SECRET : TOTP secret (optional, if using pyotp library)
- ANGEL_ONE_BASE_URL : optional override for the base URL

IMPORTANT - STATIC IP REQUIREMENT:
- NO IP whitelisting is required from Angel One side
- You MUST create your API key with a STATIC IP from your internet provider
- Contact your ISP to get a static IP address if you don't have one
- When creating the API key in SmartAPI, ensure it's configured with your static IP
- The static IP must match the IP from which you're making API calls
- If your IP changes, API calls will fail - ensure your internet connection uses static IP

IMPORTANT:
- TOTP must be enabled in your Angel One account
- MCX segment must be activated
- Test thoroughly in paper trading mode before live trading
- Real money is involved - use with caution
"""

from __future__ import annotations

import os
import time
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, List

import requests

from .broker_interface import BrokerClient

# Angel One API endpoints (check official documentation for exact URLs)
DEFAULT_ANGEL_ONE_BASE_URL = "https://apiconnect.angelbroking.com"  # Verify with Angel One docs
ANGEL_ONE_PAPER_URL = "https://apiconnect.angelbroking.com"  # Check if paper trading available

# MCX Exchange code (for commodity futures)
MCX_EXCHANGE_CODE = "MCX"  # Multi Commodity Exchange

# Angel One API Endpoints (verify with official documentation):
# - POST /rest/auth/angel/user/login - Login with TOTP
# - GET /rest/secure/user/getRMS - Account/fund details
# - GET /rest/secure/user/getPositions - List all positions
# - POST /rest/secure/order/placeOrder - Place order
# - POST /rest/secure/order/modifyOrder - Modify order
# - POST /rest/secure/order/cancelOrder - Cancel order
# - GET /rest/secure/marketData/quote - Market data (LTP, OHLC, etc.)


class AngelOneAuthError(RuntimeError):
    """Raised when Angel One credentials are missing or invalid."""


@dataclass
class AngelOneConfig:
    """Configuration wrapper for Angel One API settings."""

    api_key: str
    client_id: str
    password: str  # Trading password or MPIN
    totp_secret: Optional[str] = None  # Optional if generating TOTP externally
    base_url: str = DEFAULT_ANGEL_ONE_BASE_URL

    @classmethod
    def from_env(cls) -> "AngelOneConfig":
        """
        Load Angel One config from .env file FIRST, then fallback to environment variables.
        """
        api_key = None
        client_id = None
        password = None
        totp_secret = None
        base_url = DEFAULT_ANGEL_ONE_BASE_URL

        # PRIORITY 1: Read from .env file FIRST
        if os.path.exists(".env"):
            try:
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
                        if key == "ANGEL_ONE_API_KEY":
                            api_key = val
                        elif key == "ANGEL_ONE_CLIENT_ID":
                            client_id = val
                        elif key == "ANGEL_ONE_PASSWORD":
                            password = val
                        elif key == "ANGEL_ONE_TOTP_SECRET":
                            totp_secret = val
                        elif key == "ANGEL_ONE_BASE_URL":
                            base_url = val
            except Exception:
                pass

        # PRIORITY 2: Fallback to environment variables
        if not api_key:
            api_key = os.getenv("ANGEL_ONE_API_KEY")
        if not client_id:
            client_id = os.getenv("ANGEL_ONE_CLIENT_ID")
        if not password:
            password = os.getenv("ANGEL_ONE_PASSWORD")
        if not totp_secret:
            totp_secret = os.getenv("ANGEL_ONE_TOTP_SECRET")
        if base_url == DEFAULT_ANGEL_ONE_BASE_URL:
            base_url = os.getenv("ANGEL_ONE_BASE_URL", DEFAULT_ANGEL_ONE_BASE_URL)

        if not api_key or not client_id or not password:
            raise AngelOneAuthError(
                "Angel One credentials not found. Set ANGEL_ONE_API_KEY, ANGEL_ONE_CLIENT_ID, "
                "and ANGEL_ONE_PASSWORD in .env file or as environment variables."
            )

        return cls(
            api_key=api_key,
            client_id=client_id,
            password=password,
            totp_secret=totp_secret,
            base_url=base_url,
        )


class AngelOneClient(BrokerClient):
    """
    Angel One API client implementing the BrokerClient interface.
    
    NOTE: This is a template implementation. You need to update the API endpoints,
    request/response formats, and authentication method based on actual Angel One API documentation.
    """

    def __init__(
        self,
        config: Optional[AngelOneConfig] = None,
        api_key: Optional[str] = None,
        client_id: Optional[str] = None,
        password: Optional[str] = None,
        totp_secret: Optional[str] = None,
    ):
        """
        Initialize Angel One client.
        
        Args:
            config: Optional AngelOneConfig object
            api_key: Optional API key (overrides config)
            client_id: Optional client ID (overrides config)
            password: Optional trading password (overrides config)
            totp_secret: Optional TOTP secret (overrides config)
        """
        if config:
            self.config = config
        elif api_key and client_id and password:
            self.config = AngelOneConfig(
                api_key=api_key,
                client_id=client_id,
                password=password,
                totp_secret=totp_secret,
            )
        else:
            self.config = AngelOneConfig.from_env()

        self._session = requests.Session()
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[float] = None

        # Authenticate and get access token
        self._authenticate()

    def _generate_totp(self) -> str:
        """
        Generate TOTP using secret.
        
        Returns:
            6-digit TOTP string
        """
        if self.config.totp_secret:
            try:
                import pyotp
                # Clean TOTP secret (remove spaces, quotes, comments, convert to uppercase)
                clean_secret = self.config.totp_secret.strip().replace(" ", "").replace('"', '').replace("'", "").split("#")[0].strip().upper()
                # Remove any non-base32 characters (keep only A-Z, 2-7)
                import re
                clean_secret = re.sub(r'[^A-Z2-7]', '', clean_secret)
                if not clean_secret:
                    raise RuntimeError("TOTP secret is empty after cleaning. Check format.")
                totp = pyotp.TOTP(clean_secret)
                generated_totp = totp.now()
                return generated_totp
            except ImportError:
                raise RuntimeError(
                    "pyotp library required for TOTP generation. Install with: pip install pyotp"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to generate TOTP. Check your TOTP secret format. Error: {e}"
                )
        else:
            raise RuntimeError(
                "TOTP secret not provided. Either set ANGEL_ONE_TOTP_SECRET or generate TOTP manually."
            )

    def _authenticate(self) -> str:
        """
        Authenticate with Angel One API and get access token.
        
        Returns:
            Access token (JWT)
            
        Raises:
            AngelOneAuthError: If authentication fails
        """
        # Generate TOTP
        totp = self._generate_totp()

        # Angel One SmartAPI login endpoint (official format)
        login_url = f"{self.config.base_url}/rest/auth/angelbroking/user/v1/loginByPassword"
        
        # Ensure clientcode is uppercase (Angel One requirement)
        clientcode = str(self.config.client_id).strip().upper()
        
        login_body = {
            "clientcode": clientcode,
            "password": str(self.config.password).strip(),
            "totp": str(totp).strip(),
        }

        # Get local and public IP addresses (required by Angel One)
        try:
            # Get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "192.168.1.1"  # Fallback
        
        # Get public IP (simplified - may need actual public IP)
        public_ip = local_ip  # For now, use local IP (you may need to fetch actual public IP)
        
        # Get MAC address (required by Angel One)
        try:
            mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0,2*6,2)][::-1])
        except Exception:
            mac = "00:00:00:00:00:00"  # Fallback

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-UserType": "USER",
            "X-SourceID": "WEB",
            "X-ClientLocalIP": local_ip,
            "X-ClientPublicIP": public_ip,
            "X-MACAddress": mac,
            "X-PrivateKey": self.config.api_key,
        }

        try:
            response = self._session.post(login_url, json=login_body, headers=headers, timeout=15)
            
            # Get response data for error handling
            try:
                data = response.json()
            except ValueError:
                data = {"message": response.text}
            
            # Check for errors first
            if not response.ok:
                error_msg = data.get("message", data.get("error", f"HTTP {response.status_code}"))
                error_desc = data.get("errorcode", "")
                full_error = f"{error_msg}" + (f" (Error Code: {error_desc})" if error_desc else "")
                raise AngelOneAuthError(f"Angel One authentication failed: {full_error}")
            
            # Extract access token (verify response format with Angel One docs)
            if data.get("status") and data.get("data"):
                self._access_token = data["data"].get("jwtToken") or data["data"].get("token")
                if not self._access_token:
                    raise AngelOneAuthError("Login response did not contain access token")

                # Store refresh token if available
                self._refresh_token = data["data"].get("refreshToken")
                
                # Update session headers with token
                self._session.headers.update(
                    {
                        "Authorization": f"Bearer {self._access_token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "X-UserType": "USER",
                        "X-SourceID": "WEB",
                        "X-ClientLocalIP": local_ip,
                        "X-ClientPublicIP": public_ip,
                        "X-MACAddress": mac,
                        "X-PrivateKey": self.config.api_key,
                    }
                )

                # Extract token expiry if available
                expires_in = data["data"].get("expiresIn", 3600)  # Default 1 hour
                self._token_expiry = time.time() + expires_in

                return self._access_token
            else:
                error_msg = data.get("message", "Authentication failed")
                error_desc = data.get("errorcode", "")
                full_error = f"{error_msg}" + (f" (Error Code: {error_desc})" if error_desc else "")
                raise AngelOneAuthError(f"Angel One authentication failed: {full_error}")

        except AngelOneAuthError:
            raise  # Re-raise our custom errors
        except requests.exceptions.RequestException as e:
            # Try to extract error message from response if available
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("message", error_data.get("error", str(e)))
                except ValueError:
                    error_msg = e.response.text or str(e)
            else:
                error_msg = str(e)
            raise AngelOneAuthError(f"Failed to authenticate with Angel One: {error_msg}")

    def _reload_credentials_from_env(self) -> bool:
        """
        Reload credentials from .env file and re-authenticate.
        Returns True if credentials were reloaded, False otherwise.
        """
        try:
            if os.path.exists(".env"):
                api_key = None
                client_id = None
                password = None
                totp_secret = None

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
                        if key == "ANGEL_ONE_API_KEY":
                            api_key = val
                        elif key == "ANGEL_ONE_CLIENT_ID":
                            client_id = val
                        elif key == "ANGEL_ONE_PASSWORD":
                            password = val
                        elif key == "ANGEL_ONE_TOTP_SECRET":
                            totp_secret = val

                if api_key and client_id and password:
                    self.config.api_key = api_key
                    self.config.client_id = client_id
                    self.config.password = password
                    if totp_secret:
                        self.config.totp_secret = totp_secret
                    # Re-authenticate
                    self._authenticate()
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

        # Check if token expired and re-authenticate if needed
        if self._token_expiry and time.time() >= self._token_expiry:
            self._authenticate()

        # Check if response is HTML (error page) before parsing JSON
        resp = self._session.request(method, url, params=params, json=json_body, timeout=15)
        
        # Check for HTML error pages (API gateway rejection)
        if resp.headers.get("content-type", "").startswith("text/html"):
            html_content = resp.text
            if "Request Rejected" in html_content or "<html>" in html_content.lower():
                # Extract support ID if available
                import re
                support_id_match = re.search(r"support ID is: (\d+)", html_content)
                support_id = support_id_match.group(1) if support_id_match else "N/A"
                raise RuntimeError(
                    f"Angel One API gateway rejected request (Support ID: {support_id}). "
                    f"This usually means: (1) API key was not created with static IP from your ISP, "
                    f"(2) Wrong endpoint format, or (3) Symbol token required instead of symbol name. "
                    f"IMPORTANT: NO IP whitelisting is required from Angel One side. "
                    f"You MUST create your API key with a STATIC IP from your internet provider. "
                    f"Contact your ISP to get a static IP if you don't have one."
                )

        if resp.status_code == 401:
            # Try to reload credentials and re-authenticate
            if self._reload_credentials_from_env():
                resp = self._session.request(method, url, params=params, json=json_body, timeout=15)
                if resp.status_code == 401:
                    raise AngelOneAuthError(
                        "Angel One authentication failed (401) even after reloading from .env. "
                        "Your access token may be expired. Please check your credentials and TOTP."
                    )
            else:
                raise AngelOneAuthError(
                    "Angel One authentication failed (401). Your access token may be expired. "
                    "Please update .env file with fresh credentials."
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
            raise RuntimeError(f"Angel One API error {resp.status_code}: {full_msg}")

        return data

    @property
    def broker_name(self) -> str:
        return "angelone"

    def get_account(self) -> Dict[str, Any]:
        """Return account details."""
        # Angel One SmartAPI endpoint for RMS/account details
        response = self._request("GET", "/rest/secure/user/getRMS")

        # Handle Angel One response format (verify structure)
        if isinstance(response, dict):
            account = response.get("data", response)
        else:
            account = {}

        # Map Angel One account fields to standard format
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
        # Angel One SmartAPI endpoint for positions
        response = self._request("GET", "/rest/secure/user/getPositions")

        # Handle Angel One response format
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
        Submit an order to Angel One MCX with comprehensive validation.
        
        For MCX commodities, qty is required (not notional).
        """
        # Validation: qty is required for MCX
        if qty is None and notional is None:
            raise ValueError("For MCX commodities, qty must be provided (not notional).")

        if notional is not None:
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

        # Angel One order format (verify with documentation)
        body: Dict[str, Any] = {
            "variety": "NORMAL",  # NORMAL, STOPLOSS, AMO, etc.
            "tradingsymbol": symbol.upper(),
            "symboltoken": "",  # May need to fetch from symbol master
            "transactiontype": "BUY" if side == "buy" else "SELL",
            "exchange": MCX_EXCHANGE_CODE,  # MCX for commodities
            "ordertype": "MARKET" if order_type_lower == "market" else "LIMIT",
            "producttype": "INTRADAY",  # INTRADAY, MARGIN, or DELIVERY
            "duration": "DAY" if time_in_force.lower() == "day" else "IOC",  # Verify with docs
            "price": "0",  # For market orders
            "squareoff": "0",
            "stoploss": "0",
            "quantity": str(qty_int),  # MCX requires integer quantities (lot-based)
        }

        # Add limit price if order type is limit
        if order_type_lower == "limit" and take_profit_limit_price:
            body["price"] = str(take_profit_limit_price)

        # Add stop-loss if provided
        if stop_loss_price:
            body["stoploss"] = str(stop_loss_price)
            body["variety"] = "STOPLOSS"  # May need bracket order for stop-loss

        # Add client order ID if provided
        if client_order_id:
            body["uniqueorderid"] = client_order_id

        # Angel One SmartAPI order placement endpoint
        response = self._request("POST", "/rest/secure/order/placeOrder", json_body=body)

        # Handle response (verify format with docs)
        if isinstance(response, dict):
            order_data = response.get("data", response)
        else:
            order_data = response

        return {
            "id": order_data.get("orderid") or order_data.get("orderId"),
            "status": order_data.get("status") or "pending",
            "symbol": symbol.upper(),
            "qty": qty_int,
            "side": side.upper(),
            "order_type": order_type_lower,
        }

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order by ID."""
        # Angel One SmartAPI order cancellation endpoint
        body = {
            "variety": "NORMAL",
            "orderid": order_id,
        }

        response = self._request("POST", "/rest/secure/order/cancelOrder", json_body=body)

        if isinstance(response, dict):
            cancel_data = response.get("data", response)
        else:
            cancel_data = response

        return {
            "order_id": order_id,
            "status": cancel_data.get("status") or "cancelled",
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
        
        This creates a broker-level stop-loss order that will execute even if the system is down.
        Critical for real money trading.
        
        Args:
            symbol: Trading symbol (e.g., "GOLDDEC24")
            qty: Quantity to sell/buy when stop is triggered (must be integer for MCX)
            stop_price: Price at which to trigger the stop order
            side: "sell" for long positions (stop-loss), "buy" for short positions (stop-loss)
            time_in_force: Order time in force (default: "gtc" - good till cancelled)
            client_order_id: Optional client order ID for tracking
        """
        # MCX requires integer quantities (lot-based)
        qty_int = int(qty)
        if qty_int <= 0:
            raise ValueError(f"Quantity must be positive integer for MCX, got: {qty}")
        
        side = side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError(f"side must be 'buy' or 'sell', got: {side}")
        
        # Angel One stop-loss order format
        body: Dict[str, Any] = {
            "variety": "STOPLOSS",
            "tradingsymbol": symbol.upper(),
            "symboltoken": "",  # May need to fetch from symbol master
            "transactiontype": "SELL" if side == "sell" else "BUY",
            "exchange": MCX_EXCHANGE_CODE,
            "ordertype": "STOPLOSS_MARKET",  # Stop-loss market order
            "producttype": "INTRADAY",  # INTRADAY, MARGIN, or DELIVERY
            "duration": "DAY" if time_in_force.lower() == "day" else "IOC",
            "price": "0",  # Market order after stop is triggered
            "triggerprice": str(stop_price),  # Stop trigger price
            "quantity": str(qty_int),
        }
        
        if client_order_id:
            body["uniqueorderid"] = client_order_id
        
        response = self._request("POST", "/rest/secure/order/placeOrder", json_body=body)
        
        if isinstance(response, dict):
            order_data = response.get("data", response)
        else:
            order_data = response
        
        return {
            "id": order_data.get("orderid") or order_data.get("orderId"),
            "status": order_data.get("status") or "pending",
            "symbol": symbol.upper(),
            "qty": qty_int,
            "side": side.upper(),
            "stop_price": stop_price,
            "order_type": "stop_loss",
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
        
        This creates a broker-level take-profit order that will execute at the limit price.
        
        Args:
            symbol: Trading symbol (e.g., "GOLDDEC24")
            qty: Quantity to sell/buy (must be integer for MCX)
            limit_price: Limit price at which to execute the order
            side: "sell" for long positions, "buy" for short positions
            time_in_force: Order time in force (default: "gtc" - good till cancelled)
            client_order_id: Optional client order ID for tracking
        """
        # MCX requires integer quantities (lot-based)
        qty_int = int(qty)
        if qty_int <= 0:
            raise ValueError(f"Quantity must be positive integer for MCX, got: {qty}")
        
        side = side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError(f"side must be 'buy' or 'sell', got: {side}")
        
        # Angel One limit order format
        body: Dict[str, Any] = {
            "variety": "NORMAL",
            "tradingsymbol": symbol.upper(),
            "symboltoken": "",  # May need to fetch from symbol master
            "transactiontype": "SELL" if side == "sell" else "BUY",
            "exchange": MCX_EXCHANGE_CODE,
            "ordertype": "LIMIT",
            "producttype": "INTRADAY",  # INTRADAY, MARGIN, or DELIVERY
            "duration": "DAY" if time_in_force.lower() == "day" else "IOC",
            "price": str(limit_price),  # Limit price
            "quantity": str(qty_int),
        }
        
        if client_order_id:
            body["uniqueorderid"] = client_order_id
        
        response = self._request("POST", "/rest/secure/order/placeOrder", json_body=body)
        
        if isinstance(response, dict):
            order_data = response.get("data", response)
        else:
            order_data = response
        
        return {
            "id": order_data.get("orderid") or order_data.get("orderId"),
            "status": order_data.get("status") or "pending",
            "symbol": symbol.upper(),
            "qty": qty_int,
            "side": side.upper(),
            "limit_price": limit_price,
            "order_type": "take_profit",
        }

    def _load_scrip_master(self) -> Optional[list]:
        """
        Load Angel One scrip master JSON file (cached).
        
        The scrip master contains all instruments with their tokens.
        URL: https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json
        
        Returns:
            List of instrument dictionaries or None if failed
        """
        import json
        from pathlib import Path
        
        # Cache file location
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "angelone_scrip_master.json"
        
        # Check if cache exists and is recent (refresh daily)
        if cache_file.exists():
            import time
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 86400:  # 24 hours
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass  # Cache corrupted, re-download
        
        # Download fresh scrip master
        try:
            scrip_master_url = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
            response = requests.get(scrip_master_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Save to cache
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            return data
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load Angel One scrip master: {e}. Using API fallback.", UserWarning)
            return None

    def _get_symbol_token(self, symbol: str, exchange: str = MCX_EXCHANGE_CODE) -> Optional[str]:
        """
        Get symbol token (numeric ID) from symbol name using Angel One scrip master.
        
        Angel One SmartAPI requires symbol tokens (numeric IDs) for market data queries.
        This method looks up the token from Angel One's scrip master JSON file.
        
        Args:
            symbol: Symbol name (e.g., "GOLDDEC25")
            exchange: Exchange code (default: MCX)
            
        Returns:
            Symbol token (numeric string) or None if not found
        """
        # Method 1: Use scrip master JSON (most reliable)
        try:
            scrip_master = self._load_scrip_master()
            if scrip_master and isinstance(scrip_master, list):
                # Search for symbol in scrip master
                symbol_upper = symbol.upper()
                # Try multiple format variations (MCX contracts use different formats)
                symbol_variations = [
                    symbol_upper,  # GOLD05FEB26FUT (new format)
                ]
                
                # If symbol has FUT suffix, also try without it
                if symbol_upper.endswith("FUT"):
                    symbol_variations.append(symbol_upper[:-3])  # Remove FUT
                
                # Try variations with spaces (old format compatibility)
                for month in ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]:
                    if month in symbol_upper:
                        symbol_variations.append(symbol_upper.replace(month, f" {month}"))  # GOLD JAN26
                        symbol_variations.append(symbol_upper.replace(month, f"{month} "))  # GOLDJAN 26
                        break
                
                # Try old format (without day and FUT): GOLDJAN26
                import re
                # Match: BASESYMBOL + DAY(2digits) + MONTH(3letters) + YEAR(2digits) + FUT
                old_format_match = re.match(r"^([A-Z]+)(\d{2})([A-Z]{3})(\d{2})FUT$", symbol_upper)
                if old_format_match:
                    base, day, month, year = old_format_match.groups()
                    # Try without day and FUT: GOLDJAN26
                    symbol_variations.append(f"{base}{month}{year}")
                    # Try without FUT: GOLD05FEB26
                    symbol_variations.append(f"{base}{day}{month}{year}")
                
                # Also try reverse: if old format (GOLDJAN26), try new format
                new_format_match = re.match(r"^([A-Z]+)([A-Z]{3})(\d{2})$", symbol_upper)
                if new_format_match and not symbol_upper.endswith("FUT"):
                    base, month, year = new_format_match.groups()
                    # Try with common expiry days and FUT
                    for day in [5, 30, 27, 31, 20, 19, 18]:
                        symbol_variations.append(f"{base}{day:02d}{month}{year}FUT")
                
                for inst in scrip_master:
                    if isinstance(inst, dict):
                        # Check multiple fields for symbol match
                        trading_symbol = inst.get("tradingsymbol", inst.get("symbol", "")).upper()
                        name = inst.get("name", "").upper()
                        exch_seg = inst.get("exch_seg", inst.get("exchange", "")).upper()
                        
                        # Match symbol and exchange - try all variations
                        for symbol_variant in symbol_variations:
                            if (trading_symbol == symbol_variant or name == symbol_variant) and exch_seg == exchange.upper():
                                token = inst.get("token", inst.get("symboltoken", ""))
                                if token:
                                    return str(token)
                        
                        # DISABLED: Partial matching was too aggressive and matched wrong symbols
                        # (e.g., GOLDMJAN26 matched GOLDMAHMCOM incorrectly)
                        # Only use exact matches to avoid wrong token lookups
                        # If exact match not found, symbol doesn't exist in scrip master
        except Exception as e:
            import warnings
            warnings.warn(f"Scrip master lookup failed for {symbol}: {e}", UserWarning)
        
        # Method 2: Try API endpoints (fallback)
        try:
            # Try instruments endpoint
            response = self._request("GET", "/rest/secure/marketData/instruments", params={"exchange": exchange})
            if isinstance(response, dict):
                instruments = response.get("data", [])
                if isinstance(instruments, list):
                    for inst in instruments:
                        if isinstance(inst, dict):
                            trading_symbol = inst.get("tradingsymbol", inst.get("symbol", "")).upper()
                            if trading_symbol == symbol.upper():
                                token = inst.get("token", inst.get("symboltoken", ""))
                                if token:
                                    return str(token)
        except Exception:
            pass
        
        # Method 3: Try master data endpoint
        try:
            response = self._request("GET", "/rest/secure/marketData/master", params={"exchange": exchange})
            if isinstance(response, dict):
                data = response.get("data", [])
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            if item.get("tradingsymbol", "").upper() == symbol.upper():
                                token = item.get("token", "")
                                if token:
                                    return str(token)
        except Exception:
            pass
        
        return None

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
        
        NOTE: Angel One may require symbol TOKEN (numeric) instead of symbol NAME.
        If this fails, you may need to implement symbol token lookup from master data.
        """
        # Increase retries if force_retry is True
        if force_retry:
            max_retries = max(max_retries * 2, 10)
            retry_delay = max(retry_delay * 1.5, 2.0)

        for attempt in range(max_retries):
            try:
                # Angel One SmartAPI market data endpoint for LTP
                # CRITICAL: Angel One requires symbol TOKEN (numeric ID) not symbol NAME
                
                # Step 1: Try to get symbol token from master data
                symbol_token = self._get_symbol_token(symbol, MCX_EXCHANGE_CODE)
                
                # Step 2: Use token if available, otherwise try symbol name
                symbol_to_use = symbol_token if symbol_token else symbol.upper()
                
                # Angel One SmartAPI market data quote format
                # Try multiple endpoint formats:
                
                # Angel One SmartAPI market data quote format
                # CRITICAL: Must use symbol TOKEN (numeric) not symbol NAME
                # Format: {"mode": "LTP", "exchangeTokens": {"MCX": ["token"]}}
                
                # If we have a token, use it; otherwise try symbol name (may fail)
                if symbol_token:
                    # Use token - this is the correct format
                    body = {
                        "mode": "LTP",
                        "exchangeTokens": {
                            MCX_EXCHANGE_CODE: [symbol_token]
                        }
                    }
                else:
                    # Fallback: try with symbol name (may not work)
                    import warnings
                    warnings.warn(
                        f"Symbol token not found for {symbol}. Trying with symbol name (may fail). "
                        f"Please ensure scrip master JSON is downloaded and IP is whitelisted.",
                        UserWarning,
                    )
                    body = {
                        "mode": "LTP",
                        "exchangeTokens": {
                            MCX_EXCHANGE_CODE: [symbol.upper()]
                        }
                    }

                # Try the quote endpoint
                try:
                    response = self._request("POST", "/rest/secure/marketData/quote", json_body=body)
                except RuntimeError as api_err:
                    # If this fails, check if it's a static IP issue
                    error_str = str(api_err)
                    if "Request Rejected" in error_str or "HTML" in error_str:
                        # This is likely a static IP issue
                        import warnings
                        warnings.warn(
                            f"Angel One API rejected request for {symbol}. "
                            f"This usually means: (1) API key was not created with static IP, or (2) Symbol token lookup failed. "
                            f"IMPORTANT: NO IP whitelisting is required from Angel One. "
                            f"You MUST create your API key with a STATIC IP from your ISP. Ensure scrip master is downloaded.",
                            UserWarning,
                        )
                        # Don't try alternative endpoint if it's an IP issue
                        raise RuntimeError(
                            f"Angel One API gateway rejected request. "
                            f"IMPORTANT: NO IP whitelisting is required. "
                            f"You MUST create your API key with a STATIC IP from your internet provider. "
                            f"Symbol: {symbol}, Token: {symbol_token or 'NOT FOUND'}"
                        )
                    else:
                        # Other error - try alternative endpoint
                        try:
                            # Alternative endpoint format
                            response = self._request("POST", "/rest/secure/marketData/quote/v1", json_body=body)
                        except Exception:
                            # If both fail, re-raise original error
                            raise api_err

                # Check if response is HTML (error page)
                if isinstance(response, str) and "<html>" in response.lower():
                    # API returned HTML error page - endpoint or format is wrong
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise RuntimeError(
                            f"Angel One API returned HTML error page. "
                            f"This usually means the endpoint format is incorrect or symbol token is required. "
                            f"Symbol: {symbol}. "
                            f"Note: Angel One may require numeric symbol tokens instead of symbol names. "
                            f"Check Angel One SmartAPI documentation for correct format."
                        )

                # Handle Angel One response format (verify structure)
                if isinstance(response, dict):
                    # Check for error first
                    if response.get("status") == False or "error" in response:
                        error_msg = response.get("message", response.get("error", "Unknown error"))
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        else:
                            raise RuntimeError(f"Angel One API error: {error_msg}")
                    
                    data = response.get("data", response)
                    if isinstance(data, list) and len(data) > 0:
                        data = data[0]
                    elif not isinstance(data, dict):
                        # Data is not a dict - log it
                        if attempt == max_retries - 1:
                            raise RuntimeError(f"Unexpected response format: {type(data)} - {data}")
                elif isinstance(response, str):
                    # Response is a string (not JSON) - likely an error
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Angel One API returned string instead of JSON: {response[:200]}")
                    time.sleep(retry_delay)
                    continue
                else:
                    data = response

                # Ensure data is a dict before calling .get()
                if not isinstance(data, dict):
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Response data is not a dict: {type(data)} - {data}")
                    time.sleep(retry_delay)
                    continue

                # Extract LTP (Last Traded Price) from response
                # Try multiple field names
                price = (
                    data.get("ltp") or 
                    data.get("lastPrice") or 
                    data.get("price") or 
                    data.get("lastTradedPrice") or
                    data.get("LTP") or
                    0
                )
                
                if price and float(price) > 0:
                    price_float = float(price)
                    # Sanity check
                    if price_float > 0 and price_float < 1e10:
                        return {"price": price_float, "ltp": price_float, "p": price_float}
                    else:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                else:
                    # Log what we got for debugging
                    if attempt == max_retries - 1:
                        import warnings
                        warnings.warn(
                            f"Angel One get_last_trade: No valid price in response for {symbol}. Response: {data}",
                            UserWarning,
                        )
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    # On last attempt, log the full error
                    import warnings
                    import traceback
                    error_details = f"{str(e)}\n{traceback.format_exc()}"
                    warnings.warn(
                        f"Angel One get_last_trade failed after {max_retries} attempts for {symbol}: {error_details}",
                        UserWarning,
                    )
                    
                    # FALLBACK: Try to get price from open positions
                    try:
                        positions = self.list_positions()
                        if positions:
                            # Search for matching symbol in positions
                            for pos in positions:
                                pos_symbol = pos.get("symbol", "").upper()
                                if pos_symbol == symbol.upper():
                                    # Get LTP from position
                                    ltp = pos.get("ltp", 0)
                                    if ltp and float(ltp) > 0:
                                        price_float = float(ltp)
                                        import warnings
                                        warnings.warn(
                                            f"Using position-based LTP for {symbol}: ₹{price_float:.2f} (market data API failed)",
                                            UserWarning,
                                        )
                                        return {"price": price_float, "ltp": price_float, "p": price_float}
                    except Exception as pos_err:
                        # Position fallback also failed - log but don't raise
                        import warnings
                        warnings.warn(
                            f"Position-based price fallback also failed for {symbol}: {pos_err}",
                            UserWarning,
                        )
                    
                    return None

        return None
    
    def get_historical_candles(
        self,
        symbol: str,
        timeframe: str = "1d",
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        years: float = 7.0,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical OHLC candle data for MCX symbol from Angel One SmartAPI.
        
        This uses the /candle/data endpoint which supports MCX commodities.
        Requires symbol token (numeric ID) which is looked up automatically.
        
        Args:
            symbol: MCX trading symbol (e.g., "GOLDDEC25", "SILVERFEB24")
            timeframe: Timeframe ("1d", "1h", "5m", etc.)
            from_date: Start date (optional, defaults to years ago)
            to_date: End date (optional, defaults to now)
            years: Number of years to fetch (used if from_date not provided)
            
        Returns:
            List of candle dictionaries with: timestamp, open, high, low, close, volume
            
        Raises:
            RuntimeError: If symbol token not found or API call fails
        """
        # Get symbol token (required for historical data API)
        symbol_token = self._get_symbol_token(symbol, MCX_EXCHANGE_CODE)
        if not symbol_token:
            # Try different expiry days if current contract doesn't exist
            # Format: BASESYMBOL + DAY(2digits) + MONTH(3letters) + YEAR(2digits) + FUT
            import re
            base_match = re.match(r"^([A-Z]+)(\d{2})([A-Z]{3})(\d{2})FUT$", symbol.upper())
            if base_match:
                base_symbol, day_str, month_code, year = base_match.groups()
                day = int(day_str)
                month_map = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                           "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
                
                # Try alternative expiry days for same month
                from trading.mcx_symbol_mapper import MCX_EXPIRY_DAYS
                expiry_days = MCX_EXPIRY_DAYS.get(base_symbol, [30, 27, 31, 5])
                for alt_day in expiry_days:
                    if alt_day != day:
                        alt_contract = f"{base_symbol}{alt_day:02d}{month_code}{year}FUT"
                        alt_token = self._get_symbol_token(alt_contract, MCX_EXCHANGE_CODE)
                        if alt_token:
                            import warnings
                            warnings.warn(f"Contract {symbol} not found, using alternative expiry day {alt_contract}", UserWarning)
                            symbol_token = alt_token
                            symbol = alt_contract
                            break
                
                # If still not found, try next month contract
                if not symbol_token:
                    month_num = month_map.get(month_code, 1)
                    next_month = (month_num % 12) + 1
                    next_year = int(year) if next_month > month_num else int(year) + 1
                    month_codes = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
                    next_month_code = month_codes[next_month - 1]
                    
                    # Try next month with same expiry day
                    next_contract = f"{base_symbol}{day:02d}{next_month_code}{next_year:02d}FUT"
                    symbol_token = self._get_symbol_token(next_contract, MCX_EXCHANGE_CODE)
                    if symbol_token:
                        import warnings
                        warnings.warn(f"Current month contract {symbol} not found, using next month {next_contract}", UserWarning)
                        symbol = next_contract
                    else:
                        # Try next month with alternative expiry days
                        for alt_day in expiry_days:
                            next_contract = f"{base_symbol}{alt_day:02d}{next_month_code}{next_year:02d}FUT"
                            symbol_token = self._get_symbol_token(next_contract, MCX_EXCHANGE_CODE)
                            if symbol_token:
                                import warnings
                                warnings.warn(f"Using next month contract {next_contract} with alternative expiry day", UserWarning)
                                symbol = next_contract
                                break
        
        if not symbol_token:
            raise RuntimeError(
                f"Symbol token not found for {symbol} (and next month contract). "
                f"This might mean: (1) Contract format is wrong, (2) Contract doesn't exist in scrip master, "
                f"(3) Contract has expired, or (4) Scrip master needs to be refreshed. "
                f"Try downloading fresh scrip master from: https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
            )
        
        # Map timeframe to Angel One interval format
        interval_map = {
            "1m": "ONE_MINUTE",
            "3m": "THREE_MINUTE",
            "5m": "FIVE_MINUTE",
            "10m": "TEN_MINUTE",
            "15m": "FIFTEEN_MINUTE",
            "30m": "THIRTY_MINUTE",
            "1h": "ONE_HOUR",
            "1d": "ONE_DAY",
        }
        angel_interval = interval_map.get(timeframe.lower(), "ONE_DAY")
        
        # Set date range
        if to_date is None:
            to_date = datetime.now(timezone.utc)
        if from_date is None:
            from_date = to_date - timedelta(days=int(years * 365))
        
        # Calculate total days requested
        total_days = (to_date - from_date).days
        
        # Max days per request (as per AngelOne API documentation)
        max_days_per_request = {
            "ONE_MINUTE": 30,
            "THREE_MINUTE": 60,
            "FIVE_MINUTE": 100,
            "TEN_MINUTE": 100,
            "FIFTEEN_MINUTE": 200,
            "THIRTY_MINUTE": 200,
            "ONE_HOUR": 400,
            "ONE_DAY": 2000,
        }.get(angel_interval, 2000)  # Default to 2000 days for ONE_DAY
        
        # If request exceeds max days, we'll need to split into multiple requests
        if total_days > max_days_per_request:
            print(f"  [INFO] Requesting {total_days} days, but max is {max_days_per_request} days per request")
            print(f"  [INFO] Will split into multiple requests to fetch all data")
        
        # Format dates for Angel One API (yyyy-MM-dd hh:mm format as per official docs)
        # NOTE: Use 24-hour format (H) not 12-hour (h) - Python's %H gives 24-hour format
        from_date_str = from_date.strftime("%Y-%m-%d %H:%M")  # e.g., "2023-09-06 11:15"
        to_date_str = to_date.strftime("%Y-%m-%d %H:%M")      # e.g., "2023-09-06 12:00"
        
        # Angel One SmartAPI candle data endpoint (OFFICIAL API)
        # Endpoint: /rest/secure/angelbroking/historical/v1/getCandleData
        # Documentation: https://apiconnect.angelone.in/rest/secure/angelbroking/historical/v1/getCandleData
        # Exchange: MCX (for commodities only)
        body = {
            "exchange": MCX_EXCHANGE_CODE,  # "MCX" for commodities
            "symboltoken": symbol_token,     # Numeric token from scrip master
            "interval": angel_interval,      # ONE_MINUTE, ONE_HOUR, ONE_DAY, etc.
            "fromdate": from_date_str,       # Format: "yyyy-MM-dd hh:mm"
            "todate": to_date_str,           # Format: "yyyy-MM-dd hh:mm"
        }
        
        try:
            # Add rate limiting protection - wait between requests
            import time
            time.sleep(0.5)  # 500ms delay to avoid rate limits
            
            # If request exceeds max days, split into multiple chunks
            all_candles = []
            current_from = from_date
            chunk_number = 1
            
            while current_from < to_date:
                # Calculate chunk end date (max days per request)
                chunk_to = min(current_from + timedelta(days=max_days_per_request), to_date)
                
                if total_days > max_days_per_request:
                    print(f"  [CHUNK {chunk_number}] Fetching {current_from.date()} to {chunk_to.date()} ({max_days_per_request} days max)")
                
                # Format dates for this chunk
                chunk_from_str = current_from.strftime("%Y-%m-%d %H:%M")
                chunk_to_str = chunk_to.strftime("%Y-%m-%d %H:%M")
                
                # Request body for this chunk
                chunk_body = {
                    "exchange": MCX_EXCHANGE_CODE,  # "MCX" for commodities
                    "symboltoken": symbol_token,     # Numeric token from scrip master
                    "interval": angel_interval,      # ONE_MINUTE, ONE_HOUR, ONE_DAY, etc.
                    "fromdate": chunk_from_str,      # Format: "yyyy-MM-dd hh:mm"
                    "todate": chunk_to_str,          # Format: "yyyy-MM-dd hh:mm"
                }
                
                # Use official historical data endpoint
                response = self._request("POST", "/rest/secure/angelbroking/historical/v1/getCandleData", json_body=chunk_body)
                
                # Handle response format - check if it's HTML (error page) first
                if isinstance(response, str):
                    # API returned HTML error page instead of JSON
                    if "<html>" in response.lower() or "request rejected" in response.lower():
                        raise RuntimeError(
                            f"Angel One API returned HTML error page. "
                            f"This usually means: (1) IP not whitelisted, (2) Rate limit exceeded, "
                            f"or (3) Invalid symbol token. Response: {response[:200]}"
                        )
                    else:
                        raise RuntimeError(f"Unexpected response format: got string instead of JSON: {response[:200]}")
                
                # Handle response format
                if isinstance(response, dict):
                    if response.get("status") == False or "error" in response:
                        error_msg = response.get("message", response.get("error", "Unknown error"))
                        raise RuntimeError(f"Angel One historical data error: {error_msg}")
                    
                    data = response.get("data", [])
                    if not data or data is None:
                        # No data available for this chunk - might be expired contract or no data for date range
                        if total_days > max_days_per_request:
                            print(f"  [CHUNK {chunk_number}] No data returned for this date range")
                    else:
                        # Convert Angel One format to canonical candles
                        chunk_candles = []
                        for candle_row in data:
                            if not isinstance(candle_row, list) or len(candle_row) < 6:
                                continue
                            
                            # Angel One format: [timestamp, Open, High, Low, Close, Volume]
                            timestamp_str = candle_row[0]
                            open_price = float(candle_row[1]) if candle_row[1] else 0
                            high = float(candle_row[2]) if candle_row[2] else 0
                            low = float(candle_row[3]) if candle_row[3] else 0
                            close = float(candle_row[4]) if candle_row[4] else 0
                            volume = float(candle_row[5]) if len(candle_row) > 5 and candle_row[5] else 0
                            
                            # Parse timestamp (format: "2023-09-06T11:15:00+05:30")
                            try:
                                # Remove timezone offset and parse
                                if "+" in timestamp_str:
                                    timestamp_str = timestamp_str.split("+")[0]
                                elif "-" in timestamp_str and timestamp_str.count("-") > 2:
                                    # Has timezone offset like "2023-09-06T11:15:00-05:30"
                                    parts = timestamp_str.rsplit("-", 1)
                                    if len(parts) == 2 and ":" in parts[1]:
                                        timestamp_str = parts[0]
                                
                                dt = datetime.fromisoformat(timestamp_str.replace("T", " "))
                                if dt.tzinfo is None:
                                    # Assume IST (UTC+5:30) if no timezone
                                    dt = dt.replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
                                dt = dt.astimezone(timezone.utc)
                                timestamp_iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                            except Exception as ts_err:
                                # Skip invalid timestamps
                                continue
                            
                            # Create canonical candle format
                            candle = {
                                "timestamp": timestamp_iso,
                                "open": open_price,
                                "high": high,
                                "low": low,
                                "close": close,
                                "volume": volume,
                                "source": "angelone_mcx",
                            }
                            chunk_candles.append(candle)
                        
                        all_candles.extend(chunk_candles)
                        if total_days > max_days_per_request:
                            print(f"  [CHUNK {chunk_number}] Fetched {len(chunk_candles)} candles (total so far: {len(all_candles)})")
                
                # Move to next chunk
                current_from = chunk_to + timedelta(days=1)  # Start next chunk 1 day after previous end
                chunk_number += 1
                
                # Rate limiting between chunks
                if current_from < to_date:
                    time.sleep(0.5)  # Wait 500ms between chunks
            
            if total_days > max_days_per_request:
                print(f"  [OK] Successfully fetched {len(all_candles)} total candles across {chunk_number - 1} chunks")
            
            return all_candles
                
        except RuntimeError:
            raise  # Re-raise our custom errors
        except Exception as e:
            error_str = str(e)
            # Check for rate limiting
            if "rate" in error_str.lower() or "429" in error_str or "access denied" in error_str.lower():
                raise RuntimeError(
                    f"Angel One rate limit exceeded. Please wait a few minutes and try again. "
                    f"Original error: {error_str}"
                )
            raise RuntimeError(f"Failed to fetch historical data from Angel One: {e}")
