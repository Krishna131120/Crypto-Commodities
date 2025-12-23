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
from typing import Any, Dict, Optional

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
                # Clean TOTP secret (remove spaces, convert to uppercase)
                clean_secret = self.config.totp_secret.strip().replace(" ", "").upper()
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

        resp = self._session.request(method, url, params=params, json=json_body, timeout=15)

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
        # Increase retries if force_retry is True
        if force_retry:
            max_retries = max(max_retries * 2, 10)
            retry_delay = max(retry_delay * 1.5, 2.0)

        for attempt in range(max_retries):
            try:
                # Angel One SmartAPI market data endpoint for LTP
                # Format: {"mode": "LTP", "exchangeTokens": {"MCX": ["symbol_token"]}}
                # For MCX, we need symbol token, but for now try with symbol name
                body = {
                    "mode": "LTP",
                    "exchangeTokens": {
                        MCX_EXCHANGE_CODE: [symbol.upper()]
                    }
                }

                response = self._request("POST", "/rest/secure/marketData/quote", json_body=body)

                # Handle Angel One response format (verify structure)
                if isinstance(response, dict):
                    data = response.get("data", response)
                    if isinstance(data, list) and len(data) > 0:
                        data = data[0]
                else:
                    data = response

                # Extract LTP (Last Traded Price) from response
                price = data.get("ltp", data.get("lastPrice", data.get("price", 0)))
                if price and float(price) > 0:
                    price_float = float(price)
                    # Sanity check
                    if price_float > 0 and price_float < 1e10:
                        return {"price": price_float}
                    else:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                else:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    # On last attempt, log the error
                    import warnings
                    warnings.warn(
                        f"Angel One get_last_trade failed after {max_retries} attempts for {symbol}: {e}",
                        UserWarning,
                    )
                    return None

        return None
