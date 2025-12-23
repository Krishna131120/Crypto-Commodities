# Angel One API Integration Requirements

## Overview
This document outlines what's required to connect Angel One API to the commodities trading bot for MCX trading.

---

## 1. Angel One Account Setup

### Required:
1. **Angel One Trading Account** - Active account with MCX segment enabled
2. **API Key** - Register for API key through Angel One developer portal
3. **Client ID** - Your Angel One client ID
4. **TOTP (Time-Based One-Time Password)** - Enable 2FA via authenticator app

---

## 2. Authentication Requirements

### Credentials Needed:
- **API Key** - From Angel One developer portal
- **Client ID** - Your Angel One client ID (e.g., "A12345")
- **TOTP Secret** - Generated when enabling TOTP
- **Trading Password/MPIN** - Your trading terminal password

### Authentication Flow:
1. Generate TOTP using authenticator app (Google Authenticator, etc.)
2. Login with: Client ID + Trading Password + TOTP
3. Get access token (JWT) - valid for session duration
4. Use access token in API requests

---

## 3. API Endpoints Required

### Base URL:
- **Production:** `https://apiconnect.angelbroking.com` (or similar - check Angel One docs)
- **Paper Trading:** Check if Angel One provides sandbox environment

### Required Endpoints:
1. **Authentication:**
   - Login endpoint (with TOTP)
   - Token refresh endpoint

2. **Account:**
   - Get account/fund details
   - Get margin details

3. **Positions:**
   - List all positions
   - Get position for specific symbol

4. **Orders:**
   - Place order (market/limit)
   - Modify order
   - Cancel order
   - Get order status

5. **Market Data:**
   - Last traded price (LTP)
   - OHLC data
   - Full quote data

---

## 4. MCX Commodities Specific Requirements

### Exchange Segment:
- **Exchange Code:** `MCX` (Multi Commodity Exchange)
- **Segment:** Commodity futures

### Symbol Format:
- MCX symbols (e.g., `GOLDDEC25`, `SILVERDEC25`)
- Need to map Yahoo symbols (GC=F) to MCX symbols

### Order Requirements:
- **Quantity:** Lot-based (integer quantities)
- **Product Type:** INTRADAY, MARGIN, or DELIVERY
- **Order Type:** MARKET or LIMIT
- **Transaction Type:** BUY or SELL

---

## 5. Environment Variables Needed

Add to `.env` file:
```
ANGEL_ONE_API_KEY="your_api_key_here"
ANGEL_ONE_CLIENT_ID="your_client_id"
ANGEL_ONE_TOTP_SECRET="your_totp_secret"  # Optional if using TOTP library
ANGEL_ONE_PASSWORD="your_trading_password"  # Or MPIN
ANGEL_ONE_BASE_URL="https://apiconnect.angelbroking.com"  # Optional override
```

---

## 6. Python Package Requirements

You may need to install:
```bash
pip install smartapi-python  # If Angel One provides official SDK
# OR
pip install requests pyotp  # For manual API calls + TOTP generation
```

---

## 7. Implementation Checklist

### Angel One Client (`angelone_client.py`) Must Implement:

1. **BrokerClient Interface** (from `broker_interface.py`):
   - `get_account()` - Account details
   - `list_positions()` - All positions (filter MCX only)
   - `get_position(symbol)` - Single position
   - `submit_order(...)` - Place order
   - `cancel_order(order_id)` - Cancel order
   - `get_last_trade(symbol)` - LTP
   - `broker_name` property - Return "angelone"

2. **Authentication:**
   - Login with TOTP
   - Token refresh
   - Session management

3. **MCX Filtering:**
   - Filter positions by exchange segment = "MCX"
   - Only process MCX symbols

4. **Error Handling:**
   - Handle authentication failures
   - Handle API rate limits
   - Handle order rejections

---

## 8. Key Differences from Dhan

| Feature | Dhan | Angel One |
|---------|------|-----------|
| **Auth** | JWT token (24hr) | TOTP + Password + Token |
| **Token Refresh** | RenewToken endpoint | Re-login with TOTP |
| **Base URL** | `https://api.dhan.co` | `https://apiconnect.angelbroking.com` |
| **Headers** | `access-token`, `dhanClientId` | Check Angel One docs |
| **MCX Exchange** | `MCX` | `MCX` (same) |

---

## 9. Documentation Resources

1. **Angel One SmartAPI Docs:** https://smartapi.angelone.in/
2. **Developer Portal:** Register for API key
3. **TOTP Setup Guide:** Enable 2FA in Angel One account
4. **MCX Segment Activation:** Ensure commodity segment is active

---

## 10. Testing Checklist

Before live trading:
- [ ] Test authentication flow
- [ ] Test account balance retrieval
- [ ] Test position listing (MCX only)
- [ ] Test order placement (dry-run/small quantity)
- [ ] Test order cancellation
- [ ] Test LTP retrieval for MCX symbols
- [ ] Verify MCX symbol mapping works
- [ ] Test error handling (expired token, etc.)

---

## Next Steps

1. Register for Angel One API key
2. Enable TOTP in your Angel One account
3. Get API documentation from Angel One
4. Implement `AngelOneClient` class
5. Test with small quantities first
6. Replace Dhan references in codebase
