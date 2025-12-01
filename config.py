"""
Configuration for crypto and commodities data ingestion system.
"""
import os
from pathlib import Path

# Base data directory
BASE_DATA_DIR = Path("data/json/raw")

# Data sources configuration
BINANCE_BULK_BASE_URL = "https://data.binance.vision/data/spot/daily/klines"
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"
BINANCE_REST_BASE = "https://api.binance.com/api/v3"

# Fallback sources
COINBASE_WS_URL = "wss://ws-feed.pro.coinbase.com"
KUCOIN_WS_URL = "wss://ws-api-spot.kucoin.com"
OKX_REST_BASE = "https://www.okx.com/api/v5"

# Commodities sources
YAHOO_FINANCE_BASE = "yfinance"
INVESTING_COM_BASE = "https://www.investing.com"
TRADING_ECONOMICS_BASE = "https://api.tradingeconomics.com"

# Supported symbols (used as fallback only if user doesn't provide any)
# Users can now enter any symbols they want via interactive mode or command line
CRYPTO_SYMBOLS = []  # Empty by default - user must specify

COMMODITIES_SYMBOLS = []  # Empty by default - user must specify

# Timeframes
TIMEFRAMES = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400
}

# Retry configuration
MAX_RETRIES = 5
INITIAL_BACKOFF = 1
MAX_BACKOFF = 60
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

# WebSocket configuration
WS_RECONNECT_DELAY = 5
WS_HEARTBEAT_INTERVAL = 30
WS_TIMEOUT = 30

# Data validation
MIN_HISTORICAL_YEARS = 7
MAX_CANDLE_AGE_SECONDS = 300  # Reject candles older than 5 minutes for live data

# File rotation
DAILY_ROTATION_HOUR = 0  # UTC hour when new daily file is created

