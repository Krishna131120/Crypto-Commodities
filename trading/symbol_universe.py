"""
Symbol universe and data_symbol <-> trading_symbol mapping.

This module defines a small, curated starting universe of assets where:
- `data_symbol` is the symbol used by the ingestion/feature pipeline
  (e.g. Binance or Yahoo-style symbols like BTC-USDT, GC=F).
- `trading_symbol` is the symbol used on the Alpaca side for order
  submission (e.g. BTCUSD, GLD, USO).

We deliberately keep this mapping explicit and human-auditable so that:
- You can easily extend / edit it as you add more assets or change
  how you want to proxy commodities (e.g. GC=F -> GLD).
- The live trading engine never guesses a trading symbol; it always
  uses this mapping and will refuse to trade symbols that are not mapped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class AssetMapping:
    """Link between data/feature symbol and Alpaca trading symbol."""

    logical_name: str          # Human label, e.g. "bitcoin", "gold"
    asset_type: str            # "crypto" or "commodities"
    data_symbol: str           # Symbol used in your data/feature pipeline
    trading_symbol: str        # Symbol used at Alpaca for orders
    timeframe: str = "1d"      # Default timeframe for features/models
    horizon_profile: str = "short"  # Default horizon profile for training/trading
    enabled: bool = True       # If False, will not be traded even if models exist


# NOTE: This is a *starting* universe focused on symbols that already exist
# in your project structure. You can safely change/extend this list.
#
# Crypto:
# - Data symbols follow your existing convention (BTC-USDT, ETH-USDT, ...).
# - Trading symbols follow Alpaca's crypto format (BTCUSD, ETHUSD, ...).
#
# Commodities:
# - Data symbols are Yahoo-style futures (CL=F, GC=F, SI=F, PL=F).
# - Trading symbols are *ETF proxies* that roughly track the underlying:
#   - Crude oil:  CL=F  -> USO   (United States Oil Fund)
#   - Gold:       GC=F  -> GLD   (SPDR Gold Shares)
#   - Silver:     SI=F  -> SLV   (iShares Silver Trust)
#   - Platinum:   PL=F  -> PPLT  (Aberdeen Physical Platinum Shares)
# If you prefer to use different instruments (or if your Alpaca account
# has futures enabled), you can update these mappings.
UNIVERSE: List[AssetMapping] = [
    # Crypto
    AssetMapping(
        logical_name="bitcoin",
        asset_type="crypto",
        data_symbol="BTC-USDT",  # existing project convention
        # Alpaca TRADING API expects symbols without "/", e.g. BTCUSD.
        # Data/historical helpers will convert to BTC/USD when needed.
        trading_symbol="BTCUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="ethereum",
        asset_type="crypto",
        data_symbol="ETH-USDT",
        trading_symbol="ETHUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="solana",
        asset_type="crypto",
        data_symbol="SOL-USDT",
        trading_symbol="SOLUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    # Extended crypto universe (all traded in USD on Alpaca)
    AssetMapping(
        logical_name="aave",
        asset_type="crypto",
        data_symbol="AAVE-USDT",
        trading_symbol="AAVEUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="algorand",
        asset_type="crypto",
        data_symbol="ALGO-USDT",
        trading_symbol="ALGOUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="avalanche",
        asset_type="crypto",
        data_symbol="AVAX-USDT",
        trading_symbol="AVAXUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="basic_attention_token",
        asset_type="crypto",
        data_symbol="BAT-USDT",
        trading_symbol="BATUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="bitcoin_cash",
        asset_type="crypto",
        data_symbol="BCH-USDT",
        trading_symbol="BCHUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="chainlink",
        asset_type="crypto",
        data_symbol="LINK-USDT",
        trading_symbol="LINKUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="dai",
        asset_type="crypto",
        data_symbol="DAI-USDT",
        trading_symbol="DAIUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="dogecoin",
        asset_type="crypto",
        data_symbol="DOGE-USDT",
        trading_symbol="DOGEUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="graph",
        asset_type="crypto",
        data_symbol="GRT-USDT",
        trading_symbol="GRTUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="litecoin",
        asset_type="crypto",
        data_symbol="LTC-USDT",
        trading_symbol="LTCUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="maker",
        asset_type="crypto",
        data_symbol="MKR-USDT",
        trading_symbol="MKRUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="matic",
        asset_type="crypto",
        data_symbol="MATIC-USDT",
        trading_symbol="MATICUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="near_protocol",
        asset_type="crypto",
        data_symbol="NEAR-USDT",
        trading_symbol="NEARUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="pax_gold",
        asset_type="crypto",
        data_symbol="PAXG-USDT",
        trading_symbol="PAXGUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="shiba_inu",
        asset_type="crypto",
        data_symbol="SHIB-USDT",
        trading_symbol="SHIBUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="sushi",
        asset_type="crypto",
        data_symbol="SUSHI-USDT",
        trading_symbol="SUSHIUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="tether",
        asset_type="crypto",
        data_symbol="USDT-USDT",
        trading_symbol="USDTUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="tron",
        asset_type="crypto",
        data_symbol="TRX-USDT",
        trading_symbol="TRXUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="uniswap",
        asset_type="crypto",
        data_symbol="UNI-USDT",
        trading_symbol="UNIUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="wrapped_bitcoin",
        asset_type="crypto",
        data_symbol="WBTC-USDT",
        trading_symbol="WBTCUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="yearn_finance",
        asset_type="crypto",
        data_symbol="YFI-USDT",
        trading_symbol="YFIUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
]


def all_enabled() -> List[AssetMapping]:
    """Return all assets that are marked as enabled."""
    return [a for a in UNIVERSE if a.enabled]


def by_asset_type(asset_type: str) -> List[AssetMapping]:
    """Return enabled assets of a given type ('crypto' or 'commodities')."""
    asset_type = asset_type.lower()
    return [a for a in all_enabled() if a.asset_type == asset_type]


def find_by_data_symbol(symbol: str) -> Optional[AssetMapping]:
    """Look up mapping by data/feature symbol (case-insensitive)."""
    sym = symbol.strip().upper()
    for asset in UNIVERSE:
        if asset.data_symbol.upper() == sym:
            return asset
    return None


def find_by_trading_symbol(symbol: str) -> Optional[AssetMapping]:
    """Look up mapping by Alpaca trading symbol (case-insensitive)."""
    sym = symbol.strip().upper()
    for asset in UNIVERSE:
        if asset.trading_symbol.upper() == sym:
            return asset
    return None


def to_trading_symbol(data_symbol: str) -> Optional[str]:
    """Return Alpaca trading symbol for a given data symbol, or None."""
    asset = find_by_data_symbol(data_symbol)
    return asset.trading_symbol if asset else None


def to_data_symbol(trading_symbol: str) -> Optional[str]:
    """Return data/feature symbol for a given Alpaca trading symbol, or None."""
    asset = find_by_trading_symbol(trading_symbol)
    return asset.data_symbol if asset else None


def universe_summary() -> Dict[str, Dict[str, str]]:
    """
    Return a human-readable summary of the current universe, useful for logging.
    """
    summary: Dict[str, Dict[str, str]] = {}
    for asset in UNIVERSE:
        summary[asset.logical_name] = {
            "asset_type": asset.asset_type,
            "data_symbol": asset.data_symbol,
            "trading_symbol": asset.trading_symbol,
            "timeframe": asset.timeframe,
            "horizon_profile": asset.horizon_profile,
            "enabled": str(asset.enabled),
        }
    return summary



