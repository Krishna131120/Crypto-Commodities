"""
Symbol universe and data_symbol <-> trading_symbol mapping.

This module defines a comprehensive universe of assets where:
- `data_symbol` is the symbol used by the ingestion/feature pipeline
  (e.g. Binance or Yahoo-style symbols like BTC-USDT, GC=F, or MCX-specific symbols like MCX_GOLDM).
- `trading_symbol` is the symbol used at the broker for order submission:
  - Crypto: Alpaca format (BTCUSD, ETHUSD, etc.)
  - Commodities: MCX contract symbols (GOLD, SILVER, CRUDEOIL, etc.) - DHAN broker ONLY

IMPORTANT FOR COMMODITIES:
- ALL commodities MUST use Angel One broker (MCX exchange)
- No Alpaca fallback - commodities will raise an error if AlpacaClient is used
- MCX contract symbols are auto-generated based on trading horizon
- Includes 28+ MCX commodities: Bullion, Energy, Base Metals, Agricultural

We deliberately keep this mapping explicit and human-auditable so that:
- You can easily extend / edit it as you add more assets
- The live trading engine never guesses a trading symbol; it always
  uses this mapping and will refuse to trade symbols that are not mapped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from .mcx_symbol_mapper import get_mcx_contract_for_horizon
except ImportError:
    # Fallback if mcx_symbol_mapper not available
    def get_mcx_contract_for_horizon(yahoo_symbol: str, horizon: str) -> str:
        return yahoo_symbol  # Return as-is if mapper not available


@dataclass(frozen=True)
class AssetMapping:
    """Link between data/feature symbol and broker trading symbol."""

    logical_name: str          # Human label, e.g. "bitcoin", "gold"
    asset_type: str            # "crypto" or "commodities"
    data_symbol: str           # Symbol used in your data/feature pipeline
    trading_symbol: str        # Symbol used at broker for orders (Alpaca format or MCX format)
    timeframe: str = "1d"      # Default timeframe for features/models
    horizon_profile: str = "short"  # Default horizon profile for training/trading
    enabled: bool = True       # If False, will not be traded even if models exist
    
    def get_mcx_symbol(self, horizon: Optional[str] = None) -> str:
        """
        Get MCX contract symbol for this asset.
        
        Args:
            horizon: Trading horizon ("intraday", "short", "long")
            
        Returns:
            MCX contract symbol (e.g., "GOLDFEB24")
        """
        if self.asset_type != "commodities":
            return self.trading_symbol  # Return as-is for non-commodities
        
        horizon = horizon or self.horizon_profile
        return get_mcx_contract_for_horizon(self.data_symbol, horizon)


# NOTE: This is a *starting* universe focused on symbols that already exist
# in your project structure. You can safely change/extend this list.
#
# Crypto:
# - Data symbols follow your existing convention (BTC-USDT, ETH-USDT, ...).
# - Trading symbols follow Alpaca's crypto format (BTCUSD, ETHUSD, ...).
#
# Commodities:
# - Data symbols are Yahoo-style futures (CL=F, GC=F, SI=F, etc.) or MCX-specific symbols (MCX_*)
# - Trading symbols are MCX contract symbols (GOLD, SILVER, CRUDEOIL, etc.)
# - ALL commodities MUST use Angel One broker (MCX exchange) - no Alpaca fallback
# - MCX contract symbols are auto-generated based on horizon (e.g., GOLDDEC25 for current month)
# - Includes: Bullion (Gold, Silver variants), Energy (Crude Oil, Natural Gas), 
#   Base Metals (Aluminium, Copper, Lead, Nickel, Zinc), Agricultural (Cotton, Cardamom, etc.)
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
        enabled=False,  # ← DISABLED: 0% win rate (7 losses, 0 wins) - strategy doesn't work for this pair
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
        enabled=False,  # ← DISABLED: 0% win rate (77 losses, 0 wins, -$3,345 lost) - strategy doesn't work for this pair
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
        enabled=False,  # ← Disabled: Alpaca API returns 500 errors (not supported on paper trading)
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
    # Additional USDT pairs
    AssetMapping(
        logical_name="curve_dao",
        asset_type="crypto",
        data_symbol="CRV-USDT",
        trading_symbol="CRVUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="polkadot",
        asset_type="crypto",
        data_symbol="DOT-USDT",
        trading_symbol="DOTUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="pepe",
        asset_type="crypto",
        data_symbol="PEPE-USDT",
        trading_symbol="PEPEUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="skycoin",
        asset_type="crypto",
        data_symbol="SKY-USDT",
        trading_symbol="SKYUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="trump",
        asset_type="crypto",
        data_symbol="TRUMP-USDT",
        trading_symbol="TRUMPUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="ripple",
        asset_type="crypto",
        data_symbol="XRP-USDT",
        trading_symbol="XRPUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="tezos",
        asset_type="crypto",
        data_symbol="XTZ-USDT",
        trading_symbol="XTZUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="sandbox",
        asset_type="crypto",
        data_symbol="SAND-USDT",
        trading_symbol="SANDUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    # USDC pairs (traded as USD on Alpaca)
    AssetMapping(
        logical_name="aave_usdc",
        asset_type="crypto",
        data_symbol="AAVE-USDC",
        trading_symbol="AAVEUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="avalanche_usdc",
        asset_type="crypto",
        data_symbol="AVAX-USDC",
        trading_symbol="AVAXUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="basic_attention_token_usdc",
        asset_type="crypto",
        data_symbol="BAT-USDC",
        trading_symbol="BATUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="bitcoin_cash_usdc",
        asset_type="crypto",
        data_symbol="BCH-USDC",
        trading_symbol="BCHUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="bitcoin_usdc",
        asset_type="crypto",
        data_symbol="BTC-USDC",
        trading_symbol="BTCUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="curve_dao_usdc",
        asset_type="crypto",
        data_symbol="CRV-USDC",
        trading_symbol="CRVUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="dogecoin_usdc",
        asset_type="crypto",
        data_symbol="DOGE-USDC",
        trading_symbol="DOGEUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="polkadot_usdc",
        asset_type="crypto",
        data_symbol="DOT-USDC",
        trading_symbol="DOTUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="ethereum_usdc",
        asset_type="crypto",
        data_symbol="ETH-USDC",
        trading_symbol="ETHUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="graph_usdc",
        asset_type="crypto",
        data_symbol="GRT-USDC",
        trading_symbol="GRTUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="chainlink_usdc",
        asset_type="crypto",
        data_symbol="LINK-USDC",
        trading_symbol="LINKUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="litecoin_usdc",
        asset_type="crypto",
        data_symbol="LTC-USDC",
        trading_symbol="LTCUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="shiba_inu_usdc",
        asset_type="crypto",
        data_symbol="SHIB-USDC",
        trading_symbol="SHIBUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="skycoin_usdc",
        asset_type="crypto",
        data_symbol="SKY-USDC",
        trading_symbol="SKYUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="sushi_usdc",
        asset_type="crypto",
        data_symbol="SUSHI-USDC",
        trading_symbol="SUSHIUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="uniswap_usdc",
        asset_type="crypto",
        data_symbol="UNI-USDC",
        trading_symbol="UNIUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="tezos_usdc",
        asset_type="crypto",
        data_symbol="XTZ-USDC",
        trading_symbol="XTZUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="yearn_finance_usdc",
        asset_type="crypto",
        data_symbol="YFI-USDC",
        trading_symbol="YFIUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    # USD pairs (direct USD trading on Alpaca)
    AssetMapping(
        logical_name="usdc_usd",
        asset_type="crypto",
        data_symbol="USDC-USD",
        trading_symbol="USDCUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="usdg_usd",
        asset_type="crypto",
        data_symbol="USDG-USD",
        trading_symbol="USDGUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="usdt_usd",
        asset_type="crypto",
        data_symbol="USDT-USD",
        trading_symbol="USDTUSD",
        timeframe="1d",
        horizon_profile="short",
    ),
    # BTC pairs (Note: Alpaca may not directly support BTC-quoted pairs,
    # but we include them for data collection - trading will use USD pairs)
    AssetMapping(
        logical_name="bitcoin_cash_btc",
        asset_type="crypto",
        data_symbol="BCH-BTC",
        trading_symbol="BCHUSD",  # Use USD equivalent for trading
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="ethereum_btc",
        asset_type="crypto",
        data_symbol="ETH-BTC",
        trading_symbol="ETHUSD",  # Use USD equivalent for trading
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="litecoin_btc",
        asset_type="crypto",
        data_symbol="LTC-BTC",
        trading_symbol="LTCUSD",  # Use USD equivalent for trading
        timeframe="1d",
        horizon_profile="short",
    ),
    AssetMapping(
        logical_name="uniswap_btc",
        asset_type="crypto",
        data_symbol="UNI-BTC",
        trading_symbol="UNIUSD",  # Use USD equivalent for trading
        timeframe="1d",
        horizon_profile="short",
    ),
    # ============================================================================
    # COMMODITIES - MCX Only (Angel One broker required)
    # ============================================================================
    # 
    # BUDGET CONFIGURATION:
    # Currently configured for ₹5,000 budget - only Gold Petal & Gold Guinea enabled
    # 
    # TO ENABLE MORE COMMODITIES:
    # 1. Find the commodity below (search by name)
    # 2. Change "enabled=False," to "enabled=True," (or remove the line entirely)
    # 3. Save the file
    # 
    # RECOMMENDED PROGRESSION:
    # - ₹5,000: Gold Petal, Gold Guinea (current)
    # - ₹50,000: Add Gold Mini, Crude Oil Mini, Silver Micro
    # - ₹1,00,000+: Add Natural Gas, Zinc Mini, Copper, etc.
    # ============================================================================
    
    # Bullion (Precious Metals)
    # Standard contracts - HIGH margin, disabled for budget trading
    AssetMapping(
        logical_name="gold",
        asset_type="commodities",
        data_symbol="GC=F",
        trading_symbol="GOLD",  # MCX Gold futures - 1 kg (₹70k-100k margin)
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Disabled: Too expensive for ₹5k budget
    ),
    AssetMapping(
        logical_name="gold_mini",
        asset_type="commodities",
        data_symbol="MCX_GOLDM",
        trading_symbol="GOLDM",  # MCX Gold Mini - 100g (₹7k-10k margin)
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Enable when you have ₹50k+ budget
    ),
    AssetMapping(
        logical_name="gold_guinea",
        asset_type="commodities",
        data_symbol="MCX_GOLDGUINEA",
        trading_symbol="GOLDGUINEA",  # MCX Gold Guinea - 8g (₹600-850 margin)
        timeframe="1d",
        horizon_profile="short",
        enabled=True,  # ✅ ENABLED for ₹5k budget
    ),
    AssetMapping(
        logical_name="gold_petal",
        asset_type="commodities",
        data_symbol="MCX_GOLDPETAL",
        trading_symbol="GOLDPETAL",  # MCX Gold Petal - 1g (₹700-1k margin)
        timeframe="1d",
        horizon_profile="short",
        enabled=True,  # ✅ ENABLED for ₹5k budget
    ),
    AssetMapping(
        logical_name="silver",
        asset_type="commodities",
        data_symbol="SI=F",
        trading_symbol="SILVER",  # MCX Silver - 30kg (₹2.5L-3.8L margin)
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Disabled: Too expensive
    ),
    AssetMapping(
        logical_name="silver_mini",
        asset_type="commodities",
        data_symbol="MCX_SILVERM",
        trading_symbol="SILVERM",  # MCX Silver Mini - 5kg (₹42k-64k margin)
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Enable when you have ₹1L+ budget
    ),
    AssetMapping(
        logical_name="silver_micro",
        asset_type="commodities",
        data_symbol="MCX_SILVERMIC",
        trading_symbol="SILVERMIC",  # MCX Silver Micro - 1kg (₹8.5k-12.7k margin)
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Enable when you have ₹50k+ budget
    ),
    AssetMapping(
        logical_name="silver_1000",
        asset_type="commodities",
        data_symbol="MCX_SILVER1000",
        trading_symbol="SILVER1000",  # MCX Silver 1000 - 30kg
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Disabled: Too expensive
    ),
    AssetMapping(
        logical_name="platinum",
        asset_type="commodities",
        data_symbol="PL=F",
        trading_symbol="PLATINUM",  # MCX Platinum - Low liquidity
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Disabled: Rarely traded on MCX
    ),
    
    # Energy
    AssetMapping(
        logical_name="crude_oil",
        asset_type="commodities",
        data_symbol="CL=F",
        trading_symbol="CRUDEOIL",  # MCX Crude Oil - 100 barrels (₹36k-60k margin)
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Disabled: Too expensive for ₹5k budget
    ),
    AssetMapping(
        logical_name="crude_oil_mini",
        asset_type="commodities",
        data_symbol="MCX_CRUDEOILM",
        trading_symbol="CRUDEOILM",  # MCX Crude Oil Mini - 10 barrels (₹3.6k-6k margin)
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Enable when you have ₹50k+ budget (good volatility for ML)
    ),
    AssetMapping(
        logical_name="brent_crude",
        asset_type="commodities",
        data_symbol="BZ=F",
        trading_symbol="BRENTCRUDE",  # MCX Brent Crude - Similar to Crude Oil
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Disabled: Too expensive
    ),
    AssetMapping(
        logical_name="natural_gas",
        asset_type="commodities",
        data_symbol="NG=F",
        trading_symbol="NATURALGAS",  # MCX Natural Gas - 1250 MMBtu (₹18k-31k margin)
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Enable when you have ₹1L+ budget (high volatility)
    ),
    
    # Base Metals
    AssetMapping(
        logical_name="aluminium",
        asset_type="commodities",
        data_symbol="MCX_ALUMINIUM",
        trading_symbol="ALUMINIUM",  # MCX Aluminium - 5 MT
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Disabled: Too expensive
    ),
    AssetMapping(
        logical_name="aluminium_mini",
        asset_type="commodities",
        data_symbol="MCX_ALUMINI",
        trading_symbol="ALUMINI",  # MCX Aluminium Mini - 1 MT (₹5k-8k margin)
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Enable when you have ₹50k+ budget
    ),
    AssetMapping(
        logical_name="copper",
        asset_type="commodities",
        data_symbol="HG=F",
        trading_symbol="COPPER",  # MCX Copper (₹15k-25k margin)
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Enable when you have ₹1L+ budget (good economic indicator)
    ),
    AssetMapping(
        logical_name="lead",
        asset_type="commodities",
        data_symbol="MCX_LEAD",
        trading_symbol="LEAD",  # MCX Lead - 5 MT
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Disabled: Too expensive
    ),
    AssetMapping(
        logical_name="lead_mini",
        asset_type="commodities",
        data_symbol="MCX_LEADMINI",
        trading_symbol="LEADMINI",  # MCX Lead Mini - 1 MT (₹4k-6k margin)
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Enable when you have ₹50k+ budget
    ),
    AssetMapping(
        logical_name="nickel",
        asset_type="commodities",
        data_symbol="MCX_NICKEL",
        trading_symbol="NICKEL",  # MCX Nickel - 1 MT
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Disabled: Expensive & volatile
    ),
    AssetMapping(
        logical_name="zinc",
        asset_type="commodities",
        data_symbol="MCX_ZINC",
        trading_symbol="ZINC",  # MCX Zinc - 5 MT
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Disabled: Too expensive
    ),
    AssetMapping(
        logical_name="zinc_mini",
        asset_type="commodities",
        data_symbol="MCX_ZINCMINI",
        trading_symbol="ZINCMINI",  # MCX Zinc Mini - 1 MT (₹5k-8k margin)
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Enable when you have ₹50k+ budget
    ),
    
    # Agricultural
    AssetMapping(
        logical_name="corn",
        asset_type="commodities",
        data_symbol="ZC=F",
        trading_symbol="CORN",  # MCX Corn - Poor data quality for India
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Disabled: Low liquidity on MCX
    ),
    AssetMapping(
        logical_name="soybean",
        asset_type="commodities",
        data_symbol="ZS=F",
        trading_symbol="SOYBEAN",  # MCX Soybean - Poor data quality
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Disabled: Low liquidity on MCX
    ),
    AssetMapping(
        logical_name="wheat",
        asset_type="commodities",
        data_symbol="ZW=F",
        trading_symbol="WHEAT",  # MCX Wheat - Poor data quality
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Disabled: Low liquidity on MCX
    ),
    AssetMapping(
        logical_name="cardamom",
        asset_type="commodities",
        data_symbol="MCX_CARDAMOM",
        trading_symbol="CARDAMOM",  # MCX Cardamom - 100kg
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Disabled: Very low liquidity, wide spreads
    ),
    AssetMapping(
        logical_name="cotton",
        asset_type="commodities",
        data_symbol="CT=F",
        trading_symbol="COTTON",  # MCX Cotton
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Enable for agricultural diversification (₹2L+ budget)
    ),
    AssetMapping(
        logical_name="crude_palm_oil",
        asset_type="commodities",
        data_symbol="MCX_CPO",
        trading_symbol="CPO",  # MCX Crude Palm Oil - 10 MT
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Enable for agricultural diversification (₹1L+ budget)
    ),
    AssetMapping(
        logical_name="mentha_oil",
        asset_type="commodities",
        data_symbol="MCX_MENTHAOIL",
        trading_symbol="MENTHAOIL",  # MCX Mentha Oil - 360kg
        timeframe="1d",
        horizon_profile="short",
        enabled=False,  # ← Disabled: Very low liquidity
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



