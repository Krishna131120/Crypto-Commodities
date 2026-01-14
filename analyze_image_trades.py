"""
Analyze all trades from the provided trading screenshots.
Extracts P/L data and calculates totals per currency and overall.
"""

from collections import defaultdict
from datetime import datetime


def parse_amount(value_str):
    """Parse amount string like '+$1,234.56' or '-$1,234.56' to float."""
    value_str = value_str.replace('$', '').replace(',', '').replace('+', '')
    try:
        return float(value_str)
    except:
        return 0.0


def analyze_trades():
    """Analyze all trades from the image descriptions."""
    
    # Trades extracted from image descriptions (in descending chronological order)
    # Image 1 trades
    trades = [
        # Most recent (Image 1 - latest timestamps)
        {"action": "Sell", "symbol": "GRT/USD", "qty": 9524.231255468, "amount": 376.59, "time": "2026-01-06 23:30:08"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 46391.233, "amount": 1854.62, "time": "2026-01-06 23:30:08"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 15416.483, "amount": 621.73, "time": "2026-01-06 23:30:08"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 30950.17, "amount": 1244.12, "time": "2026-01-06 23:30:08"},
        {"action": "Sell", "symbol": "ETH/USD", "qty": 0.003428667, "amount": 11.07, "time": "2026-01-06 18:43:57"},
        {"action": "Sell", "symbol": "ETH/USD", "qty": 1.348577306, "amount": 4352.79, "time": "2026-01-06 18:43:57"},
        {"action": "Buy", "symbol": "ETH/USD", "qty": 1.354443973, "amount": -4385.13, "time": "2026-01-06 18:32:09"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 203.836201108, "amount": 1253.16, "time": "2026-01-06 18:32:07"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 503.73, "amount": 3106.43, "time": "2026-01-06 18:32:07"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 30581.6, "amount": -1368.53, "time": "2026-01-06 18:32:01"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 10189.814058173, "amount": -463.40, "time": "2026-01-06 18:32:01"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 15389.343, "amount": -686.99, "time": "2026-01-06 18:32:01"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 46305.8, "amount": -2083.30, "time": "2026-01-06 18:32:01"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 203.137116919, "amount": -1262.77, "time": "2026-01-06 18:30:48"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 505.705, "amount": -3136.28, "time": "2026-01-06 18:30:48"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 31000, "amount": 1317.29, "time": "2026-01-06 18:30:41"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 45966, "amount": 1942.19, "time": "2026-01-06 18:30:41"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 15468.898, "amount": 659.04, "time": "2026-01-06 18:30:41"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 10110.772525972, "amount": 422.08, "time": "2026-01-06 18:30:41"},
        {"action": "Sell", "symbol": "ETH/USD", "qty": 1.352806225, "amount": 4368.31, "time": "2026-01-06 18:30:36"},
        {"action": "Buy", "symbol": "ETH/USD", "qty": 1.355245668, "amount": -4390.42, "time": "2026-01-06 18:29:28"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 202.892653642, "amount": 1247.57, "time": "2026-01-06 18:29:25"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 505.01, "amount": 3115.58, "time": "2026-01-06 18:29:25"},
        {"action": "Sell", "symbol": "ETH/USD", "qty": 1.353155531, "amount": 4376.19, "time": "2026-01-06 18:27:52"},
        {"action": "Buy", "symbol": "ETH/USD", "qty": 1.355595604, "amount": -4393.17, "time": "2026-01-06 18:26:45"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 15439.6, "amount": -688.76, "time": "2026-01-06 18:26:35"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 46286.156, "amount": -2082.88, "time": "2026-01-06 18:26:35"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 30996.5, "amount": -1386.78, "time": "2026-01-06 18:26:35"},
    ]
    
    # Additional trades from other images (continuing in chronological order)
    additional_trades = [
        # Image 2 trades (18:25:15 - 18:18:27)
        {"action": "Sell", "symbol": "GRT/USD", "qty": 46402.5, "amount": 1959.75, "time": "2026-01-06 18:25:15"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 15378.0, "amount": 655.05, "time": "2026-01-06 18:25:15"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 10739.736092801, "amount": 448.36, "time": "2026-01-06 18:25:15"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 30797.95, "amount": 1308.22, "time": "2026-01-06 18:25:15"},
        {"action": "Sell", "symbol": "ETH/USD", "qty": 1.355345723, "amount": 4378.62, "time": "2026-01-06 18:25:07"},
        {"action": "Buy", "symbol": "ETH/USD", "qty": 1.357789745, "amount": -4396.92, "time": "2026-01-06 18:23:59"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 510.02, "amount": -3165.80, "time": "2026-01-06 18:23:55"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 199.15917616, "amount": -1239.60, "time": "2026-01-06 18:23:55"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 504.56, "amount": 3115.29, "time": "2026-01-06 18:22:37"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 203.462294339, "amount": 1253.10, "time": "2026-01-06 18:22:37"},
        {"action": "Sell", "symbol": "ETH/USD", "qty": 1.355647932, "amount": 4380.60, "time": "2026-01-06 18:22:23"},
        {"action": "Buy", "symbol": "ETH/USD", "qty": 1.358092499, "amount": -4400.90, "time": "2026-01-06 18:21:14"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 198.979032598, "amount": -1238.90, "time": "2026-01-06 18:21:11"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 510.32, "amount": -3168.67, "time": "2026-01-06 18:21:11"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 503.011, "amount": 3103.51, "time": "2026-01-06 18:19:52"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 208.337612705, "amount": 1281.87, "time": "2026-01-06 18:19:52"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 45826.3, "amount": -2050.87, "time": "2026-01-06 18:19:47"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 15485.344, "amount": -687.47, "time": "2026-01-06 18:19:47"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 11561.10918233, "amount": -523.02, "time": "2026-01-06 18:19:47"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 30631.741, "amount": -1363.73, "time": "2026-01-06 18:19:47"},
        {"action": "Sell", "symbol": "ETH/USD", "qty": 1.353969892, "amount": 4375.05, "time": "2026-01-06 18:19:42"},
        {"action": "Sell", "symbol": "ETH/USD", "qty": 0.004362545, "amount": 14.09, "time": "2026-01-06 18:19:42"},
        {"action": "Buy", "symbol": "ETH/USD", "qty": 1.360781845, "amount": -4411.08, "time": "2026-01-06 18:18:35"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 11871.202629541, "amount": 493.12, "time": "2026-01-06 18:18:27"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 45638.93, "amount": 1918.21, "time": "2026-01-06 18:18:27"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 15272.0, "amount": 647.52, "time": "2026-01-06 18:18:27"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 10008.329580017, "amount": -455.09, "time": "2026-01-06 18:26:35"},
        
        # More trades from other images (18:05 - 17:28 timeframe)
        {"action": "Sell", "symbol": "UNI/USD", "qty": 210.909147369, "amount": 1294.56, "time": "2026-01-06 18:05:23"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 14403.487297169, "amount": 591.35, "time": "2026-01-06 18:05:17"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 15420.34, "amount": 645.98, "time": "2026-01-06 18:05:17"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 30545.1, "amount": 1275.91, "time": "2026-01-06 18:05:17"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 45642.71, "amount": 1896.00, "time": "2026-01-06 18:05:17"},
        {"action": "Sell", "symbol": "ETH/USD", "qty": 1.379188375, "amount": 4455.43, "time": "2026-01-06 18:03:47"},
        {"action": "Buy", "symbol": "ETH/USD", "qty": 1.381675391, "amount": -4473.69, "time": "2026-01-06 18:02:39"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 220.93218931, "amount": -1371.73, "time": "2026-01-06 18:02:36"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 502.98, "amount": -3114.36, "time": "2026-01-06 18:02:36"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 15492.4, "amount": -681.25, "time": "2026-01-06 18:02:29"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 13292.332341384, "amount": -595.91, "time": "2026-01-06 18:02:29"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 30959.57, "amount": -1365.57, "time": "2026-01-06 18:02:29"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 46458.5, "amount": -2060.44, "time": "2026-01-06 18:02:29"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 505.1, "amount": 3109.34, "time": "2026-01-06 18:01:19"},
        
        # Trades with explicit P/L from image descriptions (17:58 - 17:44 timeframe)
        {"action": "Sell", "symbol": "ETH/USD", "qty": 1.361522043, "amount": 4403.53, "time": "2026-01-06 17:58:07"},
        {"action": "Buy", "symbol": "ETH/USD", "qty": 1.363977202, "amount": -4419.79, "time": "2026-01-06 17:56:59"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 15344.87, "amount": -678.55, "time": "2026-01-06 17:52:51"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 30697.314, "amount": -1361.65, "time": "2026-01-06 17:52:51"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 45596.05, "amount": -2032.88, "time": "2026-01-06 17:52:51"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 12685.979191402, "amount": -571.68, "time": "2026-01-06 17:52:51"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 12939.589315895, "amount": 535.45, "time": "2026-01-06 17:51:33"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 30401, "amount": 1280.43, "time": "2026-01-06 17:51:33"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 45664.21, "amount": 1911.58, "time": "2026-01-06 17:51:33"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 15221.79, "amount": 642.87, "time": "2026-01-06 17:51:33"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 203.439025932, "amount": -1266.78, "time": "2026-01-06 17:51:39"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 510.9, "amount": -3171.60, "time": "2026-01-06 17:51:39"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 506.1, "amount": 3131.66, "time": "2026-01-06 17:50:21"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 206.102501591, "amount": 1271.21, "time": "2026-01-06 17:50:21"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 15296.67, "amount": -676.72, "time": "2026-01-06 17:50:15"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 12113.099479759, "amount": -546.18, "time": "2026-01-06 17:50:15"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 30701.72, "amount": -1361.68, "time": "2026-01-06 17:50:15"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 46303.046, "amount": -2065.58, "time": "2026-01-06 17:50:15"},
        
        # More trades from 17:44 - 17:28 timeframe
        {"action": "Buy", "symbol": "GRT/USD", "qty": 30575.39, "amount": -1355.33, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 30569.548, "amount": 1286.60, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 45779.42, "amount": 1915.14, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 14265.485690794, "amount": 590.09, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 15198.799, "amount": 641.45, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "ETH/USD", "qty": 1.369128072, "amount": 4426.89, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "ETH/USD", "qty": 1.371596947, "amount": -4448.71, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "ETH/USD", "qty": 1.370874227, "amount": 4431.67, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "ETH/USD", "qty": 1.373346251, "amount": -4452.11, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "ETH/USD", "qty": 1.371533431, "amount": 4433.00, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "ETH/USD", "qty": 1.374006643, "amount": -4455.49, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 207.16374615, "amount": -1295.95, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 506.6, "amount": -3160.22, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 512.07, "amount": 3174.74, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 201.199578274, "amount": 1243.78, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 203.455778676, "amount": -1272.78, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 511.1, "amount": -3187.63, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 209.950303056, "amount": 1295.68, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 505.19, "amount": 3126.54, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 507.638, "amount": -3161.46, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 208.791876835, "amount": -1303.87, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 511.5, "amount": 3169.40, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 207.862366634, "amount": 1283.85, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 503.79, "amount": -3119.78, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 216.869553831, "amount": -1346.79, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 206.869308415, "amount": 1271.19, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 512.1, "amount": 3156.52, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 508.856, "amount": -3152.70, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 211.409786832, "amount": -1313.32, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 212.163284803, "amount": 1304.18, "time": "2026-01-06 17:44:51"},
        
        # Additional trades from other images
        {"action": "Sell", "symbol": "GRT/USD", "qty": 642.37, "amount": 642.37, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 1300.87, "amount": 1300.87, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 503.65, "amount": 503.65, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 1929.28, "amount": 1929.28, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 2055.71, "amount": -2055.71, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 681.02, "amount": -681.02, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 559.46, "amount": -559.46, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 1361.27, "amount": -1361.27, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 649.65, "amount": 649.65, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 1289.74, "amount": 1289.74, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 536.86, "amount": 536.86, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "GRT/USD", "qty": 1908.51, "amount": 1908.51, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 682.39, "amount": -682.39, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 2064.16, "amount": -2064.16, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "GRT/USD", "qty": 564.74, "amount": -564.74, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "ETH/USD", "qty": 11.11, "amount": 11.11, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "ETH/USD", "qty": 4399.41, "amount": 4399.41, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "ETH/USD", "qty": 4432.31, "amount": -4432.31, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "ETH/USD", "qty": 4422.50, "amount": 4422.50, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "ETH/USD", "qty": 4440.71, "amount": -4440.71, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 3164.96, "amount": -3164.96, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 1279.78, "amount": -1279.78, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 3122.07, "amount": 3122.07, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 1289.01, "amount": 1289.01, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 1309.73, "amount": -1309.73, "time": "2026-01-06 17:44:51"},
        {"action": "Buy", "symbol": "UNI/USD", "qty": 3144.31, "amount": -3144.31, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 1276.36, "amount": 1276.36, "time": "2026-01-06 17:44:51"},
        {"action": "Sell", "symbol": "UNI/USD", "qty": 3139.92, "amount": 3139.92, "time": "2026-01-06 17:44:51"},
    ]
    
    trades.extend(additional_trades)
    
    # Group by symbol
    by_symbol = defaultdict(lambda: {"buys": [], "sells": [], "total_pl": 0.0})
    
    for trade in trades:
        symbol = trade["symbol"]
        amount = trade["amount"]
        
        if trade["action"] == "Buy":
            by_symbol[symbol]["buys"].append(trade)
            by_symbol[symbol]["total_pl"] += amount  # Buy is negative
        else:  # Sell
            by_symbol[symbol]["sells"].append(trade)
            by_symbol[symbol]["total_pl"] += amount  # Sell is positive
    
    # Print results
    print("=" * 100)
    print("COMPREHENSIVE TRADE ANALYSIS FROM IMAGES")
    print("=" * 100)
    print()
    
    total_overall = 0.0
    
    for symbol in sorted(by_symbol.keys()):
        data = by_symbol[symbol]
        total_pl = data["total_pl"]
        total_overall += total_pl
        
        print(f"\n{symbol}:")
        print("-" * 100)
        print(f"  Total Buy Orders: {len(data['buys'])}")
        print(f"  Total Sell Orders: {len(data['sells'])}")
        print(f"  Total Transactions: {len(data['buys']) + len(data['sells'])}")
        print(f"  Total P/L: ${total_pl:,.2f}")
        
        # Calculate net cash flow
        total_buys = sum(t["amount"] for t in data["buys"])
        total_sells = sum(t["amount"] for t in data["sells"])
        net_cash = total_sells + total_buys  # buys are already negative
        
        print(f"  Total Cash Out (Buys): ${abs(total_buys):,.2f}")
        print(f"  Total Cash In (Sells): ${total_sells:,.2f}")
        print(f"  Net Cash Flow: ${net_cash:,.2f}")
    
    print("\n" + "=" * 100)
    print(f"OVERALL TOTAL PROFIT/LOSS: ${total_overall:,.2f}")
    print("=" * 100)
    
    return by_symbol, total_overall


if __name__ == "__main__":
    analyze_trades()
