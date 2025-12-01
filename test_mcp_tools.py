"""
Test script for MCP tools.
Tests /feed/live and /tools/fetch_logs functionality.
"""
import json
from mcp_server import tool_feed_live, tool_fetch_logs, tool_get_available_symbols


def test_feed_live():
    """Test /feed/live tool."""
    print("=" * 80)
    print("Testing /feed/live tool")
    print("=" * 80)
    
    # Test with BTC-USDT
    result = tool_feed_live(symbol="BTC-USDT", asset_type="crypto", timeframe="1d")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()


def test_fetch_logs():
    """Test /tools/fetch_logs tool."""
    print("=" * 80)
    print("Testing /tools/fetch_logs tool")
    print("=" * 80)
    
    # Fetch recent logs
    result = tool_fetch_logs(limit=10)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()


def test_list_symbols():
    """Test listing available symbols."""
    print("=" * 80)
    print("Testing list_symbols")
    print("=" * 80)
    
    symbols = tool_get_available_symbols()
    print(f"Found {len(symbols)} symbols with trained models:")
    for sym in symbols:
        print(f"  - {sym['asset_type']}/{sym['symbol']}/{sym['timeframe']}")
    print()


if __name__ == "__main__":
    print("\nMCP Tools Test Suite\n")
    
    # Test listing symbols first
    test_list_symbols()
    
    # Test feed_live
    try:
        test_feed_live()
    except Exception as e:
        print(f"Error testing feed_live: {e}\n")
    
    # Test fetch_logs
    try:
        test_fetch_logs()
    except Exception as e:
        print(f"Error testing fetch_logs: {e}\n")
    
    print("=" * 80)
    print("Test complete")
    print("=" * 80)

