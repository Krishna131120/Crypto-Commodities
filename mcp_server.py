"""
MCP (Model Context Protocol) Server for crypto/commodities prediction system.
Provides tools for accessing live predictions and bucket logs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ml.bucket_logger import get_bucket_logger
from ml.horizons import normalize_profile, DEFAULT_HORIZON_PROFILE
from core.model_paths import summary_path as build_summary_path, list_horizon_dirs, timeframe_dir


class MCPServer:
    """
    MCP Server providing tools for:
    - /feed/live: Show live prediction output from summary.json
    - /tools/fetch_logs: Pull summarized history from buckets
    """
    
    def __init__(self):
        self.bucket_logger = get_bucket_logger()
    
    def feed_live(
        self,
        symbol: str,
        asset_type: str = "crypto",
        timeframe: str = "1d",
        horizon: str = "short",
    ) -> Dict[str, Any]:
        """
        Get live prediction feed from summary.json.
        Shows the same output format as seen in summary.json.
        
        Args:
            symbol: Trading symbol (e.g., BTC-USDT)
            asset_type: Asset type (crypto or commodities)
            timeframe: Timeframe (e.g., 1d)
        
        Returns:
            Dictionary with live prediction data from summary.json
        """
        normalized_horizon = normalize_profile(horizon or DEFAULT_HORIZON_PROFILE)
        summary_path = build_summary_path(asset_type, symbol, timeframe, normalized_horizon)
        if not summary_path.exists():
            # Fallback to legacy summary if horizon-specific file absent
            legacy_dirs = list_horizon_dirs(asset_type, symbol, timeframe)
            if legacy_dirs:
                summary_path = legacy_dirs[0] / "summary.json"
        
        if not summary_path.exists():
            return {
                "error": "Model not trained",
                "message": f"No summary.json found for {asset_type}/{symbol}/{timeframe} ({normalized_horizon})",
                "path": str(summary_path)
            }
        
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            
            # Extract key information in the format shown in summary.json
            result = {
                "symbol": summary.get("symbol"),
                "asset_type": summary.get("asset_type"),
                "timeframe": summary.get("timeframe"),
                "last_updated": summary.get("last_updated"),
                
                # Main prediction (new format)
                "prediction": summary.get("prediction", {}),
                
                # Individual model predictions
                "model_predictions": summary.get("model_predictions", {}),
                
                # Consensus details
                "consensus": summary.get("consensus", {}),
                
                # Technical details (optional, for advanced users)
                "technical": summary.get("technical", {})
            }
            
            return result
        
        except Exception as exc:
            return {
                "error": "Failed to read summary",
                "message": str(exc),
                "path": str(summary_path)
            }
    
    def fetch_logs(
        self,
        symbol: Optional[str] = None,
        asset_type: Optional[str] = None,
        start_timestamp: Optional[str] = None,
        end_timestamp: Optional[str] = None,
        limit: Optional[int] = 100,
        event_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fetch summarized history from bucket logs for dashboard view.
        
        Args:
            symbol: Filter by symbol (optional)
            asset_type: Filter by asset type (optional)
            start_timestamp: Start timestamp filter (ISO format, optional)
            end_timestamp: End timestamp filter (ISO format, optional)
            limit: Maximum number of entries per type (default: 100)
            event_types: List of event types to include (predictions, trades, feedback)
                        If None, includes all types
        
        Returns:
            Dictionary with summarized history
        """
        if event_types is None:
            event_types = ["predictions", "trades", "feedback"]
        
        summary = self.bucket_logger.get_summary(
            symbol=symbol,
            asset_type=asset_type,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            limit=limit
        )
        
        # Filter by event types if specified
        filtered_summary = {
            "index": summary["index"],
            "filters": {
                "symbol": symbol,
                "asset_type": asset_type,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "limit": limit,
                "event_types": event_types
            }
        }
        
        if "predictions" in event_types:
            filtered_summary["predictions"] = summary["predictions"]
        if "trades" in event_types:
            filtered_summary["trades"] = summary["trades"]
        if "feedback" in event_types:
            filtered_summary["feedback"] = summary["feedback"]
        
        # Add summary statistics
        filtered_summary["statistics"] = {
            "total_predictions": len(filtered_summary.get("predictions", [])),
            "total_trades": len(filtered_summary.get("trades", [])),
            "total_feedback": len(filtered_summary.get("feedback", []))
        }
        
        return filtered_summary
    
    def get_available_symbols(self, asset_type: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get list of available symbols with trained models.
        
        Args:
            asset_type: Filter by asset type (crypto or commodities, optional)
        
        Returns:
            List of dictionaries with symbol information
        """
        models_dir = Path("models")
        symbols = []
        
        if not models_dir.exists():
            return symbols
        
        for asset_dir in models_dir.iterdir():
            if not asset_dir.is_dir():
                continue
            
            current_asset_type = asset_dir.name
            if asset_type and current_asset_type != asset_type:
                continue
            
            for symbol_dir in asset_dir.iterdir():
                if not symbol_dir.is_dir():
                    continue
                
                symbol = symbol_dir.name
                
                # Check for summary.json in any timeframe
                for timeframe_dir in symbol_dir.iterdir():
                    if not timeframe_dir.is_dir():
                        continue
                    
                    timeframe = timeframe_dir.name
                    horizon_dirs = list_horizon_dirs(current_asset_type, symbol, timeframe)
                    if horizon_dirs:
                        symbols.append({
                            "symbol": symbol,
                            "asset_type": current_asset_type,
                            "timeframe": timeframe,
                            "horizons": [d.name if d.parent.name == timeframe else "legacy" for d in horizon_dirs],
                        })
                        break
        
        return symbols


# Global MCP server instance
_mcp_server: Optional[MCPServer] = None


def get_mcp_server() -> MCPServer:
    """Get or create global MCP server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServer()
    return _mcp_server


# Tool functions for external use
def tool_feed_live(symbol: str, asset_type: str = "crypto", timeframe: str = "1d", horizon: str = "short") -> Dict[str, Any]:
    """
    MCP Tool: /feed/live
    Get live prediction feed from summary.json.
    """
    server = get_mcp_server()
    return server.feed_live(symbol=symbol, asset_type=asset_type, timeframe=timeframe, horizon=horizon)


def tool_fetch_logs(
    symbol: Optional[str] = None,
    asset_type: Optional[str] = None,
    start_timestamp: Optional[str] = None,
    end_timestamp: Optional[str] = None,
    limit: Optional[int] = 100,
    event_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    MCP Tool: /tools/fetch_logs
    Fetch summarized history from bucket logs.
    """
    server = get_mcp_server()
    return server.fetch_logs(
        symbol=symbol,
        asset_type=asset_type,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        limit=limit,
        event_types=event_types
    )


def tool_get_available_symbols(asset_type: Optional[str] = None) -> List[Dict[str, str]]:
    """
    MCP Tool: Get list of available symbols with trained models.
    """
    server = get_mcp_server()
    return server.get_available_symbols(asset_type=asset_type)


# CLI interface for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python mcp_server.py feed_live <symbol> [asset_type] [timeframe]")
        print("  python mcp_server.py fetch_logs [symbol] [asset_type] [limit]")
        print("  python mcp_server.py list_symbols [asset_type]")
        sys.exit(1)
    
    command = sys.argv[1]
    server = get_mcp_server()
    
    if command == "feed_live":
        symbol = sys.argv[2] if len(sys.argv) > 2 else "BTC-USDT"
        asset_type = sys.argv[3] if len(sys.argv) > 3 else "crypto"
        timeframe = sys.argv[4] if len(sys.argv) > 4 else "1d"
        
        result = server.feed_live(symbol=symbol, asset_type=asset_type, timeframe=timeframe)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif command == "fetch_logs":
        symbol = sys.argv[2] if len(sys.argv) > 2 else None
        asset_type = sys.argv[3] if len(sys.argv) > 3 else None
        limit = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 100
        
        result = server.fetch_logs(symbol=symbol, asset_type=asset_type, limit=limit)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif command == "list_symbols":
        asset_type = sys.argv[2] if len(sys.argv) > 2 else None
        symbols = server.get_available_symbols(asset_type=asset_type)
        print(json.dumps(symbols, indent=2, ensure_ascii=False))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

