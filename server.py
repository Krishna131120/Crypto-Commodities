"""
Simple HTTP server for MCP tools.
Provides REST API endpoints for /feed/live and /tools/fetch_logs.
"""
from __future__ import annotations

import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any

from mcp_server import tool_feed_live, tool_fetch_logs, tool_get_available_symbols


class MCPRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for MCP tools."""
    
    def _send_response(self, status_code: int, data: Dict[str, Any], content_type: str = "application/json"):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        
        response = json.dumps(data, indent=2, ensure_ascii=False)
        self.wfile.write(response.encode("utf-8"))
    
    def _send_error(self, status_code: int, message: str):
        """Send error response."""
        self._send_response(status_code, {"error": True, "message": message})
    
    def _parse_query_params(self, path: str) -> Dict[str, str]:
        """Parse query parameters from URL."""
        parsed = urllib.parse.urlparse(path)
        return dict(urllib.parse.parse_qsl(parsed.query))
    
    def _read_json_body(self) -> Dict[str, Any]:
        """Read and parse JSON request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        
        body = self.rfile.read(content_length)
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests."""
        path = self.path.split("?")[0]
        query_params = self._parse_query_params(self.path)
        
        try:
            if path == "/feed/live":
                # GET /feed/live?symbol=BTC-USDT&asset_type=crypto&timeframe=1d
                symbol = query_params.get("symbol", "BTC-USDT")
                asset_type = query_params.get("asset_type", "crypto")
                timeframe = query_params.get("timeframe", "1d")
                
                result = tool_feed_live(symbol=symbol, asset_type=asset_type, timeframe=timeframe)
                self._send_response(200, result)
            
            elif path == "/tools/fetch_logs":
                # GET /tools/fetch_logs?symbol=BTC-USDT&asset_type=crypto&limit=100
                symbol = query_params.get("symbol")
                asset_type = query_params.get("asset_type")
                start_timestamp = query_params.get("start_timestamp")
                end_timestamp = query_params.get("end_timestamp")
                limit = int(query_params.get("limit", 100)) if query_params.get("limit") else None
                event_types = query_params.get("event_types", "").split(",") if query_params.get("event_types") else None
                if event_types:
                    event_types = [e.strip() for e in event_types if e.strip()]
                
                result = tool_fetch_logs(
                    symbol=symbol,
                    asset_type=asset_type,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    limit=limit,
                    event_types=event_types
                )
                self._send_response(200, result)
            
            elif path == "/symbols":
                # GET /symbols?asset_type=crypto
                asset_type = query_params.get("asset_type")
                symbols = tool_get_available_symbols(asset_type=asset_type)
                self._send_response(200, {"symbols": symbols, "count": len(symbols)})
            
            elif path == "/tools/health":
                # GET /tools/health - Service + resource usage
                try:
                    import psutil
                    import os
                    from datetime import datetime, timezone
                    
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    cpu_percent = process.cpu_percent(interval=0.1)
                    
                    system_memory = psutil.virtual_memory()
                    try:
                        disk_usage = psutil.disk_usage("/")
                    except:
                        import platform
                        if platform.system() == "Windows":
                            disk_usage = psutil.disk_usage("C:\\")
                        else:
                            disk_usage = psutil.disk_usage("/")
                    
                    health_data = {
                        "status": "running",
                        "service": "MCP Server",
                        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "resources": {
                            "process": {
                                "memory_mb": round(memory_info.rss / 1024 / 1024, 2),
                                "cpu_percent": round(cpu_percent, 2),
                                "threads": process.num_threads()
                            },
                            "system": {
                                "memory_total_gb": round(system_memory.total / 1024 / 1024 / 1024, 2),
                                "memory_available_gb": round(system_memory.available / 1024 / 1024 / 1024, 2),
                                "memory_percent": round(system_memory.percent, 2),
                                "disk_total_gb": round(disk_usage.total / 1024 / 1024 / 1024, 2),
                                "disk_free_gb": round(disk_usage.free / 1024 / 1024 / 1024, 2),
                                "disk_percent": round(disk_usage.percent, 2)
                            }
                        },
                        "endpoints": {
                            "GET /feed/live": "Get live prediction feed",
                            "GET/POST /tools/fetch_logs": "Fetch bucket logs",
                            "GET /symbols": "List available symbols",
                            "GET /tools/health": "Health check with resource usage",
                            "GET /health": "Simple health check"
                        }
                    }
                    self._send_response(200, health_data)
                except ImportError:
                    # Fallback if psutil not available
                    self._send_response(200, {
                        "status": "running",
                        "service": "MCP Server",
                        "note": "Resource usage not available (psutil not installed)",
                        "endpoints": {
                            "/feed/live": "Get live prediction feed",
                            "/tools/fetch_logs": "Fetch bucket logs",
                            "/symbols": "List available symbols",
                            "/tools/health": "Health check",
                            "/health": "Simple health check"
                        }
                    })
            
            elif path == "/" or path == "/health":
                # Simple health check endpoint
                self._send_response(200, {
                    "status": "running",
                    "service": "MCP Server",
                    "endpoints": {
                        "/feed/live": "Get live prediction feed",
                        "/tools/fetch_logs": "Fetch bucket logs",
                        "/symbols": "List available symbols",
                        "/tools/health": "Health check with resource usage",
                        "/health": "Simple health check"
                    }
                })
            
            else:
                self._send_error(404, f"Endpoint not found: {path}")
        
        except Exception as exc:
            self._send_error(500, f"Internal server error: {str(exc)}")
    
    def do_POST(self):
        """Handle POST requests."""
        path = self.path.split("?")[0]
        body = self._read_json_body()
        
        try:
            # Import adapter for new endpoints
            from core.mcp_adapter import get_mcp_adapter
            adapter = get_mcp_adapter()
            
            if path == "/tools/predict":
                # POST /tools/predict
                symbols = body.get("symbols", [])
                if not symbols:
                    self._send_error(400, "symbols array is required")
                    return
                
                horizon = body.get("horizon", "long")  # Default to "long"
                risk_profile = body.get("risk_profile")
                asset_type = body.get("asset_type", "crypto")
                timeframe = body.get("timeframe", "1d")
                
                result = adapter.predict(
                    symbols=symbols,
                    horizon=horizon,
                    risk_profile=risk_profile,
                    asset_type=asset_type,
                    timeframe=timeframe
                )
                self._send_response(200, result)
            
            elif path == "/tools/scan_all":
                # POST /tools/scan_all
                asset_type = body.get("asset_type", "crypto")
                timeframe = body.get("timeframe", "1d")
                min_confidence = body.get("min_confidence", 0.5)
                limit = body.get("limit", 50)
                
                result = adapter.scan_all(
                    asset_type=asset_type,
                    timeframe=timeframe,
                    min_confidence=min_confidence,
                    limit=limit
                )
                self._send_response(200, result)
            
            elif path == "/tools/analyze":
                # POST /tools/analyze
                tickers = body.get("tickers", [])
                if not tickers:
                    self._send_error(400, "tickers array is required")
                    return
                
                asset_type = body.get("asset_type", "crypto")
                timeframe = body.get("timeframe", "1d")
                
                result = adapter.analyze(
                    tickers=tickers,
                    asset_type=asset_type,
                    timeframe=timeframe
                )
                self._send_response(200, result)
            
            elif path == "/tools/feedback":
                # POST /tools/feedback
                symbol = body.get("symbol")
                asset_type = body.get("asset_type", "crypto")
                timeframe = body.get("timeframe", "1d")
                prediction_timestamp = body.get("prediction_timestamp")
                action_taken = body.get("action_taken")
                predicted_return = body.get("predicted_return")
                predicted_price = body.get("predicted_price")
                actual_price = body.get("actual_price")
                actual_return = body.get("actual_return")
                features = body.get("features")
                
                if not all([symbol, prediction_timestamp, action_taken, predicted_return is not None, predicted_price]):
                    self._send_error(400, "Missing required fields: symbol, prediction_timestamp, action_taken, predicted_return, predicted_price")
                    return
                
                result = adapter.add_feedback(
                    symbol=symbol,
                    asset_type=asset_type,
                    timeframe=timeframe,
                    prediction_timestamp=prediction_timestamp,
                    action_taken=action_taken,
                    predicted_return=predicted_return,
                    predicted_price=predicted_price,
                    actual_price=actual_price,
                    actual_return=actual_return,
                    features=features
                )
                self._send_response(200, result)
            
            elif path == "/feed/live":
                # POST /feed/live with JSON body
                symbol = body.get("symbol", "BTC-USDT")
                asset_type = body.get("asset_type", "crypto")
                timeframe = body.get("timeframe", "1d")
                
                result = tool_feed_live(symbol=symbol, asset_type=asset_type, timeframe=timeframe)
                self._send_response(200, result)
            
            elif path == "/tools/fetch_logs":
                # POST /tools/fetch_logs with JSON body
                symbol = body.get("symbol")
                asset_type = body.get("asset_type")
                start_timestamp = body.get("start_timestamp")
                end_timestamp = body.get("end_timestamp")
                limit = body.get("limit", 100)
                event_types = body.get("event_types")
                
                result = tool_fetch_logs(
                    symbol=symbol,
                    asset_type=asset_type,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    limit=limit,
                    event_types=event_types
                )
                self._send_response(200, result)
            
            else:
                self._send_error(404, f"Endpoint not found: {path}")
        
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._send_error(500, f"Internal server error: {str(exc)}")
    
    def log_message(self, format, *args):
        """Override to customize log format."""
        print(f"[SERVER] {args[0]}")


def run_server(host: str = "localhost", port: int = 8000):
    """Run the MCP HTTP server."""
    server_address = (host, port)
    
    try:
        httpd = HTTPServer(server_address, MCPRequestHandler)
    except OSError as e:
        if "Address already in use" in str(e) or "Only one usage" in str(e):
            print(f"ERROR: Port {port} is already in use!")
            print(f"Please use a different port or stop the process using port {port}")
            print(f"\nTo use a different port: python server.py <port>")
            return
        raise
    
    print("=" * 80)
    print("MCP Server Starting")
    print("=" * 80)
    print(f"Server running on http://{host}:{port}")
    print()
    print("Available endpoints:")
    print(f"  GET/POST  http://{host}:{port}/feed/live")
    print(f"            Query params: symbol, asset_type, timeframe")
    print(f"            Example: http://{host}:{port}/feed/live?symbol=BTC-USDT&asset_type=crypto&timeframe=1d")
    print()
    print(f"  GET/POST  http://{host}:{port}/tools/fetch_logs")
    print(f"            Query params: symbol, asset_type, start_timestamp, end_timestamp, limit, event_types")
    print(f"            Example: http://{host}:{port}/tools/fetch_logs?symbol=BTC-USDT&limit=50")
    print()
    print(f"  GET       http://{host}:{port}/symbols")
    print(f"            Query params: asset_type (optional)")
    print(f"            Example: http://{host}:{port}/symbols?asset_type=crypto")
    print()
    print(f"  GET       http://{host}:{port}/health")
    print(f"            Health check endpoint")
    print()
    print("Postman Collection:")
    print("  Import MCP_Server.postman_collection.json into Postman to test all endpoints")
    print()
    print("Example curl commands:")
    print(f"  curl http://{host}:{port}/feed/live?symbol=BTC-USDT&asset_type=crypto&timeframe=1d")
    print(f"  curl http://{host}:{port}/tools/fetch_logs?limit=10")
    print(f"  curl http://{host}:{port}/symbols")
    print()
    print("=" * 80)
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    print()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Server stopped")
        print("=" * 80)
        httpd.server_close()


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    host = "localhost"
    port = 8000
    
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    if len(sys.argv) > 2:
        host = sys.argv[2]
    
    run_server(host=host, port=port)

