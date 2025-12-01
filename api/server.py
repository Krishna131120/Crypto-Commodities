"""
API Server for MCP tools.
Provides REST endpoints for predictions, scanning, analysis, and health checks.
"""
from __future__ import annotations

import json
import os
import psutil
import sys
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.mcp_adapter import get_mcp_adapter


class APIRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for API endpoints."""
    
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
        print(f"[API] Response: {status_code} - {len(response)} bytes")
    
    def _send_error(self, status_code: int, message: str):
        """Send error response."""
        self._send_response(status_code, {"error": True, "message": message})
    
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
        
        try:
            if path == "/tools/health":
                # GET /tools/health - Service + resource usage
                adapter = get_mcp_adapter()
                
                # Get system resources
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent(interval=0.1)
                
                system_memory = psutil.virtual_memory()
                try:
                    disk_usage = psutil.disk_usage("/")
                except:
                    # Windows might use different path
                    import platform
                    if platform.system() == "Windows":
                        disk_usage = psutil.disk_usage("C:\\")
                    else:
                        disk_usage = psutil.disk_usage("/")
                
                health_data = {
                    "status": "running",
                    "service": "MCP API Server",
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
                        "POST /tools/predict": "Get predictions for symbols (crypto/commodities, single/multiple, all horizons)",
                        "POST /tools/feedback": "Submit feedback for RL learning",
                        "GET /tools/health": "Health check with resource usage"
                    }
                }
                
                self._send_response(200, health_data)
            
            elif path == "/" or path == "/health":
                # Redirect to /tools/health - call the same health check logic
                adapter = get_mcp_adapter()
                
                # Get system resources
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent(interval=0.1)
                
                system_memory = psutil.virtual_memory()
                try:
                    disk_usage = psutil.disk_usage("/")
                except:
                    # Windows might use different path
                    import platform
                    if platform.system() == "Windows":
                        disk_usage = psutil.disk_usage("C:\\")
                    else:
                        disk_usage = psutil.disk_usage("/")
                
                health_data = {
                    "status": "running",
                    "service": "MCP API Server",
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
                        "POST /tools/predict": "Get predictions for symbols (crypto/commodities, single/multiple, all horizons)",
                        "POST /tools/feedback": "Submit feedback for RL learning",
                        "GET /tools/health": "Health check with resource usage"
                    }
                }
                
                self._send_response(200, health_data)
            
            else:
                self._send_error(404, f"Endpoint not found: {path}")
        
        except Exception as exc:
            self._send_error(500, f"Internal server error: {str(exc)}")
    
    def do_POST(self):
        """Handle POST requests."""
        path = self.path.split("?")[0]
        body = self._read_json_body()
        
        adapter = get_mcp_adapter()
        
        try:
            if path == "/tools/predict":
                # POST /tools/predict
                # Body: {symbols: [], horizon: int, risk_profile: str}
                symbols = body.get("symbols", [])
                if not symbols:
                    self._send_error(400, "symbols array is required")
                    return
                
                horizon = body.get("horizon")
                risk_profile = body.get("risk_profile")
                asset_type = body.get("asset_type", "crypto")
                timeframe = body.get("timeframe", "1d")
                
                print(f"[API] Predict request: {len(symbols)} symbols, horizon={horizon}, risk_profile={risk_profile}")
                
                result = adapter.predict(
                    symbols=symbols,
                    horizon=horizon,
                    risk_profile=risk_profile,
                    asset_type=asset_type,
                    timeframe=timeframe
                )
                
                print(f"[API] Predict result: {result['count']} predictions, {len(result.get('errors', []))} errors")
                self._send_response(200, result)
            
            elif path == "/tools/scan_all":
                # POST /tools/scan_all
                # Body: {asset_type: str, timeframe: str, min_confidence: float, limit: int}
                asset_type = body.get("asset_type", "crypto")
                timeframe = body.get("timeframe", "1d")
                min_confidence = body.get("min_confidence", 0.5)
                limit = body.get("limit", 50)
                horizon = body.get("horizon")
                
                print(f"[API] Scan_all request: {asset_type}/{timeframe}, horizon={horizon}, min_confidence={min_confidence}, limit={limit}")
                
                result = adapter.scan_all(
                    asset_type=asset_type,
                    timeframe=timeframe,
                    min_confidence=min_confidence,
                    limit=limit,
                    horizon=horizon,
                )
                
                print(f"[API] Scan_all result: {result['count']} symbols in shortlist")
                self._send_response(200, result)
            
            elif path == "/tools/analyze":
                # POST /tools/analyze
                # Body: {tickers: [], asset_type: str, timeframe: str}
                tickers = body.get("tickers", [])
                if not tickers:
                    self._send_error(400, "tickers array is required")
                    return
                
                asset_type = body.get("asset_type", "crypto")
                timeframe = body.get("timeframe", "1d")
                
                print(f"[API] Analyze request: {len(tickers)} tickers, {asset_type}/{timeframe}")
                
                result = adapter.analyze(
                    tickers=tickers,
                    asset_type=asset_type,
                    timeframe=timeframe
                )
                
                print(f"[API] Analyze result: {result['count']} results")
                self._send_response(200, result)
            
            elif path == "/tools/feedback":
                # POST /tools/feedback - Add feedback for RL learning
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
                
                print(f"[API] Feedback request: {symbol}, action={action_taken}")
                
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
                
                print(f"[API] Feedback result: success={result.get('success')}")
                self._send_response(200, result)
            
            else:
                self._send_error(404, f"Endpoint not found: {path}")
        
        except Exception as exc:
            import traceback
            error_msg = str(exc)
            traceback.print_exc()
            self._send_error(500, f"Internal server error: {error_msg}")
    
    def log_message(self, format, *args):
        """Override to customize log format."""
        print(f"[SERVER] {args[0]}")


def run_server(host: str = "localhost", port: int = 8000):
    """Run the API server."""
    server_address = (host, port)
    
    try:
        httpd = HTTPServer(server_address, APIRequestHandler)
    except OSError as e:
        if "Address already in use" in str(e) or "Only one usage" in str(e):
            print(f"ERROR: Port {port} is already in use!")
            print(f"Please use a different port or stop the process using port {port}")
            print(f"\nTo use a different port: python api/server.py <port>")
            return
        raise
    
    print("=" * 80)
    print("MCP API Server Starting")
    print("=" * 80)
    print(f"Server running on http://{host}:{port}")
    print()
    print("Available endpoints:")
    print(f"  POST  http://{host}:{port}/tools/predict")
    print(f"        Body: {{symbols: [], horizon: str, risk_profile: str, asset_type: str, timeframe: str}}")
    print(f"        Returns: predictions array")
    print(f"        Supports: crypto and commodities (single or multiple symbols)")
    print(f"        Horizons: 'long', 'short', 'intraday'")
    print()
    print(f"  GET   http://{host}:{port}/tools/health")
    print(f"        Returns: service status + resource usage")
    print()
    print(f"  POST  http://{host}:{port}/tools/feedback")
    print(f"        Body: {{symbol, prediction_timestamp, action_taken, predicted_return, ...}}")
    print(f"        Returns: feedback submission status")
    print()
    print("Postman Collection:")
    print("  Import MCP_API_Server.postman_collection.json into Postman")
    print("  Collection includes: Crypto/Commodities single/multiple symbols with all horizons + Feedback")
    print()
    print("Example requests:")
    print(f'  # Crypto single symbol, long horizon')
    print(f'  curl -X POST http://{host}:{port}/tools/predict -H "Content-Type: application/json" -d \'{{"symbols": ["BTC-USDT"], "horizon": "long", "asset_type": "crypto"}}\'')
    print(f'  # Commodities multiple symbols, short horizon')
    print(f'  curl -X POST http://{host}:{port}/tools/predict -H "Content-Type: application/json" -d \'{{"symbols": ["GC=F", "SI=F"], "horizon": "short", "asset_type": "commodities"}}\'')
    print(f'  curl http://{host}:{port}/tools/health')
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

