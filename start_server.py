"""
Startup script for MCP API Server.
Runs the server and shows output automatically.

IMPORTANT: Use this script or 'python api/server.py' to start the server.
Do NOT use the old 'server.py' in the root directory.
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from api.server import run_server

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    host = "localhost"
    port = 8000
    
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"ERROR: Invalid port number: {sys.argv[1]}")
            print("Usage: python start_server.py [port] [host]")
            sys.exit(1)
    if len(sys.argv) > 2:
        host = sys.argv[2]
    
    print("\n" + "=" * 80)
    print("Starting MCP API Server...")
    print("=" * 80)
    print(f"IMPORTANT: Make sure you're using api/server.py, not the old server.py")
    print()
    
    run_server(host=host, port=port)

