#!/usr/bin/env python3
"""
Quant Sim API Server

Standalone server for the Quant Sim API.
Run with: python api_server.py

The server exposes RESTful endpoints for external tools
(like iOS Scriptable widgets) to consume portfolio data.
"""

from src.api.config import APIConfig
from src.api.routes import create_app


def main():
    """Start the API server."""
    # Load configuration
    config = APIConfig.from_yaml()
    
    # Create Flask app
    app = create_app(config)
    
    print(f"=== Quant Sim API Server ===")
    print(f"Starting server on {config.host}:{config.port}")
    print(f"API Version: {config.version}")
    print(f"Auth Enabled: {config.auth_enabled}")
    print(f"")
    print(f"Available endpoints:")
    print(f"  GET  /api/{config.version}/health       - Health check")
    print(f"  GET  /api/{config.version}/summary      - Portfolio summary")
    print(f"  GET  /api/{config.version}/portfolio    - Full portfolio")
    print(f"  GET  /api/{config.version}/positions    - Position list")
    print(f"  GET  /api/{config.version}/watchlist    - Watchlist")
    print(f"  GET  /api/{config.version}/signals      - Active signals")
    print(f"  GET  /api/{config.version}/trades/recent - Recent trades")
    print(f"  GET  /api/{config.version}/risk         - Risk metrics")
    print(f"  GET  /api/{config.version}/overview     - Dashboard overview")
    print(f"  POST /api/{config.version}/auth/token   - Generate token")
    print(f"")
    print(f"Authentication:")
    print(f"  Include header: X-API-Token: <your_token>")
    print(f"  Get token: POST /api/{config.version}/auth/token with username/password")
    
    # Run the server
    app.run(
        host=config.host,
        port=config.port,
        debug=config.debug,
    )


if __name__ == "__main__":
    main()