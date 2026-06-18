# Quant Sim API Documentation

A RESTful API for consuming portfolio data, signals, and analytics from Quant Sim.
Designed for lightweight consumption by mobile widgets (iOS Scriptable) and external tools.

All API responses include:
- `timestamp` (UTC ISO timestamp of response creation)
- `updatedAt` (same format, client-friendly field for widgets)

## Quick Start

### 1. Install Dependencies

```bash
pip install flask  # Already in requirements.txt
```

### 2. Configure the API

Edit `config/settings.yaml` to configure API settings:

```yaml
api:
  host: "0.0.0.0"
  port: 8080
  debug: false
  auth_enabled: true
  rate_limit_enabled: true
  rate_limit_requests: 60
  rate_limit_window: 60
```

### 3. Start the Server

```bash
python api_server.py
```

### 4. Get an API Token (if auth enabled)

```bash
curl -X POST http://localhost:8080/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "your_user", "password": "your_password"}'
```

### 5. Make API Requests

```bash
curl http://localhost:8080/api/v1/summary \
  -H "X-API-Token: your_token_here"
```

---

## Authentication

When `auth_enabled: true`, include your API token in the `X-API-Token` header:

```
X-API-Token: <your_session_token>
```

To get a token, use the `/api/v1/auth/token` endpoint with your username and password.

**Note:** Tokens are session-based and expire after 24 hours.

---

## Endpoints

### Health Check

#### `GET /api/v1/health`

Check API health and version. No authentication required.

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-05-22T20:00:00Z",
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "api_version": "v1"
  }
}
```

---

### Portfolio Summary

#### `GET /api/v1/summary`

Returns a high-level portfolio summary including total value, P&L, and key metrics.

**Authentication:** Required when `auth_enabled: true`; optional only if auth is disabled (uses default user if disabled)

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-05-22T20:00:00Z",
  "data": {
    "total_value": 125000.00,
    "total_cost": 100000.00,
    "total_pnl": 25000.00,
    "total_pnl_percent": 25.0,
    "positions_count": 12,
    "open_trades_count": 3,
    "last_updated": "2026-05-22T19:30:00Z"
  }
}
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| total_value | float | Current total portfolio value |
| total_cost | float | Total cost basis |
| total_pnl | float | Total profit/loss in currency |
| total_pnl_percent | float | Total P&L as percentage |
| positions_count | int | Number of portfolio positions |
| open_trades_count | int | Number of open swing trades |
| last_updated | string | ISO timestamp of last update |

---

### Full Portfolio

#### `GET /api/v1/portfolio`

Returns the full portfolio with all positions and their current values.

**Authentication:** Required when `auth_enabled: true`; optional only if auth is disabled

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-05-22T20:00:00Z",
  "data": {
    "name": "default",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-05-22T19:30:00Z",
    "positions": [
      {
        "ticker": "AAPL",
        "shares": 50,
        "cost_basis": 150.00,
        "price": 175.00,
        "market_value": 8750.00,
        "cost_value": 7500.00,
        "pnl": 1250.00,
        "pnl_percent": 16.67,
        "current_weight": 7.0,
        "target_weight": 10.0
      }
    ],
    "total_value": 125000.00,
    "total_pnl": 25000.00
  }
}
```

---

### Positions List

#### `GET /api/v1/positions`

Returns a list of current portfolio positions with detailed info.

**Authentication:** Required when `auth_enabled: true`; optional only if auth is disabled

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-05-22T20:00:00Z",
  "data": [
    {
      "ticker": "AAPL",
      "shares": 50,
      "cost_basis": 150.00,
      "target_weight": 10.0,
      "current_weight": 7.0,
      "price": 175.00,
      "market_value": 8750.00,
      "pnl": 1250.00,
      "pnl_percent": 16.67
    }
  ]
}
```

---

### Watchlist

#### `GET /api/v1/watchlist`

Returns the user's watchlist with current prices and changes.
If no watchlist is configured, returns a default set of popular tickers.

**Authentication:** Required when `auth_enabled: true`; optional only if auth is disabled

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-05-22T20:00:00Z",
  "data": [
    {
      "ticker": "SPY",
      "price": 450.00,
      "change": 2.50,
      "change_percent": 0.56,
      "volume": 75000000
    }
  ]
}
```

---

### Signals & Alerts

#### `GET /api/v1/signals`

Returns active alerts and signals from the swing tracker.

**Authentication:** Required when `auth_enabled: true`; optional only if auth is disabled

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-05-22T20:00:00Z",
  "data": {
    "active_trades": [
      {
        "id": "swing_abc123",
        "ticker": "AAPL",
        "direction": "long",
        "entry_price": 170.00,
        "current_price": 175.00,
        "stop_loss": 165.00,
        "target_price": 185.00,
        "pnl_percent": 2.94,
        "days_held": 5,
        "status": "open"
      }
    ],
    "alerts": [
      {
        "type": "stop_approaching",
        "ticker": "TSLA",
        "message": "Price within 2.1% of stop loss",
        "trade_id": "swing_xyz789"
      }
    ],
    "total_open": 3
  }
}
```

**Alert Types:**
| Type | Description |
|------|-------------|
| `stop_approaching` | Price is within 3% of stop loss |

---

### Recent Trades

#### `GET /api/v1/trades/recent`

Returns recent closed trades with performance metrics.

**Authentication:** Required when `auth_enabled: true`; optional only if auth is disabled

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-05-22T20:00:00Z",
  "data": [
    {
      "id": "swing_xyz789",
      "ticker": "MSFT",
      "direction": "long",
      "setup_type": "breakout",
      "entry_price": 300.00,
      "exit_price": 320.00,
      "entry_date": "2026-05-10",
      "exit_date": "2026-05-20",
      "realized_pnl": 1000.00,
      "r_multiple": 2.0,
      "exit_reason": "target_reached",
      "holding_days": 10
    }
  ]
}
```

---

### Risk Metrics

#### `GET /api/v1/risk`

Returns risk metrics and analysis for the portfolio.

**Authentication:** Required when `auth_enabled: true`; optional only if auth is disabled

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-05-22T20:00:00Z",
  "data": {
    "portfolio_beta": null,
    "portfolio_volatility": null,
    "sharpe_ratio": null,
    "max_drawdown": null,
    "var_95": null,
    "concentration_hhi": 0.15,
    "effective_holdings": 6.67,
    "sector_exposure": {
      "Technology": 35.0,
      "Healthcare": 15.0,
      "Financials": 10.0
    },
    "risk_flags": [],
    "total_value": 125000.00
  }
}
```

**Risk Flags:**
The API automatically generates risk flags for:
- High concentration (HHI > 0.25)
- Low number of positions (< 5)
- Single position > 25% of portfolio

---

### Dashboard Overview

#### `GET /api/v1/overview`

Returns a dashboard overview combining key metrics from all modules.

**Authentication:** Required when `auth_enabled: true`; optional only if auth is disabled

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-05-22T20:00:00Z",
  "data": {
    "portfolio": {
      "total_value": 125000.00,
      "total_pnl": 25000.00,
      "positions_count": 12
    },
    "trading": {
      "open_trades": 3,
      "closed_trades_30d": 5,
      "win_rate": 65.0
    },
    "market": {
      "regime": "bull",
      "spy_price": 450.00,
      "spy_change_percent": 0.56
    },
    "recent_activity": [
      {
        "type": "trade_closed",
        "ticker": "MSFT",
        "pnl": 500.00,
        "date": "2026-05-20"
      }
    ]
  }
}
```

**Market Regimes:**
| Regime | Description |
|--------|-------------|
| `bull` | SPY positive change |
| `bear` | SPY negative change |
| `unknown` | Unable to determine |

---

### Generate Auth Token

#### `POST /api/v1/auth/token`

Generate an API token for authenticated users.

**Request Body:**
```json
{
  "username": "your_user",
  "password": "your_password"
}
```

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-05-22T20:00:00Z",
  "data": {
    "token": "abc123...",
    "expires_in": 86400,
    "user": {
      "id": 1,
      "username": "your_user"
    }
  }
}
```

---

## Error Responses

All error responses follow this format:

```json
{
  "success": false,
  "timestamp": "2026-05-22T20:00:00Z",
  "error": "Human-readable error message",
  "error_code": "machine_readable_error_code"
}
```

**Common Error Codes:**
| Code | HTTP Status | Description |
|------|-------------|-------------|
| `auth_required` | 401 | Authentication token required |
| `invalid_token` | 401 | Token is invalid or expired |
| `bad_request` | 400 | Invalid request body |
| `not_found` | 404 | Endpoint not found |
| `internal_error` | 500 | Server error |

---

## iOS Scriptable Example

```javascript
// iOS Scriptable widget example
const API_URL = "https://your-server.com/api/v1";
const API_TOKEN = "your_token_here";

async function fetchPortfolioSummary() {
  const url = `${API_URL}/summary`;
  const request = new Request(url);
  request.headers = {
    "X-API-Token": API_TOKEN,
    "Content-Type": "application/json"
  };
  
  const response = await request.loadJSON();
  
  if (response.success) {
    const data = response.data;
    return {
      value: data.total_value,
      pnl: data.total_pnl,
      pnlPercent: data.total_pnl_percent
    };
  }
  
  return null;
}

// Use in widget
const summary = await fetchPortfolioSummary();
if (summary) {
  widget.addText(`$${summary.value.toLocaleString()}`);
  widget.addText(`P&L: $${summary.pnl.toLocaleString()} (${summary.pnlPercent}%)`);
}
```

---

## Online Deployment Note (Streamlit + Scriptable)

If you run Quant Sim online, Scriptable must call the API server URL directly (for example `https://your-api-host.com/api/v1`), not the Streamlit page URL.

Recommended setup:
1. Keep Streamlit UI hosted as usual.
2. Run `api_server.py` as a separate web service (same codebase).
3. Use `X-API-Token` in Scriptable requests when `auth_enabled: true`.

---

## Rate Limiting

When `rate_limit_enabled: true`, the API limits requests to:
- `rate_limit_requests` requests per `rate_limit_window` seconds
- Default: 60 requests per minute

Rate limit headers are included in responses when limits are approached.

---

## Configuration

All API settings are configured in `config/settings.yaml`:

```yaml
api:
  host: "0.0.0.0"        # Server bind address
  port: 8080             # Server port
  debug: false           # Debug mode
  auth_enabled: true     # Require authentication
  rate_limit_enabled: true
  rate_limit_requests: 60
  rate_limit_window: 60
```

Environment variable overrides:
- `API_HOST`
- `API_PORT`
- `API_DEBUG`
- `API_BASE_PATH`
- `API_VERSION`
- `API_AUTH_ENABLED`
- `API_TOKEN_HEADER`
- `API_RATE_LIMIT`
- `API_RATE_LIMIT_REQUESTS`
- `API_RATE_LIMIT_WINDOW`
- `API_CORS_ENABLED`
- `API_CORS_ORIGINS`
- `API_DEFAULT_USER_ID`
