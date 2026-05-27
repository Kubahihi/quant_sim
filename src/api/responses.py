"""
API Response utilities.

Provides standardized response structures and helpers
for consistent JSON output across all API endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _utc_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _pick(source: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Pick the first present key from a dict."""
    for key in keys:
        if key in source:
            return source.get(key)
    return default


@dataclass
class APIResponse:
    """
    Standard API response structure.
    
    All API responses follow this format for consistency
    and easy consumption by clients like iOS Scriptable.
    """
    
    success: bool
    timestamp: str = field(default_factory=_utc_iso)
    data: Any | None = None
    error: str | None = None
    error_code: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "success": self.success,
            "timestamp": self.timestamp,
            "updatedAt": self.timestamp,
        }
        
        if self.success and self.data is not None:
            result["data"] = self.data
        
        if not self.success:
            if self.error:
                result["error"] = self.error
            if self.error_code:
                result["error_code"] = self.error_code
        
        if self.meta:
            result["meta"] = self.meta
        
        return result
    
    @classmethod
    def ok(
        cls,
        data: Any,
        meta: dict[str, Any] | None = None,
    ) -> "APIResponse":
        """Create a success response."""
        return cls(
            success=True,
            data=data,
            error=None,
            error_code=None,
            meta=meta or {},
        )
    
    @classmethod
    def error(
        cls,
        message: str,
        code: str = "internal_error",
        status_code: int = 500,
    ) -> "APIResponse":
        """Create an error response."""
        return cls(
            success=False,
            error=message,
            error_code=code,
            meta={"status_code": status_code},
        )


@dataclass
class SuccessResponse(APIResponse):
    """Convenience class for success responses."""
    
    def __init__(
        self,
        data: Any,
        meta: dict[str, Any] | None = None,
    ):
        super().__init__(success=True, data=data, meta=meta or {})


@dataclass
class ErrorResponse(APIResponse):
    """Convenience class for error responses."""
    
    def __init__(
        self,
        message: str,
        code: str = "internal_error",
        status_code: int = 500,
    ):
        super().__init__(
            success=False,
            error=message,
            error_code=code,
            meta={"status_code": status_code},
        )


def make_paginated_response(
    data: list[Any],
    total: int,
    page: int = 1,
    per_page: int = 20,
    meta: dict[str, Any] | None = None,
) -> APIResponse:
    """
    Create a paginated response with standard pagination metadata.
    
    Args:
        data: List of items for current page
        total: Total number of items across all pages
        page: Current page number (1-indexed)
        per_page: Number of items per page
        meta: Additional metadata to include
    
    Returns:
        APIResponse with pagination info
    """
    total_pages = (total + per_page - 1) // per_page
    
    pagination_meta = {
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }
    
    if meta:
        pagination_meta.update(meta)
    
    return APIResponse.ok(data=data, meta=pagination_meta)


def serialize_position(position: dict[str, Any]) -> dict[str, Any]:
    """
    Serialize a position dict to API-friendly format.
    
    Handles NaN values and ensures consistent field names.
    """
    import math
    import numpy as np
    
    def clean_value(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        return v
    
    return {
        "ticker": str(_pick(position, "ticker", "Ticker", default="")).upper(),
        "shares": clean_value(_pick(position, "shares", "Shares", default=0)),
        "cost_basis": clean_value(_pick(position, "cost_basis", "CostBasis")),
        "target_weight": clean_value(_pick(position, "target_weight", "TargetWeight")),
        "current_weight": clean_value(_pick(position, "current_weight", "CurrentWeight")),
        "price": clean_value(_pick(position, "price", "Price")),
        "market_value": clean_value(_pick(position, "market_value", "MarketValue")),
        "cost_value": clean_value(_pick(position, "cost_value", "CostValue")),
        "pnl": clean_value(_pick(position, "pnl", "PnL")),
        "pnl_percent": clean_value(_pick(position, "pnl_percent", "PnLPercent")),
        "notes": str(_pick(position, "notes", "Notes", default="") or "").strip(),
    }


def serialize_trade(trade_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Serialize a trade dict to API-friendly format.
    """
    import math
    import numpy as np
    
    def clean_value(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        return v
    
    return {
        "id": trade_dict.get("id", ""),
        "ticker": str(trade_dict.get("ticker", "")).upper(),
        "direction": trade_dict.get("direction", ""),
        "status": trade_dict.get("status", ""),
        "setup_type": trade_dict.get("setup_type", ""),
        "entry_price": clean_value(trade_dict.get("entry_price")),
        "stop_loss": clean_value(trade_dict.get("stop_loss")),
        "target_price": clean_value(trade_dict.get("target_price")),
        "position_size": clean_value(trade_dict.get("position_size")),
        "risk_percent": clean_value(trade_dict.get("risk_percent")),
        "entry_date": trade_dict.get("entry_date"),
        "exit_date": trade_dict.get("exit_date"),
        "exit_price": clean_value(trade_dict.get("exit_price")),
        "exit_reason": trade_dict.get("exit_reason"),
        "realized_pnl": clean_value(trade_dict.get("realized_pnl")),
        "r_multiple": clean_value(trade_dict.get("r_multiple")),
        "thesis": trade_dict.get("thesis", ""),
        "notes": trade_dict.get("notes", ""),
    }
