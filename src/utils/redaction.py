"""Keep credentials out of persisted and displayed analysis records."""
from collections.abc import Mapping
import re


RUN_CONFIG_FIELDS = frozenset({
    "tickers", "news_tickers", "security_types", "weights", "start_date", "end_date",
    "risk_profile", "risk_free_rate", "horizon_days", "benchmark_ticker",
    "benchmark_metrics", "rebalance_constraints", "portfolio_metrics",
    "transaction_cost_bps", "news_max_items", "sector_keywords",
})


def sanitize_run_config(config):
    """Persist only explicitly supported reproducibility parameters."""
    return redact_sensitive_data({
        key: value for key, value in config.items() if key in RUN_CONFIG_FIELDS
    })


def redact_sensitive_data(value):
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if any(part in normalized for part in ("apikey", "password", "secret", "token", "authorization", "privatekey")):
                continue
            result[key] = redact_sensitive_data(item)
        return result
    if isinstance(value, (list, tuple)):
        return [redact_sensitive_data(item) for item in value]
    return value
