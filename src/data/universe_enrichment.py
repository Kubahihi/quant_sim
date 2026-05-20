from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests as curl_requests


STANDARD_COLUMNS = [
    "Ticker",
    "Company",
    "Exchange",
    "Sector",
    "Industry",
    "MarketCap",
    "AvgVolume",
    "Price",
    "Beta",
    "PE",
    "ForwardPE",
    "PEG",
    "ROE",
    "ROA",
    "RevenueGrowth",
    "EarningsGrowth",
    "DividendYield",
    "Return52W",
    "Source",
    "SourceCount",
    "LastUpdated",
]

NUMERIC_COLUMNS = [
    "MarketCap",
    "AvgVolume",
    "Price",
    "Beta",
    "PE",
    "ForwardPE",
    "PEG",
    "ROE",
    "ROA",
    "RevenueGrowth",
    "EarningsGrowth",
    "DividendYield",
    "Return52W",
    "SourceCount",
]

TEXT_COLUMNS = [
    "Ticker",
    "Company",
    "Exchange",
    "Sector",
    "Industry",
    "Source",
    "LastUpdated",
]


ProgressCallback = Callable[[dict[str, Any]], None]
PROXY_ENV_KEYS = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]
YF_PROXY_BYPASS = {"http": "", "https": "", "all": ""}


def _configure_yfinance_cache() -> None:
    cache_dir = Path("data") / "cache" / "yfinance_tz"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(cache_dir.resolve()))
    except Exception:
        # Cache configuration failures should not block universe refresh.
        pass


_configure_yfinance_cache()


def _emit_progress(
    progress_callback: ProgressCallback | None,
    progress: float,
    stage: str,
    message: str,
    current: int | None = None,
    total: int | None = None,
) -> None:
    if progress_callback is None:
        return

    safe_progress = max(0.0, min(1.0, float(progress)))
    payload: dict[str, Any] = {
        "progress": safe_progress,
        "stage": stage,
        "message": message,
    }
    if current is not None:
        payload["current"] = int(current)
    if total is not None:
        payload["total"] = int(total)
    progress_callback(payload)


def _is_blackhole_proxy(value: str | None) -> bool:
    if not value:
        return False
    lowered = str(value).strip().lower()
    return "127.0.0.1:9" in lowered


def _has_blackhole_proxy_env() -> bool:
    return any(_is_blackhole_proxy(os.environ.get(key)) for key in PROXY_ENV_KEYS)


def _configure_yfinance_network() -> None:
    try:
        if _has_blackhole_proxy_env():
            yf.config.network.proxy = dict(YF_PROXY_BYPASS)
    except Exception:
        pass


def _build_yfinance_session() -> Any:
    _configure_yfinance_network()
    session = curl_requests.Session(impersonate="chrome")
    if _has_blackhole_proxy_env():
        session.proxies = dict(YF_PROXY_BYPASS)
    try:
        session.trust_env = False
    except Exception:
        pass
    return session


_configure_yfinance_network()


@contextmanager
def _temporary_proxy_bypass():
    """
    Temporarily disable known broken local proxy settings.

    Some environments set proxy vars to 127.0.0.1:9, which breaks
    yfinance quote/fundamental endpoints while chart downloads may still work.
    """
    original: dict[str, str] = {}
    removed: list[str] = []
    for key in PROXY_ENV_KEYS:
        value = os.environ.get(key)
        if _is_blackhole_proxy(value):
            original[key] = str(value)
            os.environ.pop(key, None)
            removed.append(key)
    try:
        yield
    finally:
        for key in removed:
            os.environ[key] = original[key]


def _chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def _pick_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _extract_symbol_history(history: pd.DataFrame, symbol: str, chunk_size: int) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()

    if isinstance(history.columns, pd.MultiIndex):
        first_level = history.columns.get_level_values(0)
        second_level = history.columns.get_level_values(1)
        if symbol in first_level:
            symbol_frame = history[symbol]
            if isinstance(symbol_frame, pd.Series):
                return symbol_frame.to_frame()
            return symbol_frame
        if symbol in second_level:
            return history.xs(symbol, axis=1, level=1, drop_level=True)
        return pd.DataFrame()

    if chunk_size == 1:
        return history

    return pd.DataFrame()


def _price_metrics_from_history(symbol_history: pd.DataFrame) -> dict[str, float]:
    if symbol_history.empty:
        return {}

    close_col = _pick_column(symbol_history, ["Adj Close", "Close", "adj close", "close"])
    volume_col = _pick_column(symbol_history, ["Volume", "volume"])

    close_series = (
        pd.to_numeric(symbol_history[close_col], errors="coerce").dropna()
        if close_col in symbol_history.columns
        else pd.Series(dtype=float)
    )
    volume_series = (
        pd.to_numeric(symbol_history[volume_col], errors="coerce").dropna()
        if volume_col in symbol_history.columns
        else pd.Series(dtype=float)
    )

    metrics: dict[str, float] = {}
    if not close_series.empty:
        metrics["Price"] = float(close_series.iloc[-1])
        base_index = max(0, len(close_series) - 252)
        base_price = float(close_series.iloc[base_index])
        if base_price > 0:
            metrics["Return52W"] = float(close_series.iloc[-1] / base_price - 1.0)

    if not volume_series.empty:
        metrics["AvgVolume"] = float(volume_series.tail(30).mean())

    return metrics


def _extract_close_returns(symbol_history: pd.DataFrame) -> pd.Series:
    close_col = _pick_column(symbol_history, ["Adj Close", "Close", "adj close", "close"])
    if close_col is None or close_col not in symbol_history.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(symbol_history[close_col], errors="coerce").dropna()
    if len(close) < 40:
        return pd.Series(dtype=float)
    returns = close.pct_change().dropna()
    return returns if not returns.empty else pd.Series(dtype=float)


def _compute_beta(symbol_history: pd.DataFrame, benchmark_returns: pd.Series) -> float | None:
    if benchmark_returns.empty:
        return None
    asset_returns = _extract_close_returns(symbol_history)
    if asset_returns.empty:
        return None

    aligned = pd.concat(
        [asset_returns.rename("asset"), benchmark_returns.rename("bench")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 30:
        return None

    bench_var = float(aligned["bench"].var())
    if bench_var <= 0:
        return None
    covariance = float(aligned["asset"].cov(aligned["bench"]))
    return covariance / bench_var


def _coerce_text_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in TEXT_COLUMNS:
        if column in output.columns:
            output[column] = output[column].astype("object")
    return output


def _merge_symbol_updates(
    base: pd.DataFrame,
    updates_by_symbol: dict[str, dict[str, object]],
    overwrite_non_null: bool,
) -> pd.DataFrame:
    if not updates_by_symbol:
        return base

    updates = pd.DataFrame.from_dict(updates_by_symbol, orient="index")
    if updates.empty:
        return base

    updates.index.name = "Ticker"
    updates = updates.reset_index()
    update_columns = [column for column in updates.columns if column != "Ticker" and column in base.columns]
    if not update_columns:
        return base

    updates = updates[["Ticker", *update_columns]].drop_duplicates(subset=["Ticker"], keep="last")
    merged = base.merge(updates, on="Ticker", how="left", suffixes=("", "_upd"))
    for column in update_columns:
        update_column = f"{column}_upd"
        if overwrite_non_null:
            merged[column] = merged[update_column].combine_first(merged[column])
        else:
            merged[column] = merged[column].combine_first(merged[update_column])
        merged = merged.drop(columns=[update_column])

    return _coerce_text_columns(merged)


def _fetch_benchmark_returns(benchmark_symbol: str) -> pd.Series:
    if not benchmark_symbol:
        return pd.Series(dtype=float)
    try:
        session = _build_yfinance_session()
        with _temporary_proxy_bypass():
            history = yf.download(
                tickers=benchmark_symbol,
                period="1y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                session=session,
            )
    except Exception:
        return pd.Series(dtype=float)
    if history.empty:
        return pd.Series(dtype=float)
    return _extract_close_returns(history)


def _fetch_price_volume_snapshot(
    tickers: list[str],
    chunk_size: int = 220,
    progress_callback: ProgressCallback | None = None,
    progress_start: float = 0.20,
    progress_end: float = 0.50,
    compute_beta: bool = False,
    benchmark_symbol: str = "SPY",
    session: Any | None = None,
) -> dict[str, dict[str, float]]:
    """Fetch price, avg volume, and 52-week return in batched requests."""
    output: dict[str, dict[str, float]] = {}
    if not tickers:
        return output

    benchmark_returns = pd.Series(dtype=float)
    if compute_beta:
        benchmark_returns = _fetch_benchmark_returns(benchmark_symbol)
    chunks = list(_chunked(tickers, chunk_size))
    total_chunks = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        chunk_progress = progress_start + (progress_end - progress_start) * (idx - 1) / max(1, total_chunks)
        _emit_progress(
            progress_callback,
            chunk_progress,
            "price_volume",
            f"Downloading price/volume history chunk {idx}/{total_chunks}",
            current=idx,
            total=total_chunks,
        )
        try:
            history = yf.download(
                tickers=chunk,
                period="1y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="ticker",
                session=session,
            )
        except Exception:
            continue

        if history.empty:
            continue

        for symbol in chunk:
            try:
                symbol_history = _extract_symbol_history(history, symbol, len(chunk))
                metrics = _price_metrics_from_history(symbol_history)
                if compute_beta:
                    beta = _compute_beta(symbol_history, benchmark_returns)
                    if beta is not None and np.isfinite(beta):
                        metrics["Beta"] = float(beta)
                if metrics:
                    output[symbol] = metrics
            except Exception:
                continue

    _emit_progress(
        progress_callback,
        progress_end,
        "price_volume",
        "Price/volume stage completed",
        current=total_chunks,
        total=total_chunks,
    )
    return output


def _fetch_fast_info(
    tickers: list[str],
    chunk_size: int = 120,
    progress_callback: ProgressCallback | None = None,
    progress_start: float = 0.50,
    progress_end: float = 0.75,
    session: Any | None = None,
) -> dict[str, dict[str, float]]:
    """Fetch lightweight fast_info fields (market cap, beta, fallback price)."""
    output: dict[str, dict[str, float]] = {}
    if not tickers:
        return output

    chunks = list(_chunked(tickers, chunk_size))
    total_chunks = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        chunk_progress = progress_start + (progress_end - progress_start) * (idx - 1) / max(1, total_chunks)
        _emit_progress(
            progress_callback,
            chunk_progress,
            "fast_info",
            f"Downloading fast info chunk {idx}/{total_chunks}",
            current=idx,
            total=total_chunks,
        )
        try:
            ticker_bundle = yf.Tickers(" ".join(chunk), session=session)
        except Exception:
            ticker_bundle = None

        for symbol in chunk:
            try:
                ticker_obj = None
                if ticker_bundle is not None:
                    ticker_obj = ticker_bundle.tickers.get(symbol)
                if ticker_obj is None:
                    ticker_obj = yf.Ticker(symbol, session=session)

                fast_info = getattr(ticker_obj, "fast_info", None) or {}
                row: dict[str, float] = {}

                def _safe_fast_info_get(*keys: str) -> float | None:
                    for key in keys:
                        try:
                            value = fast_info.get(key)
                        except Exception:
                            continue
                        if value is not None:
                            return value
                    return None

                market_cap = _safe_fast_info_get("marketCap", "market_cap")
                beta = _safe_fast_info_get("beta")
                price = _safe_fast_info_get("lastPrice", "last_price")
                shares = _safe_fast_info_get("shares")

                if market_cap is None and shares is not None and price is not None:
                    try:
                        market_cap = float(shares) * float(price)
                    except Exception:
                        market_cap = None

                if market_cap is not None:
                    row["MarketCap"] = float(market_cap)
                if beta is not None:
                    row["Beta"] = float(beta)
                if price is not None:
                    row["Price"] = float(price)

                if row:
                    output[symbol] = row
            except Exception:
                continue

    _emit_progress(
        progress_callback,
        progress_end,
        "fast_info",
        "Fast info stage completed",
        current=total_chunks,
        total=total_chunks,
    )
    return output


def _select_fast_info_tickers(base: pd.DataFrame, max_symbols: int) -> list[str]:
    if max_symbols <= 0 or base.empty or "Ticker" not in base.columns:
        return []

    candidates = base[["Ticker", "SourceCount", "MarketCap", "AvgVolume", "Price", "Beta"]].copy()
    candidates["SourceCount"] = pd.to_numeric(candidates["SourceCount"], errors="coerce")
    candidates["MarketCap"] = pd.to_numeric(candidates["MarketCap"], errors="coerce")
    candidates["AvgVolume"] = pd.to_numeric(candidates["AvgVolume"], errors="coerce")
    candidates["Price"] = pd.to_numeric(candidates["Price"], errors="coerce")
    candidates["Beta"] = pd.to_numeric(candidates["Beta"], errors="coerce")

    candidates["needs_fast_info"] = (
        candidates["MarketCap"].isna()
        | candidates["Beta"].isna()
        | candidates["Price"].isna()
    )
    candidates["has_price"] = candidates["Price"].notna()
    candidates["has_volume"] = candidates["AvgVolume"].notna()
    candidates = candidates.sort_values(
        by=["needs_fast_info", "has_price", "has_volume", "SourceCount", "AvgVolume", "Price", "Ticker"],
        ascending=[False, False, False, False, False, False, True],
        na_position="last",
    )
    candidates = candidates.loc[candidates["needs_fast_info"] & candidates["has_price"]]
    return candidates["Ticker"].astype(str).head(max_symbols).tolist()


# Persistent cache for fundamental data across refresh cycles
_FUNDAMENTAL_CACHE: dict[str, dict[str, object]] = {}
_FUNDAMENTAL_CACHE_TIMESTAMP: str | None = None


def _load_fundamental_cache() -> dict[str, dict[str, object]]:
    """Load cached fundamental data from disk if available."""
    global _FUNDAMENTAL_CACHE, _FUNDAMENTAL_CACHE_TIMESTAMP
    if _FUNDAMENTAL_CACHE:
        return _FUNDAMENTAL_CACHE

    cache_path = Path("data") / "cache" / "universe" / "fundamental_cache.json"
    if cache_path.exists():
        try:
            import json
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            _FUNDAMENTAL_CACHE = data.get("data", {})
            _FUNDAMENTAL_CACHE_TIMESTAMP = data.get("timestamp")
        except Exception:
            pass

    return _FUNDAMENTAL_CACHE


def _save_fundamental_cache(data: dict[str, dict[str, object]]) -> None:
    """Persist fundamental data cache to disk."""
    global _FUNDAMENTAL_CACHE, _FUNDAMENTAL_CACHE_TIMESTAMP
    _FUNDAMENTAL_CACHE = data
    _FUNDAMENTAL_CACHE_TIMESTAMP = _safe_timestamp_now()

    cache_path = Path("data") / "cache" / "universe" / "fundamental_cache.json"
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        cache_path.write_text(
            json.dumps({"data": data, "timestamp": _FUNDAMENTAL_CACHE_TIMESTAMP}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _safe_timestamp_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _normalize_cached_fundamental_row(cached_data: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(cached_data, dict):
        return {}
    return {
        "Company": cached_data.get("Company") or cached_data.get("company_name"),
        "Exchange": cached_data.get("Exchange") or cached_data.get("exchange"),
        "Sector": cached_data.get("Sector") or cached_data.get("sector"),
        "Industry": cached_data.get("Industry") or cached_data.get("industry"),
        "MarketCap": cached_data.get("MarketCap") or cached_data.get("marketCap"),
        "Beta": cached_data.get("Beta") or cached_data.get("beta"),
        "PE": cached_data.get("PE") or cached_data.get("trailingPE"),
        "ForwardPE": cached_data.get("ForwardPE") or cached_data.get("forwardPE"),
        "PEG": cached_data.get("PEG") or cached_data.get("pegRatio"),
        "ROE": cached_data.get("ROE") or cached_data.get("returnOnEquity"),
        "ROA": cached_data.get("ROA") or cached_data.get("returnOnAssets"),
        "RevenueGrowth": cached_data.get("RevenueGrowth") or cached_data.get("revenueGrowth"),
        "EarningsGrowth": cached_data.get("EarningsGrowth") or cached_data.get("earningsGrowth"),
        "DividendYield": cached_data.get("DividendYield") or cached_data.get("dividendYield"),
    }


def _build_cached_updates_for_tickers(
    tickers: list[str],
    cache: dict[str, dict[str, object]] | None,
) -> dict[str, dict[str, object]]:
    if not tickers or not cache:
        return {}
    cache_lookup = {str(symbol).strip().upper(): value for symbol, value in cache.items()}
    updates: dict[str, dict[str, object]] = {}
    for symbol in tickers:
        key = str(symbol).strip().upper()
        normalized = _normalize_cached_fundamental_row(cache_lookup.get(key))
        if not normalized:
            continue
        # Reuse cache when it carries at least one descriptive field.
        if (
            normalized.get("Sector")
            or normalized.get("Industry")
            or normalized.get("PE")
            or normalized.get("ForwardPE")
            or normalized.get("ROE")
            or normalized.get("RevenueGrowth")
            or normalized.get("DividendYield")
            or normalized.get("Company")
        ):
            updates[key] = normalized
    return updates


def _select_detail_tickers(
    base: pd.DataFrame,
    detail_columns: list[str],
    max_symbols: int,
    cache: dict[str, dict[str, object]] | None = None,
) -> list[str]:
    if max_symbols <= 0 or base.empty or "Ticker" not in base.columns:
        return []

    missing_score = base[detail_columns].isna().sum(axis=1)
    detail_candidates = base.loc[
        (missing_score >= 5) & pd.to_numeric(base["Price"], errors="coerce").notna(),
        ["Ticker", "MarketCap", "AvgVolume", "SourceCount", "Price", *detail_columns],
    ].copy()
    if detail_candidates.empty:
        return []

    detail_candidates["MarketCap"] = pd.to_numeric(detail_candidates["MarketCap"], errors="coerce")
    detail_candidates["AvgVolume"] = pd.to_numeric(detail_candidates["AvgVolume"], errors="coerce")
    detail_candidates["SourceCount"] = pd.to_numeric(detail_candidates["SourceCount"], errors="coerce")
    detail_candidates["Price"] = pd.to_numeric(detail_candidates["Price"], errors="coerce")

    cache_keys: set[str] = set()
    if cache:
        cache_keys = {str(symbol).strip().upper() for symbol in cache.keys()}
    detail_candidates["Cached"] = detail_candidates["Ticker"].astype(str).str.upper().isin(cache_keys)

    key_fields = ["PE", "ForwardPE", "ROE", "RevenueGrowth", "DividendYield"]
    available_key_fields = [column for column in key_fields if column in detail_candidates.columns]
    if available_key_fields:
        key_field_counts = [
            pd.to_numeric(detail_candidates[column], errors="coerce").notna().astype(int)
            for column in available_key_fields
        ]
        detail_candidates["KeyFieldCount"] = sum(key_field_counts)
    else:
        detail_candidates["KeyFieldCount"] = 0

    detail_candidates = detail_candidates.sort_values(
        by=[
            "Cached",
            "KeyFieldCount",
            "SourceCount",
            "MarketCap",
            "AvgVolume",
            "Price",
            "Ticker",
        ],
        ascending=[True, True, False, False, False, False, True],
        na_position="last",
    )
    return detail_candidates["Ticker"].astype(str).head(max_symbols).tolist()


def _fetch_detail_info(
    tickers: list[str],
    max_symbols: int = 1200,
    probe_size: int = 40,
    min_probe_success_ratio: float = 0.25,
    progress_callback: ProgressCallback | None = None,
    progress_start: float = 0.75,
    progress_end: float = 0.96,
    session: Any | None = None,
) -> dict[str, dict[str, object]]:
    """
    Fetch richer metadata with improved coverage and caching.

    Changes from previous version:
    - Increased max_symbols from 800 to 1200 for broader coverage
    - Increased probe_size from 25 to 40 for more reliable success rate assessment
    - Raised min_probe_success_ratio from 0.10 to 0.25 to avoid false aborts
    - Added persistent caching to reduce redundant API calls
    - Better error handling and recovery
    """
    output: dict[str, dict[str, object]] = {}
    if not tickers:
        return output

    # Load cached data to avoid re-fetching
    cache = _load_fundamental_cache()
    cache_hits = 0

    limited_tickers = tickers[:max_symbols]
    total_symbols = len(limited_tickers)
    probe_target = min(max(1, probe_size), total_symbols)
    probe_success = 0
    processed_symbols = 0
    consecutive_failures = 0
    max_consecutive_failures = 50  # Stop if we have 50 consecutive failures

    for idx, symbol in enumerate(limited_tickers, start=1):
        processed_symbols = idx
        if idx == 1 or idx % 10 == 0 or idx == total_symbols:
            step_progress = progress_start + (progress_end - progress_start) * idx / max(1, total_symbols)
            _emit_progress(
                progress_callback,
                step_progress,
                "detail_info",
                f"Downloading detailed fundamentals {idx}/{total_symbols} (cache hits: {cache_hits})",
                current=idx,
                total=total_symbols,
            )

        # Check cache first (support both legacy and canonical key names).
        if symbol in cache:
            normalized_cached = _normalize_cached_fundamental_row(cache[symbol])
            if normalized_cached.get("Sector") or normalized_cached.get("MarketCap") or normalized_cached.get("Company"):
                output[symbol] = normalized_cached
                cache[symbol] = normalized_cached
                cache_hits += 1
                consecutive_failures = 0
                continue

        try:
            ticker = yf.Ticker(symbol, session=session)
            info = ticker.get_info()
        except Exception:
            info = {}

        if isinstance(info, dict) and info:
            consecutive_failures = 0
            row: dict[str, object] = {
                "Company": info.get("longName") or info.get("shortName"),
                "Exchange": info.get("exchange"),
                "Sector": info.get("sector"),
                "Industry": info.get("industry"),
                "MarketCap": info.get("marketCap"),
                "Beta": info.get("beta"),
                "PE": info.get("trailingPE"),
                "ForwardPE": info.get("forwardPE"),
                "PEG": info.get("pegRatio"),
                "ROE": info.get("returnOnEquity"),
                "ROA": info.get("returnOnAssets"),
                "RevenueGrowth": info.get("revenueGrowth"),
                "EarningsGrowth": info.get("earningsGrowth"),
                "DividendYield": info.get("dividendYield"),
            }
            output[symbol] = row
            # Update cache
            cache[symbol] = row

            if idx <= probe_target:
                # Count as success if we got any meaningful data
                if row.get("Sector") or row.get("MarketCap") or row.get("Company"):
                    probe_success += 1
        else:
            consecutive_failures += 1

        # Check for consecutive failures (indicates rate limiting or network issues)
        if consecutive_failures >= max_consecutive_failures:
            _emit_progress(
                progress_callback,
                progress_end,
                "detail_info",
                f"Too many consecutive failures ({consecutive_failures}), stopping early",
                current=idx,
                total=total_symbols,
            )
            break

        if idx == probe_target and total_symbols > probe_target:
            success_ratio = probe_success / float(probe_target)
            if success_ratio < min_probe_success_ratio:
                _emit_progress(
                    progress_callback,
                    progress_end,
                    "detail_info",
                    f"Low detail info yield ({probe_success}/{probe_target}, ratio={success_ratio:.2f}), skipping remaining expensive calls",
                    current=idx,
                    total=total_symbols,
                )
                break

    # Persist cache to disk
    if cache:
        _save_fundamental_cache(cache)

    _emit_progress(
        progress_callback,
        progress_end,
        "detail_info",
        f"Detailed fundamentals stage completed (cache hits: {cache_hits}, fetched: {len(output) - cache_hits})",
        current=processed_symbols,
        total=total_symbols,
    )
    return output


def _merge_previous_snapshot(base: pd.DataFrame, previous_snapshot: pd.DataFrame | None) -> pd.DataFrame:
    if previous_snapshot is None or previous_snapshot.empty or "Ticker" not in previous_snapshot.columns:
        return _coerce_text_columns(base)

    previous = _coerce_text_columns(previous_snapshot)
    previous = previous.drop_duplicates(subset=["Ticker"], keep="first")
    previous_columns = [column for column in STANDARD_COLUMNS if column in previous.columns]
    previous = previous[previous_columns]

    merged = _coerce_text_columns(base).merge(previous, on="Ticker", how="left", suffixes=("", "_prev"))
    for column in STANDARD_COLUMNS:
        prev_column = f"{column}_prev"
        if prev_column in merged.columns and column in merged.columns:
            merged[column] = merged[column].combine_first(merged[prev_column])
            merged = merged.drop(columns=[prev_column])
    return _coerce_text_columns(merged)


def _ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in STANDARD_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan
    output = _coerce_text_columns(output)
    for column in NUMERIC_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output[STANDARD_COLUMNS]


def _compute_coverage_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """
    Compute coverage metrics for the enriched universe.

    Returns a dictionary with coverage percentages for key fields.
    """
    if frame.empty:
        return {}

    total = len(frame)
    metrics: dict[str, Any] = {
        "total_symbols": total,
    }

    # Text field coverage
    for col in ["Sector", "Industry", "Company", "Exchange"]:
        non_null = frame[col].notna() & (frame[col].astype(str).str.strip() != "") & (frame[col].astype(str).str.lower() != "nan")
        metrics[f"{col}_coverage"] = round(float(non_null.sum()) / total * 100, 1)

    # Numeric field coverage
    for col in NUMERIC_COLUMNS:
        if col in frame.columns:
            non_null = pd.to_numeric(frame[col], errors="coerce").notna()
            metrics[f"{col}_coverage"] = round(float(non_null.sum()) / total * 100, 1)

    # Price coverage (critical field)
    price_non_null = pd.to_numeric(frame["Price"], errors="coerce").notna()
    metrics["Price_coverage"] = round(float(price_non_null.sum()) / total * 100, 1)

    # Sector coverage (key metric)
    sector_non_null = frame["Sector"].notna() & (frame["Sector"].astype(str).str.strip() != "") & (frame["Sector"].astype(str).str.lower() != "nan")
    metrics["Sector_coverage"] = round(float(sector_non_null.sum()) / total * 100, 1)

    # Overall data completeness (average of key fields)
    key_fields = ["Sector", "Industry", "Company", "MarketCap", "Price", "PE", "Beta"]
    completeness_scores = []
    for col in key_fields:
        if col in frame.columns:
            if col in TEXT_COLUMNS:
                non_null = frame[col].notna() & (frame[col].astype(str).str.strip() != "") & (frame[col].astype(str).str.lower() != "nan")
            else:
                non_null = pd.to_numeric(frame[col], errors="coerce").notna()
            completeness_scores.append(float(non_null.sum()) / total)
    metrics["overall_completeness"] = round(sum(completeness_scores) / len(completeness_scores) * 100, 1) if completeness_scores else 0.0

    return metrics


def _log_coverage_metrics(metrics: dict[str, Any]) -> None:
    """Log coverage metrics for monitoring."""
    if not metrics:
        return

    print("\n" + "=" * 60)
    print("UNIVERSE COVERAGE REPORT")
    print("=" * 60)
    print(f"Total Symbols: {metrics.get('total_symbols', 'N/A'):,}")
    print("-" * 40)
    print("Key Field Coverage:")
    for key, value in sorted(metrics.items()):
        if key.endswith("_coverage") or key == "overall_completeness":
            field_name = key.replace("_coverage", "").replace("_", " ").title()
            print(f"  {field_name}: {value}%")
    print("=" * 60 + "\n")


def enrich_universe_candidates(
    candidates: pd.DataFrame,
    previous_snapshot: pd.DataFrame | None = None,
    price_chunk_size: int = 220,
    fast_info_limit: int = 1000,
    detail_limit: int = 1200,
    compute_beta_from_history: bool = False,
    progress_callback: ProgressCallback | None = None,
    report_coverage: bool = True,
) -> pd.DataFrame:
    """
    Enrich a raw ticker universe with lightweight metadata and fundamentals.

    Expensive detail calls are capped and fault tolerant. Missing fields are
    left as NaN so a single symbol cannot break the pipeline.

    Args:
        candidates: Raw candidate DataFrame from universe sources
        previous_snapshot: Previous universe snapshot for fallback data
        price_chunk_size: Number of tickers per price download batch
        fast_info_limit: Max tickers to fetch fast_info for
        detail_limit: Max tickers to fetch detailed fundamentals for
        compute_beta_from_history: Whether to compute beta from price history
        progress_callback: Optional callback for progress updates
        report_coverage: Whether to compute and log coverage metrics
    """
    if candidates.empty:
        return _ensure_columns(pd.DataFrame())

    _emit_progress(progress_callback, 0.15, "enrichment", "Preparing enrichment base")

    base = pd.DataFrame({
        "Ticker": candidates.get("ticker", pd.Series(dtype=str)).astype(str).str.upper(),
        "Company": candidates.get("company_name"),
        "Exchange": candidates.get("exchange"),
        "Sector": candidates.get("sector"),
        "Industry": candidates.get("industry"),
        "Source": candidates.get("source"),
        "SourceCount": candidates.get("source_count"),
    })
    base = base.dropna(subset=["Ticker"])
    base = base[base["Ticker"].str.len() > 0]
    base = base.drop_duplicates(subset=["Ticker"], keep="first").reset_index(drop=True)

    for column in STANDARD_COLUMNS:
        if column not in base.columns:
            base[column] = np.nan

    base = _coerce_text_columns(base)
    base["LastUpdated"] = datetime.now(timezone.utc).isoformat()
    base = _merge_previous_snapshot(base, previous_snapshot)

    tickers = base["Ticker"].tolist()
    fundamental_cache = _load_fundamental_cache()
    cached_updates = _build_cached_updates_for_tickers(tickers, fundamental_cache)
    if cached_updates:
        base = _merge_symbol_updates(base, cached_updates, overwrite_non_null=True)

    _emit_progress(progress_callback, 0.18, "enrichment", f"Universe prepared: {len(tickers)} tickers")
    _emit_progress(progress_callback, 0.19, "enrichment", "Applying network compatibility settings")
    yf_session = _build_yfinance_session()
    with _temporary_proxy_bypass():
        price_snapshot = _fetch_price_volume_snapshot(
            tickers,
            chunk_size=price_chunk_size,
            progress_callback=progress_callback,
            progress_start=0.20,
            progress_end=0.50,
            compute_beta=compute_beta_from_history,
            session=yf_session,
        )

    base = _merge_symbol_updates(base, price_snapshot, overwrite_non_null=True)

    # Request rich info only for symbols that still miss most descriptive fields.
    detail_columns = [
        "Sector",
        "Industry",
        "PE",
        "ForwardPE",
        "PEG",
        "ROE",
        "ROA",
        "RevenueGrowth",
        "EarningsGrowth",
        "DividendYield",
    ]
    detail_tickers = _select_detail_tickers(
        base,
        detail_columns=detail_columns,
        max_symbols=int(detail_limit),
        cache=fundamental_cache,
    )

    with _temporary_proxy_bypass():
        detail_info = _fetch_detail_info(
            detail_tickers,
            max_symbols=detail_limit,
            progress_callback=progress_callback,
            progress_start=0.50,
            progress_end=0.82,
            session=yf_session,
        )

    base = _merge_symbol_updates(base, detail_info, overwrite_non_null=True)

    # Fill remaining lightweight fields for symbols still missing market metadata.
    with _temporary_proxy_bypass():
        fast_info_tickers = _select_fast_info_tickers(base, max_symbols=int(fast_info_limit))
        if fast_info_tickers:
            fast_info = _fetch_fast_info(
                fast_info_tickers,
                progress_callback=progress_callback,
                progress_start=0.82,
                progress_end=0.96,
                session=yf_session,
            )
        else:
            fast_info = {}

    base = _merge_symbol_updates(base, fast_info, overwrite_non_null=False)

    base["LastUpdated"] = datetime.now(timezone.utc).isoformat()
    _emit_progress(progress_callback, 0.98, "enrichment", "Finalizing enriched universe dataframe")

    result = _ensure_columns(base)

    # Compute and log coverage metrics
    if report_coverage:
        metrics = _compute_coverage_metrics(result)
        _log_coverage_metrics(metrics)

    return result
