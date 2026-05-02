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
                threads=True,
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


def _fetch_detail_info(
    tickers: list[str],
    max_symbols: int = 800,
    probe_size: int = 25,
    min_probe_success_ratio: float = 0.10,
    progress_callback: ProgressCallback | None = None,
    progress_start: float = 0.75,
    progress_end: float = 0.96,
    session: Any | None = None,
) -> dict[str, dict[str, object]]:
    """
    Fetch richer metadata with strict cap to keep daily refresh bounded.

    This call is intentionally capped because yfinance `info` is expensive.
    """
    output: dict[str, dict[str, object]] = {}
    if not tickers:
        return output

    limited_tickers = tickers[:max_symbols]
    total_symbols = len(limited_tickers)
    probe_target = min(max(1, probe_size), total_symbols)
    probe_success = 0
    processed_symbols = 0

    for idx, symbol in enumerate(limited_tickers, start=1):
        processed_symbols = idx
        if idx == 1 or idx % 10 == 0 or idx == total_symbols:
            step_progress = progress_start + (progress_end - progress_start) * idx / max(1, total_symbols)
            _emit_progress(
                progress_callback,
                step_progress,
                "detail_info",
                f"Downloading detailed fundamentals {idx}/{total_symbols}",
                current=idx,
                total=total_symbols,
            )
        try:
            ticker = yf.Ticker(symbol, session=session)
            info = ticker.get_info()
        except Exception:
            info = {}

        if isinstance(info, dict) and info:
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
            if idx <= probe_target:
                probe_success += 1

        if idx == probe_target and total_symbols > probe_target:
            success_ratio = probe_success / float(probe_target)
            if success_ratio < min_probe_success_ratio:
                _emit_progress(
                    progress_callback,
                    progress_end,
                    "detail_info",
                    f"Low detail info yield ({probe_success}/{probe_target}), skipping remaining expensive calls",
                    current=idx,
                    total=total_symbols,
                )
                break

    _emit_progress(
        progress_callback,
        progress_end,
        "detail_info",
        "Detailed fundamentals stage completed",
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


def enrich_universe_candidates(
    candidates: pd.DataFrame,
    previous_snapshot: pd.DataFrame | None = None,
    price_chunk_size: int = 220,
    fast_info_limit: int = 1500,
    detail_limit: int = 800,
    compute_beta_from_history: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """
    Enrich a raw ticker universe with lightweight metadata and fundamentals.

    Expensive detail calls are capped and fault tolerant. Missing fields are
    left as NaN so a single symbol cannot break the pipeline.
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
        fast_info_tickers = _select_fast_info_tickers(base, max_symbols=int(fast_info_limit))
        if fast_info_tickers:
            fast_info = _fetch_fast_info(
                fast_info_tickers,
                progress_callback=progress_callback,
                progress_start=0.50,
                progress_end=0.75,
                session=yf_session,
            )
        else:
            fast_info = {}

    base = _merge_symbol_updates(base, fast_info, overwrite_non_null=False)

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
    missing_score = base[detail_columns].isna().sum(axis=1)
    detail_candidates = base.loc[
        (missing_score >= 5) & pd.to_numeric(base["Price"], errors="coerce").notna(),
        ["Ticker", "MarketCap", "AvgVolume", "SourceCount", "Price"],
    ].copy()
    detail_candidates["MarketCap"] = pd.to_numeric(detail_candidates["MarketCap"], errors="coerce")
    detail_candidates["AvgVolume"] = pd.to_numeric(detail_candidates["AvgVolume"], errors="coerce")
    detail_candidates["SourceCount"] = pd.to_numeric(detail_candidates["SourceCount"], errors="coerce")
    detail_candidates["Price"] = pd.to_numeric(detail_candidates["Price"], errors="coerce")
    detail_candidates = detail_candidates.sort_values(
        by=["SourceCount", "MarketCap", "AvgVolume", "Price", "Ticker"],
        ascending=[False, False, False, False, True],
        na_position="last",
    )
    detail_tickers = detail_candidates["Ticker"].astype(str).tolist()
    with _temporary_proxy_bypass():
        detail_info = _fetch_detail_info(
            detail_tickers,
            max_symbols=detail_limit,
            progress_callback=progress_callback,
            progress_start=0.75,
            progress_end=0.96,
            session=yf_session,
        )

    base = _merge_symbol_updates(base, detail_info, overwrite_non_null=True)

    base["LastUpdated"] = datetime.now(timezone.utc).isoformat()
    _emit_progress(progress_callback, 0.98, "enrichment", "Finalizing enriched universe dataframe")
    return _ensure_columns(base)
