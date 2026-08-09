from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import html
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Sequence, Tuple
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

from .results import NewsItemResult, NewsResult

logger = logging.getLogger(__name__)

_NEWS_CACHE: Dict[str, Tuple[datetime, List["RawNewsItem"], List[str], str]] = {}
_CACHE_TTL_SECONDS = 15 * 60
_CACHE_SCHEMA_VERSION = "news_v4"
_DEFAULT_FETCH_TIME_BUDGET_SECONDS = 8.0
_DEFAULT_MIN_ITEMS_BEFORE_EARLY_STOP = 20
_PRIMARY_PROVIDER_GRACE_SECONDS = 1.0
_MAX_PROVIDER_WORKERS = 3

_MACRO_KEYWORDS = [
    "earnings",
    "inflation",
    "rates",
    "guidance",
    "fed",
    "cpi",
    "growth",
    "recession",
    "volatility",
]


@dataclass
class RawNewsItem:
    title: str
    published_at: str
    source: str
    url: str
    summary: str
    raw_text: str = ""
    query_context: str = ""


@dataclass
class _ProviderFetchResult:
    items: List[RawNewsItem]
    error: str = ""


class NewsProvider:
    provider_name = "base"

    def fetch(self, tickers: List[str], start_date: datetime, end_date: datetime, context: Dict[str, Any] | None = None) -> List[RawNewsItem]:
        raise NotImplementedError


class MissingAPIKeyError(RuntimeError):
    pass


class SentimentAnalyzer:
    def score(self, text: str) -> float:
        raise NotImplementedError


class VaderSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        value = self._analyzer.polarity_scores(text or "").get("compound", 0.0)
        return float(max(-1.0, min(1.0, value)))


class LexiconSentimentAnalyzer(SentimentAnalyzer):
    _POSITIVE = {
        "beat",
        "beats",
        "bullish",
        "growth",
        "strong",
        "upside",
        "upgrade",
        "record",
        "optimistic",
        "outperform",
        "tailwind",
        "guidance",
    }
    _NEGATIVE = {
        "miss",
        "misses",
        "bearish",
        "weak",
        "downgrade",
        "downside",
        "risk",
        "recession",
        "uncertain",
        "selloff",
        "headwind",
        "lawsuit",
    }

    def score(self, text: str) -> float:
        tokens = re.findall(r"[a-zA-Z']+", (text or "").lower())
        if not tokens:
            return 0.0
        pos = sum(1 for token in tokens if token in self._POSITIVE)
        neg = sum(1 for token in tokens if token in self._NEGATIVE)
        denom = max(1, pos + neg)
        return float(max(-1.0, min(1.0, (pos - neg) / denom)))


def _build_sentiment_analyzer() -> SentimentAnalyzer:
    try:
        return VaderSentimentAnalyzer()
    except Exception:
        logger.info("VADER sentiment analyzer unavailable, using lexicon fallback.")
        return LexiconSentimentAnalyzer()


def _coerce_fetch_time_budget(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = _DEFAULT_FETCH_TIME_BUDGET_SECONDS
    if not np.isfinite(parsed):
        parsed = _DEFAULT_FETCH_TIME_BUDGET_SECONDS
    return max(0.01, min(60.0, parsed))


def _remaining_fetch_time(context: Dict[str, Any] | None) -> float | None:
    if not context:
        return None
    deadline = context.get("_news_fetch_deadline_monotonic")
    if deadline is None:
        return None
    try:
        return max(0.0, float(deadline) - time.monotonic())
    except (TypeError, ValueError):
        return None


def _request_timeout(context: Dict[str, Any] | None, maximum: float = 10.0) -> float:
    remaining = _remaining_fetch_time(context)
    if remaining is not None:
        if remaining <= 0:
            raise TimeoutError("news fetch time budget exhausted")
        return max(0.001, min(float(maximum), remaining))
    return float(maximum)


def _fetch_budget_exhausted(context: Dict[str, Any] | None) -> bool:
    remaining = _remaining_fetch_time(context)
    return remaining is not None and remaining <= 0


def _sufficient_item_count(context: Dict[str, Any] | None) -> int:
    raw_value = (context or {}).get(
        "_news_sufficient_items",
        _DEFAULT_MIN_ITEMS_BEFORE_EARLY_STOP,
    )
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return _DEFAULT_MIN_ITEMS_BEFORE_EARLY_STOP


class NewsApiProvider(NewsProvider):
    provider_name = "newsapi"

    @staticmethod
    def _request_json(url: str, api_key: str, timeout: float = 10.0) -> Dict[str, Any]:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "quant-sim/1.0 (+streamlit)",
                "Accept": "application/json",
                "X-Api-Key": api_key,
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _parse_articles(payload: Dict[str, Any], query_context: str = "") -> List[RawNewsItem]:
        rows = payload.get("articles", []) or []
        items: List[RawNewsItem] = []
        for row in rows:
            title = _clean_text(str(row.get("title", "")))
            source = str((row.get("source") or {}).get("name", "newsapi")).strip() or "newsapi"
            url_value = str(row.get("url", "")).strip()
            if not title or not url_value:
                continue
            published_at = str(row.get("publishedAt", "")).strip() or datetime.now(timezone.utc).isoformat()
            summary = _clean_text(str(row.get("description", "")))
            raw_text = _clean_text(str(row.get("content", "")))
            items.append(
                RawNewsItem(
                    title=title,
                    source=source,
                    published_at=published_at,
                    url=url_value,
                    summary=summary,
                    raw_text=raw_text,
                    query_context=query_context,
                )
            )
        return items

    def fetch(self, tickers: List[str], start_date: datetime, end_date: datetime, context: Dict[str, Any] | None = None) -> List[RawNewsItem]:
        context = context or {}
        api_key = str(context.get("news_api_key") or os.getenv("NEWSAPI_KEY") or "").strip()
        if not api_key:
            raise MissingAPIKeyError("NEWSAPI_KEY is missing; skipping NewsAPI provider.")

        keywords = _normalized_keywords(context)
        query_terms = [*tickers[:3], *keywords[:4]]
        query = " OR ".join(term for term in query_terms if term) or "markets"
        query_context = " ".join(term for term in query_terms if term)

        params = {
            "q": query,
            "from": start_date.date().isoformat(),
            "to": end_date.date().isoformat(),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": "20",
        }
        url = "https://newsapi.org/v2/everything?" + urllib.parse.urlencode(params)
        try:
            payload = self._request_json(
                url,
                api_key=api_key,
                timeout=_request_timeout(context),
            )
            if payload.get("status") == "ok":
                parsed = self._parse_articles(payload, query_context=query_context)
                if parsed:
                    return parsed
                # Broader fallback when query is too specific and returns no rows.
                fallback_params = {
                    "q": query_terms[0] if query_terms else "markets",
                    "language": "en",
                    "pageSize": "20",
                    "sortBy": "publishedAt",
                }
                fallback_url = "https://newsapi.org/v2/everything?" + urllib.parse.urlencode(fallback_params)
                fallback_payload = self._request_json(
                    fallback_url,
                    api_key=api_key,
                    timeout=_request_timeout(context),
                )
                if fallback_payload.get("status") == "ok":
                    return self._parse_articles(fallback_payload, query_context=query_context)
                return []
            message = str(payload.get("message", "NewsAPI response status is not ok"))
            raise RuntimeError(message)
        except urllib.error.HTTPError as exc:
            if exc.code != 426:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="ignore")[:200]
                except Exception:
                    detail = ""
                raise RuntimeError(f"NewsAPI HTTP {exc.code}: {detail or exc.reason}") from exc

            # Some environments return 426 for /everything; retry with top-headlines.
            logger.warning("NewsAPI /everything returned 426; retrying with /top-headlines.")
            headline_params = {
                "q": query_terms[0] if query_terms else "markets",
                "language": "en",
                "pageSize": "20",
                "category": "business",
            }
            fallback_url = "https://newsapi.org/v2/top-headlines?" + urllib.parse.urlencode(headline_params)
            try:
                payload = self._request_json(
                    fallback_url,
                    api_key=api_key,
                    timeout=_request_timeout(context),
                )
                if payload.get("status") == "ok":
                    return self._parse_articles(payload, query_context=query_context)
                message = str(payload.get("message", "NewsAPI fallback response status is not ok"))
                raise RuntimeError(message)
            except Exception as fallback_exc:
                raise RuntimeError(
                    "NewsAPI returned HTTP 426 (Upgrade Required) and fallback request failed. "
                    "Check NEWSAPI_KEY plan/permissions or rely on fallback providers."
                ) from fallback_exc
        except Exception as exc:
            logger.warning("NewsAPI fetch failed: %s", exc)
            raise


class YahooNewsProvider(NewsProvider):
    provider_name = "yfinance"

    def fetch(self, tickers: List[str], start_date: datetime, end_date: datetime, context: Dict[str, Any] | None = None) -> List[RawNewsItem]:
        context = context or {}
        try:
            import yfinance as yf
        except Exception as exc:
            logger.warning("yfinance import failed in YahooNewsProvider: %s", exc)
            raise

        raw_items: List[RawNewsItem] = []
        sufficient_items = _sufficient_item_count(context)
        for ticker in tickers[:5]:
            if _fetch_budget_exhausted(context):
                logger.info("Yahoo news stopped after reaching the total fetch time budget.")
                break
            try:
                yf_ticker = yf.Ticker(ticker)
                items = getattr(yf_ticker, "news", []) or []
            except Exception as exc:
                logger.warning("yfinance news fetch failed for %s: %s", ticker, exc)
                items = []

            for item in items[:12]:
                try:
                    provider = str(item.get("publisher") or "yfinance")
                    timestamp = item.get("providerPublishTime")
                    if timestamp is None:
                        published = datetime.now(timezone.utc).isoformat()
                    else:
                        published = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()
                    raw_items.append(
                        RawNewsItem(
                            title=_clean_text(str(item.get("title", ""))),
                            published_at=published,
                            source=provider,
                            url=str(item.get("link", "")),
                            summary=_clean_text(str(item.get("summary", "")))[:800],
                            query_context=str(ticker),
                        )
                    )
                except Exception as exc:
                    logger.warning("yfinance news item parsing failed: %s", exc)
                    continue
            if len(_dedupe_raw_items(raw_items)) >= sufficient_items:
                break
        return _dedupe_raw_items(raw_items)


class GoogleRssNewsProvider(NewsProvider):
    provider_name = "google_rss"

    def fetch(self, tickers: List[str], start_date: datetime, end_date: datetime, context: Dict[str, Any] | None = None) -> List[RawNewsItem]:
        context = context or {}
        keywords = _normalized_keywords(context)
        queries: List[str] = []
        for ticker in tickers[:8]:
            queries.append(f"{ticker} stock")
            for keyword in keywords[:4]:
                queries.append(f"{ticker} {keyword}")
        if tickers or keywords:
            queries.append(" ".join([*tickers[:4], *keywords[:4]]).strip())
        queries.append("markets macro rates inflation earnings")

        unique_queries: List[str] = []
        seen_queries: set[str] = set()
        for query in queries:
            normalized = query.strip().lower()
            if not normalized or normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            unique_queries.append(query.strip())

        all_items: List[RawNewsItem] = []
        fetch_errors: List[str] = []
        sufficient_items = _sufficient_item_count(context)
        for query in unique_queries[:20]:
            if _fetch_budget_exhausted(context):
                fetch_errors.append("news fetch time budget exhausted")
                break
            rss_url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(query)
            try:
                with urllib.request.urlopen(
                    rss_url,
                    timeout=_request_timeout(context),
                ) as response:
                    payload = response.read()
                root = ET.fromstring(payload)
            except Exception as exc:
                fetch_errors.append(f"{query}: {exc}")
                logger.warning("Google RSS fetch/parse failed for query '%s': %s", query, exc)
                continue

            for node in root.findall(".//item")[:100]:
                try:
                    title = _clean_text(node.findtext("title") or "")
                    link = (node.findtext("link") or "").strip()
                    pub_date = (node.findtext("pubDate") or "").strip()
                    source = _clean_text(node.findtext("source") or "google_rss")
                    description = _clean_text(node.findtext("description") or "")
                    if not title or not link:
                        continue
                    published_iso = _parse_rss_time(pub_date)
                    all_items.append(
                        RawNewsItem(
                            title=title,
                            source=source,
                            published_at=published_iso,
                            url=link,
                            summary=description[:1200],
                            query_context=query,
                        )
                    )
                except Exception as exc:
                    logger.warning("Google RSS item parsing failed: %s", exc)
                    continue
            if len(_dedupe_raw_items(all_items)) >= sufficient_items:
                break

        deduped = _dedupe_raw_items(all_items)
        if deduped:
            return deduped
        if fetch_errors:
            raise RuntimeError("; ".join(fetch_errors[:3]))
        return []


class SampleNewsProvider(NewsProvider):
    provider_name = "sample"

    def fetch(self, tickers: List[str], start_date: datetime, end_date: datetime, context: Dict[str, Any] | None = None) -> List[RawNewsItem]:
        context = context or {}
        ticker = (tickers or ["MARKET"])[0]
        keywords = _normalized_keywords(context)
        key = keywords[0] if keywords else "earnings"
        now = datetime.now(timezone.utc)
        return [
            RawNewsItem(
                title=f"{ticker} outlook update around {key}",
                source="sample",
                published_at=now.isoformat(),
                url=f"https://example.com/{ticker.lower()}/1",
                summary=f"Sample fallback article for {ticker} with {key} context.",
                raw_text=f"{ticker} shows mixed but stable tone around {key} and macro conditions.",
                query_context=f"{ticker} {key}",
            ),
            RawNewsItem(
                title=f"{ticker} risk scenario: rates and inflation watch",
                source="sample",
                published_at=(now - timedelta(hours=8)).isoformat(),
                url=f"https://example.com/{ticker.lower()}/2",
                summary=f"Fallback scenario note for {ticker} covering rates and inflation.",
                raw_text=f"{ticker} could react to rates, inflation and guidance revisions.",
                query_context=f"{ticker} rates inflation",
            ),
        ]


def _parse_rss_time(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return datetime.now(timezone.utc).isoformat()
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"):
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            continue
    return datetime.now(timezone.utc).isoformat()


def _dedupe_raw_items(items: Sequence[RawNewsItem]) -> List[RawNewsItem]:
    dedup: Dict[str, RawNewsItem] = {}
    for item in items:
        key = f"{item.title.strip().lower()}|{item.url.strip().lower()}"
        if key and key not in dedup:
            dedup[key] = item
    return list(dedup.values())


def _clean_text(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_match_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    lowered_text = (text or "").lower()
    lowered_term = (term or "").lower()
    if lowered_term in lowered_text:
        return True
    normalized_text = _normalize_match_text(lowered_text)
    normalized_term = _normalize_match_text(lowered_term)
    if not normalized_term:
        return False
    return normalized_term in normalized_text


def _sanitize_raw_item(item: RawNewsItem) -> RawNewsItem | None:
    title = (item.title or "").strip()
    summary = (item.summary or "").strip()
    raw_text = (item.raw_text or "").strip()
    url = (item.url or "").strip()
    source = (item.source or "unknown").strip() or "unknown"

    if not title:
        title = summary[:120].strip()
    if not title:
        title = raw_text[:120].strip()
    if not title:
        return None

    published_at = (item.published_at or "").strip()
    try:
        published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        published_at = published.astimezone(timezone.utc).isoformat()
    except Exception:
        published_at = datetime.now(timezone.utc).isoformat()

    return RawNewsItem(
        title=title,
        published_at=published_at,
        source=source,
        url=url,
        summary=summary,
        raw_text=raw_text,
        query_context=(item.query_context or "").strip(),
    )


def _normalized_keywords(context: Dict[str, Any]) -> List[str]:
    raw = [str(value).strip().lower() for value in context.get("sector_keywords", []) if str(value).strip()]
    merged = [*raw, *_MACRO_KEYWORDS]
    dedup: Dict[str, None] = {}
    for item in merged:
        if item not in dedup:
            dedup[item] = None
    return list(dedup.keys())


def _sentiment_label(score: float) -> str:
    if score > 0.2:
        return "positive"
    if score < -0.2:
        return "negative"
    return "neutral"


def _recency_score(published_at: str) -> float:
    try:
        published = datetime.fromisoformat((published_at or "").replace("Z", "+00:00"))
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        age_hours = max(0.0, (datetime.now(timezone.utc) - published.astimezone(timezone.utc)).total_seconds() / 3600.0)
    except Exception:
        age_hours = 72.0
    # Half-life style decay around ~3 days.
    return float(max(0.05, min(1.0, np.exp(-age_hours / 72.0))))


def _match_count(text: str, terms: Sequence[str]) -> int:
    unique_terms = []
    seen: set[str] = set()
    for term in terms:
        key = str(term or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_terms.append(key)
    return sum(1 for term in unique_terms if _contains_term(text, term))


def _why_it_matters(item: RawNewsItem, sentiment_label: str, relevance_score: float, ticker_hits: int, keyword_hits: int, recency: float) -> str:
    text = f"{item.title} {item.summary} {item.raw_text}".lower()
    topic_hits = [kw for kw in ["earnings", "inflation", "rates", "guidance", "recession", "volatility"] if kw in text]
    topic_text = ", ".join(topic_hits[:2]) if topic_hits else "macro backdrop"
    recency_text = "fresh" if recency > 0.65 else "dated"
    return (
        f"{sentiment_label} tone; relevance {relevance_score:.2f} from ticker hits={ticker_hits}, "
        f"keyword hits={keyword_hits}, {recency_text} timing; key topic: {topic_text}."
    )


def _score_news_item(item: RawNewsItem, context: Dict[str, Any], analyzer: SentimentAnalyzer) -> NewsItemResult:
    text = f"{item.title} {item.summary} {item.raw_text} {item.query_context}".lower()
    tickers = [str(t).lower() for t in context.get("tickers", []) if str(t).strip()]
    keywords = _normalized_keywords(context)

    ticker_hits = _match_count(text, tickers)
    keyword_hits = _match_count(text, keywords)
    ticker_component = 1.0 if ticker_hits > 0 else 0.0
    keyword_component = min(1.0, keyword_hits / 3.0)
    recency_component = _recency_score(item.published_at)

    match_strength = float(0.7 * ticker_component + 0.3 * keyword_component)
    relevance = float(match_strength * (0.85 + 0.15 * recency_component))
    relevance = float(max(0.0, min(1.0, relevance)))

    short_summary = item.summary.strip() or item.title.strip()
    if len(short_summary) > 260:
        short_summary = short_summary[:260].rstrip() + "..."

    sentiment = float(analyzer.score(f"{item.title}. {item.summary}. {item.raw_text}"))
    label = _sentiment_label(sentiment)

    return NewsItemResult(
        title=item.title,
        published_at=item.published_at,
        source=item.source,
        url=item.url,
        summary=short_summary,
        relevance_score=float(round(relevance, 4)),
        why_it_matters=_why_it_matters(item, label, relevance, ticker_hits, keyword_hits, recency_component),
        sentiment_score=float(round(sentiment, 4)),
        sentiment_label=label,
        tags=["news", label],
        raw_text=item.raw_text[:1200] if item.raw_text else "",
    )


def _cache_key(
    tickers: Sequence[str],
    start_date: datetime,
    end_date: datetime,
    context: Dict[str, Any],
    provider_names: Sequence[str],
    fetch_time_budget_seconds: float = _DEFAULT_FETCH_TIME_BUDGET_SECONDS,
    min_items_before_early_stop: int = _DEFAULT_MIN_ITEMS_BEFORE_EARLY_STOP,
) -> str:
    payload = {
        "schema": _CACHE_SCHEMA_VERSION,
        "tickers": [str(value).upper() for value in tickers],
        "start": start_date.date().isoformat(),
        "end": end_date.date().isoformat(),
        "keywords": _normalized_keywords(context),
        "providers": list(provider_names),
        "fetch_time_budget_seconds": round(float(fetch_time_budget_seconds), 3),
        "min_items_before_early_stop": int(min_items_before_early_stop),
    }
    return json.dumps(payload, sort_keys=True)


def _clean_fetch_errors(errors: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    for item in errors:
        text = str(item).strip()
        if not text:
            continue
        if "empty result" in text.lower():
            continue
        cleaned.append(text)
    return cleaned


def clear_news_cache() -> None:
    _NEWS_CACHE.clear()


def _fetch_provider_safely(
    provider: NewsProvider,
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
    context: Dict[str, Any],
) -> _ProviderFetchResult:
    try:
        items = provider.fetch(tickers, start_date, end_date, context=context) or []
        normalized = _dedupe_raw_items(items)
        sanitized = [
            prepared
            for prepared in (_sanitize_raw_item(item) for item in normalized)
            if prepared is not None
        ]
        return _ProviderFetchResult(items=sanitized)
    except MissingAPIKeyError as exc:
        message = f"{provider.provider_name}: {exc}"
        logger.info(message)
        return _ProviderFetchResult(items=[], error=message)
    except Exception as exc:
        message = f"{provider.provider_name} fetch failed: {exc}"
        logger.warning(message)
        return _ProviderFetchResult(items=[], error=message)


def _fetch_raw_news(
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
    context: Dict[str, Any],
    providers: Sequence[NewsProvider],
    fetch_time_budget_seconds: float = _DEFAULT_FETCH_TIME_BUDGET_SECONDS,
    min_items_before_early_stop: int = _DEFAULT_MIN_ITEMS_BEFORE_EARLY_STOP,
) -> Tuple[List[RawNewsItem], List[str], str]:
    provider_list = list(providers)
    provider_names = [provider.provider_name for provider in provider_list]
    time_budget = _coerce_fetch_time_budget(fetch_time_budget_seconds)
    sufficient_items = max(1, int(min_items_before_early_stop))
    key = _cache_key(
        tickers,
        start_date,
        end_date,
        context,
        provider_names,
        fetch_time_budget_seconds=time_budget,
        min_items_before_early_stop=sufficient_items,
    )
    now = datetime.now(timezone.utc)
    cached = _NEWS_CACHE.get(key)
    if cached:
        cached_at, cached_items, cached_errors, used_provider = cached
        if (now - cached_at).total_seconds() <= _CACHE_TTL_SECONDS:
            return list(cached_items), _clean_fetch_errors(cached_errors), used_provider

    if not provider_list:
        _NEWS_CACHE[key] = (now, [], [], "none")
        return [], [], "none"

    deadline = time.monotonic() + time_budget
    provider_context = {
        **context,
        "_news_fetch_deadline_monotonic": deadline,
        "_news_sufficient_items": sufficient_items,
    }
    executor = ThreadPoolExecutor(
        max_workers=min(_MAX_PROVIDER_WORKERS, len(provider_list)),
        thread_name_prefix="news-provider",
    )
    futures: Dict[int, Future[_ProviderFetchResult]] = {}
    results: Dict[int, _ProviderFetchResult] = {}
    early_stop_index: int | None = None

    def submit_provider(index: int) -> None:
        futures[index] = executor.submit(
            _fetch_provider_safely,
            provider_list[index],
            tickers,
            start_date,
            end_date,
            dict(provider_context),
        )

    def collect_completed(completed: Sequence[Future[_ProviderFetchResult]]) -> None:
        completed_set = set(completed)
        for index, future in futures.items():
            if future not in completed_set or index in results:
                continue
            try:
                results[index] = future.result()
            except Exception as exc:
                message = f"{provider_names[index]} fetch failed: {exc}"
                logger.warning(message)
                results[index] = _ProviderFetchResult(items=[], error=message)

    try:
        # Give the preferred provider a short head start. A quick, sufficiently
        # complete response avoids launching fallbacks; a slow primary cannot
        # consume the entire news budget before fallbacks start concurrently.
        submit_provider(0)
        primary_grace = min(
            _PRIMARY_PROVIDER_GRACE_SECONDS,
            time_budget * 0.2,
            max(0.0, deadline - time.monotonic()),
        )
        if primary_grace > 0:
            completed, _ = wait(
                [futures[0]],
                timeout=primary_grace,
            )
            collect_completed(list(completed))

        if 0 in results and len(_dedupe_raw_items(results[0].items)) >= sufficient_items:
            early_stop_index = 0
        else:
            for index in range(1, len(provider_list)):
                if time.monotonic() >= deadline:
                    break
                submit_provider(index)

            pending = {
                future
                for index, future in futures.items()
                if index not in results
            }
            remaining = deadline - time.monotonic()
            if pending and remaining > 0:
                completed, _ = wait(pending, timeout=remaining)
                collect_completed(list(completed))

        if early_stop_index is not None:
            for index, future in futures.items():
                if index > early_stop_index:
                    future.cancel()
            results = {
                index: result
                for index, result in results.items()
                if index <= early_stop_index
            }
        else:
            # Capture futures that completed at the deadline boundary before
            # classifying the remaining work as timed out.
            collect_completed([future for future in futures.values() if future.done()])
            timeout_message = (
                f"fetch timed out after {time_budget:.2f}s total news budget"
            )
            for index, provider_name in enumerate(provider_names):
                if index in results:
                    continue
                future = futures.get(index)
                if future is not None:
                    future.cancel()
                results[index] = _ProviderFetchResult(
                    items=[],
                    error=f"{provider_name} {timeout_message}",
                )
    finally:
        # Never wait beyond the caller-visible budget. Built-in HTTP providers
        # also receive the shared deadline so their in-flight work exits soon.
        executor.shutdown(wait=False, cancel_futures=True)

    errors: List[str] = []
    successful_providers: List[str] = []
    aggregated: List[RawNewsItem] = []
    for index, provider in enumerate(provider_list):
        result = results.get(index)
        if result is None:
            continue
        if result.error:
            errors.append(result.error)
        if result.items:
            aggregated.extend(result.items)
            successful_providers.append(provider.provider_name)
        elif not result.error:
            logger.info("%s returned empty result for current query.", provider.provider_name)

    deduped = _dedupe_raw_items(aggregated)
    used_provider = ",".join(successful_providers[:3]) if deduped else "none"

    clean_errors = _clean_fetch_errors(errors)
    _NEWS_CACHE[key] = (
        datetime.now(timezone.utc),
        deduped,
        clean_errors,
        used_provider,
    )
    return deduped, clean_errors, used_provider


def build_news_analysis(
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
    context: Dict[str, Any],
    provider: NewsProvider | None = None,
    analyzer: SentimentAnalyzer | None = None,
    max_items: int = 120,
    fetch_time_budget_seconds: float = _DEFAULT_FETCH_TIME_BUDGET_SECONDS,
    min_items_before_early_stop: int = _DEFAULT_MIN_ITEMS_BEFORE_EARLY_STOP,
) -> NewsResult:
    use_context = dict(context or {})
    sentiment_analyzer = analyzer or _build_sentiment_analyzer()
    fetch_budget = _coerce_fetch_time_budget(fetch_time_budget_seconds)
    target_item_count = min(
        max(1, int(max_items)),
        max(1, int(min_items_before_early_stop)),
    )

    if provider is None:
        real_providers: List[NewsProvider] = [
            NewsApiProvider(),
            YahooNewsProvider(),
            GoogleRssNewsProvider(),
        ]
        raw_items, errors, used_provider = _fetch_raw_news(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            context=use_context,
            providers=real_providers,
            fetch_time_budget_seconds=fetch_budget,
            min_items_before_early_stop=target_item_count,
        )
        if not raw_items:
            sample_items, sample_errors, sample_provider = _fetch_raw_news(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                context=use_context,
                providers=[SampleNewsProvider()],
                fetch_time_budget_seconds=fetch_budget,
                min_items_before_early_stop=target_item_count,
            )
            raw_items = sample_items
            errors = [*errors, *sample_errors]
            used_provider = sample_provider if sample_items else used_provider
    else:
        raw_items, errors, used_provider = _fetch_raw_news(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            context=use_context,
            providers=[provider],
            fetch_time_budget_seconds=fetch_budget,
            min_items_before_early_stop=target_item_count,
        )
    errors = _clean_fetch_errors(errors)

    if not raw_items:
        result_context = {
            **use_context,
            "sentiment_score": 0.0,
            "sentiment_dispersion": 0.0,
            "relevance_coverage": 0.0,
            "provider_used": used_provider,
            "fetch_errors": errors,
        }
        message = "; ".join(errors) if errors else "No news returned for selected inputs."
        return NewsResult(
            available=True,
            items=[],
            context=result_context,
            sentiment_score=0.0,
            sentiment_dispersion=0.0,
            error=message,
        )

    scored: List[NewsItemResult] = []
    for item in raw_items:
        try:
            scored.append(_score_news_item(item, use_context, sentiment_analyzer))
        except Exception as exc:
            message = f"scoring failed for '{item.title[:50]}': {exc}"
            logger.warning(message)
            errors.append(message)

    scored.sort(key=lambda item: item.relevance_score, reverse=True)
    selected = scored[: max(1, int(max_items))]
    weighted_total = float(sum(item.relevance_score for item in selected))
    if weighted_total > 0:
        sentiment_score = float(sum(item.sentiment_score * item.relevance_score for item in selected) / weighted_total)
    else:
        sentiment_score = float(np.mean([item.sentiment_score for item in selected])) if selected else 0.0
    sentiment_dispersion = float(np.std([item.sentiment_score for item in selected])) if selected else 0.0
    relevance_coverage = float(np.mean([item.relevance_score for item in selected])) if selected else 0.0

    result_context = {
        **use_context,
        "sentiment_score": sentiment_score,
        "sentiment_dispersion": sentiment_dispersion,
        "relevance_coverage": relevance_coverage,
        "provider_used": used_provider,
        "fetch_errors": errors,
    }
    return NewsResult(
        available=True,
        items=selected,
        context=result_context,
        sentiment_score=sentiment_score,
        sentiment_dispersion=sentiment_dispersion,
        error="; ".join(errors[:3]) if errors else "",
    )


def build_news_rows_for_ui(news: NewsResult | None) -> List[Dict[str, Any]]:
    if news is None:
        return []
    rows: List[Dict[str, Any]] = []
    for item in news.items:
        indicator = "green" if item.sentiment_score > 0.2 else "red" if item.sentiment_score < -0.2 else "amber"
        rows.append(
            {
                "Title": item.title,
                "Published": item.published_at,
                "Source": item.source,
                "Relevance": float(item.relevance_score),
                "Sentiment": float(item.sentiment_score),
                "Sentiment Label": item.sentiment_label,
                "Sentiment Color": indicator,
                "Summary": item.summary,
                "Why it matters": item.why_it_matters,
                "URL": item.url,
            }
        )
    return rows


def recent_window_endpoints(days: int = 30) -> tuple[datetime, datetime]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(1, int(days)))
    return start, end
