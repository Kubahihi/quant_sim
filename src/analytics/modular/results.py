from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


@dataclass
class BaseResult:
    name: str
    family: str
    available: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelResult(BaseResult):
    pass


@dataclass
class SignalResult(BaseResult):
    direction: str = "neutral"
    score: float = 0.0


@dataclass
class NewsItemResult:
    title: str
    published_at: str
    source: str
    url: str
    summary: str
    relevance_score: float
    why_it_matters: str
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    raw_text: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NewsResult:
    available: bool
    items: List[NewsItemResult] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    sentiment_score: float = 0.0
    sentiment_dispersion: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "items": [item.to_dict() for item in self.items],
            "context": self.context,
            "sentiment_score": self.sentiment_score,
            "sentiment_dispersion": self.sentiment_dispersion,
            "error": self.error,
        }


@dataclass
class SummaryResult:
    generated_at: str
    composite_score: float
    regime_label: str
    confidence: float
    highlights: List[str]
    model_snapshot: Dict[str, Any]
    signal_snapshot: Dict[str, Any]
    risk_flags: List[str] = field(default_factory=list)
    strongest_signals: List[Dict[str, Any]] = field(default_factory=list)
    agreement_score: float = 0.0
    disagreement_score: float = 0.0
    uncertainty: float = 0.0
    expected_return_view: float = 0.0
    expected_risk_view: float = 0.0
    regime_interpretation: str = ""
    drawdown_implication: str = ""
    volatility_implication: str = ""
    recent_changes: List[str] = field(default_factory=list)
    news_sentiment: float = 0.0
    news_sentiment_dispersion: float = 0.0
    top_relevant_news: List[Dict[str, Any]] = field(default_factory=list)
    news_implication: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    config: Dict[str, Any]
    universe: List[str]
    date_range: Dict[str, str]
    outputs: Dict[str, Any]
    metrics: Dict[str, Any]
    summary: Dict[str, Any]
    news: Dict[str, Any]
    sentiment: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def now(run_id: str, **kwargs: Any) -> "RunRecord":
        return RunRecord(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
