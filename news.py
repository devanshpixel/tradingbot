from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any

import feedparser
import requests
from textblob import TextBlob


@dataclass(frozen=True)
class SentimentResult:
    polarity: float  # -1..+1
    label: str  # POSITIVE/NEGATIVE/NEUTRAL
    confidence_boost: float  # -15..+15 (used by signal engine)
    sample_size: int


def fetch_google_news_rss(symbol: str, max_items: int = 10) -> list[dict[str, Any]]:
    """
    Google News RSS is query-driven. For NSE symbols we query the base name too.
    """
    base = symbol.replace(".NS", "")
    q = f"{base} NSE"
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    items: list[dict[str, Any]] = []
    for e in (feed.entries or [])[:max_items]:
        items.append(
            {
                "source": "google_rss",
                "title": getattr(e, "title", "") or "",
                "summary": getattr(e, "summary", "") or "",
                "link": getattr(e, "link", "") or "",
                "published": getattr(e, "published", "") or "",
            }
        )
    return items


def fetch_finnhub_news(symbol: str, api_key: str, max_items: int = 10) -> list[dict[str, Any]]:
    """
    Finnhub Company News endpoint. For NSE symbols, Finnhub coverage varies.
    """
    if not api_key:
        return []
    to_dt = _dt.date.today()
    from_dt = to_dt - _dt.timedelta(days=7)
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": symbol.replace(".NS", ""),
        "from": from_dt.isoformat(),
        "to": to_dt.isoformat(),
        "token": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            return []
        out: list[dict[str, Any]] = []
        for a in data[:max_items]:
            out.append(
                {
                    "source": "finnhub",
                    "title": str(a.get("headline", "") or ""),
                    "summary": str(a.get("summary", "") or ""),
                    "link": str(a.get("url", "") or ""),
                    "published": str(a.get("datetime", "") or ""),
                }
            )
        return out
    except Exception:
        return []


def _polarity(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    try:
        return float(TextBlob(t).sentiment.polarity)
    except Exception:
        return 0.0


def sentiment_from_articles(articles: list[dict[str, Any]]) -> SentimentResult:
    if not articles:
        return SentimentResult(polarity=0.0, label="NEUTRAL", confidence_boost=0.0, sample_size=0)

    scores: list[float] = []
    for a in articles:
        title = str(a.get("title", "") or "")
        summary = str(a.get("summary", "") or "")
        scores.append(_polarity(f"{title}. {summary}"))

    if not scores:
        return SentimentResult(polarity=0.0, label="NEUTRAL", confidence_boost=0.0, sample_size=0)

    avg = sum(scores) / len(scores)
    if avg > 0.12:
        label = "POSITIVE"
    elif avg < -0.12:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    # Map sentiment to a bounded confidence boost used by the signal engine.
    boost = max(-15.0, min(15.0, avg * 60.0))
    return SentimentResult(polarity=float(avg), label=label, confidence_boost=float(boost), sample_size=len(scores))

