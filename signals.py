from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

from news import SentimentResult


def _fetch(symbol: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        period="1d",
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns={c: c.lower() for c in df.columns})
    return df


def _to_1d_series(df: pd.DataFrame, col: str, default: Any = None) -> pd.Series:
    """
    Return a single Series for a column that may come back as DataFrame due to
    duplicate names in yfinance output.
    """
    if col not in df.columns:
        if isinstance(default, pd.Series):
            return default
        if default is None:
            return pd.Series(index=df.index, dtype=float)
        return pd.Series(default, index=df.index, dtype=float)

    out = df[col].squeeze()
    if isinstance(out, pd.DataFrame):
        out = out.iloc[:, 0]
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=df.index)
    return out


def _hold_row(symbol: str, sentiment: SentimentResult) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "signal": "HOLD",
        "confidence": 0.0,
        "entry": None,
        "stop_loss": None,
        "target": None,
        "rr": None,
        "rsi": None,
        "macd_hist": None,
        "bb_pos": None,
        "vol_spike_x": None,
        "sentiment": sentiment.label,
        "sent_polarity": round(sentiment.polarity, 3),
        "sent_samples": sentiment.sample_size,
        "score_total": 0.0,
        "score_tech": 0.0,
        **{f"sent_{k}": v for k, v in asdict(sentiment).items() if k not in {"label", "polarity"}},
    }


def _levels_from_price(price: float, atr_proxy_pct: float, direction: str) -> tuple[float, float, float]:
    entry = float(price)
    move = max(0.2, float(atr_proxy_pct)) / 100.0 * entry
    if direction == "LONG":
        stop = entry - 1.0 * move
        target = entry + 1.8 * move
    else:
        stop = entry + 1.0 * move
        target = entry - 1.8 * move
    return round(entry, 2), round(stop, 2), round(target, 2)


def _rr(entry: float, stop: float, target: float, side: str) -> float:
    if side == "LONG":
        risk = max(1e-9, entry - stop)
        reward = target - entry
    else:
        risk = max(1e-9, stop - entry)
        reward = entry - target
    return round(reward / risk, 2)


def generate_signal_row(symbol: str, interval: str, sentiment: SentimentResult) -> dict[str, Any]:
    df = _fetch(symbol=symbol, interval=interval)
    if df.empty or len(df) < 35:
        return _hold_row(symbol, sentiment)

    close = pd.to_numeric(_to_1d_series(df, "close"), errors="coerce")
    high = pd.to_numeric(_to_1d_series(df, "high", default=close), errors="coerce")
    low = pd.to_numeric(_to_1d_series(df, "low", default=close), errors="coerce")
    vol = pd.to_numeric(_to_1d_series(df, "volume", default=0.0), errors="coerce").fillna(0.0)

    close_clean = close.dropna()
    high_clean = high.dropna()
    low_clean = low.dropna()
    if close_clean.empty or high_clean.empty or low_clean.empty:
        return _hold_row(symbol, sentiment)

    last_close = float(close_clean.iloc[-1])

    rsi_series = RSIIndicator(close=close, window=14).rsi().dropna()
    macd_hist_series = MACD(close=close, window_slow=26, window_fast=12, window_sign=9).macd_diff().dropna()
    bb = BollingerBands(close=close, window=20, window_dev=2)
    bb_high_series = bb.bollinger_hband().dropna()
    bb_low_series = bb.bollinger_lband().dropna()

    if rsi_series.empty or macd_hist_series.empty or bb_high_series.empty or bb_low_series.empty:
        return _hold_row(symbol, sentiment)

    rsi = float(rsi_series.iloc[-1])
    macd_hist = float(macd_hist_series.iloc[-1])
    bb_high = float(bb_high_series.iloc[-1])
    bb_low = float(bb_low_series.iloc[-1])
    bb_pos = 0.5 if bb_high == bb_low else (last_close - bb_low) / (bb_high - bb_low)

    vol_avg = float(vol.tail(20).mean()) if len(vol) >= 20 else float(vol.mean())
    vol_spike = float(vol.iloc[-1] / vol_avg) if vol_avg > 0 else 0.0

    day_range = float((high_clean.iloc[-1] - low_clean.iloc[-1]) / max(1e-9, last_close) * 100.0)

    score = 0.0
    if rsi <= 30:
        score += 18
    elif rsi >= 70:
        score -= 18
    else:
        score += (50 - rsi) * 0.18

    score += max(-20.0, min(20.0, macd_hist * 120.0))
    score += max(-12.0, min(12.0, (0.5 - bb_pos) * 24.0))
    score += max(0.0, min(12.0, (vol_spike - 1.0) * 6.0))

    score_total = score + float(sentiment.confidence_boost)

    if score_total >= 20:
        side = "LONG"
        signal = "BUY"
    elif score_total <= -20:
        side = "SHORT"
        signal = "SHORT"
    else:
        side = "FLAT"
        signal = "HOLD"

    confidence = min(100.0, max(0.0, abs(score_total) * 2.2))

    entry = stop = target = rr = None
    if side in {"LONG", "SHORT"}:
        entry, stop, target = _levels_from_price(price=last_close, atr_proxy_pct=max(0.6, day_range), direction=side)
        rr = _rr(entry=entry, stop=stop, target=target, side=side)

    return {
        "symbol": symbol,
        "signal": signal,
        "confidence": round(float(confidence), 1),
        "entry": entry,
        "stop_loss": stop,
        "target": target,
        "rr": rr,
        "rsi": round(rsi, 1),
        "macd_hist": round(macd_hist, 4),
        "bb_pos": round(float(bb_pos), 3),
        "vol_spike_x": round(vol_spike, 2),
        "sentiment": sentiment.label,
        "sent_polarity": round(sentiment.polarity, 3),
        "sent_samples": sentiment.sample_size,
        "score_total": round(float(score_total), 2),
        "score_tech": round(float(score), 2),
        **{f"sent_{k}": v for k, v in asdict(sentiment).items() if k not in {"label", "polarity"}},
    }

