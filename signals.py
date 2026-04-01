from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, time
from typing import Any

import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo
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


def _market_is_bearish(interval: str) -> bool:
    idx = _fetch(symbol="^NSEI", interval=interval)
    if idx.empty:
        return False
    close = pd.to_numeric(_to_1d_series(idx, "close"), errors="coerce").dropna()
    if len(close) < 3:
        return False
    return bool(close.iloc[-1] < close.iloc[0])


def _in_intraday_signal_window() -> bool:
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata")).time()
    return time(9, 15) <= now_ist <= time(11, 30)


def _detect_candlestick_pattern(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> tuple[str, float]:
    o = pd.to_numeric(open_, errors="coerce").dropna()
    h = pd.to_numeric(high, errors="coerce").dropna()
    l = pd.to_numeric(low, errors="coerce").dropna()
    c = pd.to_numeric(close, errors="coerce").dropna()
    n = min(len(o), len(h), len(l), len(c))
    if n < 2:
        return "NONE", 0.0

    o = o.iloc[-n:]
    h = h.iloc[-n:]
    l = l.iloc[-n:]
    c = c.iloc[-n:]

    o1, h1, l1, c1 = float(o.iloc[-1]), float(h.iloc[-1]), float(l.iloc[-1]), float(c.iloc[-1])
    body1 = abs(c1 - o1)
    rng1 = max(1e-9, h1 - l1)
    upper1 = h1 - max(o1, c1)
    lower1 = min(o1, c1) - l1

    # Doji
    if body1 <= 0.1 * rng1:
        return "DOJI", 2.0

    # Hammer (bullish reversal)
    if lower1 >= 2.0 * body1 and upper1 <= body1 and c1 > o1:
        return "HAMMER", 6.0

    if n >= 2:
        o0, c0 = float(o.iloc[-2]), float(c.iloc[-2])
        prev_bear = c0 < o0
        prev_bull = c0 > o0
        curr_bear = c1 < o1
        curr_bull = c1 > o1

        # Engulfing
        if prev_bear and curr_bull and o1 <= c0 and c1 >= o0:
            return "BULL_ENGULFING", 7.0
        if prev_bull and curr_bear and o1 >= c0 and c1 <= o0:
            return "BEAR_ENGULFING", -7.0

    if n >= 3:
        o2, c2 = float(o.iloc[-3]), float(c.iloc[-3])
        o1m, c1m = float(o.iloc[-2]), float(c.iloc[-2])
        # Morning star approximation
        if c2 < o2 and abs(c1m - o1m) <= 0.25 * max(1e-9, abs(c2 - o2)) and c1 > o1 and c1 > (o2 + c2) / 2:
            return "MORNING_STAR", 8.0

    return "NONE", 0.0


def _support_resistance(low: pd.Series, high: pd.Series, lookback: int = 20) -> tuple[float | None, float | None]:
    low_n = pd.to_numeric(low, errors="coerce").dropna().tail(lookback)
    high_n = pd.to_numeric(high, errors="coerce").dropna().tail(lookback)
    if low_n.empty or high_n.empty:
        return None, None
    return float(low_n.min()), float(high_n.max())


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
        "candlestick": "NONE",
        "support": None,
        "resistance": None,
        "market_bias": "NEUTRAL",
        "time_window_ok": False,
        "volume_confirmed": False,
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
    time_window_ok = _in_intraday_signal_window()
    market_bearish = _market_is_bearish(interval=interval)

    df = _fetch(symbol=symbol, interval=interval)
    if df.empty or len(df) < 35:
        row = _hold_row(symbol, sentiment)
        row["time_window_ok"] = time_window_ok
        row["market_bias"] = "BEARISH" if market_bearish else "BULLISH_OR_NEUTRAL"
        return row

    close = pd.to_numeric(_to_1d_series(df, "close"), errors="coerce")
    open_ = pd.to_numeric(_to_1d_series(df, "open", default=close), errors="coerce")
    high = pd.to_numeric(_to_1d_series(df, "high", default=close), errors="coerce")
    low = pd.to_numeric(_to_1d_series(df, "low", default=close), errors="coerce")
    vol = pd.to_numeric(_to_1d_series(df, "volume", default=0.0), errors="coerce").fillna(0.0)

    close_clean = close.dropna()
    high_clean = high.dropna()
    low_clean = low.dropna()
    if close_clean.empty or high_clean.empty or low_clean.empty:
        row = _hold_row(symbol, sentiment)
        row["time_window_ok"] = time_window_ok
        row["market_bias"] = "BEARISH" if market_bearish else "BULLISH_OR_NEUTRAL"
        return row

    last_close = float(close_clean.iloc[-1])

    rsi_series = RSIIndicator(close=close, window=14).rsi().dropna()
    macd_hist_series = MACD(close=close, window_slow=26, window_fast=12, window_sign=9).macd_diff().dropna()
    bb = BollingerBands(close=close, window=20, window_dev=2)
    bb_high_series = bb.bollinger_hband().dropna()
    bb_low_series = bb.bollinger_lband().dropna()

    if rsi_series.empty or macd_hist_series.empty or bb_high_series.empty or bb_low_series.empty:
        row = _hold_row(symbol, sentiment)
        row["time_window_ok"] = time_window_ok
        row["market_bias"] = "BEARISH" if market_bearish else "BULLISH_OR_NEUTRAL"
        return row

    rsi = float(rsi_series.iloc[-1])
    macd_hist = float(macd_hist_series.iloc[-1])
    bb_high = float(bb_high_series.iloc[-1])
    bb_low = float(bb_low_series.iloc[-1])
    bb_pos = 0.5 if bb_high == bb_low else (last_close - bb_low) / (bb_high - bb_low)

    vol_avg = float(vol.tail(20).mean()) if len(vol) >= 20 else float(vol.mean())
    vol_last = float(vol.iloc[-1]) if len(vol) else 0.0
    vol_spike = float(vol_last / vol_avg) if vol_avg > 0 else 0.0

    day_range = float((high_clean.iloc[-1] - low_clean.iloc[-1]) / max(1e-9, last_close) * 100.0)
    support, resistance = _support_resistance(low=low, high=high, lookback=20)
    pattern_name, pattern_score = _detect_candlestick_pattern(open_=open_, high=high, low=low, close=close)

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
    score += pattern_score

    if support is not None and resistance is not None and resistance > support:
        sr_pos = (last_close - support) / max(1e-9, resistance - support)
        # close near support = slightly bullish, near resistance = slightly bearish
        score += max(-8.0, min(8.0, (0.5 - sr_pos) * 16.0))

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

    # Volume-direction confirmation: require participation in the signaled direction.
    recent_mean = float(close.tail(5).mean()) if len(close) >= 5 else last_close
    bullish_pressure = last_close >= recent_mean
    bearish_pressure = last_close <= recent_mean
    volume_confirmed = vol_spike >= 1.05 and ((side == "LONG" and bullish_pressure) or (side == "SHORT" and bearish_pressure))

    if side in {"LONG", "SHORT"} and not volume_confirmed:
        side = "FLAT"
        signal = "HOLD"

    # Market regime filter: avoid BUY in bearish NIFTY conditions.
    if side == "LONG" and market_bearish:
        side = "FLAT"
        signal = "HOLD"

    # Time quality filter: only actionable signals during early session.
    if side in {"LONG", "SHORT"} and not time_window_ok:
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
        "candlestick": pattern_name,
        "support": round(float(support), 2) if support is not None else None,
        "resistance": round(float(resistance), 2) if resistance is not None else None,
        "market_bias": "BEARISH" if market_bearish else "BULLISH_OR_NEUTRAL",
        "time_window_ok": time_window_ok,
        "volume_confirmed": volume_confirmed if side in {"LONG", "SHORT"} else False,
        "sentiment": sentiment.label,
        "sent_polarity": round(sentiment.polarity, 3),
        "sent_samples": sentiment.sample_size,
        "score_total": round(float(score_total), 2),
        "score_tech": round(float(score), 2),
        **{f"sent_{k}": v for k, v in asdict(sentiment).items() if k not in {"label", "polarity"}},
    }

