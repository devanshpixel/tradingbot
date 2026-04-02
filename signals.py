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
    try:
        df = yf.download(
            tickers=symbol,
            period="1d",
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()
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
        "entry": 0.0,
        "stop_loss": 0.0,
        "target": 0.0,
        "rr": 0.0,
        "rsi": 50.0,
        "macd_hist": 0.0,
        "bb_pos": 0.5,
        "vol_spike_x": 0.0,
        "candlestick": "NONE",
        "support": 0.0,
        "resistance": 0.0,
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
    try:
        df = _fetch(symbol=symbol, interval=interval)
        if df.empty or len(df) < 20:
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
        macd_obj = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        macd_line_series = macd_obj.macd().dropna()
        macd_signal_series = macd_obj.macd_signal().dropna()
        macd_hist_series = macd_obj.macd_diff().dropna()
        bb = BollingerBands(close=close, window=20, window_dev=2)
        bb_high_series = bb.bollinger_hband().dropna()
        bb_low_series = bb.bollinger_lband().dropna()

        rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
        rsi_prev = float(rsi_series.iloc[-2]) if len(rsi_series) >= 2 else rsi
        macd_hist = float(macd_hist_series.iloc[-1]) if not macd_hist_series.empty else 0.0
        macd_line = float(macd_line_series.iloc[-1]) if not macd_line_series.empty else 0.0
        macd_signal = float(macd_signal_series.iloc[-1]) if not macd_signal_series.empty else 0.0
        macd_line_prev = float(macd_line_series.iloc[-2]) if len(macd_line_series) >= 2 else macd_line
        macd_signal_prev = float(macd_signal_series.iloc[-2]) if len(macd_signal_series) >= 2 else macd_signal
        bb_high = float(bb_high_series.iloc[-1]) if not bb_high_series.empty else last_close
        bb_low = float(bb_low_series.iloc[-1]) if not bb_low_series.empty else last_close
        bb_pos = 0.5 if bb_high == bb_low else (last_close - bb_low) / (bb_high - bb_low)

        vol_avg = float(vol.tail(20).mean()) if len(vol) >= 20 else float(vol.mean())
        vol_last = float(vol.iloc[-1]) if len(vol) else 0.0
        vol_spike = float(vol_last / vol_avg) if vol_avg > 0 else 0.0

        # Intraday VWAP (session cumulative)
        typical_price = (high.fillna(close) + low.fillna(close) + close) / 3.0
        vwap_num = (typical_price * vol).cumsum()
        vwap_den = vol.cumsum().replace(0, pd.NA)
        vwap_series = (vwap_num / vwap_den).dropna()
        vwap = float(vwap_series.iloc[-1]) if not vwap_series.empty else last_close

        day_range = float((high_clean.iloc[-1] - low_clean.iloc[-1]) / max(1e-9, last_close) * 100.0)
        support, resistance = _support_resistance(low=low, high=high, lookback=20)
        pattern_name, pattern_score = _detect_candlestick_pattern(open_=open_, high=high, low=low, close=close)

        rsi_rising = rsi > rsi_prev
        rsi_falling = rsi < rsi_prev
        macd_cross_up = macd_line_prev <= macd_signal_prev and macd_line > macd_signal
        macd_cross_down = macd_line_prev >= macd_signal_prev and macd_line < macd_signal
        volume_spike_ok = vol_spike > 1.5
        price_above_vwap = last_close > vwap
        price_below_vwap = last_close < vwap

        buy_setup = (40.0 <= rsi <= 60.0) and rsi_rising and macd_cross_up and volume_spike_ok and price_above_vwap
        sell_setup = (((rsi > 60.0) and rsi_falling) or ((rsi < 40.0) and rsi_falling)) and macd_cross_down and volume_spike_ok and price_below_vwap

        side = "LONG" if buy_setup else ("SHORT" if sell_setup else "FLAT")
        signal = "BUY" if side == "LONG" else ("SELL" if side == "SHORT" else "HOLD")

        # Confidence = indicator alignment + volume strength + trend strength
        alignment_count = sum(
            [
                bool((40.0 <= rsi <= 60.0) and rsi_rising) if side == "LONG" else bool((((rsi > 60.0) and rsi_falling) or ((rsi < 40.0) and rsi_falling))),
                bool(macd_cross_up) if side == "LONG" else bool(macd_cross_down),
                bool(price_above_vwap) if side == "LONG" else bool(price_below_vwap),
            ]
        ) if side in {"LONG", "SHORT"} else 0
        indicator_alignment = (alignment_count / 3.0) * 60.0
        volume_strength = min(25.0, max(0.0, (vol_spike - 1.0) * 25.0))
        trend_strength = min(15.0, abs(macd_hist) * 300.0)
        confidence = indicator_alignment + volume_strength + trend_strength
        score = (indicator_alignment * 0.5) + (volume_strength * 0.3) + (trend_strength * 0.2)
        score_total = score + float(sentiment.confidence_boost) * 0.2

        volume_confirmed = volume_spike_ok

        if side == "LONG" and market_bearish:
            confidence *= 0.6
        if side in {"LONG", "SHORT"} and not time_window_ok:
            confidence *= 0.8

        entry = stop = target = rr = 0.0
        if side in {"LONG", "SHORT"}:
            entry, stop, target = _levels_from_price(price=last_close, atr_proxy_pct=max(0.6, day_range), direction=side)
            rr = _rr(entry=entry, stop=stop, target=target, side=side)

        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": round(float(confidence), 1),
            "entry": round(float(entry), 2),
            "stop_loss": round(float(stop), 2),
            "target": round(float(target), 2),
            "rr": round(float(rr), 2),
            "rsi": round(float(rsi), 1),
            "macd_hist": round(float(macd_hist), 4),
            "bb_pos": round(float(bb_pos), 3),
            "vol_spike_x": round(float(vol_spike), 2),
            "candlestick": pattern_name,
            "support": round(float(support), 2) if support is not None else round(float(last_close), 2),
            "resistance": round(float(resistance), 2) if resistance is not None else round(float(last_close), 2),
            "market_bias": "BEARISH" if market_bearish else "BULLISH_OR_NEUTRAL",
            "time_window_ok": time_window_ok,
            "volume_confirmed": bool(volume_confirmed),
            "vwap": round(float(vwap), 2),
            "sentiment": sentiment.label,
            "sent_polarity": round(sentiment.polarity, 3),
            "sent_samples": sentiment.sample_size,
            "score_total": round(float(score_total), 2),
            "score_tech": round(float(score), 2),
            **{f"sent_{k}": v for k, v in asdict(sentiment).items() if k not in {"label", "polarity"}},
        }
    except Exception:
        row = _hold_row(symbol, sentiment)
        row["time_window_ok"] = time_window_ok
        row["market_bias"] = "BEARISH" if market_bearish else "BULLISH_OR_NEUTRAL"
        return row

