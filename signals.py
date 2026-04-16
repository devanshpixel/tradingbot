from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, time
from typing import Any

import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
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


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> float:
    """True Range average over `window` bars."""
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr_val = float(tr.dropna().tail(window).mean())
    return atr_val if atr_val > 0 else float(c.dropna().iloc[-1]) * 0.005


def _entry_timing(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: float,
    direction: str,
    lookback: int = 12,
) -> tuple[str, float, float]:
    """
    Classify the current bar's entry quality.

    Returns
    -------
    timing : str
        "PULLBACK" | "BREAKOUT" | "CHASING" | "EXTENDED" | "NO_BREAKOUT"
    entry_offset : float
        Fraction to add/subtract from last_close to get the ideal entry price.
        Positive = higher (for longs entering at breakout level + buffer).
    confidence_mult : float
        Multiply the raw confidence score by this factor.
    """
    c = pd.to_numeric(close, errors="coerce").dropna()
    h = pd.to_numeric(high, errors="coerce").dropna()
    l = pd.to_numeric(low, errors="coerce").dropna()

    min_len = lookback + 4
    if len(c) < min_len or atr <= 0:
        return "NO_BREAKOUT", 0.0, 0.80

    last_close = float(c.iloc[-1])
    # Reference window: bars before the most recent 2 bars (avoid including the signal bar)
    ref_high = float(h.iloc[-(lookback + 2): -2].max())
    ref_low  = float(l.iloc[-(lookback + 2): -2].min())

    if direction == "LONG":
        pivot = ref_high
        if last_close <= pivot:
            return "NO_BREAKOUT", 0.0, 0.65

        extension = (last_close - pivot) / atr

        # Detect pullback: price broke out then retraced back near the pivot
        recent_high = float(h.iloc[-4: -1].max()) if len(h) >= 4 else last_close
        pulled_back = recent_high > last_close * 1.003  # price was higher, now pulled back
        near_pivot  = extension < 0.6                  # still close to breakout level

        if pulled_back and near_pivot:
            # Best scenario: entering on a pullback to the breakout level
            # Entry = just above the pivot
            entry_price = pivot * 1.001
            offset = (entry_price - last_close) / last_close
            return "PULLBACK", offset, 1.20

        if extension <= 1.0:
            return "BREAKOUT", 0.0, 1.05      # fresh breakout — decent entry
        if extension <= 2.0:
            return "CHASING", 0.0, 0.70       # move already underway, risky
        return "EXTENDED", 0.0, 0.30          # far too late, strong penalty

    else:  # SHORT
        pivot = ref_low
        if last_close >= pivot:
            return "NO_BREAKOUT", 0.0, 0.65

        extension = (pivot - last_close) / atr

        recent_low = float(l.iloc[-4: -1].min()) if len(l) >= 4 else last_close
        bounced_up = recent_low < last_close * 0.997
        near_pivot = extension < 0.6

        if bounced_up and near_pivot:
            entry_price = pivot * 0.999
            offset = (entry_price - last_close) / last_close
            return "PULLBACK", offset, 1.20

        if extension <= 1.0:
            return "BREAKOUT", 0.0, 1.05
        if extension <= 2.0:
            return "CHASING", 0.0, 0.70
        return "EXTENDED", 0.0, 0.30


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

        # ADX needs at least 28 rows (14-period smoothing twice); fall back to 50 if insufficient
        _adx_window = 14
        adx_series = pd.Series(dtype=float)
        if len(close.dropna()) >= _adx_window * 2:
            try:
                adx_obj = ADXIndicator(high=high, low=low, close=close, window=_adx_window)
                adx_series = adx_obj.adx().dropna()
            except Exception:
                pass
        adx = float(adx_series.iloc[-1]) if not adx_series.empty else 50.0

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

        # Gate: if market is sideways (ADX < 20) skip directional signals entirely
        _ADX_SIDEWAYS = 20.0
        _ADX_TRENDING = 25.0
        if adx < _ADX_SIDEWAYS:
            row = _hold_row(symbol, sentiment)
            row.update({
                "time_window_ok": time_window_ok,
                "market_bias": "BEARISH" if market_bearish else "BULLISH_OR_NEUTRAL",
                "adx": round(float(adx), 1),
                "rsi": round(float(rsi), 1),
                "regime": "SIDEWAYS",
            })
            return row

        atr_val = _atr(high=high, low=low, close=close, window=14)

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
        volume_spike_ok = vol_spike > 1.3
        price_above_vwap = last_close > vwap
        price_below_vwap = last_close < vwap

        # --- Point-based scoring for both directions ---
        buy_pts = 0.0
        sell_pts = 0.0

        # RSI
        if rsi < 35:
            buy_pts += 18.0
        elif rsi < 45 and rsi_rising:
            buy_pts += 12.0
        elif rsi < 55 and rsi_rising:
            buy_pts += 7.0
        if rsi > 65:
            sell_pts += 18.0
        elif rsi > 55 and rsi_falling:
            sell_pts += 12.0
        elif rsi > 45 and rsi_falling:
            sell_pts += 7.0

        # MACD
        if macd_cross_up:
            buy_pts += 16.0
        elif macd_hist > 0:
            buy_pts += 8.0
        if macd_cross_down:
            sell_pts += 16.0
        elif macd_hist < 0:
            sell_pts += 8.0

        # MACD histogram magnitude (trend strength)
        hist_boost = min(10.0, abs(macd_hist) * 200.0)
        if macd_hist > 0:
            buy_pts += hist_boost
        elif macd_hist < 0:
            sell_pts += hist_boost

        # VWAP
        if price_above_vwap:
            buy_pts += 10.0
        if price_below_vwap:
            sell_pts += 10.0

        # Bollinger Bands position
        if bb_pos < 0.2:
            buy_pts += 8.0
        elif bb_pos > 0.8:
            sell_pts += 8.0

        # Volume
        vol_bonus = min(15.0, max(0.0, (vol_spike - 1.0) * 12.0))
        buy_pts += vol_bonus if volume_spike_ok else 0.0
        sell_pts += vol_bonus if volume_spike_ok else 0.0

        # Candlestick pattern
        bull_patterns = {"HAMMER", "BULL_ENGULFING", "MORNING_STAR"}
        bear_patterns = {"BEAR_ENGULFING"}
        if pattern_name in bull_patterns:
            buy_pts += abs(pattern_score)
        elif pattern_name in bear_patterns:
            sell_pts += abs(pattern_score)
        # DOJI is direction-neutral — slight boost to whichever is leading
        elif pattern_name == "DOJI":
            buy_pts += 1.5
            sell_pts += 1.5

        # Sentiment
        sent_boost = float(sentiment.confidence_boost)
        if sent_boost > 0:
            buy_pts += min(8.0, sent_boost * 0.15)
        elif sent_boost < 0:
            sell_pts += min(8.0, abs(sent_boost) * 0.15)

        # Market bias penalty
        if market_bearish:
            buy_pts *= 0.75

        # Determine direction
        _SIGNAL_THRESHOLD = 20.0
        if buy_pts >= _SIGNAL_THRESHOLD and buy_pts > sell_pts:
            side = "LONG"
        elif sell_pts >= _SIGNAL_THRESHOLD and sell_pts > buy_pts:
            side = "SHORT"
        else:
            side = "FLAT"
        signal = "BUY" if side == "LONG" else ("SELL" if side == "SHORT" else "HOLD")

        # Confidence: scale the winning side's score to 0–100
        _MAX_PTS = 85.0
        raw_pts = buy_pts if side == "LONG" else (sell_pts if side == "SHORT" else max(buy_pts, sell_pts))
        confidence = min(99.0, (raw_pts / _MAX_PTS) * 100.0)

        # ADX trend-strength multiplier: strong trend boosts confidence, weak trend penalises
        if adx >= 35.0:
            adx_mult = 1.15
        elif adx >= _ADX_TRENDING:
            adx_mult = 1.0 + (adx - _ADX_TRENDING) / (_ADX_TRENDING * 2.0)
        else:
            # ADX in [_ADX_SIDEWAYS, _ADX_TRENDING) — muted but not blocked
            adx_mult = 0.80
        confidence = min(99.0, confidence * adx_mult)
        regime = "STRONG_TREND" if adx >= 35.0 else ("TRENDING" if adx >= _ADX_TRENDING else "WEAK_TREND")

        # Time window and volume adjustments
        volume_confirmed = volume_spike_ok
        if side == "LONG" and market_bearish:
            confidence *= 0.85
        if side in {"LONG", "SHORT"} and not time_window_ok:
            confidence *= 0.90

        score = raw_pts
        score_total = score + sent_boost * 0.1

        # --- Entry timing quality ---
        entry_timing = "N/A"
        extension_atr = 0.0
        entry = stop = target = rr = 0.0

        if side in {"LONG", "SHORT"}:
            entry_timing, entry_offset, timing_mult = _entry_timing(
                close=close, high=high, low=low,
                atr=atr_val, direction=side, lookback=12,
            )
            confidence = min(99.0, confidence * timing_mult)

            # Extended entries are suppressed — treat as HOLD
            if entry_timing in {"EXTENDED", "NO_BREAKOUT"}:
                row = _hold_row(symbol, sentiment)
                row.update({
                    "time_window_ok": time_window_ok,
                    "market_bias": "BEARISH" if market_bearish else "BULLISH_OR_NEUTRAL",
                    "adx": round(float(adx), 1),
                    "rsi": round(float(rsi), 1),
                    "regime": regime,
                    "entry_timing": entry_timing,
                    "extension_atr": round(extension_atr, 2),
                    "atr": round(float(atr_val), 2),
                })
                return row

            # Adjust entry price: pullback entries use the pivot level, breakouts use last_close
            adjusted_entry = last_close * (1.0 + entry_offset)
            entry, stop, target = _levels_from_price(
                price=adjusted_entry,
                atr_proxy_pct=max(0.6, float(atr_val / max(1e-9, last_close) * 100.0)),
                direction=side,
            )
            rr = _rr(entry=entry, stop=stop, target=target, side=side)

            # Compute extension for output
            if side == "LONG":
                c_clean = pd.to_numeric(close, errors="coerce").dropna()
                h_clean = pd.to_numeric(high, errors="coerce").dropna()
                if len(h_clean) >= 14:
                    pivot_h = float(h_clean.iloc[-14:-2].max())
                    extension_atr = round((last_close - pivot_h) / max(atr_val, 1e-9), 2)
            else:
                l_clean = pd.to_numeric(low, errors="coerce").dropna()
                if len(l_clean) >= 14:
                    pivot_l = float(l_clean.iloc[-14:-2].min())
                    extension_atr = round((pivot_l - last_close) / max(atr_val, 1e-9), 2)

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
            "adx": round(float(adx), 1),
            "regime": regime,
            "entry_timing": entry_timing,
            "extension_atr": round(float(extension_atr), 2),
            "atr": round(float(atr_val), 2),
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

