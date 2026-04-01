from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import yfinance as yf


def default_nse_universe() -> list[str]:
    """
    Lightweight default NSE universe (NIFTY 50-ish) for scanning.
    Symbols include the `.NS` suffix for yfinance.
    """
    return [
        "ADANIENT.NS",
        "ADANIPORTS.NS",
        "APOLLOHOSP.NS",
        "ASIANPAINT.NS",
        "AXISBANK.NS",
        "BAJAJ-AUTO.NS",
        "BAJFINANCE.NS",
        "BAJAJFINSV.NS",
        "BEL.NS",
        "BHARTIARTL.NS",
        "BPCL.NS",
        "BRITANNIA.NS",
        "CIPLA.NS",
        "COALINDIA.NS",
        "DIVISLAB.NS",
        "DRREDDY.NS",
        "EICHERMOT.NS",
        "GRASIM.NS",
        "HCLTECH.NS",
        "HDFCBANK.NS",
        "HDFCLIFE.NS",
        "HEROMOTOCO.NS",
        "HINDALCO.NS",
        "HINDUNILVR.NS",
        "ICICIBANK.NS",
        "INDUSINDBK.NS",
        "INFY.NS",
        "ITC.NS",
        "JSWSTEEL.NS",
        "KOTAKBANK.NS",
        "LT.NS",
        "M&M.NS",
        "MARUTI.NS",
        "NESTLEIND.NS",
        "NTPC.NS",
        "ONGC.NS",
        "POWERGRID.NS",
        "RELIANCE.NS",
        "SBILIFE.NS",
        "SBIN.NS",
        "SHRIRAMFIN.NS",
        "SUNPHARMA.NS",
        "TATACONSUM.NS",
        # "TATAMOTORS.NS" intentionally excluded (delisted / no data)
        "TATASTEEL.NS",
        "TCS.NS",
        "TECHM.NS",
        "TITAN.NS",
        "ULTRACEMCO.NS",
        "WIPRO.NS",
    ]


@dataclass(frozen=True)
class ScanConfig:
    interval: str = "5m"
    period: str = "1d"
    min_price: float = 50.0
    min_rows: int = 30


def _safe_last(series_like: Any) -> float | None:
    """
    Return the last numeric value from a 1D-like input.

    Some data sources (or duplicated column names after flattening) can yield a
    DataFrame instead of a Series, which breaks `pd.to_numeric` (expects 1D).
    """
    if series_like is None:
        return None

    if isinstance(series_like, pd.DataFrame):
        if series_like.empty or series_like.shape[1] == 0:
            return None
        # Pick the first column that yields a non-empty numeric series.
        for i in range(series_like.shape[1]):
            # Use iloc to avoid duplicate column labels returning a DataFrame again.
            s = pd.to_numeric(series_like.iloc[:, i], errors="coerce").dropna()
            if not s.empty:
                return float(s.iloc[-1])
        return None

    numeric = pd.to_numeric(series_like, errors="coerce")
    s = pd.Series(numeric).dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _fetch_intraday(symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        period=period,
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


def scan_universe(
    universe: Iterable[str],
    interval: str = "5m",
    min_price: float = 50.0,
) -> pd.DataFrame:
    """
    Score symbols by simple intraday opportunity heuristics:
    - volume spike vs recent average
    - momentum (last close vs first close of day)
    """
    cfg = ScanConfig(interval=interval, min_price=min_price)
    rows: list[dict] = []
    for symbol in universe:
        df = _fetch_intraday(symbol=symbol, interval=cfg.interval, period=cfg.period)
        if df.empty or len(df) < cfg.min_rows:
            continue
        # Normalise "close" column to a 1D Series even if duplicates created a DataFrame
        close_like = df["close"] if "close" in df.columns else pd.Series(dtype=float)
        if isinstance(close_like, pd.DataFrame):
            close_like = close_like.iloc[:, 0]

        last_close = _safe_last(close_like)
        if last_close is None or last_close < cfg.min_price:
            continue

        close_numeric = pd.to_numeric(close_like, errors="coerce").dropna()
        if close_numeric.empty:
            continue
        first_close = float(close_numeric.iloc[0])
        momentum_pct = (last_close / first_close - 1.0) * 100.0 if first_close else 0.0

        # Normalise "volume" similarly to handle potential duplicate columns
        vol_like = df["volume"] if "volume" in df.columns else pd.Series(dtype=float)
        if isinstance(vol_like, pd.DataFrame):
            vol_like = vol_like.iloc[:, 0]

        v = pd.to_numeric(vol_like, errors="coerce").fillna(0.0)
        vol_last = float(v.iloc[-1])
        vol_avg = float(v.tail(20).mean()) if len(v) >= 20 else float(v.mean())
        vol_spike = (vol_last / vol_avg) if vol_avg > 0 else 0.0

        # Filter for high-volume spike and strong, clear intraday momentum.
        if vol_spike <= 1.1:
            continue
        if abs(momentum_pct) <= 0.3:
            continue

        score = (abs(momentum_pct) * 0.7) + (min(vol_spike, 10.0) * 3.0)
        direction_hint = "UP" if momentum_pct >= 0 else "DOWN"

        rows.append(
            {
                "symbol": symbol,
                "last": round(last_close, 2),
                "momentum_%": round(momentum_pct, 2),
                "vol_spike_x": round(vol_spike, 2),
                "direction": direction_hint,
                "scan_score": round(score, 2),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("scan_score", ascending=False).reset_index(drop=True)

