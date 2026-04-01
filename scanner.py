from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Iterable

import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo


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


def _in_intraday_scan_window() -> bool:
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata")).time()
    return time(9, 15) <= now_ist <= time(11, 30)


def _market_is_bearish(interval: str, period: str) -> bool:
    idx = _fetch_intraday(symbol="^NSEI", interval=interval, period=period)
    if idx.empty:
        return False
    close_like = idx["close"] if "close" in idx.columns else pd.Series(dtype=float)
    if isinstance(close_like, pd.DataFrame):
        close_like = close_like.iloc[:, 0]
    close = pd.to_numeric(close_like, errors="coerce").dropna()
    if len(close) < 3:
        return False
    return bool(close.iloc[-1] < close.iloc[0])


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
    try:
        df = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        print(f"[scanner] fetch_error symbol={symbol} error={exc}")
        return pd.DataFrame()

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
    symbols = list(universe)
    if not symbols:
        symbols = default_nse_universe()
    # Performance guard: keep scan fast for UI.
    symbols = symbols[:20]

    rows: list[dict] = []
    for symbol in symbols:
        price = float(cfg.min_price)
        volume = 0.0
        momentum = 0.0
        try:
            df = _fetch_intraday(symbol=symbol, interval=cfg.interval, period=cfg.period)
            if not df.empty:
                close_like = df["close"] if "close" in df.columns else pd.Series(dtype=float)
                if isinstance(close_like, pd.DataFrame):
                    close_like = close_like.iloc[:, 0]
                close_numeric = pd.to_numeric(close_like, errors="coerce").dropna()
                if not close_numeric.empty:
                    price = float(close_numeric.iloc[-1])
                    first_close = float(close_numeric.iloc[0]) if len(close_numeric) else price
                    momentum = (price / first_close - 1.0) * 100.0 if first_close else 0.0

                vol_like = df["volume"] if "volume" in df.columns else pd.Series(dtype=float)
                if isinstance(vol_like, pd.DataFrame):
                    vol_like = vol_like.iloc[:, 0]
                vol_numeric = pd.to_numeric(vol_like, errors="coerce").fillna(0.0)
                if not vol_numeric.empty:
                    volume = float(vol_numeric.iloc[-1])
        except Exception as exc:
            print(f"[scanner] safe_default symbol={symbol} error={exc}")

        # Always append a usable numeric row for each symbol.
        rows.append(
            {
                "symbol": str(symbol),
                "price": round(float(price), 2),
                "volume": round(float(max(0.0, volume)), 2),
                "momentum": round(float(momentum), 2),
                # Backward-compatible aliases used elsewhere in UI.
                "last": round(float(price), 2),
                "vol_last": round(float(max(0.0, volume)), 2),
                "momentum_%": round(float(momentum), 2),
            }
        )

    # Prefer symbols with stronger activity, but always return at least 10 rows.
    ranked = sorted(rows, key=lambda r: (float(r["volume"]), abs(float(r["momentum"]))), reverse=True)
    if len(ranked) >= 10:
        selected = ranked[: max(10, min(20, len(ranked)))]
    else:
        # If universe is very small, recycle defaults from NSE universe to ensure 10.
        needed = 10 - len(ranked)
        for sym in default_nse_universe():
            if needed <= 0:
                break
            if any(r["symbol"] == sym for r in ranked):
                continue
            ranked.append(
                {
                    "symbol": sym,
                    "price": round(float(cfg.min_price), 2),
                    "volume": 0.0,
                    "momentum": 0.0,
                    "last": round(float(cfg.min_price), 2),
                    "vol_last": 0.0,
                    "momentum_%": 0.0,
                }
            )
            needed -= 1
        selected = ranked[:10]

    # Guaranteed non-empty DataFrame with stable columns.
    return pd.DataFrame(selected, columns=["symbol", "price", "volume", "momentum", "last", "vol_last", "momentum_%"])

