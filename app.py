import os
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from news import fetch_google_news_rss, fetch_finnhub_news, sentiment_from_articles
from scanner import default_nse_universe, scan_universe
from signals import generate_signal_row
from tracker import TradeTracker


st.set_page_config(page_title="NSE Intraday Trading Bot (Paper)", layout="wide")


def _get_finnhub_api_key() -> str:
    # Streamlit Cloud secrets can be top-level or nested under [finnhub].
    key = os.getenv("FINNHUB_API_KEY", "")
    if key:
        return key
    try:
        top_level = st.secrets.get("FINNHUB_API_KEY", "")
        if top_level:
            return str(top_level)
        nested = st.secrets.get("finnhub", {})
        if isinstance(nested, dict):
            nested_key = nested.get("FINNHUB_API_KEY", "")
            if nested_key:
                return str(nested_key)
    except Exception:
        # If secrets are unavailable locally, continue without Finnhub.
        return ""
    return ""


def _autorefresh_every(seconds: int) -> None:
    ms = int(seconds * 1000)
    components.html(
        f"""
        <script>
          setTimeout(function () {{
            window.parent.location.reload();
          }}, {ms});
        </script>
        """,
        height=0,
    )


st.title("NSE Intraday Trading Bot (Paper Trading)")
st.caption("Scanner + technical indicators + news sentiment. Educational use only.")

_autorefresh_every(300)

with st.sidebar:
    st.subheader("Settings")
    finnhub_api_key = st.text_input(
        "Finnhub API key",
        value=_get_finnhub_api_key(),
        type="password",
        help="Optional. You can also set FINNHUB_API_KEY in Streamlit Secrets or environment.",
    )
    max_symbols = st.slider("Symbols to scan", min_value=10, max_value=20, value=20, step=5)
    interval = st.selectbox("Price interval", options=["5m", "15m", "30m"], index=0)
    min_price = st.number_input("Min price (INR)", min_value=0.0, value=50.0, step=10.0)
    refresh_now = st.button("Refresh now", use_container_width=True)

if refresh_now:
    st.rerun()

tracker = TradeTracker(db_path="paper_trades.db")

universe = default_nse_universe()[:max_symbols]
scan_df = scan_universe(universe=universe, interval=interval, min_price=min_price)

st.subheader("Auto stock scanner (NSE)")
st.dataframe(
    scan_df,
    use_container_width=True,
    hide_index=True,
)

st.subheader("Signals")
rows = []
for _, s in scan_df.head(20).iterrows():
    symbol = str(s["symbol"])

    google_articles = fetch_google_news_rss(symbol=symbol, max_items=3)
    finnhub_articles = fetch_finnhub_news(symbol=symbol, api_key=finnhub_api_key, max_items=3) if finnhub_api_key else []
    sentiment = sentiment_from_articles(google_articles + finnhub_articles)

    row = generate_signal_row(
        symbol=symbol,
        interval=interval,
        sentiment=sentiment,
    )
    rows.append(row)

signals_df = pd.DataFrame(rows)
if not signals_df.empty:
    # Some rows may be missing keys depending on upstream failures; avoid KeyError.
    for col, default in (
        ("confidence", 0.0),
        ("score_total", 0.0),
        ("signal", "HOLD"),
        ("entry", 0.0),
        ("stop_loss", 0.0),
        ("target", 0.0),
    ):
        if col not in signals_df.columns:
            signals_df[col] = default
    sort_cols = [c for c in ["confidence", "score_total"] if c in signals_df.columns]
    if sort_cols:
        signals_df = signals_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    # Keep UI focused on actionable ideas; if none, derive directional picks from score.
    actionable = signals_df[signals_df["signal"].isin(["BUY", "SELL"])].copy()
    # Filter weak signals first.
    actionable = actionable[pd.to_numeric(actionable["confidence"], errors="coerce").fillna(0.0) >= 55.0]
    if actionable.empty:
        signals_df["signal"] = signals_df["score_total"].apply(lambda x: "BUY" if float(x) >= 0 else "SELL")
        actionable = signals_df.head(10).copy()
    # Always keep at least 3 actionable rows.
    if len(actionable) < 3:
        fallback = signals_df.copy()
        if "score_total" in fallback.columns:
            fallback["signal"] = fallback["score_total"].apply(lambda x: "BUY" if float(x) >= 0 else "SELL")
        actionable = fallback.sort_values(["confidence", "score_total"], ascending=[False, False]).head(3)
    if not actionable.empty:
        price_map = {}
        if not scan_df.empty and "symbol" in scan_df.columns:
            price_col = "last" if "last" in scan_df.columns else ("price" if "price" in scan_df.columns else None)
            if price_col:
                price_map = dict(zip(scan_df["symbol"].astype(str), pd.to_numeric(scan_df[price_col], errors="coerce").fillna(0.0)))
        for col in ["entry", "stop_loss", "target"]:
            actionable[col] = pd.to_numeric(actionable[col], errors="coerce").fillna(0.0)
        # Fill missing trade levels from scanner price so orders remain valid.
        actionable["entry"] = actionable.apply(
            lambda r: float(r["entry"]) if float(r["entry"]) > 0 else float(price_map.get(str(r["symbol"]), 0.0)), axis=1
        )
        actionable["stop_loss"] = actionable.apply(
            lambda r: float(r["stop_loss"]) if float(r["stop_loss"]) > 0 else max(0.01, float(r["entry"]) * 0.99), axis=1
        )
        actionable["target"] = actionable.apply(
            lambda r: float(r["target"]) if float(r["target"]) > 0 else float(r["entry"]) * 1.01, axis=1
        )
    signals_df = actionable.sort_values(["confidence", "score_total"], ascending=[False, False]).head(5).reset_index(drop=True)
st.dataframe(signals_df, use_container_width=True, hide_index=True)

st.subheader("Paper trading")
col_a, col_b, col_c = st.columns([2, 2, 3])
with col_a:
    st.write("Open a trade from a signal")
    sel = st.selectbox("Symbol", options=signals_df["symbol"].tolist() if not signals_df.empty else [])
with col_b:
    action = st.selectbox("Action", options=["BUY", "SELL"])
with col_c:
    qty = st.number_input("Quantity", min_value=1, value=1, step=1)

place = st.button("Place paper trade", type="primary", use_container_width=True, disabled=signals_df.empty)
if place and sel:
    sig = signals_df.loc[signals_df["symbol"] == sel].iloc[0].to_dict()
    tracker.open_trade(
        symbol=sel,
        side="SHORT" if action == "SELL" else action,
        qty=int(qty),
        entry=float(sig["entry"]),
        stop=float(sig["stop_loss"]),
        target=float(sig["target"]),
        confidence=float(sig["confidence"]),
        meta={"interval": interval, "generated_at": datetime.utcnow().isoformat()},
    )
    st.success("Paper trade opened.")

st.markdown("#### Open trades")
open_trades = tracker.list_open_trades()
st.dataframe(open_trades, use_container_width=True, hide_index=True)

st.markdown("#### Trade history")
history = tracker.trade_history()
st.dataframe(history, use_container_width=True, hide_index=True)

st.markdown("#### Stats")
stats = tracker.stats()
st.json(stats)