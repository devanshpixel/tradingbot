# NSE Intraday Scanner & Paper Trading Dashboard

## Overview
An educational NSE (National Stock Exchange of India) intraday stock scanner and paper trading dashboard built with Streamlit. It provides real-time (5-minute refresh) stock scanning, technical analysis signals, news sentiment analysis, and simulated trade tracking.

## Tech Stack
- **Language:** Python 3.12
- **UI Framework:** Streamlit (runs on port 5000)
- **Data:** yfinance (Yahoo Finance), feedparser (Google News RSS), Finnhub API (optional)
- **Technical Analysis:** `ta` library (RSI, MACD, Bollinger Bands, VWAP)
- **NLP/Sentiment:** TextBlob, NLTK
- **Database:** SQLite (`paper_trades.db`) via standard `sqlite3`
- **Package Manager:** pip

## Project Structure
- `app.py` — Main Streamlit application entry point
- `scanner.py` — NSE universe definition and stock scanning logic
- `signals.py` — Technical analysis, candlestick patterns, and signal generation
- `news.py` — News fetching and sentiment analysis
- `tracker.py` — SQLite trade tracker (open/close trades, stats)
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — Streamlit server config (port 5000, dark theme)

## Running the App
The app runs via the "Start application" workflow:
```
streamlit run app.py
```

## Configuration
- **Streamlit server:** port 5000, host 0.0.0.0, headless mode
- **Optional:** Set `FINNHUB_API_KEY` env var or Streamlit secret for Finnhub news data
- **Database:** SQLite file `paper_trades.db` (auto-created on first run)
