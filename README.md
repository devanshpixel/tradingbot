## NSE intraday scanner + paper trading (Streamlit)

Educational intraday dashboard for Indian stocks (NSE) that combines:
- Auto scanning (volume spike + momentum)
- Technical analysis (RSI, MACD, Bollinger Bands, volume)
- News sentiment (Google News RSS + optional Finnhub) using TextBlob
- Paper trading tracker (local SQLite) with history + win-rate + P/L

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m textblob.download_corpora
```

### Run

```bash
streamlit run app.py
```

### Finnhub API key (optional)

- **Streamlit Secrets**: create `.streamlit/secrets.toml`

```toml
FINNHUB_API_KEY = "paste-your-key-here"
```

- **Or environment variable**:

```powershell
$env:FINNHUB_API_KEY="paste-your-key-here"
```

### Files

- `app.py`: Streamlit UI (auto refreshes every 5 minutes)
- `scanner.py`: NSE universe + scanner ranking
- `signals.py`: indicator engine + confidence + entry/stop/target + RR
- `news.py`: Google RSS + Finnhub fetch + TextBlob sentiment
- `tracker.py`: SQLite-based paper trading tracker

