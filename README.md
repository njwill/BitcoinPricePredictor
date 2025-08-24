# BitcoinPricePredictor

A lightweight, Streamlit-based app for exploring Bitcoin price data, running quick technical studies, and generating human-readable insights. The project is organized as modular Python files (fetching data, applying indicators, drawing charts, optional AI commentary, and simple persistence) so you can reuse pieces in notebooks or other apps.

See the tool in action at [theBTCcourse.com](https://predict.thebtccourse.com/).

---

## ✨ Features

* **Streamlit UI** for quick, interactive exploration.
* **Data ingestion** module for fetching and updating BTC price history.
* **Technical analysis** helpers (e.g., moving averages / momentum style indicators).
* **Charting** utilities (candlesticks + overlays).
* **Optional AI commentary** to summarize signals in plain English.
* **Simple persistence** (e.g., local DB) for caching data and storing results.
* **Social copy generator** to turn insights into shareable blurbs.

---

## 📦 Project structure

```
.
├── app.py                     # Streamlit entry point
├── data_fetcher.py            # Pull/refresh price data
├── technical_analysis.py      # Indicator calculations & transforms
├── chart_generator.py         # Plotting utilities (candles/overlays)
├── ai_analysis.py             # Optional AI-driven narrative/insights
├── social_media_generator.py  # Optional "share this" text generator
├── database.py                # Local persistence/caching helpers
├── utils.py                   # Common helpers
├── .streamlit/                # Streamlit configuration
├── pyproject.toml             # Project metadata & deps (managed with uv)
└── uv.lock                    # Locked dependency set
```

*File list based on the current repository contents (Aug 24, 2025). ([GitHub][1])*

---

## 🚀 Quick start

### Prereqs

* Python 3.10+ recommended
* [Streamlit](https://streamlit.io/)
* (Optional) [uv](https://docs.astral.sh/uv/) for fast, reproducible installs

### 1) Clone

```bash
git clone https://github.com/njwill/BitcoinPricePredictor.git
cd BitcoinPricePredictor
```

### 2A) Install (recommended) — using `uv`

```bash
# If you don’t have uv:
# pipx install uv   # or: pip install uv

uv sync                # creates a .venv and installs from pyproject/uv.lock
uv run streamlit run app.py
```

### 2B) Install — using plain `pip`

If you prefer `pip`, create a virtualenv and install the project (editable mode reads deps from `pyproject.toml`):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
streamlit run app.py
```

---

## ⚙️ Configuration

Create a `.env` in the project root (optional but useful):

```dotenv
# Optional keys if you enable AI commentary or social text
OPENAI_API_KEY=<your-key>

# Optional database override (defaults to local file if not set)
DATABASE_URL=sqlite:///bitcoin.db

# Streamlit tweaks (optional)
STREAMLIT_SERVER_PORT=8501
```

*If you don’t use the AI or social features, you can omit the related keys.*

---

## 🧠 How it works (modules at a glance)

* **`data_fetcher.py`** – pulls historical BTC OHLCV data and handles refresh windows.
* **`technical_analysis.py`** – computes indicators (e.g., moving averages / momentum features) you can plug into views or models.
* **`chart_generator.py`** – candlestick charts with overlays & annotations for quick visual inspection.
* **`ai_analysis.py`** *(optional)* – turns raw signals into a concise narrative for humans.
* **`social_media_generator.py`** *(optional)* – drafts shareable blurbs (tweets/posts) from the current readout.
* **`database.py`** – basic persistence/caching to avoid refetching/ recomputing on every run.
* **`app.py`** – Streamlit UI that stitches everything together.

---

## 🧪 Extending & reuse

* Use the modules directly in notebooks:

  ```python
  from data_fetcher import get_price_data
  from technical_analysis import add_indicators
  from chart_generator import plot_candles

  df = get_price_data(symbol="BTC-USD", lookback="3y")
  df = add_indicators(df)
  fig = plot_candles(df, overlays=["SMA20","SMA50","SMA200"])
  ```
* Swap in your own model or indicator set by adding functions to `technical_analysis.py` and calling them from `app.py`.

---

## 🤖 Notes on “prediction”

This project is intended for **exploration and education**. If you plug in ML models, keep proper train/validation/test splits, out-of-sample evaluation, and walk-forward backtests in mind. Financial markets are noisy; use outputs responsibly.

---

## 🧰 Scripts & common tasks

```bash
# Start the Streamlit app
streamlit run app.py

# Freeze deps from your active venv (pip users only; optional)
pip freeze > requirements.txt
```

---

## 🗺️ Roadmap (suggested)

* Backtests and walk-forward evaluation helpers
* Configurable data providers
* More indicators & overlays out of the box
* Export charts and reports
* CI checks (lint/format/test) on PRs

---

## 🤝 Contributing

Issues and PRs are welcome! Please open a discussion/issue describing your change before submitting a PR.

---

## ⚠️ Disclaimer

Nothing here is financial advice. Use at your own risk.

---

[1]: https://github.com/njwill/BitcoinPricePredictor "GitHub - njwill/BitcoinPricePredictor"
