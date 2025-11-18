# AI Financial Trend Forecaster – MVP 🚀

Interactive Streamlit app that pulls live market data with `yfinance`, layers on classic technical indicators, and builds a quick baseline price forecast.

This is a **minimum viable product (MVP)** – intentionally small, focused, and shippable.
Future versions will expand the modeling options and UX.

---

## 🔗 Live App

**Streamlit:** https://YOUR-STREAMLIT-URL-HERE

> Best viewed on desktop. Mobile works, but the charts deserve more screen real estate.

---

## 🧠 What it does

Given one or more stock tickers (e.g., `AAPL, MSFT`), the app will:

- ✅ Fetch historical price data with `yfinance`
- ✅ Compute technical indicators:
  - Moving Averages – **MA20 / MA50 / MA200**
  - Exponential Moving Averages – **EMA12 / EMA26**
  - **RSI (14)**  
  - **MACD (12, 26, 9)**
- ✅ Plot:
  - Price + MAs/EMAs
  - MACD panel
- ✅ Generate a **baseline forecast** using:
  - **Linear Regression on the last _N_ days** (configurable)
- ✅ Provide plain-English insights about:
  - Trend direction
  - MA crossover context
  - Momentum signals

There is an **ARIMA(1,1,1)** branch in the code, but it is **explicitly disabled** in this MVP to keep the surface area small and avoid half-wired models in production.

---

## 🏗 Tech stack

- **Python 3.12+**
- **Streamlit** – app UI
- **pandas / NumPy** – data wrangling
- **yfinance** – market data
- **scikit-learn** – linear regression model
- **plotly** – interactive charts

---

## 📁 Project structure

```bash
fintrend-forecaster/
├─ app.py            # Main Streamlit app
├─ requirements.txt  # Python dependencies
├─ .gitignore        # Ignore venv, cache, etc.
└─ README.md         # You are here
