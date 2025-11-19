# AI Financial Trend Forecaster – MVP 🚀

Interactive Streamlit app that pulls live market data with `yfinance`, layers on classic technical indicators, and builds a quick baseline price forecast.

This is a **minimum viable product (MVP)** – intentionally small, focused, and shippable.
Future versions will expand the modeling options and UX.

# 📘 Roadmap

This project is intentionally released as a **Minimum Viable Product (MVP)**.  
The goal is to demonstrate:

- Data ingestion from live market sources  
- Indicator computation  
- Baseline modeling  
- Interactive visualization  
- A clean, deployable Streamlit architecture  

---

## 🚀 Future Enhancements

The following milestones will expand the modeling and analytics capabilities:

### 🔮 Forecasting Models
- Full ARIMA(1,1,1) implementation  
- Prophet  
- LSTM baseline model  
- Gradient Boosting Regressor  

### 📊 Technical Indicators
- Bollinger Bands  
- Stochastic Oscillator  
- OBV (On-Balance Volume)  
- Volume profile heatmaps  

### 🧠 Model Quality & Diagnostics
- Improved differencing pipeline  
- Handle edge-case index alignment  
- Add model diagnostics and warnings panel  
- “Model Details” sidebar for transparency  

### 🎨 UI & Experience Improvements
- Theme switcher  
- Downloadable charts  
- Export-to-Excel support  
- Component-based chart rendering  

### ⚙️ Performance & Architecture
- Caching layers for large tickers  
- API-ready extraction of model predictions  

---

## 📝 Notes  
ARIMA is present in the codebase but intentionally **disabled in the MVP** until the forecasting pipeline and index alignment logic is fully production-ready.


## 🔗 Live App

**Streamlit:** [https://voltaireravencroft-fintrend-forecaster-app-a2btfx.streamlit.app]

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

---

## 🛠 Local Development

If you want to run this project locally:

### 1. Clone the repo
```bash
git clone https://github.com/voltaireravencroft/fintrend-forecaster.git
cd fintrend-forecaster
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### For Windows
```bash
.venv\Scripts\activate
```

### For Mac/Linux
```bash
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```
## 📛 Naming Note

## "Ravencroft" is a professional/brand identity.  
## License copyright is issued under my current legal name, **Michael Galvan**.