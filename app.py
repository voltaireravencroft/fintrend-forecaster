import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

# Try to use 'ta'; fall back to manual calcs if not available

try:
    import ta
    TA_AVAILABLE = True
except Exception:
    TA_AVAILABLE = False

# -------------
# Page Config
# -------------

st.set_page_config(page_title="AI Financial Trend Forecaster (MVP)" , layout="wide")
st.title("AI Financial Trend Forecaster - MVP 🚀")
st.caption("yfinance + technicals + quick baseline forecast + plain-English insights")

# -----------------
# Sidebar controls
# -----------------

with st.sidebar:
    st.header("Settings")
    tickers = st.text_input("Tickers (comma-separated)" , value="AAPL,MSFT")
    period = st.selectbox("History range", ["1y", "2y", "5y", "max"], index=0)
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    st.subheader("Indicators")
    show_ma = st.checkbox("MA (20/50/200)", value=True)
    show_ema = st.checkbox("EMA (12/26)", value=True)
    show_rsi = st.checkbox("RSI (14)", value=True)
    show_macd = st.checkbox("MACD (12,26,9)", value=True)

    st.subheader("Forecast")
    model_choice = st.radio("Model", ["Linear Regression (last N days)", "ARIMA(1,1,1) - disabled in MVP",], 
                            index=0,)
    horizon = st.slider("Forecast horizon (days)", 5, 60, 30, 5)
    lookback = st.slider("Regression lookback (days)", 30, 240, 90, 10, help="Only for Linear Regression")
    
    st.subheader("Export")
    want_export = st.checkbox("Enable CSV export of indicators", value=True)

# ---------
# Helpers
# ---------
@st.cache_data(show_spinner=False)
def fetch_prices(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.rename(columns=str.title)
        df.index.name = "Date"
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    close = pd.Series(np.asarray(out["Close"]).ravel(), index=out.index, name="Close")

    # MAs
    out["MA20"] = close.rolling(20).mean()
    out["MA50"] = close.rolling(50).mean()
    out["MA200"] = close.rolling(200).mean()
    # EMAs
    out["EMA12"] = close.ewm(span=12, adjust=False).mean()
    out["EMA26"] = close.ewm(span=26, adjust=False).mean()
    # RSI & MACD
    if TA_AVAILABLE:
        out["RS14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        out["MACD"] = macd.macd()
        out["MACD_SIGNAL"] = macd.macd_signal()
        out["MACD_HIST"] = macd.macd_diff()
    else:
    # Simple RSI fallback
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss.replace(0, np.nan))
        out["RSI14"] = 100 - (100 / (1 + rs))
    # Simple MACD fallback
        out["MACD"] = out["EMA12"] - out["EMA26"]
        out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
        out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]
    return out

def linear_regression_forecast(df: pd.DataFrame, horizon: int, lookback: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    tail = df.tail(lookback).reset_index()
    tail["t"] = np.arange(len(tail))
    y = tail["Close"].values.reshape(-1, 1)
    X = tail["t"].values.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(X, y)
    # Predict next horizon
    future_t = np.arange(len(tail), len(tail) + horizon).reshape(-1, 1)
    preds = lr.predict(future_t).ravel()
    last_date = tail["Date"].iloc[-1]
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)
    return pd.DataFrame({"Date": future_dates, "Forecast": preds}).set_index("Date")

def arima_forecast(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    series = df["Close"].dropna()
    if len(series) < 30:
        return pd.DataFrame()
    try:
        model = ARIMA(series, order=(1,1,1))
        res = model.fit()
        fc = res.forecast(steps=horizon)
        return pd.DataFrame({"Forecast": fc})
    except Exception:
        return pd.DataFrame()
    
def insight_rules(row: dict, prev_row: dict):
    notes = []
    # Trend crosses
    vals = [row.get("MA20"), row.get("MA50"), prev_row.get("MA20"), prev_row.get("MA50")]
    have_ma = all(pd.notna(v) for v in vals)

    if have_ma:
        if row["MA20"] >= row["MA50"] and prev_row["MA20"] < prev_row["MA50"]:
            notes.append("Bullish: MA20 crossed above MA50.")
        if row["MA20"] <=row["MA50"] and prev_row["MA20"] > prev_row["MA50"]:
            notes.append("Bearish: MA20 crossed below MA50.")

    # RSI zones
    rsi = row.get("RSI14")
    if pd.notna(rsi):
        if rsi < 30: notes.append("RSI < 30: Potentially oversold.")
        elif rsi > 70: notes.append("RSI > 70: Potentially overbought.")
    # MACD signal
    macd, sig = row.get("MACD"), row.get("MACD_SIGNAL")
    if pd.notna(macd) and pd.notna(sig):
        if macd > sig: notes.append("MACD above signal: momentum positive.")
        elif macd < sig: notes.append("MACD below signal: momentum negative.")
    return notes

def summarize_insights(df: pd.DataFrame) -> list:
    if df.empty or "Close" not in df.columns:
        return ["Not enough data for insights."]
    
    last = df.iloc[-1].to_dict()
    prev = df.iloc[-2].to_dict() if len(df) >=2 else last
    
    notes = insight_rules(last, prev)

    # --- Scalarize closes to avoid "truth value of a Series is ambiguous"
    def _as_scalar(x):
        if isinstance(x, pd.Series):
            return float(x.iloc[0])
        arr = np.asarray(x)
        return float(np.ravel(arr)[0])

    last_close = _as_scalar(df["Close"].iloc[-1])
    week_ago_val = df["Close"].iloc[-5] if len(df) >= 5 else df["Close"].iloc[0]
    week_ago = _as_scalar(week_ago_val)

    if not np.isnan(last_close) and not np.isnan(week_ago) and week_ago != 0:
        change = (last_close - week_ago) / week_ago * 100.0
        notes.append(f"5-period change: {change:+.2f}%.")

    if not notes:
        notes.append("No strong technical signals right now; trend appears neutral.")
    return notes

def build_price_chart(df: pd.DataFrame, settings) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price"
    ))
    if settings["show_ma"]:
        for col in ["MA20", "MA50", "MA200"]:
            if col in df:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
    if settings["show_ema"]:
        for col in ["EMA12", "EMA26"]:
            if col in df:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
    return fig

def build_macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="MACD Hist"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal"))
    fig.update_layout(height=220, legend=dict(orientation="h"))
    return fig

def build_rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "RSI14" in df.columns and df["RSI14"].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14"))
        fig.add_hline(y=70, line_dash="dot")
        fig.add_hline(y=30, line_dash="dot")
    else:
        # Optional: show a flat note so the layout doesn't jump
        fig.add_annotation(text="RSI not available for current data range",
                           xref="paper", yref="paper", x=0.02, y=0.8, showarrow=False)
    fig.update_layout(height=220, legend=dict(orientation="h"))
    return fig

# --------
# Main
# --------

symbols = [s.strip().upper() for s in st.sidebar.text_input("", value=tickers).split(",") if s.strip()]
if not symbols:
    st.stop()

for sym in symbols:
    st.subheader(sym)
    data = fetch_prices(sym, period, interval)
    if data.empty:
        st.warning(f"No data returned for {sym}.")
        continue

    data = add_indicators(data)
    insights = summarize_insights(data)

# Forecast

fc = None

if model_choice.startswith("Linear"):
    # ✅ Use the working linear model
    fc = linear_regression_forecast(data, horizon, lookback)

elif model_choice.startswith("ARIMA"):
    # 🚫 ARIMA path is disabled for this MVP
    st.warning(
        "ARIMA is not wired up correctly yet in this MVP build, "
        "so it is disabled for now.\n\n"
        "The app currently uses a linear baseline model only."
    )
    st.stop()  # prevents any of the ARIMA code from running


# Charts

cols = st.columns([7, 3])
with cols[0]:
    fig_price = build_price_chart(
        data,
        {"show_ma": show_ma, "show_ema": show_ema}
    )
    st.plotly_chart(fig_price, use_container_width=True)

    if show_macd:
        st.plotly_chart(build_macd_chart(data), use_container_width=True)
if show_rsi:
    if "RSI14" in data.columns and data["RSI14"].notna().any():
        st.plotly_chart(build_rsi_chart(data), use_container_width=True)
    else:
        st.info("RSI not available for this symbol/range yet.")

# ----- Forecast overlay (robust) -----
if not fc.empty:
    # Make sure we have a clean 'Actual' column
    actual = data[["Close"]].copy()
actual.columns = ["Actual"]
# (rest stays the same)
combo = pd.concat([actual, fc], axis=0)

last_hist = data.index[-1]
past = combo.loc[combo.index <= last_hist].copy()
futr = combo.loc[combo.index >  last_hist].copy()

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(x=past.index, y=past["Actual"], name="Actual"))
if "Forecast" in futr.columns and futr["Forecast"].notna().any():
    fig_fc.add_trace(go.Scatter(x=futr.index, y=futr["Forecast"], name="Forecast"))
st.plotly_chart(fig_fc, use_container_width=True)
combo = pd.concat([actual, fc], axis=0)

    # Split past vs future by last historical date
last_hist = data.index[-1]
past = combo.loc[combo.index <= last_hist].copy()
futr = combo.loc[combo.index >  last_hist].copy()

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(x=past.index, y=past["Actual"], name="Actual"))

if "Forecast" in futr.columns and futr["Forecast"].notna().any():
        fig_fc.add_trace(go.Scatter(x=futr.index, y=futr["Forecast"], name="Forecast"))

fig_fc.update_layout(height=300, title="Baseline Forecast", legend=dict(orientation="h"))
st.plotly_chart(fig_fc, use_container_width=True)
    
with cols[1]:
    st.markdown("### Insight Summary")
    for n in insights:
        st.markdown(f" - {n}")
    
    if not fc.empty:
        last_close = data["Close"].iloc[-1]
        last_fc = fc["Forecast"].iloc[-1]
delta_pct = (last_fc - last_close) / last_close * 100

# make sure it's a plain float, not a Series/ndarray
import pandas as pd
import numpy as np

if isinstance(delta_pct, pd.Series):
    delta_pct = delta_pct.iloc[0]
else:
    delta_pct = float(np.asarray(delta_pct).ravel()[0])

st.metric("Forecast Δ to horizon", f"{delta_pct:+.2f}%")

if want_export:
    # start from the indicators / price data
    enriched = data.copy()

    if not fc.empty:
        # flatten both, align by row position, and concat side-by-side
        base_exp = enriched.reset_index(drop=True)
        fc_exp = fc.reset_index(drop=True)

        # if lengths differ, trim to the shorter one so concat doesn't complain
        min_len = min(len(base_exp), len(fc_exp))
        base_exp = base_exp.iloc[:min_len].copy()
        fc_exp = fc_exp.iloc[:min_len].copy()

        enriched = pd.concat([base_exp, fc_exp], axis=1)
    else:
        # no forecast; still flatten for a clean CSV
        enriched = enriched.reset_index()

    csv_bytes = enriched.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download indicators CSV",
        data=csv_bytes,
        file_name=f"{sym}_indicators.csv",
        mime="text/csv",
    )


st.divider()
