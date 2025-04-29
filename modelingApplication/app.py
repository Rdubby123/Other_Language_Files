import streamlit as st
import yfinance as yf
import plotly.graph_objs as go

st.title("STOCK VIEW")

st.sidebar.header("Search & Timeframe")
#st.sidebar.footer("Ryan DuBrueler")
ticker   = st.sidebar.text_input("Ticker", "AAPL")
period   = st.sidebar.selectbox("Period", ["1d","5d","1mo","3mo","1y","5y"])
interval = st.sidebar.selectbox("Interval", ["1m","5m","15m","1h","1d"])

@st.cache_data(ttl=300)
def load_data(ticker, period, interval):
    return yf.Ticker(ticker).history(period=period, interval=interval)

df = load_data(ticker, period, interval)

# ── NEW: drop days with no price change ─────────────────────────────────────
# Option A: drop when Open == Close
df = df[df['Close'] != df['Open']]                            # :contentReference[oaicite:3]{index=3}
# Option B: drop when no change vs prior close
# df = df[df['Close'].diff() != 0]                             # :contentReference[oaicite:4]{index=4}
# ─────────────────────────────────────────────────────────────────────────────

if not df.empty:
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'],   close=df['Close']
    )])
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data (all bars had zero net change).")
