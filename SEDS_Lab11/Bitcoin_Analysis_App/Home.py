import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Bitcoin Analysis App", layout="wide")

st.title("Bitcoin Price Analysis App")


# Callback function to update data
def update_data():
    st.session_state["btc_data"] = fetch_bitcoin_data(
        "BTC-USD", st.session_state["start_date"], st.session_state["end_date"]
    )


# Sidebar inputs with session state management
if "start_date" not in st.session_state:
    st.session_state["start_date"] = pd.Timestamp("2024-01-01")
if "end_date" not in st.session_state:
    st.session_state["end_date"] = pd.Timestamp("2024-12-01")

st.sidebar.header("Settings")
st.sidebar.date_input("Start Date", key="start_date", on_change=update_data)
st.sidebar.date_input("End Date", key="end_date", on_change=update_data)


# Fetch data function
@st.cache_data
def fetch_bitcoin_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)


# Load data into session state
if "btc_data" not in st.session_state:
    st.session_state["btc_data"] = fetch_bitcoin_data(
        "BTC-USD", st.session_state["start_date"], st.session_state["end_date"]
    )

btc_data = st.session_state["btc_data"]

# Display data
if btc_data.empty:
    st.warning("No data available for the selected date range.")
else:
    st.write(f"Loaded {len(btc_data)} records.")
    st.dataframe(btc_data)
