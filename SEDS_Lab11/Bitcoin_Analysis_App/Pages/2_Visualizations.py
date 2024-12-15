import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def display_visualizations(data):
    st.header("Bitcoin Price Trend")
    fig = px.line(data, x=data.index, y="Close", title="Bitcoin Price Trend")
    st.plotly_chart(fig, use_container_width=True)


def display_correlation(data):
    st.subheader("Correlation Matrix")
    correlation_matrix = data.corr()
    st.write(correlation_matrix)
    # fig_corr = px.imshow(
    #     correlation_matrix, text_auto=True, title="Correlation Heatmap"
    # )
    # st.plotly_chart(fig_corr)
    # selected_features = ["Open", "High", "Low", "Close"]
    # corr = data[selected_features].corr()

    # Plot the heatmap
    # fig, ax = plt.subplots()
    # sns.heatmap(corr, annot=True, ax=ax)
    # st.write(fig)
    # fig_corr = px.imshow(
    #     corr,
    #     title="Correlation Matrix",
    # )
    # st.plotly_chart(fig_corr)


# Daily Returns Histogram
def display_daily_returns_histogram(data):
    st.subheader("Daily Returns Distribution")

    # Calculate daily returns
    data["Daily Return"] = data["Close"].pct_change()

    # Plot histogram of daily returns
    fig = px.histogram(
        data["Daily Return"].dropna(),
        nbins=50,
        title="Histogram of Daily Returns",
        labels={"value": "Daily Return"},
        color_discrete_sequence=["dodgerblue"],
    )
    st.plotly_chart(fig, use_container_width=True)


#  Volatility Over Time
def display_volatility(data):
    st.subheader("Volatility Over Time")

    # Calculate rolling volatility
    data["Volatility"] = data["Daily Return"].rolling(window=30).std()

    # Plot volatility
    fig = px.line(
        data,
        x=data.index,
        y="Volatility",
        title="30-Day Rolling Volatility",
        labels={"Volatility": "Volatility"},
        line_shape="spline",
    )
    st.plotly_chart(fig, use_container_width=True)


# Display Candlestick Patterns
def display_candlestick(data):
    st.subheader("Candlestick Chart")

    # Create a candlestick chart
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                increasing_line_color="green",
                decreasing_line_color="red",
            )
        ]
    )
    fig.update_layout(
        title="Bitcoin Candlestick Chart", xaxis_title="Date", yaxis_title="Price (USD)"
    )
    st.plotly_chart(fig, use_container_width=True)


# Display Relative Strength Index (RSI)
def display_rsi(data):
    st.subheader("Relative Strength Index (RSI)")

    # Calculate RSI
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # Plot RSI
    fig = px.line(
        data,
        x=data.index,
        y="RSI",
        title="Relative Strength Index (RSI)",
        labels={"RSI": "RSI Value"},
    )
    fig.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold")
    st.plotly_chart(fig, use_container_width=True)


# Volume Analysis
def display_volume(data):
    st.subheader("Trading Volume")

    # Plot trading volume
    fig = px.bar(
        data,
        x=data.index,
        y="Volume",
        title="Bitcoin Trading Volume",
        labels={"Volume": "Volume Traded (Units)"},
        color_discrete_sequence=["indianred"],
    )
    st.plotly_chart(fig, use_container_width=True)


# Main function
def main():
    st.title("Visualizations")
    if "btc_data" in st.session_state:
        btc_data = st.session_state.get("btc_data", None)
        # display_visualizations(btc_data)
        display_correlation(btc_data)
        display_daily_returns_histogram(btc_data)
        display_volatility(btc_data)
        display_candlestick(btc_data)
        display_rsi(btc_data)
        # display_volume(btc_data)

    else:
        st.warning("No data available. Please go to the Home page to load data.")


if __name__ == "__main__":
    main()
