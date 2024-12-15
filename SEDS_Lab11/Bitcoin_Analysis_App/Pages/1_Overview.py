import streamlit as st


def display_overview(data):
    st.header("Descriptive Statistics")

    # Display basic statistics
    st.subheader("Basic Statistics")
    st.write(data.describe())

    # Key Metrics
    st.subheader("Key Metrics")
    min_price = data["Close"].min()
    max_price = data["Close"].max()
    avg_price = data["Close"].mean()
    st.metric(label="Minimum Close Price", value=f"${min_price[0]:.2f}")
    st.metric(label="Maximum Close Price", value=f"${max_price[0]:.2f}")
    st.metric(label="Average Close Price", value=f"${avg_price[0]:.2f}")

    # Price Range
    st.subheader("Price Range")
    price_range = max_price - min_price
    st.write(f"The price range for the selected period is **${price_range[0]:.2f}**.")


def display_filters(data):
    st.subheader("Filter Data")

    # Date range filter
    date_range = st.slider(
        "Select Date Range",
        min_value=data.index.min().date(),
        max_value=data.index.max().date(),
        value=(data.index.min().date(), data.index.max().date()),
    )
    filtered_data = data.loc[date_range[0] : date_range[1]]

    # Price filter
    price_range = st.slider(
        "Select Close Price Range",
        min_value=float(data["Close"].min()),
        max_value=float(data["Close"].max()),
        value=(float(data["Close"].min()), float(data["Close"].max())),
    )
    filtered_data = filtered_data[
        (filtered_data["Close"] >= price_range[0])
        & (filtered_data["Close"] <= price_range[1])
    ]

    st.write(f"Filtered Data ({len(filtered_data)} records):")
    st.dataframe(filtered_data)
    return filtered_data


def display_trends(data):
    st.subheader("Price Trends")

    # Calculate daily price change
    data["Daily Change"] = data["Close"] - data["Open"]
    max_change_date = data["Daily Change"].idxmax()
    min_change_date = data["Daily Change"].idxmin()

    st.write(
        f"The highest daily gain occurred on **{max_change_date.date()}**: **${data.loc[max_change_date, 'Daily Change'][0]:.2f}**."
    )
    st.write(
        f"The highest daily loss occurred on **{min_change_date.date()}**: **${data.loc[min_change_date, 'Daily Change'][0]:.2f}**."
    )


# Main function
def main():
    st.title("Overview")
    if "btc_data" in st.session_state:
        btc_data = st.session_state.get("btc_data", None)
        display_overview(btc_data)
        filtered_data = display_filters(btc_data)
        display_trends(filtered_data)
    else:
        st.warning("No data available. Please go to the Home page to load data.")


if __name__ == "__main__":
    main()
