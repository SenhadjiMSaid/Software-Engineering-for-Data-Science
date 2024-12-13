import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_data(start_date, end_date):
    """
    Load Bitcoin price data using yfinance

    Args:
        start_date (datetime): Start date for data retrieval
        end_date (datetime): End date for data retrieval

    Returns:
        pd.DataFrame: Bitcoin price data
    """
    try:
        # Fetch Bitcoin data
        data = yf.download("BTC-USD", start=start_date, end=end_date)

        # Ensure the DataFrame is not empty
        if data.empty:
            st.error("No data retrieved. Please check your date range.")
            return None

        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def plot_price_trend(data):
    """
    Create interactive price trend visualization

    Args:
        data (pd.DataFrame): Bitcoin price DataFrame

    Returns:
        plotly graph object
    """
    try:
        # Reset index to make Date a column
        df = data.reset_index()

        # Create figure with secondary y-axis
        fig = go.Figure()

        # Add Close Price Line
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                mode="lines",
                name="Closing Price",
                line=dict(color="blue", width=2),
            )
        )

        # Add Volume Bar Chart
        fig.add_trace(
            go.Bar(
                x=df["Date"],
                y=df["Volume"],
                name="Trading Volume",
                yaxis="y2",
                opacity=0.3,
                marker_color="lightgray",
            )
        )

        # Update layout for dual y-axis
        fig.update_layout(
            title="Bitcoin Price and Volume Trend",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            yaxis2=dict(title="Trading Volume", overlaying="y", side="right"),
            height=600,
        )

        return fig
    except Exception as e:
        st.error(f"Error creating price trend plot: {e}")
        return None


def get_descriptive_stats(data):
    """
    Calculate comprehensive descriptive statistics

    Args:
        data (pd.DataFrame): Bitcoin price DataFrame

    Returns:
        pd.DataFrame: Descriptive statistics
    """
    try:
        stats = pd.DataFrame(
            {
                "Metric": [
                    "Total Trading Days",
                    "Average Close Price",
                    "Median Close Price",
                    "Minimum Price",
                    "Maximum Price",
                    "Price Standard Deviation",
                    "Total Trading Volume",
                    "Average Daily Volume",
                ],
                "Value": [
                    len(data),
                    data["Close"].mean(),
                    data["Close"].median(),
                    data["Close"].min(),
                    data["Close"].max(),
                    data["Close"].std(),
                    data["Volume"].sum(),
                    data["Volume"].mean(),
                ],
            }
        )
        return stats
    except Exception as e:
        st.error(f"Error calculating descriptive stats: {e}")
        return pd.DataFrame()


def plot_statistical_analysis(data):
    """
    Create advanced statistical visualizations

    Args:
        data (pd.DataFrame): Bitcoin price DataFrame

    Returns:
        plotly graph object
    """
    try:
        # Reset index to make Date a column
        df = data.reset_index()

        # Rolling Statistics
        rolling_windows = [7, 30, 90]

        fig = go.Figure()

        for window in rolling_windows:
            rolling_mean = data["Close"].rolling(window=window).mean()
            rolling_std = data["Close"].rolling(window=window).std()

            # Moving Average
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=rolling_mean,
                    mode="lines",
                    name=f"{window}-Day Moving Average",
                )
            )

            # Standard Deviation Band
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=rolling_std,
                    mode="lines",
                    name=f"{window}-Day Std Deviation",
                    line=dict(dash="dot"),
                )
            )

        fig.update_layout(
            title="Rolling Statistical Analysis",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=600,
        )

        return fig
    except Exception as e:
        st.error(f"Error creating statistical analysis plot: {e}")
        return None


def prepare_data_for_ml(data):
    """
    Prepare data for machine learning prediction

    Args:
        data (pd.DataFrame): Bitcoin price DataFrame

    Returns:
        tuple: X and y for machine learning model
    """
    try:
        # Reset index to make Date a column
        df = data.reset_index()

        # Create features based on days since first date
        df["Days"] = (df["Date"] - df["Date"].min()).dt.days

        X = df[["Days"]]
        y = df["Close"]

        return X, y
    except Exception as e:
        st.error(f"Error preparing data for ML: {e}")
        return None, None


def train_evaluate_model(X, y):
    """
    Train and evaluate a simple linear regression model

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable

    Returns:
        tuple: Trained model, MSE, R-squared score
    """
    try:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return model, mse, r2
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None


def main():
    st.title("Bitcoin Price Analysis Dashboard")

    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", datetime.now())

    # Load data
    data = load_data(start_date, end_date)
    print(data)

    if data is not None:
        # Tabs for different analysis
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Price Trend",
                "Descriptive Stats",
                "Statistical Analysis",
                "ML Prediction",
            ]
        )

        with tab1:
            price_trend_fig = plot_price_trend(data)
            if price_trend_fig:
                st.plotly_chart(price_trend_fig, use_container_width=True)

        with tab2:
            stats = get_descriptive_stats(data)
            if not stats.empty:
                st.dataframe(stats, use_container_width=True)

        with tab3:
            stat_analysis_fig = plot_statistical_analysis(data)
            if stat_analysis_fig:
                st.plotly_chart(stat_analysis_fig, use_container_width=True)

        with tab4:
            st.subheader("Simple Linear Regression Model")
            X, y = prepare_data_for_ml(data)

            if X is not None and y is not None:
                model, mse, r2 = train_evaluate_model(X, y)

                if model is not None:
                    st.write(f"Mean Squared Error: {mse:.2f}")
                    st.write(f"R-squared Score: {r2:.2f}")

                    # Reset index for prediction dates
                    df = data.reset_index()

                    # Plot predictions
                    future_days = 30
                    future_dates = pd.date_range(
                        start=df["Date"].iloc[-1], periods=future_days
                    )
                    future_X = pd.DataFrame(
                        {"Days": (future_dates - df["Date"].min()).days}
                    )
                    future_predictions = model.predict(future_X)

                    pred_fig = go.Figure()
                    pred_fig.add_trace(
                        go.Scatter(
                            x=df["Date"],
                            y=df["Close"],
                            mode="lines",
                            name="Actual Price",
                        )
                    )
                    pred_fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=future_predictions,
                            mode="lines",
                            name="Predicted Price",
                        )
                    )
                    pred_fig.update_layout(
                        title="Bitcoin Price Prediction (Next 30 Days)",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                    )
                    st.plotly_chart(pred_fig, use_container_width=True)
    else:
        st.warning(
            "Unable to retrieve Bitcoin price data. Please check your internet connection or date range."
        )


if __name__ == "__main__":
    main()
