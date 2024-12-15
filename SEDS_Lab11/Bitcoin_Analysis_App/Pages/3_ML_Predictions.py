import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def train_and_predict(data):
    data["Next_Close"] = data["Close"].shift(-1)
    data.dropna(inplace=True)

    # Features and target variable
    X = data[["Open", "High", "Low", "Close"]]
    y = data["Next_Close"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    return model, X_test, y_test, predictions, mse, mae


def predict_future(model, recent_data, days=365):
    st.subheader(f"Future Predictions for the Next {days} Days")

    future_dates = pd.date_range(
        start=recent_data.index[-1], periods=days + 1, freq="D"
    )[1:]
    future_predictions = []

    def get_value(data, key):
        for col in data:
            if col[0] == key:
                return data[col]
        raise KeyError(f"Key '{key}' not found in recent_data")

    current_data = recent_data.iloc[-1].to_dict()
    for _ in range(days):
        features = np.array(
            [
                [
                    get_value(current_data, "Open"),
                    get_value(current_data, "High"),
                    get_value(current_data, "Low"),
                    get_value(current_data, "Close"),
                ]
            ]
        )
        next_close = model.predict(features)[0]
        future_predictions.append(next_close)

        current_data[("Open", "BTC-USD")] = current_data[("Close", "BTC-USD")]
        current_data[("High", "BTC-USD")] = max(
            current_data[("Open", "BTC-USD")], next_close
        )
        current_data[("Low", "BTC-USD")] = min(
            current_data[("Open", "BTC-USD")], next_close
        )
        current_data[("Close", "BTC-USD")] = next_close

    future_df = pd.DataFrame(
        {"Date": future_dates, "Predicted_Close": future_predictions}
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=future_df["Date"],
            y=future_df["Predicted_Close"],
            mode="lines",
            name="Predicted Close Prices",
            line=dict(color="green", width=2),
        )
    )

    fig.update_layout(
        title=f"Bitcoin Price Predictions for the Next {days} Days",
        xaxis_title="Date",
        yaxis_title="Closing Price (USD)",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.write(future_df)


def display_actual_vs_predicted(y_test, predictions):
    st.subheader("Actual vs Predicted Prices")

    # Create a time index for plotting
    test_indices = y_test.index

    # Plotly figure
    fig = go.Figure()

    # Add actual values
    fig.add_trace(
        go.Scatter(
            x=test_indices,
            y=y_test,
            mode="lines",
            name="Actual Prices",
            line=dict(color="blue", width=2),
        )
    )

    # Add predicted values
    fig.add_trace(
        go.Scatter(
            x=test_indices,
            y=predictions,
            mode="lines",
            name="Predicted Prices",
            line=dict(color="orange", width=2, dash="dash"),
        )
    )

    # Update layout for better readability
    fig.update_layout(
        title="Actual vs Predicted Bitcoin Prices",
        xaxis_title="Date",
        yaxis_title="Closing Price (USD)",
        legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0)", bordercolor="black"),
        template="plotly_white",
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def display_prediction_errors(y_test, predictions):
    st.subheader("Prediction Errors")
    errors = y_test - predictions
    fig = px.bar(
        x=y_test.index,
        y=errors,
        title="Prediction Errors",
        labels={"x": "Date", "y": "Error (USD)"},
        color_discrete_sequence=["crimson"],
    )
    st.plotly_chart(fig, use_container_width=True)


def display_ml_predictions(data):
    st.subheader("Machine Learning Prediction Model")

    # Train the model and get results
    model, X_test, y_test, predictions, mse, mae = train_and_predict(data)

    # Display model evaluation metrics
    st.write("### Model Evaluation Metrics")
    st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")
    st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f}")

    # Enhanced visualization
    display_actual_vs_predicted(y_test, predictions)
    display_prediction_errors(y_test, predictions)

    # Future predictions
    predict_future(model, data, days=365)


def main():
    st.title("Bitcoin Price Prediction")

    if "btc_data" in st.session_state:
        btc_data = st.session_state["btc_data"]
        display_ml_predictions(btc_data)
    else:
        st.warning("No data available. Please go to the Home page to load data.")


if __name__ == "__main__":
    main()
