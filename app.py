import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os


# In your Streamlit app:
model_dir = "lstm_models"  # Directly use the directory with .h5 models

# Load models
for filename in os.listdir(model_dir):
    if filename.endswith(".h5"):
        try:
            ticker = filename.split("_")[0]
            model_path = os.path.join(model_dir, filename)
            ticker_models[ticker] = load_model(
                model_path,
                compile=False,  # Disable compilation
                custom_objects={'InputLayer': None}  # Bypass input layer issues
            )
            print(f"Successfully loaded {ticker} model")
        except Exception as e:
            st.error(f"Error loading {filename}: {str(e)}")
            
# Streamlit app title with emoji
st.title("📈 Nigerian Stock Price Predictor App 📊")

# Dropdown menu to select a stock ticker
stock_tickers = list(ticker_models.keys())  # List of available tickers
selected_stock = st.selectbox("Select a Stock Ticker:", stock_tickers)

# Input for the number of days for forecast
forecast_days = st.number_input("Enter the Number of Days for Forecast:", min_value=1, max_value=365, value=30)

# Fetch historical data for the selected stock from the preprocessed dataset
try:
    # Extract the full data for the selected stock
    full_data = df[df['Ticker'] == selected_stock][['Price']].rename(columns={'Price': 'Close'})
    
    if full_data.empty:
        st.error(f"No data available for {selected_stock}. Please try another ticker.")
    else:
        st.success(f"Data for {selected_stock} loaded successfully! ✅")

        # Display stock data
        st.subheader(f"Stock Data for {selected_stock} 📈")
        st.write(full_data)

        # Rolling Average Visualizations
        st.subheader(f"Rolling Averages for {selected_stock} 🔄")

        def plot_rolling_average(data, window, title):
            rolling_avg = data['Close'].rolling(window).mean()
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(data.index, data['Close'], label="Original Close Price", color="blue")
            ax.plot(rolling_avg.index, rolling_avg, label=f"{window}-Day Rolling Avg", color="orange")
            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True)
            return fig

        # Plot 100-day rolling average
        st.pyplot(plot_rolling_average(full_data, 100, f"{selected_stock} - 100-Day Rolling Average"))

        # Plot 200-day rolling average
        st.pyplot(plot_rolling_average(full_data, 200, f"{selected_stock} - 200-Day Rolling Average"))

        # Plot 250-day rolling average
        st.pyplot(plot_rolling_average(full_data, 250, f"{selected_stock} - 250-Day Rolling Average"))

        # Combine 100-day and 250-day rolling averages
        st.subheader(f"100-Day vs 250-Day Rolling Averages for {selected_stock} 🔄")
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(full_data.index, full_data['Close'].rolling(100).mean(), label="100-Day Rolling Avg", color="green")
        ax.plot(full_data.index, full_data['Close'].rolling(250).mean(), label="250-Day Rolling Avg", color="red")
        ax.set_title(f"{selected_stock} - 100-Day vs 250-Day Rolling Averages")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Forecast future prices using the pre-trained LSTM model
        st.subheader(f"Forecasting {forecast_days} Days Ahead for {selected_stock} 📅")

        # Prepare the last 7 days of data for forecasting
        look_back = 7
        last_data = full_data['Close'].values[-look_back:]
        last_data = last_data.reshape((1, look_back, 1))  # Reshape to match LSTM input format

        # Load the pre-trained LSTM model for the selected stock
        model_lstm = ticker_models[selected_stock]

        # Generate predictions for the specified forecast horizon
        predictions = []
        current_input = last_data.copy()

        for _ in range(forecast_days):
            pred = model_lstm.predict(current_input)[0][0]  # Predict one day ahead
            predictions.append(pred)
            # Update the input for the next prediction
            current_input = np.append(current_input[:, 1:, :], [[pred]], axis=1)

        # Create a DataFrame for forecasted prices
        forecast_dates = pd.date_range(start=full_data.index[-1], periods=forecast_days + 1, freq='D')[1:]
        forecast_data = pd.DataFrame(
            {"Forecasted Close": predictions},
            index=forecast_dates
        )

        # Display forecasted prices
        st.subheader(f"Forecasted Prices for {forecast_days} Days Ahead 📅")
        st.write(forecast_data)

        # Plot original close price vs forecasted prices
        st.subheader(f"Original Close Price vs Forecasted Prices for {forecast_days} Days Ahead 📊")
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(full_data.index, full_data['Close'], label="Original Close Price", color="blue")
        ax.plot(forecast_data.index, forecast_data['Forecasted Close'], label="Forecasted Prices", color="orange", linestyle="--")
        ax.set_title(f"{selected_stock} - Original vs Forecasted Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

except Exception as e:
    st.error(f"Error processing data for {selected_stock}: {e}")
