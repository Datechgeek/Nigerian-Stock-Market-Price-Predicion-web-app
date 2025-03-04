import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Initialize ticker_models dictionary
ticker_models = {}

# Load models with error handling
model_dir = "lstm_models"
if os.path.exists(model_dir):
    for filename in os.listdir(model_dir):
        if filename.endswith(".h5"):
            try:
                ticker = filename.split("_")[0]
                model_path = os.path.join(model_dir, filename)
                ticker_models[ticker] = load_model(
                    model_path,
                    compile=False,
                    custom_objects={'InputLayer': None}
                )
                st.success(f"✅ Successfully loaded {ticker} model")
            except Exception as e:
                st.error(f"❌ Error loading {filename}: {str(e)}")
else:
    st.error(f"🚨 Model directory '{model_dir}' not found!")

# Streamlit app title
st.title("📈 Nigerian Stock Price Predictor App 📊")

# Only show UI if models are loaded
if ticker_models:
    # Dropdown menu
    selected_stock = st.selectbox(
        "Select a Stock Ticker:", 
        options=list(ticker_models.keys())
    
    # Forecast days input
    forecast_days = st.number_input(
        "Enter the Number of Days for Forecast:", 
        min_value=1, 
        max_value=365, 
        value=30
    )

    try:
        # Load data (replace 'df' with your actual DataFrame)
        # Ensure you have your data loading logic here
        full_data = df[df['Ticker'] == selected_stock][['Price']].rename(columns={'Price': 'Close'})
        
        if full_data.empty:
            st.warning(f"⚠️ No data available for {selected_stock}")
        else:
            # Data Display Section
            st.success(f"✅ Data for {selected_stock} loaded successfully!")
            st.subheader(f"{selected_stock} Historical Prices 📈")
            st.line_chart(full_data['Close'])

            # Rolling Averages Section
            st.subheader("Technical Analysis 🔄")
            windows = [50, 100, 200]
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(full_data.index, full_data['Close'], label='Close Price', color='navy')
            
            for window in windows:
                rolling_avg = full_data['Close'].rolling(window).mean()
                ax.plot(rolling_avg.index, rolling_avg, 
                        label=f'{window}-Day MA', 
                        linestyle='--')
            
            ax.set_title(f"{selected_stock} Moving Averages")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Forecasting Section
            st.subheader(f"{forecast_days}-Day Price Forecast 📅")
            look_back = 7
            last_data = full_data['Close'].values[-look_back:]
            
            if len(last_data) >= look_back:
                current_input = last_data.reshape((1, look_back, 1))
                predictions = []
                
                for _ in range(forecast_days):
                    pred = ticker_models[selected_stock].predict(current_input)[0][0]
                    predictions.append(pred)
                    current_input = np.append(current_input[:, 1:, :], [[pred]], axis=1)
                
                # Create forecast DataFrame
                forecast_dates = pd.date_range(
                    start=full_data.index[-1] + pd.DateOffset(1),
                    periods=forecast_days
                )
                forecast_df = pd.DataFrame(
                    {'Forecast': predictions},
                    index=forecast_dates
                )
                
                # Display forecast
                st.line_chart(forecast_df)
                st.write("Detailed Forecast Data:")
                st.dataframe(forecast_df.style.format("{:.2f}"))
                
                # Combined plot
                fig, ax = plt.subplots(figsize=(15, 6))
                ax.plot(full_data.index[-60:], full_data['Close'].values[-60:], 
                        label='Historical Price', color='blue')
                ax.plot(forecast_df.index, forecast_df['Forecast'], 
                        label='Forecast', color='orange', linestyle='--')
                ax.set_title(f"{selected_stock} Price Forecast")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("⚠️ Insufficient historical data for forecasting")

    except Exception as e:
        st.error(f"🚨 Error processing {selected_stock}: {str(e)}")
else:
    st.warning("⚠️ No models available. Please train models first.")

# Add footer
st.markdown("---")
st.markdown("### 📊 Market Prediction powered by LSTM Neural Networks")
