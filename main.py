import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import json
from datetime import datetime, timedelta
from flask import Flask, jsonify
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline
def get_data():
    today = datetime.now()
    future_dates = [(today + timedelta(days=i)).strftime("%d-%m-%y") for i in range(1, 14)]

    # Activate once data is retrieved
    # dates, prices = fetch_data()
    dates = np.array(['16-04-23', '17-04-23', '18-04-23', '19-04-23', '20-04-23', '21-04-23', '22-04-23', '23-04-23', '24-04-23', '25-04-23'])
    prices = np.array([200, 280, 210, 350, 440, 400, 390, 450, 450, 475])

    y = prices.reshape(-1, 1)

    # Create SARIMA model
    sarima_model = SARIMAX(y.flatten(), order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    sarima_model_fit = sarima_model.fit()

    # Create XGBoost model
    xgb_model = XGBRegressor(n_estimators=100, max_depth=3)
    xgb_model.fit(y[:-2], y[2:])

    # Forecast future prices using SARIMA
    next_prices_sarima = sarima_model_fit.forecast(steps=13).tolist()

    # Forecast future prices using XGBoost
    xgb_input = np.append(y[-2:], np.array(next_prices_sarima[:-2]).reshape(-1, 1), axis=0)
    next_prices_xgb = xgb_model.predict(xgb_input).tolist()

    # Append the forecasted prices to the array of prices
    future_prices = np.append(prices, next_prices_sarima)

    # Plot the actual prices, SARIMA predictions, and XGBoost predictions
    plt.figure(figsize=(10, 6))
    plt.plot(dates, prices, marker='o', label='Actual Prices')
    plt.plot(future_dates, future_prices[-13:], marker='o', label='SARIMA Predictions')
    plt.plot(future_dates, next_prices_xgb[-13:], marker='o', label='XGBoost Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price Forecast')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Create a list of tuples containing the dates and forecasted prices
    data_list = [(date, price) for date, price in zip(future_dates, future_prices[-13:])]

    # Create a dictionary with the list of tuples as the value
    data_dict = {
        "forecasted_prices": data_list
    }

    # Convert the dictionary to a JSON string
    json_str = json.dumps(data_dict)

    return jsonify(json_str)



get_data()
