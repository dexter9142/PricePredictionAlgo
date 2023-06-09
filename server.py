import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

app = Flask(__name__)

class PriceResponse:
    def __init__(self, date, price):
        self.date = date
        self.price = price

price_responses = []  # Array to store PriceResponse objects
dates = []  # Array to store dates
prices = []
future_dates = []

def fetch_data():
    # Retrieve data from source and return the dates and prices
    date_strings = [
        '16-04-23', '17-04-23', '18-04-23', '19-04-23', '20-04-23', '21-04-23', '22-04-23', '23-04-23', '24-04-23',
        '25-04-23'
    ]
    dates = [datetime.strptime(date, "%d-%m-%y") for date in date_strings]
    prices = np.array([200, 280, 210, 350, 440, 400, 390, 450, 450, 475])
    return dates, prices

def forecast_prices():
    global dates, prices, future_dates

    today = datetime.now()
    future_dates = [(today + timedelta(days=i)).strftime("%d-%m-%y") for i in range(1, 14)]

    if not dates or not prices:
        dates, prices = fetch_data()

    y = np.array(prices).reshape(-1, 1)

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

    # Append the forecasted prices to the array of prices starting from the next day
    forecasted_prices = np.append(prices, next_prices_sarima[1:])

    # Create a list of dictionaries containing the dates and forecasted prices
    data_list = []
    for date, price in zip(dates, forecasted_prices):
        formatted_date = date.strftime("%Y-%m-%d")
        data_list.append({
            "date": formatted_date,
            "price": price
        })

    return data_list

@app.route('/data')
def get_data():
    json_data = json.dumps(forecast_prices())
    return json_data


@app.route('/prices', methods=['POST'])
def process_prices():
    # Get the JSON data from the request
    payload = request.get_json()
    global dates, prices

    # Extract the dates and prices from the payload
    user_dates = [item['date'] for item in payload]
    user_prices = [item['price'] for item in payload]

    # Convert dates to datetime objects
    user_dates = [datetime.strptime(date, "%Y-%m-%d") for date in user_dates]

    # Perform any necessary data preprocessing or modeling steps here
    # ...

    # Create SARIMA model using user-provided prices
    y = np.array(user_prices).reshape(-1, 1)
    sarima_model = SARIMAX(y.flatten(), order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    sarima_model_fit = sarima_model.fit()

    # Forecast prices for the provided dates using SARIMA
    forecasted_prices = sarima_model_fit.forecast(steps=len(user_dates)).tolist()

    # Format the forecasted prices to have one digit after the decimal point
    forecasted_prices = [round(price, 1) for price in forecasted_prices]

    # Create a list of dictionaries containing the dates and forecasted prices
    response = []

    # Append the user-provided prices to the response
    for date, price in zip(user_dates, user_prices):
        formatted_date = date.strftime("%Y-%m-%d")
        response.append({
            "date": formatted_date,
            "price": price
        })

    # Get the last date from the user-provided dates
    last_date = user_dates[-1]

    # Generate the next day's date
    next_date = last_date + timedelta(days=1)

    # Append the forecasted prices to the response starting from the next day
    for price in forecasted_prices:
        formatted_date = next_date.strftime("%Y-%m-%d")
        response.append({
            "date": formatted_date,
            "price": price
        })
        next_date += timedelta(days=1)

    # Convert the response to JSON
    json_data = json.dumps(response)

    # Return the JSON data
    return json_data




if __name__ == '__main__':
    app.run(debug=True)
