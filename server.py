import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
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

@app.route('/data')
def get_data():
    global dates, prices  # Declare as global variables

    today = datetime.now()
    future_dates = [(today + timedelta(days=i)).strftime("%d-%m-%y") for i in range(1, 14)]

    if not dates or not prices:
        # Activate once data is retrieved
        # dates, prices = fetch_data()
        date_strings = [
            '16-04-23', '17-04-23', '18-04-23', '19-04-23', '20-04-23', '21-04-23', '22-04-23', '23-04-23', '24-04-23',
            '25-04-23'
        ]
        dates = [datetime.strptime(date, "%d-%m-%y") for date in date_strings]
        prices = np.array([200, 280, 210, 350, 440, 400, 390, 450, 450, 475])

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

    # Append the forecasted prices to the array of prices
    future_prices = np.append(prices, next_prices_sarima)
    date_strings = [date.strftime("%d-%m-%y") for date in dates]

    # Plot the actual prices, SARIMA predictions, and XGBoost predictions
    plt.figure(figsize=(10, 6))
    plt.plot(date_strings, prices, marker='o', label='Actual Prices')
    plt.plot(future_dates, future_prices[-13:], marker='o', label='SARIMA Predictions')
    plt.plot(future_dates, next_prices_xgb[-13:], marker='o', label='XGBoost Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price Forecast')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Create a list of dictionaries containing the dates and forecasted prices
    data_list = []
    for date, price in zip(future_dates, next_prices_sarima):
        formatted_date = datetime.strptime(date, "%d-%m-%y").strftime("%Y-%m-%d")
        data_list.append({
            "date": formatted_date,
            "price": price
        })

    # Convert the list of dictionaries to JSON
    json_data = json.dumps(data_list)

    # Return the JSON response
    return json_data

@app.route('/prices', methods=['POST'])
def process_prices():
    # Get the JSON data from the request
    payload = request.get_json()

    # Extract the dates and prices from the payload
    global dates, prices
    dates = [item['date'] for item in payload]
    prices = [item['price'] for item in payload]

    # Convert dates to datetime objects
    dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates]

    # Convert prices to a regular Python list
    prices = list(prices)

    # Perform any necessary data preprocessing or modeling steps here
    # ...

    # Clear the price_responses array
    price_responses.clear()

    # Create PriceResponse objects and append them to the price_responses array
    for date, price in zip(dates, prices):
        price_responses.append(PriceResponse(date.strftime("%Y-%m-%d"), price))

    # Generate the response in the desired format
    response = []
    for price_response in price_responses:
        response.append({
            "date": price_response.date,
            "price": price_response.price
        })

    # Convert the response to JSON and return it
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
