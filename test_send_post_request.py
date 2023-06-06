import requests
import json

url = 'http://127.0.0.1:5000/prices'
data = [
    {
        "date": "2023-05-02",
        "price": 9000.0
    },
    {
        "date": "2023-05-03",
        "price": 8800.0
    },
    {
        "date": "2023-05-04",
        "price": 8700.0
    },
    {
        "date": "2023-05-05",
        "price": 9000.0
    },
    {
        "date": "2023-05-06",
        "price": 8599.0
    },
    {
        "date": "2023-05-07",
        "price": 9500.0
    },
    {
        "date": "2023-05-08",
        "price": 9999.0
    },
    {
        "date": "2023-05-09",
        "price": 9999.0
    },
    {
        "date": "2023-05-10",
        "price": 9999.0
    },
    {
        "date": "2023-05-11",
        "price": 9999.0
    }
]

response = requests.post(url, json=data)

# Check if the response was successful (status code 200)
if response.status_code == 200:
    try:
        json_data = response.json()
        print(json_data)
    except json.JSONDecodeError:
        print("Error: Failed to parse response as JSON")
else:
    print("Error:", response.status_code)
