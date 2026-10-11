import requests

url = "http://127.0.0.1:5000/predict"

payload = {
    "study_hours": 6,
    "attendance_percentage": 85
}

try:
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        print("Success! Response from Flask API:")
        print(response.json())
    else:
        print(f"Error {response.status_code}:")
        print(response.json())

except requests.exceptions.ConnectionError:
    print("Connection failed! Make sure your Flask app (app.py) is running in another terminal.")