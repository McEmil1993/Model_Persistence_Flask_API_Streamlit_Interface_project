import requests

url = "http://127.0.0.1:5000/predict"

# Valid test
r1 = requests.post(url, json={"study_hours": 7, "attendance_percentage": 85})
print("VALID TEST")
print("Status:", r1.status_code)
print(r1.json())

# Invalid test (attendance is over 100)
r2 = requests.post(url, json={"study_hours": 7, "attendance_percentage": 150})
print("INVALID TEST")
print("Status:", r2.status_code)
print(r2.json())