import json
import urllib.request
import urllib.error

url = "http://127.0.0.1:5000/predict"


def send(payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            print(r.status, r.read().decode())
    except urllib.error.HTTPError as e:
        print(e.code, e.read().decode())


print("Test 1: valid input")
send({"study_hours": 6, "attendance_percentage": 85})

print("\nTest 2: attendance = 120 (dapat mag-error)")
send({"study_hours": 6, "attendance_percentage": 120})