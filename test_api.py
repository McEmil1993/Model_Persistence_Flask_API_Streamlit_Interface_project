import json
from app import app

client = app.test_client()

# Valid request
response = client.post(
    "/predict",
    json={"study_hours": 6, "attendance_percentage": 90},
)
assert response.status_code == 200
payload = response.get_json()
assert payload["study_hours"] == 6.0
assert payload["attendance_percentage"] == 90.0
assert payload["prediction_label"] in {"Pass", "Fail"}

# Invalid attendance
response = client.post(
    "/predict",
    json={"study_hours": 6, "attendance_percentage": 101},
)
assert response.status_code == 400
assert "between 0 and 100" in response.get_json()["error"]

print("All API tests passed.")