import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_predict_pass(client):
    response = client.post("/predict", json={"study_hours": 8, "attendance_percentage": 90})
    assert response.status_code == 200
    assert response.get_json()["prediction"] == "Pass"


def test_predict_fail(client):
    response = client.post("/predict", json={"study_hours": 1, "attendance_percentage": 20})
    assert response.status_code == 200
    assert response.get_json()["prediction"] == "Fail"


def test_attendance_matters(client):
    # Same study hours, but high attendance tips the prediction to Pass
    low = client.post("/predict", json={"study_hours": 3, "attendance_percentage": 40}).get_json()
    high = client.post("/predict", json={"study_hours": 3, "attendance_percentage": 80}).get_json()
    assert low["prediction"] == "Fail"
    assert high["prediction"] == "Pass"


def test_missing_study_hours(client):
    response = client.post("/predict", json={"attendance_percentage": 80})
    assert response.status_code == 400
    assert "study_hours" in response.get_json()["error"]


def test_missing_attendance(client):
    response = client.post("/predict", json={"study_hours": 5})
    assert response.status_code == 400
    assert "attendance_percentage" in response.get_json()["error"]


def test_attendance_out_of_range(client):
    response = client.post("/predict", json={"study_hours": 5, "attendance_percentage": 150})
    assert response.status_code == 400


def test_non_numeric_study_hours(client):
    response = client.post("/predict", json={"study_hours": "many", "attendance_percentage": 80})
    assert response.status_code == 400


def test_non_json_body(client):
    response = client.post("/predict", data="not json", content_type="text/plain")
    assert response.status_code == 400


def test_root_gives_usage(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.get_json()
    assert data["usage"]["endpoint"] == "/predict"
    assert data["usage"]["method"] == "POST"


def test_unknown_route_returns_json_404(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert "error" in response.get_json()
