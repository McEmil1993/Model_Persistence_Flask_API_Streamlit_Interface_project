"""Automated tests. Run with:  python -m pytest -v"""
import pytest

from app import app


@pytest.fixture
def client():
    return app.test_client()


def post(client, payload):
    return client.post("/predict", json=payload)


def test_valid_request(client):
    r = post(client, {"study_hours": 8, "attendance_percentage": 90})
    assert r.status_code == 200
    body = r.get_json()
    assert body["prediction"] == "pass"
    assert body["study_hours"] == 8 and body["attendance_percentage"] == 90


def test_low_inputs_fail(client):
    r = post(client, {"study_hours": 1, "attendance_percentage": 45})
    assert r.get_json()["prediction"] == "fail"


@pytest.mark.parametrize("bad", [-1, 101, 150])
def test_attendance_out_of_range(client, bad):
    r = post(client, {"study_hours": 5, "attendance_percentage": bad})
    assert r.status_code == 400


def test_missing_field(client):
    assert post(client, {"study_hours": 5}).status_code == 400


def test_non_numeric(client):
    assert post(client, {"study_hours": "abc", "attendance_percentage": 80}).status_code == 400


def test_home_page_loads(client):
    assert client.get("/").status_code == 200


def upload(client, text):
    import io
    return client.post("/predict-batch", data={"file": (io.BytesIO(text.encode()), "c.csv")},
                       content_type="multipart/form-data")


def test_batch_scores_and_flags_invalid_rows(client):
    csv = "name,study_hours,attendance_percentage\nAna,8,90\nBen,1,45\nCy,5,150\n"
    body = upload(client, csv).get_json()
    assert body["summary"] == {"pass": 1, "fail": 1, "invalid": 1}
    assert [r["prediction"] for r in body["results"]] == ["pass", "fail", "invalid"]


def test_batch_missing_column(client):
    assert upload(client, "name,study_hours\nAna,5\n").status_code == 400


def test_batch_no_file(client):
    assert client.post("/predict-batch").status_code == 400


def test_unknown_route_is_json(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.get_json()["error"]


def upload_bytes(client, raw):
    import io
    return client.post("/predict-batch", data={"file": (io.BytesIO(raw), "c.csv")},
                       content_type="multipart/form-data")


def test_batch_excel_utf8_bom(client):
    raw = "\ufeffname,study_hours,attendance_percentage\nAna,8,90\n".encode("utf-8")
    r = upload_bytes(client, raw)
    assert r.status_code == 200 and r.get_json()["summary"]["pass"] == 1


def test_batch_windows_encoding_and_semicolons(client):
    raw = "name;study_hours;attendance_percentage\nJos\u00e9;8;90\n".encode("cp1252")
    r = upload_bytes(client, raw)
    assert r.status_code == 200 and r.get_json()["results"][0]["name"] == "Jos\u00e9"


def test_batch_friendly_headers_and_percent_sign(client):
    raw = b"Student Name,Study Hours,Attendance\nAna,8,90%\n"
    r = upload_bytes(client, raw)
    assert r.status_code == 200 and r.get_json()["results"][0]["attendance_percentage"] == 90


def test_batch_returns_model_curve(client):
    body = upload(client, "name,study_hours,attendance_percentage\nAna,8,90\nBen,1,50\n").get_json()
    assert len(body["curve"]) == 49 and body["curve_attendance"] == 70
    probs = [c["pass_probability"] for c in body["curve"]]
    assert probs == sorted(probs) and probs[0] < 0.5 < probs[-1]


def test_single_prediction_includes_curve(client):
    body = post(client, {"study_hours": 5, "attendance_percentage": 85}).get_json()
    assert len(body["curve"]) == 49
    assert body["curve"][0]["pass_probability"] < 0.5 < body["curve"][-1]["pass_probability"]