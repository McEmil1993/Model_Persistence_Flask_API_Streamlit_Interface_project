"""Verification test suite for Exercise 1, 2, and 3."""

import json
from pathlib import Path
import joblib
import numpy as np

from app import app
import train_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"


def test_exercise_1():
    print("=== Testing Exercise 1: Model Persistence ===")
    assert MODEL_PATH.exists(), f"Model file {MODEL_PATH} must exist."
    
    # Test loading
    clf = joblib.load(MODEL_PATH)
    assert clf is not None, "Loaded model should not be None."
    
    # Test three predictions with different study hours
    test_cases = [
        (1.5, 55.0, "FAIL"),
        (4.5, 80.0, "PASS"),
        (9.0, 95.0, "PASS"),
    ]
    for hrs, att, expected in test_cases:
        proba = float(clf.predict_proba([[hrs, att]])[0][1])
        verdict = "PASS" if proba >= 0.5 else "FAIL"
        print(f"  Sample: hours={hrs}, attendance={att}% -> {verdict} (P={proba:.4f}) [expected {expected}]")
        assert verdict == expected, f"Expected {expected} for ({hrs}, {att}), got {verdict}"
    print("  [PASS] Exercise 1 passed successfully!\n")


def test_exercise_2():
    print("=== Testing Exercise 2: Flask API ===")
    client = app.test_client()
    
    # 1. Test GET /
    res = client.get("/")
    assert res.status_code == 200, f"GET / returned {res.status_code}"
    
    # 2. Test valid POST /predict
    payload = {"study_hours": 4.5, "attendance_percentage": 80.0}
    res = client.post("/predict", json=payload)
    assert res.status_code == 200, f"POST /predict returned {res.status_code}: {res.data}"
    data = res.get_json()
    assert "study_hours" in data and data["study_hours"] == 4.5
    assert "attendance_percentage" in data and data["attendance_percentage"] == 80.0
    assert "prediction" in data and data["prediction"] == 1
    assert "result" in data and data["result"] == "PASS"
    assert "probability_pass" in data
    print(f"  Valid response: {data}")
    
    # 3. Test invalid attendance (> 100)
    res = client.post("/predict", json={"study_hours": 4.5, "attendance_percentage": 105})
    assert res.status_code == 400
    assert "0-100" in res.get_json()["error"]
    print(f"  Attendance > 100 error handled: {res.get_json()['error']}")

    # 4. Test invalid attendance (< 0)
    res = client.post("/predict", json={"study_hours": 4.5, "attendance_percentage": -10})
    assert res.status_code == 400
    assert "0-100" in res.get_json()["error"]
    print(f"  Attendance < 0 error handled: {res.get_json()['error']}")

    # 5. Test invalid study hours (< 0)
    res = client.post("/predict", json={"study_hours": -2, "attendance_percentage": 75})
    assert res.status_code == 400
    print(f"  Study hours < 0 error handled: {res.get_json()['error']}")

    # 6. Test missing fields
    res = client.post("/predict", json={"study_hours": 5.0})
    assert res.status_code == 400
    print(f"  Missing field error handled: {res.get_json()['error']}")

    print("  [PASS] Exercise 2 passed successfully!\n")


def test_exercise_3_file_inspection():
    print("=== Testing Exercise 3: Streamlit Interface ===")
    app_file = BASE_DIR / "streamlit_app.py"
    assert app_file.exists(), "streamlit_app.py must exist."
    content = app_file.read_text(encoding="utf-8")

    # Check required elements
    assert "st.text_input(" in content or "text_input" in content, "Missing student name input"
    assert "Study hours" in content, "Missing study hours input"
    assert "Run prediction" in content or "button" in content, "Missing prediction button"
    assert "verdict" in content or "result" in content, "Missing result message"
    assert "How was this generated" in content or "explanation" in content.lower(), "Missing explanation"
    assert "bar_chart" in content or "area_chart" in content or "line_chart" in content, "Missing bonus chart"

    print("  [PASS] Exercise 3 code contains all required elements (name input, study hours, button, result, explanation, chart)!\n")


if __name__ == "__main__":
    test_exercise_1()
    test_exercise_2()
    test_exercise_3_file_inspection()
    print("ALL 3 EXERCISES VERIFIED SUCCESSFULLY!")
