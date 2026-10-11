from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model_v2.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Run 'py train_model_v2.py' first.")

model = joblib.load(MODEL_PATH)
print("Model loaded successfully.\n")

for hours, attendance in [(2, 50), (5, 75), (9, 95)]:
    sample = pd.DataFrame({
        "study_hours": [hours],
        "attendance_percentage": [attendance],
    })
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]
    result = "PASS" if prediction == 1 else "FAIL"
    print(
        f"Study hours: {hours}, Attendance: {attendance}% -> {result} "
        f"(pass probability: {probability:.2f})"
    )