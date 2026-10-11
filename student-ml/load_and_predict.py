import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_pass_model.joblib")

# Load the saved model (NO training happens here)
model = joblib.load(MODEL_PATH)

# Three predictions using different study hours
for hours in [1, 5, 9]:
    X = pd.DataFrame({"study_hours": [hours]})
    result = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    label = "PASS" if result == 1 else "FAIL"
    print(f"Study hours = {hours}: {label} (probability of passing = {prob:.2f})")
