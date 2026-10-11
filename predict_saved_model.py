import joblib
import pandas as pd

# --- Load the saved model (no training here) ---
model = joblib.load("student_pass_model.joblib")
print("Model loaded successfully.\n")

# --- Three predictions with different study hours ---
for hours in [2, 5, 9]:
    sample = pd.DataFrame({"study_hours": [hours]})
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]
    result = "PASS" if prediction == 1 else "FAIL"
    print(f"Study hours: {hours} -> {result} (pass probability: {probability:.2f})")