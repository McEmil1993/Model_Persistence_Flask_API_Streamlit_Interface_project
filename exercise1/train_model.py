"""Exercise 1 - Model Persistence
Train a pass/fail model from study hours, save it with joblib.dump(),
then load it with joblib.load() and make three predictions.
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model.joblib"   # <- new filename

# --- 1. Training data (study hours -> passed? 1 = pass, 0 = fail) ---
data = pd.DataFrame({
    "study_hours": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8],
    "passed":      [0,   0, 0,   0, 0,   0, 1,   0, 1,   1, 1,   1, 1, 1],
})
X = data[["study_hours"]]
y = data["passed"]

# --- 2. Train ---
model = LogisticRegression()
model.fit(X, y)
print(f"Training accuracy: {model.score(X, y):.2f}")

# --- 3. Save the trained model ---
joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# --- 4. Load it back (no retraining!) ---
loaded_model = joblib.load(MODEL_FILE)
print(f"Model loaded from {MODEL_FILE}")

# --- 5. Make three predictions with different study hours ---
for hours in [1.0, 3.5, 6.0]:
    sample = pd.DataFrame({"study_hours": [hours]})
    pred = loaded_model.predict(sample)[0]
    prob = loaded_model.predict_proba(sample)[0][1]
    print(f"Study hours = {hours:>4}  ->  {'PASS' if pred == 1 else 'FAIL'}  "
          f"(pass probability {prob:.2f})")