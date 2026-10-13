"""
Exercise 1 - Model Persistence
Trains a Pass/Fail model using study hours only, SAVES it with joblib.dump(),
then LOADS it with joblib.load() and makes 3 predictions WITHOUT retraining.
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model.joblib"   # <-- new filename

# ---------- 1. Training data ----------
data = pd.DataFrame({
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "passed":      [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],   # 0 = Fail, 1 = Pass
})
X = data[["study_hours"]]
y = data["passed"]

# ---------- 2. Train ----------
model = LogisticRegression()
model.fit(X, y)

# ---------- 3. Save with joblib.dump() ----------
joblib.dump(model, MODEL_FILE)
print(f"Model trained and saved as '{MODEL_FILE}'")

# ---------- 4. Load with joblib.load() ----------
loaded_model = joblib.load(MODEL_FILE)
print("Model loaded from file (no training done here).\n")

# ---------- 5. Three predictions with different study hours ----------
for hours in [1.5, 5, 9]:
    sample = pd.DataFrame({"study_hours": [hours]})
    pred = loaded_model.predict(sample)[0]
    prob = loaded_model.predict_proba(sample)[0][1]
    label = "Pass" if pred == 1 else "Fail"
    print(f"Study hours: {hours:>4} -> {label} (pass probability: {prob:.2f})")
