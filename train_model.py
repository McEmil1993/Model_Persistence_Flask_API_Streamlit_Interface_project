"""Exercise 1 - Model Persistence (study hours only)."""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model.joblib"

# ---------- 1. TRAIN ----------
# Training data: study hours -> 0 = Fail, 1 = Pass
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

# ---------- 2. SAVE (joblib.dump) ----------
joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# ---------- 3. LOAD (joblib.load) ----------
# Pretend this is a brand new program: no training happens below.
loaded_model = joblib.load(MODEL_FILE)
print(f"Model loaded from {MODEL_FILE}")

# ---------- 4. THREE PREDICTIONS ----------
for hours in [2, 5, 9]:
    pred = loaded_model.predict([[hours]])[0]
    prob = loaded_model.predict_proba([[hours]])[0][1]
    label = "Pass" if pred == 1 else "Fail"
    print(f"Study hours = {hours}: {label} (pass probability {prob:.2f})")

# ---------- 5. EXPLANATION ----------
print(
    "\nWhy no retraining? Training already happened once and the learned "
    "parameters (coefficients and intercept) were stored inside the .joblib "
    "file. Loading the file restores those parameters, so predict() only "
    "applies the learned formula to new input."
)
