"""Exercise 1 - Model Persistence
Trains a simple model (study hours -> Pass/Fail), saves it with joblib.dump(),
loads it back with joblib.load(), and makes 3 predictions without retraining.
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model.joblib"   # new filename

# ---------- 1. Training data ----------
data = pd.DataFrame({
    "study_hours": [0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 10],
    "passed":      [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
})
X = data[["study_hours"]]
y = data["passed"]

# ---------- 2. Train ----------
model = LogisticRegression()
model.fit(X, y)

# ---------- 3. Save (model persistence) ----------
joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# ---------- 4. Load the saved model (no training happens here) ----------
loaded_model = joblib.load(MODEL_FILE)
print("Model loaded from file.\n")

# ---------- 5. Three predictions with different study hours ----------
for hours in [1, 4, 8]:
    new_student = pd.DataFrame({"study_hours": [hours]})
    pred = loaded_model.predict(new_student)[0]
    prob = loaded_model.predict_proba(new_student)[0][1]
    label = "Pass" if pred == 1 else "Fail"
    print(f"Study hours: {hours:>2} -> {label} (probability of passing: {prob:.2f})")

print("""
Why no retraining is needed:
Training (fit) is where the model learns its parameters (the coefficients and
intercept) from the data. joblib.dump() stores those learned parameters in a
file. joblib.load() restores the exact same trained model, so predict() just
applies the already-learned rule to new input. No learning step is repeated.
""")
