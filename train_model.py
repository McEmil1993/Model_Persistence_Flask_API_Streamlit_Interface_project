"""Exercise 1 - Model persistence (study hours only)."""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model.joblib"

# ---- Training data: hours studied -> 0 = fail, 1 = pass ----
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

# ---- Save the trained model ----
joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# ---- Load it back (no retraining!) ----
loaded_model = joblib.load(MODEL_FILE)

# ---- Three predictions with different study hours ----
for hours in (2, 5.5, 9):
    pred = loaded_model.predict([[hours]])[0]
    prob = loaded_model.predict_proba([[hours]])[0][1]
    label = "PASS" if pred == 1 else "FAIL"
    print(f"Study hours: {hours:>4} -> {label} (pass probability: {prob:.2f})")

# Why no retraining? Training (fit) only happens once. It finds the model's
# learned parameters (coefficients + intercept). joblib.dump() stores those
# parameters on disk, and joblib.load() restores the exact same fitted model.
# Prediction only plugs new inputs into the already-learned formula.
