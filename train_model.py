"""Exercise 1 - Model persistence (study hours only)."""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model.joblib"

# ---- 1. Training data (study hours -> pass = 1 / fail = 0) ----
data = pd.DataFrame({
    "study_hours": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10],
    "passed":      [0,   0, 0,   0, 0,   0, 1,   0, 1,   1, 1, 1, 1, 1, 1],
})
X = data[["study_hours"]]
y = data["passed"]

model = LogisticRegression()
model.fit(X, y)

# ---- 2. Save the model with joblib.dump() ----
joblib.dump(model, MODEL_FILE)
print(f"Model trained and saved to {MODEL_FILE}")

# ---- 3. Load it back with joblib.load() (no retraining) ----
loaded_model = joblib.load(MODEL_FILE)

# ---- 4. Three predictions with different study hours ----
for hours in [1.0, 3.5, 8.0]:
    sample = pd.DataFrame({"study_hours": [hours]})
    pred = loaded_model.predict(sample)[0]
    prob = loaded_model.predict_proba(sample)[0][1]
    print(f"{hours:>4} h -> {'PASS' if pred == 1 else 'FAIL'} (P(pass) = {prob:.2f})")

# ---- 5. Why no retraining is needed ----
# fit() learns the coefficient and intercept, and joblib.dump() stores that
# fitted object (including those learned numbers) on disk. joblib.load()
# restores the same object, so predict() only applies the stored formula
# to new input. Training is the slow, one-off step; prediction is cheap.