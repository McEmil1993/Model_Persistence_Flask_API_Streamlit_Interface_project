"""Exercise 2 - retrain the model with study hours AND attendance."""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model_v2.joblib"

rng = np.random.default_rng(42)
n = 200
hours = rng.uniform(0, 10, n)
attendance = rng.uniform(30, 100, n)
# Synthetic rule: both hours and attendance help, plus some noise
score = 0.6 * hours + 0.05 * attendance + rng.normal(0, 0.8, n)
passed = (score > 5.5).astype(int)

X = pd.DataFrame({"study_hours": hours, "attendance_percentage": attendance})
model = LogisticRegression(max_iter=1000).fit(X, passed)
joblib.dump(model, MODEL_FILE)
print(f"Saved {MODEL_FILE}; training accuracy = {model.score(X, passed):.2f}")