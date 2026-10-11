import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model_attendance.joblib"

rng = np.random.default_rng(42)
n = 300
hours = rng.uniform(0, 10, n)
attendance = rng.uniform(40, 100, n)

score = 0.6 * hours + 0.05 * (attendance - 70) + rng.normal(0, 0.8, n)
passed = (score > 3).astype(int)

X = np.column_stack([hours, attendance])
model = LogisticRegression(max_iter=1000)
model.fit(X, passed)
print(f"Training accuracy: {model.score(X, passed):.2f}")

joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")
