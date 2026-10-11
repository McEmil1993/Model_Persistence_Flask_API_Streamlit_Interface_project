import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model.joblib"

rng = np.random.default_rng(42)
hours = rng.uniform(0, 10, 200).reshape(-1, 1)
passed = (hours.ravel() + rng.normal(0, 1.2, 200) > 5).astype(int)

model = LogisticRegression()
model.fit(hours, passed)
print(f"Training accuracy: {model.score(hours, passed):.2f}")

joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")
