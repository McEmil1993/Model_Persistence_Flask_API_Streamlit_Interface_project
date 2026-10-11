import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. Create sample training data (study hours + attendance)
rng = np.random.default_rng(42)
n = 300
study_hours = rng.uniform(0, 10, n)
attendance = rng.uniform(40, 100, n)

# Pass depends on BOTH features (plus a little randomness)
score = 6 * study_hours + 0.5 * attendance + rng.normal(0, 5, n)
passed = (score >= 60).astype(int)

data = pd.DataFrame({
    "study_hours": study_hours,
    "attendance_percentage": attendance,
    "passed": passed,
})

X = data[["study_hours", "attendance_percentage"]]
y = data["passed"]

# 2. Train the model
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X, y)

# 3. Save it
joblib.dump(model, "student_pass_model_v2.joblib")
print("Model saved as student_pass_model_v2.joblib")
print(f"Training accuracy: {model.score(X, y):.2f}")