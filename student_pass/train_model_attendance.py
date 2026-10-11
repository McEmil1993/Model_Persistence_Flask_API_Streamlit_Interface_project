"""Exercise 2 (note) - retrain with study hours AND attendance."""
import joblib
from sklearn.linear_model import LogisticRegression

from data import load_data

MODEL_FILE = "student_pass_model_attendance.joblib"

df = load_data()
X = df[["study_hours", "attendance_percentage"]]
y = df["passed"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")
