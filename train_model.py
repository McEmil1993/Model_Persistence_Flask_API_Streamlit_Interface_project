"""Exercise 1 (+ retraining for Exercise 2): train, save, load, predict."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_FILE = "student_pass_model.joblib"
FEATURES = ["study_hours", "attendance_percentage"]

# ---- 1. Train -------------------------------------------------------------
df = pd.read_csv("data.csv")
X = df[FEATURES]
y = df["passed"]  # 1 = Pass, 0 = Fail

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X, y)
print(f"Training accuracy: {model.score(X, y):.2f}")

# ---- 2. Save with joblib.dump() ------------------------------------------
joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# ---- 3. Load with joblib.load() ------------------------------------------
loaded_model = joblib.load(MODEL_FILE)
print(f"Model loaded from {MODEL_FILE}")

# ---- 4. Three predictions with different study hours ---------------------
# Attendance is held at 85% so only study hours changes.
test_students = pd.DataFrame(
    {"study_hours": [1, 5, 9], "attendance_percentage": [85, 85, 85]}
)
predictions = loaded_model.predict(test_students)
for hours, pred in zip(test_students["study_hours"], predictions):
    print(f"Study hours: {hours} -> {'Pass' if pred == 1 else 'Fail'}")

# ---- 5. Why no retraining is needed --------------------------------------
# Training is where the model learns its parameters (the coefficients and
# intercept of the logistic regression, plus the scaler's mean and spread).
# joblib.dump() stores those learned values in the file. joblib.load()
# restores the exact same trained model, so predict() only applies the stored
# values to the new inputs. No .fit() call, no training data needed.
