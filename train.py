import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

# --- Data ---
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "passed":      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
}
df = pd.DataFrame(data)

X = df[["study_hours"]]     # 2D DataFrame
y = df["passed"]

# --- Train ---
model = LogisticRegression()
model.fit(X, y)

# --- Save ---
joblib.dump(model, "student_pass_model.joblib")
print("Model saved as student_pass_model.joblib")

# --- Load ---
loaded_model = joblib.load("student_pass_model.joblib")
print("Model loaded successfully")

# --- Predict (fixed) ---
test_hours = [2, 5.5, 9]

for h in test_hours:
    X_input = np.array([[h]])          # 2D: shape (1, 1)
    prediction = loaded_model.predict(X_input)[0]
    probability = loaded_model.predict_proba(X_input)[0][1]
    result = "PASS" if prediction == 1 else "FAIL"
    print(f"Study hours: {h:>4} | Prediction: {result} | "
          f"Pass probability: {probability:.2f}")