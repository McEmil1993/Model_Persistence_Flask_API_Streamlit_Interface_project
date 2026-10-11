import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# --- 1. Training data (study hours -> pass/fail) ---
data = pd.DataFrame({
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "passed":      [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
})

X = data[["study_hours"]]
y = data["passed"]

# --- 2. Train the model ---
model = LogisticRegression()
model.fit(X, y)

# --- 3. Save the model with a new filename ---
MODEL_FILE = "student_pass_model.joblib"
joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")