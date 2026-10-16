import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0,   0,   0,   0,   1,   1,   1,   1,   1,   1])

model = LogisticRegression()
model.fit(X, y)
print("[OK] Model trained successfully.")

joblib.dump(model, "student_pass_model.joblib")
print("[OK] Model saved as student_pass_model.joblib.")

loaded_model = joblib.load("student_pass_model.joblib")
print("[OK] Model loaded from disk.")

study_hours_list = [2, 5, 9]
print("\n--- Predictions (no retraining needed) ---")
for hours in study_hours_list:
    prediction = loaded_model.predict([[hours]])[0]
    probability = loaded_model.predict_proba([[hours]])[0][1]
    label = "Pass" if prediction == 1 else "Fail"
    print(f"Study Hours: {hours:>2}  ->  Prediction: {label:<4}  (Pass probability: {probability:.2%})")