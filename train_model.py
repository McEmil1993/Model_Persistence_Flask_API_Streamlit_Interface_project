"""
Exercise 1 - Model Persistence
Trains a model and saves it as student_pass_model.joblib
"""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# ---- Training Data ----
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([0,0,0,0,1,1,1,1,1,1])

# ---- Train ----
model = LogisticRegression()
model.fit(X, y)
print("[OK] Model trained.")

# ---- Save ----
joblib.dump(model, "student_pass_model.joblib")
print("[OK] Saved to student_pass_model.joblib")

# ---- Load ----
loaded_model = joblib.load("student_pass_model.joblib")
print("[OK] Loaded from disk.")

# ---- 3 Predictions ----
for h in [2, 5, 8]:
    pred = loaded_model.predict([[h]])[0]
    label = "Pass" if pred == 1 else "Fail"
    print(f"Study hours = {h}  ->  {label}")

print("\nEXPLANATION: The model does NOT need to be retrained because")
print("all learned parameters are stored inside the .joblib file.")
