"""Exercise 1 - Model Persistence (study hours only)."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model.joblib"   # new filename

# ---- 1. Training data ----
data = pd.DataFrame({
    "study_hours": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8],
    "passed":      [0,   0, 0,   0, 0,   0, 1,   1, 1, 1, 1, 1],
})
X = data[["study_hours"]]
y = data["passed"]

# ---- 2. Train ----
model = LogisticRegression()
model.fit(X, y)

# ---- 3. Save with joblib.dump() ----
joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# ---- 4. Load with joblib.load() (no retraining!) ----
loaded_model = joblib.load(MODEL_FILE)
print("Model loaded from file.\n")

# ---- 5. Three predictions with different study hours ----
for hours in [1, 3.5, 6]:
    sample = pd.DataFrame({"study_hours": [hours]})
    pred = loaded_model.predict(sample)[0]
    prob = loaded_model.predict_proba(sample)[0][1]
    print(f"Study hours: {hours:>4} -> {'Pass' if pred == 1 else 'Fail'} "
          f"(probability of passing: {prob:.2f})")

# ---- 6. Explanation ----
print("""
Why no retraining is needed:
The learned parameters (coefficient and intercept) were stored inside
student_pass_model.joblib when we called joblib.dump(). Loading the file
restores the exact same trained model, so predict() just applies the
already-learned pattern to new input. Training only needs to happen again
if the data changes or we want to improve the model.
""")
