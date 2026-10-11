"""Exercise 1 - Model Persistence (study hours only)."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from data import load_data

MODEL_FILE = "student_pass_model.joblib"   # new filename

# ---- 1. Train ----
df = load_data()
X = df[["study_hours"]]
y = df["passed"]

model = LogisticRegression()
model.fit(X, y)

# ---- 2. Save with joblib.dump() ----
joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# ---- 3. Load with joblib.load() ----
loaded_model = joblib.load(MODEL_FILE)
print("Model loaded back from disk")

# ---- 4. Three predictions with different study hours ----
for hours in (1.0, 4.0, 8.0):
    sample = pd.DataFrame({"study_hours": [hours]})
    pred = loaded_model.predict(sample)[0]
    prob = loaded_model.predict_proba(sample)[0][1]
    label = "PASS" if pred == 1 else "FAIL"
    print(f"Study hours: {hours:>4} -> {label} (pass probability {prob:.2f})")

# ---- Why no retraining is needed ----
# Training (fit) is the step that learns the coefficients and intercept from
# the data. joblib.dump() stores the fitted model object, including those
# learned values. joblib.load() restores that object exactly, so predict()
# only applies the stored values to the new input. Nothing is learned during
# prediction, so there is no need to call fit() again.
