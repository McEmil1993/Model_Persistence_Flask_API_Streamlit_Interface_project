"""Train, save, load, and predict with the student pass model."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# File where the trained model is saved.
MODEL_FILE = "student_pass_model.joblib"

# Training data: one row per student.
data = pd.DataFrame({
    "study_hours": [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7],
    "passed":      [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
})

FEATURES = ["study_hours"]

# ----- Train and save -----
model = LogisticRegression()
model.fit(data[FEATURES], data["passed"])

joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# ----- Load -----
loaded_model = joblib.load(MODEL_FILE)

# ----- Predict -----
samples = pd.DataFrame({"study_hours": [2, 5, 8]})

predictions = loaded_model.predict(samples[FEATURES])
probabilities = loaded_model.predict_proba(samples[FEATURES])[:, 1]

print("\nPredictions:")
for hours, pred, prob in zip(samples["study_hours"], predictions, probabilities):
    label = "PASS" if pred == 1 else "FAIL"
    print(f"  {hours} study hours -> {label} (pass probability {prob:.3f})")

# ----- Explanation -----
print("\nWhy the model does not need to be trained again:")
print("  - joblib.dump() saved the learned weights (coef_ and intercept_) to the file.")
print("  - joblib.load() restored the same fitted model with those weights.")
print("  - predict() uses the saved weights to score the new study hours.")
print("  - No learning happens during prediction, so fit() does not need to run again")
print("    unless the training data or the features change.")

print("\nWhy the results look this way:")
print("  - The model turns study hours into a score, then into a pass probability")
print("    between 0 and 1.")
print("  - A probability of 0.5 or higher is predicted as PASS.")
print("  - 2 hours: most training students with 2 to 3 hours failed, so FAIL.")
print("  - 5 hours: most training students with 5 hours passed, so PASS is likely.")
print("  - 8 hours: nearly every training student with 6 or more hours passed, so PASS.")
print("  - These are estimates from a small sample, not guarantees for any real student.")