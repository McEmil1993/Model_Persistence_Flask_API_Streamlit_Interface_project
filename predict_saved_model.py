"""Exercise 1 - load the saved model and make three predictions.

No training happens here: we only call joblib.load() and model.predict().
"""
import joblib
import pandas as pd

MODEL_FILE = "student_pass_model.joblib"

model = joblib.load(MODEL_FILE)  # load the already-trained model from disk

for hours in (1.0, 5.0, 9.0):
    X = pd.DataFrame({"study_hours": [hours]})
    label = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])
    result = "PASS" if label == 1 else "FAIL"
    print(f"Study hours: {hours:>4} -> {result} (pass probability {prob:.2f})")

print(
    "\nWhy no retraining? joblib.dump() stored the fitted model, including its "
    "learned coefficients and scaler. joblib.load() restores that exact state, "
    "so predict() only applies the learned parameters to new inputs."
)
