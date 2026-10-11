"""Exercise 1 - Model Persistence (Rayla)

Loads the saved model and asks a helper function for three predictions.
This file must NOT retrain the model - there is no fit() call here.

Run:  python predict_from_file.py
"""

import joblib
import numpy as np

MODEL_FILE = "rayla_pass_model.joblib"

# Load ONLY - no training happens in this file
model = joblib.load(MODEL_FILE)


def guess(hours):
    """Return Pass/Fail for one study-hours value."""
    features = np.array([[hours]], dtype=float)  # one-row 2D array
    return "Pass" if int(model.predict(features)[0]) == 1 else "Fail"


# Low, medium, and high study hours
samples = {"low (1.0 h)": 1.0, "medium (5.0 h)": 5.0, "high (9.0 h)": 9.0}

for label, value in samples.items():
    print(f"{label:>15}  ->  {guess(value)}")
