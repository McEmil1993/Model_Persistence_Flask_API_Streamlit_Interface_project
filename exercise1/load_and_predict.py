"""Exercise 1 - Model Persistence (Rusel)

Loads the saved model and makes three predictions.
This file must NOT retrain the model - there is no fit() call here.

Run:  python load_and_predict.py
"""

import joblib
import pandas as pd

MODEL_FILE = "rusel_pass_model.joblib"

# Load ONLY - no training happens in this file
model = joblib.load(MODEL_FILE)

# Low, medium, and high study hours
hours = [1.5, 5.0, 9.5]

rows = []
for h in hours:
    row = pd.DataFrame({"study_hours": [h]})  # one-row DataFrame
    prediction = model.predict(row)[0]
    label = "Pass" if prediction == 1 else "Fail"
    rows.append({"study_hours": h, "prediction": label})

print(pd.DataFrame(rows).to_string(index=False))
