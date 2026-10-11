"""Exercise 1 - LOAD the saved model and make three predictions.

Run:  python load_and_predict.py   (after train_model.py)
Note there is no .fit() call here.
"""
import joblib
import numpy as np

import model_lib  # noqa: F401  (needed so joblib can rebuild the model class)

model = joblib.load("student_pass_model.joblib")

for h in (2, 5, 8):
    X = np.array([[h]])
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0])
    print(f"Study hours: {h:>2} -> {'PASS' if pred else 'FAIL'} (pass probability {prob:.2f})")
