"""Exercise 1 - load the saved model and predict WITHOUT retraining."""
import joblib
import pandas as pd

model = joblib.load("student_pass_model.joblib")

for hours in [1, 4, 7]:
    X = pd.DataFrame({"study_hours": [hours]})
    result = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    print(f"{hours} study hours -> {'PASS' if result == 1 else 'FAIL'} "
          f"(pass probability: {prob:.0%})")
