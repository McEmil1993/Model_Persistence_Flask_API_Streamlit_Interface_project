import joblib
import pandas as pd

# Load the saved model. No training happens in this file.
model = joblib.load("student_pass_model.joblib")

for hours in [1, 4.5, 7]:
    X = pd.DataFrame({"study_hours": [hours]})
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    print(f"{hours} hours -> {'Pass' if pred == 1 else 'Fail'} (probability of passing: {prob:.2f})")